from __future__ import annotations

import argparse
import csv
import hashlib
import html
import importlib.util
import json
import os
import random
import re
import subprocess
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
from matplotlib import font_manager


REPO_ROOT = Path(__file__).resolve().parents[1]
CRUD_ROOT = Path(__file__).resolve().parent
OFFICIAL_ROOT = CRUD_ROOT / "crud_rag_official"
SPLIT_DATA_PATH = OFFICIAL_ROOT / "data" / "crud_split" / "split_merged.json"
DOCS_ROOT = OFFICIAL_ROOT / "data" / "80000_docs"
OUTPUT_ROOT = CRUD_ROOT / "outputs"
RUNTIME_ROOT = CRUD_ROOT / "runtime"
CORPUS_ROOT = CRUD_ROOT / "corpus"
BACKEND_ROOT = REPO_ROOT / "backend"
FONT_PATH = REPO_ROOT / "experiments" / ".assets" / "fonts" / "SourceHanSansSC-Regular.otf"
LOCAL_RERANK_MODEL_PATH = BACKEND_ROOT / "data" / "models" / "rerank" / "BAAI--bge-reranker-v2-m3"
NO_CONTEXT_ANSWER = "根据当前检索到的知识库内容，没有找到足够相关的参考资料，因此我暂时无法给出可靠回答。"
REFUSAL_MARKERS = (
    "无法确定",
    "无法回答",
    "无法给出可靠回答",
    "没有找到足够相关",
    "资料不足",
    "没有相关资料",
    "无法从提供的上下文",
    "未提及",
    "未涉及",
    "insufficient context",
    "cannot determine",
)


@dataclass
class CorpusDoc:
    doc_id: int
    title: str
    content: str


@dataclass
class CrudCase:
    case_id: str
    split: str
    query: str
    reference: str
    answers: tuple[str, ...]
    positive_hashes: tuple[str, ...]
    should_refuse: bool = False
    anchor: str = ""
    expected_doc_ids: tuple[int, ...] = ()


@dataclass
class QueryResult:
    system: str
    case_id: str
    split: str
    query: str
    response: str
    source_doc_ids: list[int]
    source_titles: list[str]
    retrieved_contexts: list[str]
    latency_ms: float
    predicted_refusal: bool
    expected_refusal: bool
    accuracy_hit: int
    retrieval_hit: int
    text_similarity: float


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_compare_module():
    return _load_module(REPO_ROOT / "llamaindex_rag_eval" / "compare_rag_systems.py", "crud_compare_helper")


def _load_llamaindex_eval_module():
    return _load_module(REPO_ROOT / "llamaindex_rag_eval" / "llamaindex_rag_eval.py", "crud_llamaindex_eval_helper")


def _normalize_text(value: str) -> str:
    lowered = html.unescape(str(value or "")).strip().lower()
    lowered = re.sub(r"\s+", "", lowered)
    lowered = re.sub(r"[，,。.!！？?；;：:\"'“”‘’（）()【】《》<>]", "", lowered)
    return lowered


def _canonical_text(value: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(str(value or "")).strip())


def _hash_text(value: str) -> str:
    return hashlib.sha1(_canonical_text(value).encode("utf-8")).hexdigest()


def _contains_refusal(text: str) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return True
    if normalized == _normalize_text(NO_CONTEXT_ANSWER):
        return True
    return any(_normalize_text(marker) in normalized for marker in REFUSAL_MARKERS)


def _text_similarity(left: str, right: str) -> float:
    a = _normalize_text(left)
    b = _normalize_text(right)
    if not a or not b:
        return 0.0
    if a == b or a in b or b in a:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def _truncate(text: str, limit: int = 2200) -> str:
    return str(text or "").strip()[:limit]


def _title_from_text(text: str, doc_id: int) -> str:
    cleaned = _canonical_text(text)
    cleaned = re.sub(r"^原标题[:：]\s*", "", cleaned)
    cleaned = re.sub(r"^\[\s*\d{4}.*?\]\s*[，,]?", "", cleaned)
    cleaned = cleaned.replace("正文：", " ").strip()
    return f"CRUD_DOC_{doc_id:05d}_{cleaned[:36]}" if cleaned else f"CRUD_DOC_{doc_id:05d}"


def _iter_corpus_texts() -> Iterable[str]:
    for path in sorted(DOCS_ROOT.iterdir()):
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = _canonical_text(line)
                if text:
                    yield text


def _evenly_sample(items: Sequence[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0 or len(items) <= limit:
        return list(items)
    step = len(items) / limit
    return [items[min(len(items) - 1, int(index * step))] for index in range(limit)]


def _load_split_data() -> dict[str, list[dict[str, Any]]]:
    if not SPLIT_DATA_PATH.exists():
        raise FileNotFoundError(f"未找到 CRUD-RAG 切分数据: {SPLIT_DATA_PATH}")
    raw = json.loads(SPLIT_DATA_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError("CRUD-RAG 切分数据格式异常。")
    return {str(key): list(value) for key, value in raw.items()}


def _positive_hashes_from_texts(texts: Sequence[str]) -> tuple[str, ...]:
    return tuple(_hash_text(text) for text in texts if _canonical_text(text))


def _build_positive_cases(
    split_data: dict[str, list[dict[str, Any]]],
    *,
    summary_samples: int,
    qa_1doc_samples: int,
    qa_2doc_samples: int,
    qa_3doc_samples: int,
    hallu_samples: int,
) -> tuple[list[CrudCase], dict[str, str]]:
    cases: list[CrudCase] = []
    positive_texts_by_hash: dict[str, str] = {}

    def add_case(case: CrudCase, source_texts: Sequence[str]) -> None:
        cases.append(case)
        for text in source_texts:
            normalized = _canonical_text(text)
            if not normalized:
                continue
            positive_texts_by_hash[_hash_text(normalized)] = normalized

    for index, item in enumerate(_evenly_sample(split_data["event_summary"], summary_samples), start=1):
        source_text = _canonical_text(item.get("text", ""))
        event = str(item.get("event") or "").strip()
        summary = str(item.get("summary") or "").strip()
        query = f"请根据知识库概括这件事的核心内容：{event}"
        add_case(
            CrudCase(
                case_id=f"event_summary_{index:03d}",
                split="event_summary",
                query=query,
                reference=summary,
                answers=(summary,),
                positive_hashes=_positive_hashes_from_texts((source_text,)),
                anchor=event or query,
            ),
            (source_text,),
        )

    for split_name, sample_limit, news_keys in (
        ("questanswer_1doc", qa_1doc_samples, ("news1",)),
        ("questanswer_2docs", qa_2doc_samples, ("news1", "news2")),
        ("questanswer_3docs", qa_3doc_samples, ("news1", "news2", "news3")),
    ):
        for index, item in enumerate(_evenly_sample(split_data[split_name], sample_limit), start=1):
            news_texts = tuple(_canonical_text(item.get(key, "")) for key in news_keys)
            answer = str(item.get("answers") or "").strip()
            question = str(item.get("questions") or "").strip()
            event = str(item.get("event") or "").strip()
            add_case(
                CrudCase(
                    case_id=f"{split_name}_{index:03d}",
                    split=split_name,
                    query=question,
                    reference=answer,
                    answers=(answer,),
                    positive_hashes=_positive_hashes_from_texts(news_texts),
                    anchor=event or question,
                ),
                news_texts,
            )

    for index, item in enumerate(_evenly_sample(split_data["hallu_modified"], hallu_samples), start=1):
        beginning = str(item.get("newsBeginning") or "").strip()
        hallucinated = str(item.get("hallucinatedContinuation") or "").strip()
        corrected = str(item.get("hallucinatedMod") or "").strip()
        full_text = _canonical_text(f"{beginning}\n{item.get('newsRemainder', '')}")
        query = (
            "请根据知识库判断并纠正下面这段新闻续写中的错误，只输出纠正后的文本。"
            f"\n新闻开头：{beginning}\n续写：{hallucinated}"
        )
        add_case(
            CrudCase(
                case_id=f"hallu_modified_{index:03d}",
                split="hallu_modified",
                query=query,
                reference=corrected,
                answers=(corrected,),
                positive_hashes=_positive_hashes_from_texts((full_text,)),
                anchor=beginning[:80] or query[:80],
            ),
            (full_text,),
        )

    return cases, positive_texts_by_hash


def _build_negative_cases(anchors: Sequence[str], limit: int) -> list[CrudCase]:
    negatives: list[CrudCase] = []
    for index, anchor in enumerate(_evenly_sample([{"anchor": value} for value in anchors if value], limit), start=1):
        subject = str(anchor["anchor"]).strip()
        query = f"关于“{subject}”，文中负责人的电子邮箱地址是什么？如果资料没有提供，请直接说明无法确定。"
        negatives.append(
            CrudCase(
                case_id=f"negative_rejection_{index:03d}",
                split="negative_rejection",
                query=query,
                reference="",
                answers=(),
                positive_hashes=(),
                should_refuse=True,
                anchor=subject,
            )
        )
    return negatives


def _build_corpus_subset(
    positive_texts_by_hash: dict[str, str],
    *,
    distractor_count: int,
    seed: int,
) -> tuple[list[CorpusDoc], dict[str, int]]:
    positive_hashes = set(positive_texts_by_hash.keys())
    matched_positives: dict[str, str] = {}
    reservoir: list[str] = []
    rng = random.Random(seed)
    seen_distractors = 0

    for text in _iter_corpus_texts():
        text_hash = _hash_text(text)
        if text_hash in positive_hashes and text_hash not in matched_positives:
            matched_positives[text_hash] = text
            continue

        if distractor_count <= 0:
            continue
        seen_distractors += 1
        if len(reservoir) < distractor_count:
            reservoir.append(text)
            continue
        replace_at = rng.randint(0, seen_distractors - 1)
        if replace_at < distractor_count:
            reservoir[replace_at] = text

    docs: list[CorpusDoc] = []
    hash_to_doc_id: dict[str, int] = {}

    def append_doc(text: str) -> None:
        doc_id = len(docs) + 1
        docs.append(CorpusDoc(doc_id=doc_id, title=_title_from_text(text, doc_id), content=text))
        hash_to_doc_id[_hash_text(text)] = doc_id

    for positive_hash, text in positive_texts_by_hash.items():
        append_doc(matched_positives.get(positive_hash, text))

    for text in reservoir:
        text_hash = _hash_text(text)
        if text_hash in hash_to_doc_id:
            continue
        append_doc(text)

    return docs, hash_to_doc_id


def _write_corpus_jsonl(docs: Sequence[CorpusDoc], path: Path) -> None:
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for doc in docs:
            handle.write(json.dumps(asdict(doc), ensure_ascii=False) + "\n")


def _runtime_manifest_path(runtime_dir: Path) -> Path:
    return runtime_dir / "build_manifest.json"


def _build_runtime(runtime_dir: Path, corpus_jsonl: Path, manifest: dict[str, Any]) -> None:
    helper_code = r"""
import asyncio
import json
import sys
from pathlib import Path

repo_root = Path(sys.argv[1]).resolve()
runtime_dir = Path(sys.argv[2]).resolve()
corpus_jsonl = Path(sys.argv[3]).resolve()

sys.path.insert(0, str(repo_root / "backend"))

from app.db_init import init_db
from app.database import AsyncSessionLocal
from app.models import Document
from app.services.rag.vector_index_service import _extract_core_entities, _first_paragraph, get_vector_store
from fastapi.concurrency import run_in_threadpool
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter


async def main() -> None:
    await init_db()
    records = [json.loads(line) for line in corpus_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    async with AsyncSessionLocal() as session:
        documents = []
        for record in records:
            doc = Document(title=str(record["title"]), content=str(record["content"]))
            session.add(doc)
            documents.append((doc, record))
        await session.flush()

        headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        vector_store = get_vector_store()
        pending = []
        chunk_count = 0
        for index, (doc, record) in enumerate(documents, start=1):
            content = str(record["content"])
            md_header_splits = markdown_splitter.split_text(content)
            final_splits = text_splitter.split_documents(md_header_splits)
            if not final_splits and content:
                final_splits = text_splitter.create_documents([content])

            entity_prefix = _extract_core_entities(str(record["title"]), _first_paragraph(content))
            for split in final_splits:
                if entity_prefix:
                    split.page_content = f"{entity_prefix}\n{split.page_content}"
                if not split.metadata:
                    split.metadata = {}
                split.metadata["source"] = str(record["title"])
                split.metadata["doc_id"] = int(doc.id)
                pending.append(split)

            if len(pending) >= 512:
                await run_in_threadpool(vector_store.add_documents, pending)
                chunk_count += len(pending)
                pending = []

            if index % 200 == 0 or index == len(documents):
                print(f"indexed {index}/{len(documents)} docs, chunks={chunk_count + len(pending)}", flush=True)

        if pending:
            await run_in_threadpool(vector_store.add_documents, pending)
            chunk_count += len(pending)

        await session.commit()
    (runtime_dir / "runtime_stats.json").write_text(json.dumps({"document_count": len(records), "chunk_count": chunk_count}, ensure_ascii=False, indent=2), encoding="utf-8")


asyncio.run(main())
"""
    env = os.environ.copy()
    env["DATA_DIR"] = str(runtime_dir)
    env["REDIS_PORT"] = "6399"
    env["AI_WARMUP_ON_STARTUP"] = "false"
    if LOCAL_RERANK_MODEL_PATH.exists():
        env["RERANK_MODEL"] = str(LOCAL_RERANK_MODEL_PATH)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(BACKEND_ROOT) + (os.pathsep + existing if existing else "")
    log_path = OUTPUT_ROOT / "crud_runtime_build.log"
    _ensure_dir(log_path.parent)
    with log_path.open("w", encoding="utf-8") as log_handle:
        subprocess.run(
            [sys.executable, "-c", helper_code, str(REPO_ROOT), str(runtime_dir), str(corpus_jsonl)],
            cwd=str(REPO_ROOT),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )
    _runtime_manifest_path(runtime_dir).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _ensure_runtime(runtime_dir: Path, corpus_jsonl: Path, manifest: dict[str, Any]) -> None:
    manifest_path = _runtime_manifest_path(runtime_dir)
    if manifest_path.exists() and (runtime_dir / "wiki.db").exists() and (runtime_dir / "chroma_db").exists():
        current = json.loads(manifest_path.read_text(encoding="utf-8"))
        if current == manifest:
            return
    if runtime_dir.exists():
        import shutil

        shutil.rmtree(runtime_dir)
    _ensure_dir(runtime_dir)
    _build_runtime(runtime_dir, corpus_jsonl, manifest)


def _assign_expected_doc_ids(cases: Sequence[CrudCase], hash_to_doc_id: dict[str, int]) -> None:
    for case in cases:
        case.expected_doc_ids = tuple(hash_to_doc_id[item] for item in case.positive_hashes if item in hash_to_doc_id)


def _call_internal_rag(
    base_url: str,
    case: CrudCase,
    *,
    doc_by_id: dict[int, CorpusDoc],
    llamaindex_module: Any,
) -> QueryResult:
    payload = json.dumps(
        {"messages": [{"role": "user", "content": case.query}], "use_rag": True},
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/api/v1/agent/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=600) as response:
        data = json.loads(response.read().decode("utf-8"))
    latency_ms = (time.perf_counter() - started) * 1000.0
    sources = list(data.get("sources") or [])
    source_doc_ids = [int(item["doc_id"]) for item in sources if item.get("doc_id") is not None]
    source_titles = [str(item.get("title") or "") for item in sources if item.get("title")]
    contexts = [_truncate(doc_by_id[doc_id].content) for doc_id in source_doc_ids if doc_id in doc_by_id]
    answer = str(data.get("response") or "").strip()
    similarity = max((_text_similarity(answer, ref) for ref in case.answers), default=0.0)
    accuracy_hit = int(llamaindex_module._match_any_answer(answer, case.answers)) if case.answers else 0
    retrieval_hit = int(any(doc_id in case.expected_doc_ids for doc_id in source_doc_ids)) if case.expected_doc_ids else 0
    return QueryResult(
        system="internal_rag",
        case_id=case.case_id,
        split=case.split,
        query=case.query,
        response=answer,
        source_doc_ids=source_doc_ids,
        source_titles=source_titles,
        retrieved_contexts=contexts,
        latency_ms=latency_ms,
        predicted_refusal=_contains_refusal(answer),
        expected_refusal=case.should_refuse,
        accuracy_hit=accuracy_hit,
        retrieval_hit=retrieval_hit,
        text_similarity=similarity,
    )


def _call_llamaindex_rag(
    query_engine: Any,
    case: CrudCase,
    *,
    doc_by_id: dict[int, CorpusDoc],
    llamaindex_module: Any,
) -> QueryResult:
    started = time.perf_counter()
    response = query_engine.query(case.query)
    latency_ms = (time.perf_counter() - started) * 1000.0
    answer = str(getattr(response, "response", response) or "").strip()
    source_doc_ids: list[int] = []
    source_titles: list[str] = []
    contexts: list[str] = []
    for item in list(getattr(response, "source_nodes", []) or [])[:3]:
        node = getattr(item, "node", item)
        metadata = dict(getattr(node, "metadata", {}) or {})
        raw_doc_id = metadata.get("doc_id")
        if raw_doc_id is not None:
            try:
                doc_id = int(raw_doc_id)
                source_doc_ids.append(doc_id)
                if doc_id in doc_by_id:
                    contexts.append(_truncate(doc_by_id[doc_id].content))
            except Exception:
                pass
        title = str(metadata.get("title") or "").strip()
        if title:
            source_titles.append(title)
        elif raw_doc_id is not None:
            source_titles.append(f"CRUD_DOC_{raw_doc_id}")
    similarity = max((_text_similarity(answer, ref) for ref in case.answers), default=0.0)
    accuracy_hit = int(llamaindex_module._match_any_answer(answer, case.answers)) if case.answers else 0
    retrieval_hit = int(any(doc_id in case.expected_doc_ids for doc_id in source_doc_ids)) if case.expected_doc_ids else 0
    return QueryResult(
        system="llamaindex",
        case_id=case.case_id,
        split=case.split,
        query=case.query,
        response=answer,
        source_doc_ids=source_doc_ids,
        source_titles=source_titles,
        retrieved_contexts=contexts,
        latency_ms=latency_ms,
        predicted_refusal=_contains_refusal(answer),
        expected_refusal=case.should_refuse,
        accuracy_hit=accuracy_hit,
        retrieval_hit=retrieval_hit,
        text_similarity=similarity,
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    _ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * ratio))))
    return float(ordered[index])


def _ragas_summary(
    results: list[QueryResult],
    case_map: dict[str, CrudCase],
    *,
    case_ids: Sequence[str],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    if not case_ids:
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "ragas_sample_count": 0,
        }, []
    compare_module = _load_compare_module()
    li_module = compare_module._load_llamaindex_module()
    ragas = li_module._require_ragas()
    dataset_rows = []
    ordered_ids: list[str] = []
    for case_id in case_ids:
        case = case_map[case_id]
        result = next(item for item in results if item.case_id == case_id)
        dataset_rows.append(
            {
                "user_input": case.query,
                "response": result.response,
                "retrieved_contexts": result.retrieved_contexts,
                "reference": case.reference,
            }
        )
        ordered_ids.append(case_id)
    dataset = compare_module.EvaluationDataset.from_list(dataset_rows)
    run_config = None
    run_config_cls = getattr(ragas, "RunConfig", None)
    if run_config_cls is None:
        try:
            from ragas import RunConfig as run_config_cls
        except Exception:
            run_config_cls = None
    if run_config_cls is not None:
        run_config = run_config_cls(timeout=180, max_retries=1, max_wait=30, max_workers=1)
    metric_result = compare_module.evaluate(
        dataset=dataset,
        metrics=[
            ragas.Faithfulness(max_retries=1),
            ragas.AnswerRelevancy(strictness=1),
            ragas.ContextPrecision(max_retries=1),
            ragas.ContextRecall(max_retries=1),
        ],
        llm=ragas.LlamaIndexLLMWrapper(li_module._build_ollama_llm()),
        embeddings=ragas.LlamaIndexEmbeddingsWrapper(li_module._build_ollama_embedding()),
        run_config=run_config,
        raise_exceptions=False,
        show_progress=False,
        batch_size=4,
    )
    summary = {
        "faithfulness": round(float(metric_result._repr_dict["faithfulness"]), 4),
        "answer_relevancy": round(float(metric_result._repr_dict["answer_relevancy"]), 4),
        "context_precision": round(float(metric_result._repr_dict["context_precision"]), 4),
        "context_recall": round(float(metric_result._repr_dict["context_recall"]), 4),
        "ragas_sample_count": len(dataset_rows),
    }
    details: list[dict[str, Any]] = []
    for case_id, score_row in zip(ordered_ids, metric_result.scores):
        row = {"case_id": case_id}
        row.update({key: round(float(value), 4) for key, value in score_row.items()})
        details.append(row)
    return summary, details


def _summarize_system(
    system_name: str,
    results: list[QueryResult],
    case_map: dict[str, CrudCase],
    *,
    ragas_case_ids: Sequence[str],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    ragas_summary, ragas_rows = _ragas_summary(results, case_map, case_ids=ragas_case_ids)
    latency_values = [item.latency_ms for item in results]
    noise_cases = [item for item in results if item.split == "hallu_modified"]
    negative_cases = [item for item in results if item.split == "negative_rejection"]
    integration_cases = [item for item in results if item.split in {"questanswer_2docs", "questanswer_3docs"}]
    summary = {
        "system": system_name,
        **ragas_summary,
        "noise_robustness": round(sum(item.text_similarity for item in noise_cases) / len(noise_cases), 4) if noise_cases else 0.0,
        "negative_rejection": round(
            sum(1 for item in negative_cases if item.predicted_refusal) / len(negative_cases), 4
        ) if negative_cases else 0.0,
        "information_integration": round(
            sum(item.accuracy_hit for item in integration_cases) / len(integration_cases), 4
        ) if integration_cases else 0.0,
        "latency_p50_ms": round(_percentile(latency_values, 0.50), 2),
        "latency_p95_ms": round(_percentile(latency_values, 0.95), 2),
        "sample_count": len(results),
    }
    detail_rows = [
        {
            "system": item.system,
            "case_id": item.case_id,
            "split": item.split,
            "query": item.query,
            "response": item.response,
            "predicted_refusal": int(item.predicted_refusal),
            "expected_refusal": int(item.expected_refusal),
            "accuracy_hit": item.accuracy_hit,
            "retrieval_hit": item.retrieval_hit,
            "text_similarity": round(item.text_similarity, 4),
            "latency_ms": round(item.latency_ms, 2),
            "source_titles": " | ".join(item.source_titles),
        }
        for item in results
    ]
    return summary, detail_rows, ragas_rows


def _configure_matplotlib() -> None:
    if FONT_PATH.exists():
        font_manager.fontManager.addfont(str(FONT_PATH))
        plt.rcParams["font.family"] = "Source Han Sans SC"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["svg.fonttype"] = "path"


def _render_svg(summary_rows: list[dict[str, Any]], output_path: Path) -> None:
    _configure_matplotlib()
    labels = {"internal_rag": "你的 RAG", "llamaindex": "LlamaIndex"}
    colors = {"internal_rag": "#0E5A8A", "llamaindex": "#D9822B"}

    retrieval_metrics = [("context_precision", "Context Precision"), ("context_recall", "Context Recall")]
    generation_metrics = [("faithfulness", "Faithfulness"), ("answer_relevancy", "Answer Relevancy")]
    end_to_end_metrics = [
        ("noise_robustness", "Noise Robustness"),
        ("negative_rejection", "Negative Rejection"),
        ("information_integration", "Information Integration"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16.8, 5.4), gridspec_kw={"width_ratios": [1.0, 1.0, 1.4]})
    bar_width = 0.36
    for ax, title, metrics in (
        (axes[0], "检索质量", retrieval_metrics),
        (axes[1], "生成质量", generation_metrics),
        (axes[2], "端到端表现", end_to_end_metrics),
    ):
        x_positions = list(range(len(metrics)))
        for offset, row in enumerate(summary_rows):
            values = [row[key] for key, _ in metrics]
            system = row["system"]
            ax.bar(
                [x + (offset - 0.5) * bar_width for x in x_positions],
                values,
                width=bar_width,
                color=colors[system],
                label=labels[system],
            )
            for index, value in enumerate(values):
                ax.text(index + (offset - 0.5) * bar_width, value + 0.015, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([label for _key, label in metrics], rotation=16, ha="right")
        ax.set_ylim(0, 1.08)
        ax.set_title(title)
        ax.grid(axis="y", linestyle="--", alpha=0.25)
    axes[0].legend(frameon=False, loc="upper left")

    fig.suptitle("你的 RAG vs LlamaIndex 在 CRUD-RAG 上的对比评测", fontsize=16, fontweight="bold")
    fig.tight_layout()
    _ensure_dir(output_path.parent)
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="在 CRUD-RAG 上比较你的 RAG 与 LlamaIndex。")
    parser.add_argument("--summary-samples", type=int, default=10)
    parser.add_argument("--qa-1doc-samples", type=int, default=10)
    parser.add_argument("--qa-2doc-samples", type=int, default=10)
    parser.add_argument("--qa-3doc-samples", type=int, default=10)
    parser.add_argument("--hallu-samples", type=int, default=10)
    parser.add_argument("--negative-samples", type=int, default=16)
    parser.add_argument("--distractor-count", type=int, default=1200)
    parser.add_argument("--ragas-sample-count", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    compare_module = _load_compare_module()
    llamaindex_module = compare_module._load_llamaindex_module()

    split_data = _load_split_data()
    positive_cases, positive_texts_by_hash = _build_positive_cases(
        split_data,
        summary_samples=args.summary_samples,
        qa_1doc_samples=args.qa_1doc_samples,
        qa_2doc_samples=args.qa_2doc_samples,
        qa_3doc_samples=args.qa_3doc_samples,
        hallu_samples=args.hallu_samples,
    )
    anchor_pool = [case.anchor for case in positive_cases if case.anchor]
    negative_cases = _build_negative_cases(anchor_pool, args.negative_samples)
    all_cases = positive_cases + negative_cases

    docs, hash_to_doc_id = _build_corpus_subset(
        positive_texts_by_hash,
        distractor_count=args.distractor_count,
        seed=args.seed,
    )
    _assign_expected_doc_ids(all_cases, hash_to_doc_id)
    doc_by_id = {doc.doc_id: doc for doc in docs}

    run_key = hashlib.sha1(
        json.dumps(
            {
                "summary_samples": args.summary_samples,
                "qa_1doc_samples": args.qa_1doc_samples,
                "qa_2doc_samples": args.qa_2doc_samples,
                "qa_3doc_samples": args.qa_3doc_samples,
                "hallu_samples": args.hallu_samples,
                "negative_samples": args.negative_samples,
                "distractor_count": args.distractor_count,
                "seed": args.seed,
                "positive_hash_count": len(positive_texts_by_hash),
            },
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()[:12]
    runtime_dir = _ensure_dir(RUNTIME_ROOT / f"crud_runtime_{run_key}")
    corpus_jsonl = CORPUS_ROOT / f"crud_corpus_{run_key}.jsonl"
    _write_corpus_jsonl(docs, corpus_jsonl)
    manifest = {
        "run_key": run_key,
        "document_count": len(docs),
        "distractor_count": args.distractor_count,
        "corpus_hash": hashlib.sha1(corpus_jsonl.read_bytes()).hexdigest(),
    }
    _ensure_runtime(runtime_dir, corpus_jsonl, manifest)

    active_docs = [compare_module.ActiveDocument(doc_id=doc.doc_id, title=doc.title, content=doc.content) for doc in docs]
    backend_log = OUTPUT_ROOT / "crud_backend_server.log"
    process, port = compare_module._start_backend_process(runtime_dir, 8020, backend_log)
    base_url = f"http://127.0.0.1:{port}"

    try:
        query_engine, _ = compare_module.build_llamaindex_query_engine(active_docs, f"crud_compare_index_{run_key}")
        warmup_query = positive_cases[0].query
        compare_module._warmup_internal_backend(base_url, warmup_query)
        compare_module._warmup_llamaindex_query_engine(query_engine, warmup_query)

        internal_results: list[QueryResult] = []
        llamaindex_results: list[QueryResult] = []
        total = len(all_cases)
        for index, case in enumerate(all_cases, start=1):
            if index == 1 or index % 10 == 0 or index == total:
                print(f"[CRUD-RAG] evaluating {index}/{total}", flush=True)
            internal_results.append(_call_internal_rag(base_url, case, doc_by_id=doc_by_id, llamaindex_module=llamaindex_module))
            llamaindex_results.append(_call_llamaindex_rag(query_engine, case, doc_by_id=doc_by_id, llamaindex_module=llamaindex_module))

        quality_candidates = [
            case.case_id
            for case in positive_cases
            if case.split in {"event_summary", "questanswer_1doc", "questanswer_2docs", "questanswer_3docs"}
        ]
        ragas_case_ids = [item["case_id"] for item in _evenly_sample([{"case_id": case_id} for case_id in quality_candidates], args.ragas_sample_count)]
        case_map = {case.case_id: case for case in all_cases}

        internal_summary, internal_details, internal_ragas = _summarize_system(
            "internal_rag",
            internal_results,
            case_map,
            ragas_case_ids=ragas_case_ids,
        )
        llama_summary, llama_details, llama_ragas = _summarize_system(
            "llamaindex",
            llamaindex_results,
            case_map,
            ragas_case_ids=ragas_case_ids,
        )

        summary_rows = [internal_summary, llama_summary]
        detail_rows = internal_details + llama_details
        ragas_rows = [
            {"system": "internal_rag", **row} for row in internal_ragas
        ] + [
            {"system": "llamaindex", **row} for row in llama_ragas
        ]
        output_json = OUTPUT_ROOT / "crud_rag_comparison_summary.json"
        output_csv = OUTPUT_ROOT / "crud_rag_comparison_summary.csv"
        detail_csv = OUTPUT_ROOT / "crud_rag_comparison_details.csv"
        ragas_csv = OUTPUT_ROOT / "crud_rag_comparison_ragas_details.csv"
        svg_path = OUTPUT_ROOT / "crud_rag_comparison_summary.svg"
        sample_path = OUTPUT_ROOT / "crud_rag_sample_manifest.json"

        _write_csv(output_csv, summary_rows)
        _write_csv(detail_csv, detail_rows)
        _write_csv(ragas_csv, ragas_rows)
        output_json.write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding="utf-8")
        sample_path.write_text(
            json.dumps(
                {
                    "run_key": run_key,
                    "runtime_dir": str(runtime_dir),
                    "corpus_jsonl": str(corpus_jsonl),
                    "document_count": len(docs),
                    "positive_case_count": len(positive_cases),
                    "negative_case_count": len(negative_cases),
                    "ragas_case_ids": ragas_case_ids,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        _render_svg(summary_rows, svg_path)

        print(json.dumps(summary_rows, ensure_ascii=False, indent=2))
        print(f"SVG: {svg_path}")
    finally:
        compare_module._stop_backend_process(process)


if __name__ == "__main__":
    main()
