from __future__ import annotations

import argparse
import csv
import hashlib
import html
import importlib.util
import json
import os
import shutil
import socket
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
from matplotlib import font_manager
from ragas import EvaluationDataset, evaluate
from ragas.metrics import AnswerRelevancy, ContextPrecision, Faithfulness


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
SQLITE_PATH = BACKEND_ROOT / "data" / "wiki.db"
OUTPUT_ROOT = REPO_ROOT / "llamaindex_rag_eval" / "outputs"
FONT_PATH = REPO_ROOT / "experiments" / ".assets" / "fonts" / "SourceHanSansSC-Regular.otf"
INTERNAL_RAG_URL = "http://127.0.0.1:8000/api/v1/agent/chat"
RGB_DATASET_DIR = REPO_ROOT / "llamaindex_rag_eval" / "data" / "rgb"
RGB_RUNTIME_DIR = RGB_DATASET_DIR / "internal_rag_runtime"
RGB_RUNTIME_PORT = 8012
LOCAL_RUNTIME_PORT = 8010
RGB_RAGAS_SAMPLE_SIZE = 50
LOCAL_RERANK_MODEL_PATH = BACKEND_ROOT / "data" / "models" / "rerank" / "BAAI--bge-reranker-v2-m3"
NO_CONTEXT_ANSWER = "根据当前检索到的知识库内容，没有找到足够相关的参考资料，因此我暂时无法给出可靠回答。"
REFUSAL_MARKERS = (
    "无法确定",
    "无法回答",
    "无法给出可靠回答",
    "无法根据",
    "没有找到足够相关",
    "资料不足",
    "没有相关资料",
    "无法从提供的上下文",
    "未提及",
    "没有提到",
    "未涉及",
    "i cannot determine",
    "insufficient context",
)


@dataclass(frozen=True)
class ActiveDocument:
    doc_id: int
    title: str
    content: str


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    query: str
    reference: str
    answers: tuple[str, ...]
    should_refuse: bool = False
    expected_titles: tuple[str, ...] = ()
    positive_ctxs: tuple[dict[str, str], ...] = ()
    dataset: str = "local"


@dataclass
class QueryResult:
    system: str
    dataset: str
    case_id: str
    query: str
    response: str
    source_titles: list[str]
    retrieved_contexts: list[str]
    latency_ms: float
    predicted_refusal: bool
    expected_refusal: bool
    retrieval_hit: bool


@dataclass
class DatasetRunResult:
    name: str
    summary_rows: list[dict[str, Any]]
    detailed_rows: list[dict[str, Any]]
    quality_rows: list[dict[str, Any]]
    svg_path: Path
    refusal_svg_path: Path | None = None
    extra_files: list[Path] = field(default_factory=list)


LOCAL_POSITIVE_CASES: tuple[EvalCase, ...] = (
    EvalCase(
        case_id="shor_fact",
        query="Shor算法能解决什么问题",
        reference="Shor算法能在多项式时间内解决大整数质因数分解问题。",
        answers=("Shor算法能在多项式时间内解决大整数质因数分解问题。",),
        expected_titles=("量子计算与Shor算法",),
    ),
    EvalCase(
        case_id="shor_ibm_2023",
        query="2023年IBM使用多少量子比特处理器实现简化版Shor算法",
        reference="2023年IBM使用127量子比特处理器实现了简化版Shor算法。",
        answers=("2023年IBM使用127量子比特处理器实现了简化版Shor算法。",),
        expected_titles=("量子计算与Shor算法",),
    ),
    EvalCase(
        case_id="mtdna_maternal",
        query="线粒体DNA为什么通常呈母系遗传",
        reference="因为受精过程中精子线粒体会被泛素化标记，并在胚胎早期被选择性降解。",
        answers=("因为受精过程中精子线粒体会被泛素化标记，并在胚胎早期被选择性降解。",),
        expected_titles=("线粒体DNA的母系遗传机制",),
    ),
    EvalCase(
        case_id="mtdna_length_genes",
        query="mtDNA全长多少碱基对，编码多少个基因",
        reference="mtDNA全长16569个碱基对，编码37个基因。",
        answers=("mtDNA全长16569个碱基对，编码37个基因。",),
        expected_titles=("线粒体DNA的母系遗传机制",),
    ),
    EvalCase(
        case_id="qingmiao_definition",
        query="青苗法是什么",
        reference="青苗法是官府在青黄不接时向农民贷款，收获后加息20%偿还。",
        answers=("青苗法是官府在青黄不接时向农民贷款，收获后加息20%偿还。",),
        expected_titles=("北宋熙宁变法中的经济政策",),
    ),
    EvalCase(
        case_id="xining_revenue",
        query="变法期间国库岁入从多少增至多少",
        reference="变法期间国库岁入从熙宁初年的6000万贯增至8000万贯。",
        answers=("变法期间国库岁入从熙宁初年的6000万贯增至8000万贯。",),
        expected_titles=("北宋熙宁变法中的经济政策",),
    ),
    EvalCase(
        case_id="archive_official_name",
        query="雾潮镇档案馆的正式名称是什么",
        reference="雾潮镇档案馆的正式名称是雾潮地方文书保藏中心。",
        answers=("雾潮镇档案馆的正式名称是雾潮地方文书保藏中心。",),
        expected_titles=("雾潮镇的档案馆",),
    ),
    EvalCase(
        case_id="archive_ledger_why",
        query="研究者为什么重视A-17-204蓝布账册",
        reference="因为这本账册记录了1949年至1952年间的物资往来，并频繁出现沈见川的名字，帮助研究者修正地方史。",
        answers=("因为这本账册记录了1949年至1952年间的物资往来，并频繁出现沈见川的名字，帮助研究者修正地方史。",),
        expected_titles=("雾潮镇的档案馆",),
    ),
    EvalCase(
        case_id="han_qiming_role",
        query="韩启明与档案馆有什么关系",
        reference="韩启明是参与档案馆修缮与设备验收的工程师，后来还设计了低成本编号规则。",
        answers=("韩启明是参与档案馆修缮与设备验收的工程师，后来还设计了低成本编号规则。",),
        expected_titles=("雾潮镇的档案馆",),
    ),
    EvalCase(
        case_id="chen_heting_donation",
        query="陈鹤汀家族捐出了哪些材料",
        reference="陈鹤汀家族向档案馆捐出了87封信件、3本日记和11张黑白照片。",
        answers=("陈鹤汀家族向档案馆捐出了87封信件、3本日记和11张黑白照片。",),
        expected_titles=("雾潮镇的档案馆",),
    ),
)


LOCAL_NEGATIVE_CASES: tuple[EvalCase, ...] = (
    EvalCase("boundary_shenyang", "沈阳", "", (), True),
    EvalCase("boundary_korea", "韩国", "", (), True),
    EvalCase("boundary_singapore_view", "新加坡的风景", "", (), True),
    EvalCase("boundary_singapore_archive", "新加坡的档案馆", "", (), True),
    EvalCase("boundary_korea_singapore", "韩国与新加坡的关系", "", (), True),
    EvalCase("boundary_old_doc", "RAG测试文档的默认聊天模型是什么", "", (), True),
)


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_llamaindex_module():
    module_path = REPO_ROOT / "llamaindex_rag_eval" / "llamaindex_rag_eval.py"
    spec = importlib.util.spec_from_file_location("llamaindex_rag_eval_module", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_documents(sqlite_path: Path) -> list[ActiveDocument]:
    connection = sqlite3.connect(str(sqlite_path))
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            """
            SELECT id, title, content
            FROM documents
            WHERE is_active = 1
            ORDER BY id
            """
        ).fetchall()
    finally:
        connection.close()
    return [
        ActiveDocument(
            doc_id=int(row["id"]),
            title=str(row["title"] or "").strip(),
            content=html.unescape(str(row["content"] or "").strip()),
        )
        for row in rows
        if str(row["content"] or "").strip()
    ]


def build_llamaindex_query_engine(active_docs: list[ActiveDocument], persist_subdir: str) -> tuple[Any, dict[int, ActiveDocument]]:
    module = _load_llamaindex_module()
    li = module._require_llamaindex()
    embed_model = module._build_ollama_embedding()
    llm = module._build_ollama_llm()
    splitter = li.SentenceSplitter(
        chunk_size=module.CONFIG.chunk_size,
        chunk_overlap=module.CONFIG.chunk_overlap,
    )
    module._configure_global_settings(llm=llm, embed_model=embed_model, splitter=splitter)

    documents = [
        li.Document(
            text=doc.content,
            metadata={
                "title": doc.title,
                "doc_id": doc.doc_id,
            },
        )
        for doc in active_docs
    ]
    nodes = splitter.get_nodes_from_documents(documents)

    persist_path = _ensure_dir(OUTPUT_ROOT / persist_subdir)
    collection_name = f"{module.CONFIG.chroma_collection}_{persist_subdir.replace('-', '_')}"
    chroma_collection = module._open_chroma_collection(
        persist_dir=persist_path,
        collection_name=collection_name,
        reset=True,
    )
    vector_store = li.ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = li.StorageContext.from_defaults(vector_store=vector_store)
    vector_index = li.VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )
    hybrid_retriever = module._build_hybrid_retriever(vector_index=vector_index, nodes=nodes)
    reranker = li.SentenceTransformerRerank(
        model=module.CONFIG.rerank_model,
        top_n=module.CONFIG.rerank_top_n,
        keep_retrieval_score=True,
        cross_encoder_kwargs={"max_length": module.CONFIG.rerank_max_length},
    )
    query_engine = li.RetrieverQueryEngine.from_args(
        retriever=hybrid_retriever,
        llm=llm,
        node_postprocessors=[reranker, module._build_final_topk_postprocessor()],
        response_mode="compact",
    )
    query_engine = module._attach_runtime_handles(
        query_engine=query_engine,
        llm=llm,
        embed_model=embed_model,
        persist_dir=persist_path,
    )
    return query_engine, {doc.doc_id: doc for doc in active_docs}


def _contains_refusal(text: str) -> bool:
    lowered = str(text or "").strip().lower()
    if not lowered:
        return True
    if lowered == NO_CONTEXT_ANSWER.lower():
        return True
    return any(marker in lowered for marker in REFUSAL_MARKERS)


def _truncate_context(text: str, limit: int = 2200) -> str:
    return str(text or "").strip()[:limit]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    _ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
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


def _title_hit(source_titles: list[str], expected_titles: tuple[str, ...]) -> bool:
    if not expected_titles:
        return len(source_titles) == 0
    source_set = set(source_titles)
    return all(title in source_set for title in expected_titles)


def _pick_port(preferred: int) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        if sock.connect_ex(("127.0.0.1", preferred)) != 0:
            return preferred
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _healthcheck(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return payload.get("status") == "ok"
    except Exception:
        return False


def _build_backend_env(data_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["DATA_DIR"] = str(data_dir)
    env["REDIS_PORT"] = "6399"
    env["AI_WARMUP_ON_STARTUP"] = "false"
    if LOCAL_RERANK_MODEL_PATH.exists():
        env["RERANK_MODEL"] = str(LOCAL_RERANK_MODEL_PATH)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(BACKEND_ROOT) + (os.pathsep + existing if existing else "")
    return env


def _wait_for_health(base_url: str, process: subprocess.Popen[str], timeout_sec: float = 120.0) -> None:
    deadline = time.time() + timeout_sec
    health_url = f"{base_url}/health"
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"后端进程提前退出，退出码: {process.returncode}")
        if _healthcheck(health_url):
            return
        time.sleep(1.0)
    raise TimeoutError(f"后端健康检查超时: {health_url}")


def _start_backend_process(data_dir: Path, preferred_port: int, log_path: Path) -> tuple[subprocess.Popen[str], int]:
    port = _pick_port(preferred_port)
    log_file = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=str(BACKEND_ROOT),
        env=_build_backend_env(data_dir),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        _wait_for_health(f"http://127.0.0.1:{port}", process)
    except Exception:
        process.terminate()
        raise
    return process, port


def _stop_backend_process(process: subprocess.Popen[str] | None) -> None:
    if process is None:
        return
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def _warmup_internal_backend(base_url: str, query: str, timeout: int = 600) -> None:
    payload = json.dumps(
        {"messages": [{"role": "user", "content": query}]},
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/api/v1/agent/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response.read()
    except Exception:
        pass


def _warmup_llamaindex_query_engine(query_engine: Any, query: str) -> None:
    try:
        query_engine.query(query)
    except Exception:
        pass


def _hash_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rgb_runtime_manifest_path(runtime_dir: Path) -> Path:
    return runtime_dir / "runtime_manifest.json"


def _build_rgb_runtime(rgb_path: Path, runtime_dir: Path) -> None:
    helper_code = r"""
import asyncio
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

repo_root = Path(sys.argv[1]).resolve()
rgb_path = Path(sys.argv[2]).resolve()
runtime_dir = Path(sys.argv[3]).resolve()
sys.path.insert(0, str(repo_root / "backend"))

spec = importlib.util.spec_from_file_location("li_eval_helper", repo_root / "llamaindex_rag_eval" / "llamaindex_rag_eval.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

from app.db_init import init_db
from app.database import AsyncSessionLocal
from app.models import Document
from app.services.rag.vector_index_service import _extract_core_entities, _first_paragraph, get_vector_store
from fastapi.concurrency import run_in_threadpool
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter


async def main() -> None:
    await init_db()
    cases = module.load_rgb_testset(str(rgb_path))
    dedup = []
    seen = set()
    for case in cases:
        for contexts in (case["positive_ctxs"], case["negative_ctxs"]):
            for context in contexts:
                text = str(context.get("text") or "").strip()
                if not text:
                    continue
                key = hashlib.sha1(text.encode("utf-8")).hexdigest()
                if key in seen:
                    continue
                seen.add(key)
                title = str(context.get("title") or key).strip() or key
                dedup.append((title, text))

    async with AsyncSessionLocal() as session:
        documents = []
        for title, text in dedup:
            doc = Document(title=title, content=text)
            session.add(doc)
            documents.append(doc)
        await session.flush()

        headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        vector_store = get_vector_store()
        pending_splits = []
        indexed_chunk_count = 0
        total = len(documents)
        for index, doc in enumerate(documents, start=1):
            content = str(doc.content or "")
            md_header_splits = markdown_splitter.split_text(content)
            final_splits = text_splitter.split_documents(md_header_splits)
            if not final_splits and content:
                final_splits = text_splitter.create_documents([content])

            entity_prefix = _extract_core_entities(str(doc.title or ""), _first_paragraph(content))
            for split in final_splits:
                if entity_prefix:
                    split.page_content = f"{entity_prefix}\n{split.page_content}"
                if not split.metadata:
                    split.metadata = {}
                split.metadata["source"] = str(doc.title or "")
                split.metadata["doc_id"] = int(doc.id)
                pending_splits.append(split)

            if len(pending_splits) >= 512:
                await run_in_threadpool(vector_store.add_documents, pending_splits)
                indexed_chunk_count += len(pending_splits)
                pending_splits = []

            if index % 250 == 0 or index == total:
                print(f"prepared {index}/{total} documents, chunks={indexed_chunk_count + len(pending_splits)}", flush=True)

        if pending_splits:
            await run_in_threadpool(vector_store.add_documents, pending_splits)
            indexed_chunk_count += len(pending_splits)
        await session.commit()

    manifest = {
        "document_count": len(dedup),
        "chunk_count": indexed_chunk_count,
        "rgb_hash": hashlib.sha1(rgb_path.read_bytes()).hexdigest(),
    }
    (runtime_dir / "build_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


asyncio.run(main())
"""
    env = _build_backend_env(runtime_dir)
    build_log_path = OUTPUT_ROOT / "rgb_runtime_build.log"
    with build_log_path.open("w", encoding="utf-8") as log_file:
        subprocess.run(
            [sys.executable, "-c", helper_code, str(REPO_ROOT), str(rgb_path), str(runtime_dir)],
            cwd=str(REPO_ROOT),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )


def _ensure_rgb_internal_runtime(rgb_path: Path) -> Path:
    manifest_path = _rgb_runtime_manifest_path(RGB_RUNTIME_DIR)
    expected_hash = _hash_file(rgb_path)
    if (
        manifest_path.exists()
        and (RGB_RUNTIME_DIR / "wiki.db").exists()
        and (RGB_RUNTIME_DIR / "chroma_db").exists()
    ):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("rgb_hash") == expected_hash:
            return RGB_RUNTIME_DIR

    if RGB_RUNTIME_DIR.exists():
        shutil.rmtree(RGB_RUNTIME_DIR)
    RGB_RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    _build_rgb_runtime(rgb_path, RGB_RUNTIME_DIR)
    manifest_path.write_text(
        json.dumps({"rgb_hash": expected_hash}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return RGB_RUNTIME_DIR


def _load_rgb_cases(module: Any, rgb_path: Path) -> list[EvalCase]:
    raw_cases = module.load_rgb_testset(str(rgb_path))
    cases: list[EvalCase] = []
    for item in raw_cases:
        answers = tuple(str(value).strip() for value in item["answers"] if str(value).strip())
        positive_ctxs = tuple({"title": str(ctx["title"]), "text": str(ctx["text"])} for ctx in item["positive_ctxs"])
        cases.append(
            EvalCase(
                case_id=f"rgb_{item['id']}",
                query=str(item["question"]).strip(),
                reference="；".join(answers),
                answers=answers,
                positive_ctxs=positive_ctxs,
                dataset="rgb",
            )
        )
    return cases


def _select_evenly_spaced_case_ids(cases: Sequence[EvalCase], max_count: int) -> set[str]:
    if len(cases) <= max_count:
        return {case.case_id for case in cases}
    step = len(cases) / max_count
    indices = {min(len(cases) - 1, int(index * step)) for index in range(max_count)}
    return {cases[index].case_id for index in sorted(indices)}


def call_internal_rag(base_url: str, case: EvalCase, doc_by_id: dict[int, ActiveDocument], li_module: Any) -> QueryResult:
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
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=600) as response:
        data = json.loads(response.read().decode("utf-8"))
    latency_ms = (time.perf_counter() - start) * 1000.0
    sources = list(data.get("sources") or [])
    source_titles = [str(item.get("title") or "") for item in sources if item.get("title")]
    retrieved_contexts: list[str] = []
    for item in sources:
        raw_doc_id = item.get("doc_id")
        if raw_doc_id is None:
            continue
        doc = doc_by_id.get(int(raw_doc_id))
        if doc is not None:
            retrieved_contexts.append(_truncate_context(doc.content))
    answer = str(data.get("response") or "").strip()
    if case.dataset == "rgb":
        retrieval_hit = li_module._retrieval_hits_positive(retrieved_contexts, case.positive_ctxs)
    else:
        retrieval_hit = _title_hit(source_titles, case.expected_titles) if not case.should_refuse else False
    return QueryResult(
        system="internal_rag",
        dataset=case.dataset,
        case_id=case.case_id,
        query=case.query,
        response=answer,
        source_titles=source_titles,
        retrieved_contexts=retrieved_contexts,
        latency_ms=latency_ms,
        predicted_refusal=_contains_refusal(answer),
        expected_refusal=case.should_refuse,
        retrieval_hit=retrieval_hit,
    )


def call_llamaindex_rag(query_engine: Any, case: EvalCase, doc_by_id: dict[int, ActiveDocument], li_module: Any) -> QueryResult:
    start = time.perf_counter()
    response = query_engine.query(case.query)
    latency_ms = (time.perf_counter() - start) * 1000.0
    answer = str(getattr(response, "response", response) or "").strip()
    source_titles: list[str] = []
    retrieved_contexts: list[str] = []
    for item in list(getattr(response, "source_nodes", []) or [])[:3]:
        node = getattr(item, "node", item)
        metadata = dict(getattr(node, "metadata", {}) or {})
        raw_doc_id = metadata.get("doc_id")
        title = str(metadata.get("title") or "").strip()
        if title:
            source_titles.append(title)
        doc = None
        if raw_doc_id is not None:
            try:
                doc = doc_by_id.get(int(raw_doc_id))
            except (TypeError, ValueError):
                doc = None
        if doc is not None:
            retrieved_contexts.append(_truncate_context(doc.content))
            continue
        node_text = str(getattr(node, "text", "") or "").strip()
        if node_text:
            retrieved_contexts.append(_truncate_context(node_text))
    if case.dataset == "rgb":
        retrieval_hit = li_module._retrieval_hits_positive(retrieved_contexts, case.positive_ctxs)
    else:
        retrieval_hit = _title_hit(source_titles, case.expected_titles) if not case.should_refuse else False
    return QueryResult(
        system="llamaindex",
        dataset=case.dataset,
        case_id=case.case_id,
        query=case.query,
        response=answer,
        source_titles=source_titles,
        retrieved_contexts=retrieved_contexts,
        latency_ms=latency_ms,
        predicted_refusal=_contains_refusal(answer),
        expected_refusal=case.should_refuse,
        retrieval_hit=retrieval_hit,
    )


def score_quality(
    system_name: str,
    dataset_name: str,
    results: list[QueryResult],
    case_map: dict[str, EvalCase],
    quality_case_ids: set[str],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    dataset_rows = []
    case_order: list[str] = []
    for result in results:
        case = case_map[result.case_id]
        if case.should_refuse or result.case_id not in quality_case_ids:
            continue
        dataset_rows.append(
            {
                "user_input": case.query,
                "response": result.response,
                "retrieved_contexts": result.retrieved_contexts,
                "reference": case.reference,
            }
        )
        case_order.append(case.case_id)
    if not dataset_rows:
        return {
            "dataset": dataset_name,
            "system": system_name,
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "quality_sample_count": 0,
        }, []

    module = _load_llamaindex_module()
    ragas = module._require_ragas()
    dataset = EvaluationDataset.from_list(dataset_rows)
    llm = ragas.LlamaIndexLLMWrapper(module._build_ollama_llm())
    embeddings = ragas.LlamaIndexEmbeddingsWrapper(module._build_ollama_embedding())
    metric_result = evaluate(
        dataset=dataset,
        metrics=[Faithfulness(), AnswerRelevancy(), ContextPrecision()],
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=False,
        show_progress=False,
    )
    summary = {
        "dataset": dataset_name,
        "system": system_name,
        "faithfulness": round(float(metric_result._repr_dict["faithfulness"]), 4),
        "answer_relevancy": round(float(metric_result._repr_dict["answer_relevancy"]), 4),
        "context_precision": round(float(metric_result._repr_dict["context_precision"]), 4),
        "quality_sample_count": len(dataset_rows),
    }
    detailed_rows = []
    for case_id, score_row in zip(case_order, metric_result.scores):
        row = {"dataset": dataset_name, "system": system_name, "case_id": case_id}
        row.update({key: round(float(value), 4) for key, value in score_row.items()})
        detailed_rows.append(row)
    return summary, detailed_rows


def score_refusal(system_name: str, results: list[QueryResult]) -> dict[str, float]:
    tp = sum(1 for item in results if item.expected_refusal and item.predicted_refusal)
    fp = sum(1 for item in results if (not item.expected_refusal) and item.predicted_refusal)
    fn = sum(1 for item in results if item.expected_refusal and (not item.predicted_refusal))
    tn = sum(1 for item in results if (not item.expected_refusal) and (not item.predicted_refusal))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    false_refusal_rate = fp / (fp + tn) if (fp + tn) else 0.0
    return {
        "system": system_name,
        "refusal_precision": round(precision, 4),
        "refusal_recall": round(recall, 4),
        "refusal_f1": round(f1, 4),
        "false_refusal_rate": round(false_refusal_rate, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def score_latency(system_name: str, results: list[QueryResult]) -> dict[str, float]:
    latencies = [item.latency_ms for item in results]
    return {
        "system": system_name,
        "latency_p50_ms": round(_percentile(latencies, 0.50), 2),
        "latency_p95_ms": round(_percentile(latencies, 0.95), 2),
        "latency_avg_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
    }


def summarize_dataset(
    dataset_name: str,
    system_name: str,
    results: list[QueryResult],
    case_map: dict[str, EvalCase],
    quality_case_ids: set[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    module = _load_llamaindex_module()
    quality_summary, quality_rows = score_quality(system_name, dataset_name, results, case_map, quality_case_ids)
    positive_results = [result for result in results if not case_map[result.case_id].should_refuse]
    accuracy_hits = 0
    for result in positive_results:
        case = case_map[result.case_id]
        accuracy_hits += int(module._match_any_answer(result.response, case.answers))
    retrieval_hits = sum(int(result.retrieval_hit) for result in positive_results)
    summary = {
        **quality_summary,
        **score_latency(system_name, results),
        "sample_count": len(results),
        "positive_sample_count": len(positive_results),
        "accuracy": round(accuracy_hits / len(positive_results), 4) if positive_results else 0.0,
        "retrieval_hit_rate_at_3": round(retrieval_hits / len(positive_results), 4) if positive_results else 0.0,
    }
    if any(item.expected_refusal for item in results):
        summary.update(score_refusal(system_name, results))
    return summary, quality_rows


def _build_detail_rows(results: list[QueryResult], case_map: dict[str, EvalCase]) -> list[dict[str, Any]]:
    module = _load_llamaindex_module()
    rows: list[dict[str, Any]] = []
    for result in results:
        case = case_map[result.case_id]
        accuracy_hit = int(module._match_any_answer(result.response, case.answers)) if not case.should_refuse else 0
        rows.append(
            {
                "dataset": result.dataset,
                "system": result.system,
                "case_id": result.case_id,
                "query": result.query,
                "expected_refusal": int(result.expected_refusal),
                "predicted_refusal": int(result.predicted_refusal),
                "accuracy_hit": accuracy_hit,
                "retrieval_hit_at_3": int(result.retrieval_hit),
                "latency_ms": round(result.latency_ms, 2),
                "source_titles": " | ".join(result.source_titles),
                "response": result.response,
            }
        )
    return rows


def _configure_matplotlib() -> None:
    if FONT_PATH.exists():
        font_manager.fontManager.addfont(str(FONT_PATH))
        plt.rcParams["font.family"] = "Source Han Sans SC"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["svg.fonttype"] = "path"


def render_summary_svg(
    summary_rows: list[dict[str, Any]],
    path: Path,
    title: str,
    metric_specs: Sequence[tuple[str, str]],
    latency_metrics: Sequence[tuple[str, str]],
) -> None:
    _configure_matplotlib()
    systems = [row["system"] for row in summary_rows]
    labels = {
        "internal_rag": "你的 RAG",
        "llamaindex": "LlamaIndex",
    }
    colors = {
        "internal_rag": "#0E5A8A",
        "llamaindex": "#D9822B",
    }
    fig, (ax_quality, ax_latency) = plt.subplots(1, 2, figsize=(12.6, 5.6), gridspec_kw={"width_ratios": [1.8, 1.0]})

    bar_width = 0.36
    x_positions = list(range(len(metric_specs)))
    for offset, system in enumerate(systems):
        ax_quality.bar(
            [x + (offset - 0.5) * bar_width for x in x_positions],
            [next(row.get(key, 0.0) for row in summary_rows if row["system"] == system) for key, _label in metric_specs],
            width=bar_width,
            color=colors[system],
            label=labels[system],
        )
    ax_quality.set_xticks(x_positions)
    ax_quality.set_xticklabels([label for _key, label in metric_specs], rotation=16, ha="right")
    ax_quality.set_ylim(0, 1.05)
    ax_quality.set_title("质量与检索效果")
    ax_quality.grid(axis="y", linestyle="--", alpha=0.25)
    ax_quality.legend(frameon=False, loc="upper left")

    latency_positions = list(range(len(latency_metrics)))
    for offset, system in enumerate(systems):
        values = [next(row.get(key, 0.0) for row in summary_rows if row["system"] == system) for key, _label in latency_metrics]
        ax_latency.bar(
            [x + (offset - 0.5) * bar_width for x in latency_positions],
            values,
            width=bar_width,
            color=colors[system],
            label=labels[system],
        )
        for index, value in enumerate(values):
            ax_latency.text(index + (offset - 0.5) * bar_width, value + 6, f"{value:.0f}", ha="center", va="bottom", fontsize=9)
    ax_latency.set_xticks(latency_positions)
    ax_latency.set_xticklabels([label for _key, label in latency_metrics])
    ax_latency.set_title("端到端延迟 (ms)")
    ax_latency.grid(axis="y", linestyle="--", alpha=0.25)

    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.tight_layout()
    _ensure_dir(path.parent)
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)


def render_refusal_svg(summary_rows: list[dict[str, Any]], path: Path) -> None:
    _configure_matplotlib()
    labels = {
        "internal_rag": "你的 RAG",
        "llamaindex": "LlamaIndex",
    }
    colors = {
        "internal_rag": "#0E5A8A",
        "llamaindex": "#D9822B",
    }
    metrics = [
        ("refusal_recall", "正确拒答率"),
        ("false_refusal_rate", "误拒答率"),
        ("refusal_f1", "拒答 F1"),
    ]
    systems = [row["system"] for row in summary_rows]
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    x_positions = list(range(len(metrics)))
    bar_width = 0.36
    for offset, system in enumerate(systems):
        values = [next(row.get(key, 0.0) for row in summary_rows if row["system"] == system) for key, _label in metrics]
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
    ax.set_xticklabels([label for _key, label in metrics])
    ax.set_ylim(0, 1.05)
    ax.set_title("拒答专项对比")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    _ensure_dir(path.parent)
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)


def render_multi_dataset_svg(dataset_rows: dict[str, list[dict[str, Any]]], path: Path) -> None:
    _configure_matplotlib()
    dataset_labels = {
        "local": "本地知识库",
        "rgb": "RGB 公共基准",
    }
    system_labels = {
        "internal_rag": "你的 RAG",
        "llamaindex": "LlamaIndex",
    }
    colors = {
        "internal_rag": "#0E5A8A",
        "llamaindex": "#D9822B",
    }
    quality_metrics = [
        ("accuracy", "Accuracy"),
        ("faithfulness", "Faithfulness"),
        ("answer_relevancy", "Answer Relevancy"),
        ("context_precision", "Context Precision"),
        ("retrieval_hit_rate_at_3", "Retrieval Hit@3"),
    ]
    latency_metrics = [
        ("latency_p50_ms", "P50"),
        ("latency_p95_ms", "P95"),
    ]
    dataset_names = [name for name in ("local", "rgb") if name in dataset_rows]
    fig, axes = plt.subplots(len(dataset_names), 2, figsize=(13.0, 5.2 * len(dataset_names)), gridspec_kw={"width_ratios": [1.9, 1.0]})
    if len(dataset_names) == 1:
        axes = [axes]

    bar_width = 0.36
    for row_index, dataset_name in enumerate(dataset_names):
        summary_rows = dataset_rows[dataset_name]
        ax_quality, ax_latency = axes[row_index]
        systems = [row["system"] for row in summary_rows]
        quality_positions = list(range(len(quality_metrics)))
        for offset, system in enumerate(systems):
            ax_quality.bar(
                [x + (offset - 0.5) * bar_width for x in quality_positions],
                [next(row.get(key, 0.0) for row in summary_rows if row["system"] == system) for key, _label in quality_metrics],
                width=bar_width,
                color=colors[system],
                label=system_labels[system],
            )
        ax_quality.set_xticks(quality_positions)
        ax_quality.set_xticklabels([label for _key, label in quality_metrics], rotation=16, ha="right")
        ax_quality.set_ylim(0, 1.05)
        ax_quality.set_title(f"{dataset_labels.get(dataset_name, dataset_name)}: 质量对比")
        ax_quality.grid(axis="y", linestyle="--", alpha=0.25)
        if row_index == 0:
            ax_quality.legend(frameon=False, loc="upper left")

        latency_positions = list(range(len(latency_metrics)))
        for offset, system in enumerate(systems):
            values = [next(row.get(key, 0.0) for row in summary_rows if row["system"] == system) for key, _label in latency_metrics]
            ax_latency.bar(
                [x + (offset - 0.5) * bar_width for x in latency_positions],
                values,
                width=bar_width,
                color=colors[system],
            )
            for index, value in enumerate(values):
                ax_latency.text(index + (offset - 0.5) * bar_width, value + 6, f"{value:.0f}", ha="center", va="bottom", fontsize=9)
        ax_latency.set_xticks(latency_positions)
        ax_latency.set_xticklabels([label for _key, label in latency_metrics])
        ax_latency.set_title(f"{dataset_labels.get(dataset_name, dataset_name)}: 延迟对比 (ms)")
        ax_latency.grid(axis="y", linestyle="--", alpha=0.25)

    fig.suptitle("你的 RAG vs LlamaIndex 多数据集对比", fontsize=16, fontweight="bold")
    fig.tight_layout()
    _ensure_dir(path.parent)
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)


def _run_local_dataset() -> DatasetRunResult:
    active_docs = _load_documents(SQLITE_PATH)
    if not active_docs:
        raise RuntimeError("未找到启用中的知识库文档，无法执行本地知识库对比。")

    module = _load_llamaindex_module()
    log_path = OUTPUT_ROOT / "local_backend_server.log"
    process, port = _start_backend_process(SQLITE_PATH.parent, LOCAL_RUNTIME_PORT, log_path)
    base_url = f"http://127.0.0.1:{port}"

    try:
        query_engine, doc_by_id = build_llamaindex_query_engine(active_docs, "compare_index_local")
        warmup_query = LOCAL_POSITIVE_CASES[0].query
        _warmup_internal_backend(base_url, warmup_query)
        _warmup_llamaindex_query_engine(query_engine, warmup_query)

        cases = list(LOCAL_POSITIVE_CASES + LOCAL_NEGATIVE_CASES)
        case_map = {case.case_id: case for case in cases}
        quality_case_ids = {case.case_id for case in LOCAL_POSITIVE_CASES}

        internal_results: list[QueryResult] = []
        llamaindex_results: list[QueryResult] = []
        for case in cases:
            internal_results.append(call_internal_rag(base_url, case, doc_by_id, module))
            llamaindex_results.append(call_llamaindex_rag(query_engine, case, doc_by_id, module))

        internal_summary, internal_quality_rows = summarize_dataset("local", "internal_rag", internal_results, case_map, quality_case_ids)
        llama_summary, llama_quality_rows = summarize_dataset("local", "llamaindex", llamaindex_results, case_map, quality_case_ids)

        summary_rows = [internal_summary, llama_summary]
        detailed_rows = _build_detail_rows(internal_results + llamaindex_results, case_map)
        quality_rows = internal_quality_rows + llama_quality_rows

        summary_csv = OUTPUT_ROOT / "rag_comparison_summary.csv"
        detail_csv = OUTPUT_ROOT / "rag_comparison_details.csv"
        quality_csv = OUTPUT_ROOT / "rag_comparison_ragas_details.csv"
        summary_json = OUTPUT_ROOT / "rag_comparison_summary.json"
        summary_svg = OUTPUT_ROOT / "rag_comparison_summary.svg"
        refusal_svg = OUTPUT_ROOT / "rag_comparison_refusal.svg"

        _write_csv(summary_csv, summary_rows)
        _write_csv(detail_csv, detailed_rows)
        _write_csv(quality_csv, quality_rows)
        summary_json.write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding="utf-8")
        render_summary_svg(
            summary_rows,
            summary_svg,
            "你的 RAG vs LlamaIndex 本地知识库对比评测",
            metric_specs=[
                ("accuracy", "Accuracy"),
                ("faithfulness", "Faithfulness"),
                ("answer_relevancy", "Answer Relevancy"),
                ("context_precision", "Context Precision"),
                ("retrieval_hit_rate_at_3", "Retrieval Hit@3"),
            ],
            latency_metrics=[("latency_p50_ms", "P50"), ("latency_p95_ms", "P95")],
        )
        render_refusal_svg(summary_rows, refusal_svg)

        return DatasetRunResult(
            name="local",
            summary_rows=summary_rows,
            detailed_rows=detailed_rows,
            quality_rows=quality_rows,
            svg_path=summary_svg,
            refusal_svg_path=refusal_svg,
            extra_files=[summary_csv, detail_csv, quality_csv, summary_json],
        )
    finally:
        _stop_backend_process(process)


def _run_rgb_dataset() -> DatasetRunResult:
    module = _load_llamaindex_module()
    rgb_path = Path(module.download_rgb_dataset(str(RGB_DATASET_DIR))).resolve()
    runtime_dir = _ensure_rgb_internal_runtime(rgb_path)
    active_docs = _load_documents(runtime_dir / "wiki.db")
    if not active_docs:
        raise RuntimeError("RGB 临时语料库为空，无法执行 RGB 对比。")

    log_path = OUTPUT_ROOT / "rgb_backend_server.log"
    process, port = _start_backend_process(runtime_dir, RGB_RUNTIME_PORT, log_path)
    base_url = f"http://127.0.0.1:{port}"

    try:
        query_engine, doc_by_id = build_llamaindex_query_engine(active_docs, "compare_index_rgb")
        cases = _load_rgb_cases(module, rgb_path)
        quality_case_ids = _select_evenly_spaced_case_ids(cases, RGB_RAGAS_SAMPLE_SIZE)
        case_map = {case.case_id: case for case in cases}

        _warmup_internal_backend(base_url, cases[0].query)
        _warmup_llamaindex_query_engine(query_engine, cases[0].query)

        internal_results: list[QueryResult] = []
        llamaindex_results: list[QueryResult] = []
        total = len(cases)
        for index, case in enumerate(cases, start=1):
            if index == 1 or index % 25 == 0 or index == total:
                print(f"[RGB] evaluating {index}/{total}", flush=True)
            internal_results.append(call_internal_rag(base_url, case, doc_by_id, module))
            llamaindex_results.append(call_llamaindex_rag(query_engine, case, doc_by_id, module))

        internal_summary, internal_quality_rows = summarize_dataset("rgb", "internal_rag", internal_results, case_map, quality_case_ids)
        llama_summary, llama_quality_rows = summarize_dataset("rgb", "llamaindex", llamaindex_results, case_map, quality_case_ids)

        summary_rows = [internal_summary, llama_summary]
        detailed_rows = _build_detail_rows(internal_results + llamaindex_results, case_map)
        quality_rows = internal_quality_rows + llama_quality_rows

        summary_csv = OUTPUT_ROOT / "rgb_rag_comparison_summary.csv"
        detail_csv = OUTPUT_ROOT / "rgb_rag_comparison_details.csv"
        quality_csv = OUTPUT_ROOT / "rgb_rag_comparison_ragas_details.csv"
        summary_json = OUTPUT_ROOT / "rgb_rag_comparison_summary.json"
        summary_svg = OUTPUT_ROOT / "rgb_rag_comparison_summary.svg"

        _write_csv(summary_csv, summary_rows)
        _write_csv(detail_csv, detailed_rows)
        _write_csv(quality_csv, quality_rows)
        summary_json.write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding="utf-8")
        render_summary_svg(
            summary_rows,
            summary_svg,
            "你的 RAG vs LlamaIndex RGB 公共基准对比",
            metric_specs=[
                ("accuracy", "Accuracy"),
                ("faithfulness", "Faithfulness"),
                ("answer_relevancy", "Answer Relevancy"),
                ("context_precision", "Context Precision"),
                ("retrieval_hit_rate_at_3", "Retrieval Hit@3"),
            ],
            latency_metrics=[("latency_p50_ms", "P50"), ("latency_p95_ms", "P95")],
        )

        return DatasetRunResult(
            name="rgb",
            summary_rows=summary_rows,
            detailed_rows=detailed_rows,
            quality_rows=quality_rows,
            svg_path=summary_svg,
            extra_files=[
                summary_csv,
                detail_csv,
                quality_csv,
                summary_json,
                rgb_path,
                runtime_dir / "build_manifest.json",
                OUTPUT_ROOT / "rgb_runtime_build.log",
                OUTPUT_ROOT / "rgb_backend_server.log",
            ],
        )
    finally:
        _stop_backend_process(process)


def main() -> None:
    parser = argparse.ArgumentParser(description="比较你的 RAG 与 LlamaIndex 在不同数据集上的表现。")
    parser.add_argument(
        "--datasets",
        default="local,rgb",
        help="逗号分隔的数据集列表，可选值: local,rgb",
    )
    args = parser.parse_args()

    requested = {item.strip().lower() for item in args.datasets.split(",") if item.strip()}
    valid = {"local", "rgb"}
    unknown = requested - valid
    if unknown:
        raise ValueError(f"不支持的数据集: {', '.join(sorted(unknown))}")

    _ensure_dir(OUTPUT_ROOT)
    results: list[DatasetRunResult] = []
    if "local" in requested:
        results.append(_run_local_dataset())
    if "rgb" in requested:
        results.append(_run_rgb_dataset())

    combined_rows = [row for dataset_result in results for row in dataset_result.summary_rows]
    dataset_summary_csv = OUTPUT_ROOT / "dataset_comparison_summary.csv"
    dataset_summary_json = OUTPUT_ROOT / "dataset_comparison_summary.json"
    _write_csv(dataset_summary_csv, combined_rows)
    dataset_summary_json.write_text(json.dumps(combined_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    if len(results) > 1:
        render_multi_dataset_svg(
            {dataset_result.name: dataset_result.summary_rows for dataset_result in results},
            OUTPUT_ROOT / "dataset_comparison_summary.svg",
        )

    print(json.dumps({"summary": combined_rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
