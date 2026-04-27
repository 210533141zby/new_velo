from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class PipelineConfig:
    # 自研系统对齐配置
    ollama_base_url: str = "http://127.0.0.1:11434"
    embedding_model: str = "nomic-embed-text:latest"
    llm_model: str = "qwen2.5:7b-instruct"
    llm_temperature: float = 0.3
    chunk_size: int = 1000
    chunk_overlap: int = 200
    vector_top_k: int = 50
    bm25_top_k: int = 50
    fusion_top_k: int = 50
    rrf_k: int = 60
    rerank_model: str = "/root/Velo/backend/data/models/rerank/BAAI--bge-reranker-v2-m3"
    rerank_top_n: int = 30
    rerank_max_length: int = 1400
    final_source_count: int = 3
    chroma_collection: str = "llamaindex_rag_eval"
    ragas_default_questions: int = 50
    rgb_file_name: str = "zh_refine.json"


CONFIG = PipelineConfig()

RGB_RAW_URLS = (
    "https://raw.githubusercontent.com/chen700564/RGB/master/data/zh_refine.json",
    "https://raw.githubusercontent.com/chen700564/RGB/main/data/zh_refine.json",
    "https://raw.githubusercontent.com/chen700564/RGB/master/data/zh.json",
    "https://raw.githubusercontent.com/chen700564/RGB/main/data/zh.json",
)

REFUSAL_MARKERS = (
    "无法确定",
    "无法回答",
    "无法给出可靠回答",
    "没有找到足够相关",
    "资料不足",
    "没有相关资料",
    "无法从提供的上下文",
    "cannot determine",
    "insufficient context",
)


def _require_requests():
    try:
        import requests
    except ImportError as exc:  # pragma: no cover - 依赖缺失时给出清晰提示
        raise RuntimeError(
            "缺少 requests，请先安装依赖：pip install requests"
        ) from exc
    return requests


def _require_llamaindex():
    try:
        from llama_index.core import (
            Document,
            Settings,
            SimpleDirectoryReader,
            StorageContext,
            VectorStoreIndex,
        )
        from llama_index.core.node_parser import SentenceSplitter
        from llama_index.core.postprocessor.types import BaseNodePostprocessor
        from llama_index.core.query_engine import RetrieverQueryEngine
        from llama_index.core.retrievers import QueryFusionRetriever, VectorIndexRetriever
        from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
        from llama_index.embeddings.ollama import OllamaEmbedding
        from llama_index.llms.ollama import Ollama
        from llama_index.retrievers.bm25 import BM25Retriever
        from llama_index.vector_stores.chroma import ChromaVectorStore

        try:
            from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank
        except ImportError:
            from llama_index.core.postprocessor import SentenceTransformerRerank  # type: ignore

        import chromadb
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "缺少 LlamaIndex 评估依赖。请安装："
            " pip install llama-index-core llama-index-llms-ollama "
            "llama-index-embeddings-ollama llama-index-vector-stores-chroma "
            "llama-index-retrievers-bm25 llama-index-postprocessor-sbert-rerank chromadb"
        ) from exc

    return SimpleNamespace(
        Document=Document,
        Settings=Settings,
        SimpleDirectoryReader=SimpleDirectoryReader,
        StorageContext=StorageContext,
        VectorStoreIndex=VectorStoreIndex,
        SentenceSplitter=SentenceSplitter,
        BaseNodePostprocessor=BaseNodePostprocessor,
        RetrieverQueryEngine=RetrieverQueryEngine,
        QueryFusionRetriever=QueryFusionRetriever,
        VectorIndexRetriever=VectorIndexRetriever,
        NodeWithScore=NodeWithScore,
        QueryBundle=QueryBundle,
        TextNode=TextNode,
        OllamaEmbedding=OllamaEmbedding,
        Ollama=Ollama,
        BM25Retriever=BM25Retriever,
        ChromaVectorStore=ChromaVectorStore,
        SentenceTransformerRerank=SentenceTransformerRerank,
        chromadb=chromadb,
    )


def _require_ragas():
    try:
        from ragas import EvaluationDataset
        from ragas.dataset_schema import SingleTurnSample
        from ragas.integrations.llama_index import evaluate as ragas_llamaindex_evaluate
        from ragas.llms import LlamaIndexLLMWrapper
        from ragas.embeddings import LlamaIndexEmbeddingsWrapper

        try:
            from ragas.testset import TestsetGenerator
        except ImportError:
            from ragas.testset.synthesizers.generate import TestsetGenerator  # type: ignore

        try:
            from ragas.testset.synthesizers.testset_schema import Testset
        except ImportError:
            from ragas.testset import Testset  # type: ignore

        from ragas.metrics import Faithfulness

        try:
            from ragas.metrics import AnswerRelevancy
        except ImportError:
            from ragas.metrics import ResponseRelevancy as AnswerRelevancy  # type: ignore

        try:
            from ragas.metrics import ContextPrecision
        except ImportError:
            from ragas.metrics import LLMContextPrecisionWithReference as ContextPrecision  # type: ignore

        try:
            from ragas.metrics import ContextRecall
        except ImportError:
            from ragas.metrics import LLMContextRecall as ContextRecall  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "缺少 RAGAS 依赖。请安装：pip install ragas"
        ) from exc

    return SimpleNamespace(
        EvaluationDataset=EvaluationDataset,
        SingleTurnSample=SingleTurnSample,
        ragas_llamaindex_evaluate=ragas_llamaindex_evaluate,
        LlamaIndexLLMWrapper=LlamaIndexLLMWrapper,
        LlamaIndexEmbeddingsWrapper=LlamaIndexEmbeddingsWrapper,
        TestsetGenerator=TestsetGenerator,
        Testset=Testset,
        Faithfulness=Faithfulness,
        AnswerRelevancy=AnswerRelevancy,
        ContextPrecision=ContextPrecision,
        ContextRecall=ContextRecall,
    )


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _normalize_text(text: str) -> str:
    lowered = str(text or "").strip().lower()
    lowered = re.sub(r"\s+", "", lowered)
    lowered = re.sub(r"[，,。.!！？?；;：:\"'“”‘’（）()【】《》<>]", "", lowered)
    return lowered


def _slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", str(value or "")).strip("_").lower() or "default"


def _hash_text(text: str) -> str:
    return hashlib.md5(str(text).encode("utf-8")).hexdigest()


def _build_ollama_embedding():
    li = _require_llamaindex()
    return li.OllamaEmbedding(
        model_name=CONFIG.embedding_model,
        base_url=CONFIG.ollama_base_url,
    )


def _build_ollama_llm():
    li = _require_llamaindex()
    return li.Ollama(
        model=CONFIG.llm_model,
        temperature=CONFIG.llm_temperature,
        base_url=CONFIG.ollama_base_url,
        request_timeout=120.0,
    )


def _configure_global_settings(llm: Any, embed_model: Any, splitter: Any) -> None:
    li = _require_llamaindex()
    li.Settings.llm = llm
    li.Settings.embed_model = embed_model
    li.Settings.text_splitter = splitter


def _iter_markdown_files(docs_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in docs_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".md", ".markdown"}
    )


def _load_markdown_documents(docs_dir: str | Path) -> list[Any]:
    li = _require_llamaindex()
    docs_path = Path(docs_dir).expanduser().resolve()
    if not docs_path.exists():
        raise FileNotFoundError(f"Markdown 目录不存在: {docs_path}")
    markdown_files = _iter_markdown_files(docs_path)
    if not markdown_files:
        raise FileNotFoundError(f"目录下没有 Markdown 文档: {docs_path}")
    reader = li.SimpleDirectoryReader(input_files=[str(path) for path in markdown_files])
    return reader.load_data()


def _normalize_rgb_contexts(raw_items: Sequence[Any], prefix: str) -> list[dict[str, str]]:
    contexts: list[dict[str, str]] = []
    for idx, item in enumerate(raw_items or []):
        if isinstance(item, str):
            text = item.strip()
            if text:
                contexts.append({"title": f"{prefix}_{idx}", "text": text})
            continue
        if isinstance(item, dict):
            title = str(
                item.get("title")
                or item.get("source")
                or item.get("name")
                or f"{prefix}_{idx}"
            ).strip()
            text = str(
                item.get("text")
                or item.get("content")
                or item.get("contents")
                or item.get("body")
                or ""
            ).strip()
            if text:
                contexts.append({"title": title or f"{prefix}_{idx}", "text": text})
    return contexts


def download_rgb_dataset(save_dir: str) -> str:
    requests = _require_requests()
    save_path = _ensure_dir(Path(save_dir).expanduser().resolve()) / CONFIG.rgb_file_name
    if save_path.exists() and save_path.stat().st_size > 0:
        return str(save_path)

    last_error: Exception | None = None
    for url in RGB_RAW_URLS:
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            save_path.write_text(response.text, encoding="utf-8")
            return str(save_path)
        except Exception as exc:  # pragma: no cover - 网络失败时继续尝试其他 URL
            last_error = exc
            continue

    raise RuntimeError(f"RGB 数据集下载失败: {last_error}")


def load_rgb_testset(file_path: str) -> list[dict[str, Any]]:
    path = Path(file_path).expanduser().resolve()
    raw_text = path.read_text(encoding="utf-8").strip()
    if not raw_text:
        raise ValueError(f"RGB 文件为空: {path}")

    try:
        raw_data = json.loads(raw_text)
        if not isinstance(raw_data, list):
            raw_data = [raw_data]
    except json.JSONDecodeError:
        raw_data = [json.loads(line) for line in raw_text.splitlines() if line.strip()]

    cases: list[dict[str, Any]] = []
    for index, item in enumerate(raw_data):
        question = str(item.get("question") or item.get("query") or "").strip()
        answer = item.get("answer") or item.get("answers") or []
        if isinstance(answer, str):
            answers = [answer.strip()] if answer.strip() else []
        else:
            answers = [str(v).strip() for v in answer if str(v).strip()]
        positive_ctxs = _normalize_rgb_contexts(
            item.get("positive_ctxs") or item.get("positive") or [],
            prefix=f"positive_{index}",
        )
        negative_ctxs = _normalize_rgb_contexts(
            item.get("negative_ctxs") or item.get("negative") or [],
            prefix=f"negative_{index}",
        )
        if question:
            cases.append(
                {
                    "id": item.get("id", index),
                    "question": question,
                    "answers": answers,
                    "positive_ctxs": positive_ctxs,
                    "negative_ctxs": negative_ctxs,
                }
            )
    if not cases:
        raise ValueError(f"RGB 文件没有解析出有效样本: {path}")
    return cases


def generate_ragas_testset(
    docs_dir: str,
    llm: Any,
    embed_model: Any,
    num_questions: int = 50,
) -> list[dict[str, Any]]:
    ragas = _require_ragas()
    documents = _load_markdown_documents(docs_dir)
    generator = ragas.TestsetGenerator(
        llm=ragas.LlamaIndexLLMWrapper(llm),
        embedding_model=ragas.LlamaIndexEmbeddingsWrapper(embed_model),
    )
    testset = generator.generate_with_llamaindex_docs(
        documents=documents,
        testset_size=num_questions,
    )
    if hasattr(testset, "to_list"):
        return testset.to_list()
    if hasattr(testset, "to_evaluation_dataset"):
        dataset = testset.to_evaluation_dataset()
        if hasattr(dataset, "to_list"):
            return dataset.to_list()
    raise RuntimeError("RAGAS 生成结果不支持导出为列表，请检查版本。")


def _node_record(node: Any) -> dict[str, Any]:
    node_id = getattr(node, "node_id", None) or getattr(node, "id_", None)
    return {
        "id": node_id,
        "text": getattr(node, "text", ""),
        "metadata": dict(getattr(node, "metadata", {}) or {}),
    }


def _save_node_records(nodes: Sequence[Any], persist_dir: Path) -> Path:
    path = persist_dir / "nodes.json"
    payload = [_node_record(node) for node in nodes]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _load_nodes_from_records(persist_dir: Path) -> list[Any]:
    li = _require_llamaindex()
    path = persist_dir / "nodes.json"
    records = json.loads(path.read_text(encoding="utf-8"))
    nodes: list[Any] = []
    for record in records:
        nodes.append(
            li.TextNode(
                text=str(record.get("text") or ""),
                metadata=dict(record.get("metadata") or {}),
                id_=record.get("id"),
            )
        )
    return nodes


def _build_documents_from_rgb_cases(cases: Sequence[dict[str, Any]]) -> list[Any]:
    li = _require_llamaindex()
    dedup: dict[str, Any] = {}
    for case in cases:
        for label, contexts in (
            ("positive", case["positive_ctxs"]),
            ("negative", case["negative_ctxs"]),
        ):
            for context in contexts:
                text = str(context.get("text") or "").strip()
                if not text:
                    continue
                key = _hash_text(text)
                if key in dedup:
                    continue
                dedup[key] = li.Document(
                    text=text,
                    metadata={
                        "title": context.get("title") or key,
                        "rgb_label": label,
                        "rgb_hash": key,
                    },
                )
    return list(dedup.values())


def _open_chroma_collection(persist_dir: Path, collection_name: str, reset: bool):
    li = _require_llamaindex()
    chroma_dir = _ensure_dir(persist_dir / "chroma_db")
    client = li.chromadb.PersistentClient(path=str(chroma_dir))
    if reset:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
    collection = client.get_or_create_collection(collection_name)
    return collection


def _write_manifest(
    persist_dir: Path,
    collection_name: str,
    source_dir: str,
    document_count: int,
    node_count: int,
) -> Path:
    manifest = {
        "collection_name": collection_name,
        "source_dir": source_dir,
        "document_count": document_count,
        "node_count": node_count,
        "config": asdict(CONFIG),
    }
    path = persist_dir / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _load_manifest(persist_dir: Path) -> dict[str, Any]:
    path = persist_dir / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"索引元数据不存在: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_index(docs_dir: str, persist_dir: str) -> dict[str, Any]:
    li = _require_llamaindex()
    docs = _load_markdown_documents(docs_dir)
    embed_model = _build_ollama_embedding()
    llm = _build_ollama_llm()
    splitter = li.SentenceSplitter(
        chunk_size=CONFIG.chunk_size,
        chunk_overlap=CONFIG.chunk_overlap,
    )
    _configure_global_settings(llm=llm, embed_model=embed_model, splitter=splitter)
    nodes = splitter.get_nodes_from_documents(docs)

    persist_path = _ensure_dir(Path(persist_dir).expanduser().resolve())
    collection_name = f"{CONFIG.chroma_collection}_{_slugify(persist_path.name)}"
    chroma_collection = _open_chroma_collection(
        persist_dir=persist_path,
        collection_name=collection_name,
        reset=True,
    )
    vector_store = li.ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = li.StorageContext.from_defaults(vector_store=vector_store)
    li.VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )
    _save_node_records(nodes, persist_path)
    _write_manifest(
        persist_dir=persist_path,
        collection_name=collection_name,
        source_dir=str(Path(docs_dir).expanduser().resolve()),
        document_count=len(docs),
        node_count=len(nodes),
    )
    return {
        "persist_dir": str(persist_path),
        "collection_name": collection_name,
        "document_count": len(docs),
        "node_count": len(nodes),
    }


def _build_hybrid_retriever(vector_index: Any, nodes: Sequence[Any]) -> Any:
    li = _require_llamaindex()
    vector_retriever = li.VectorIndexRetriever(
        index=vector_index,
        similarity_top_k=CONFIG.vector_top_k,
    )
    bm25_retriever = li.BM25Retriever.from_defaults(
        nodes=list(nodes),
        similarity_top_k=CONFIG.bm25_top_k,
    )
    # LlamaIndex 的 reciprocal_rerank 实际就是 RRF，内部常数 k 固定为 60。
    return li.QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        llm=None,
        mode="reciprocal_rerank",
        similarity_top_k=CONFIG.fusion_top_k,
        num_queries=1,  # 明确关闭 query rewrite / query expansion
        use_async=True,
    )


def _build_final_topk_postprocessor():
    li = _require_llamaindex()

    class FinalTopKPostprocessor(li.BaseNodePostprocessor):
        top_k: int = CONFIG.final_source_count

        @classmethod
        def class_name(cls) -> str:
            return "FinalTopKPostprocessor"

        def _postprocess_nodes(
            self,
            nodes: list[Any],
            query_bundle: Any | None = None,
        ) -> list[Any]:
            return list(nodes[: self.top_k])

    return FinalTopKPostprocessor()


def _attach_runtime_handles(query_engine: Any, llm: Any, embed_model: Any, persist_dir: Path) -> Any:
    query_engine._velo_llm = llm
    query_engine._velo_embed_model = embed_model
    query_engine._velo_persist_dir = str(persist_dir)
    return query_engine


def load_query_engine(persist_dir: str) -> Any:
    li = _require_llamaindex()
    persist_path = Path(persist_dir).expanduser().resolve()
    manifest = _load_manifest(persist_path)
    embed_model = _build_ollama_embedding()
    llm = _build_ollama_llm()
    splitter = li.SentenceSplitter(
        chunk_size=CONFIG.chunk_size,
        chunk_overlap=CONFIG.chunk_overlap,
    )
    _configure_global_settings(llm=llm, embed_model=embed_model, splitter=splitter)

    chroma_collection = _open_chroma_collection(
        persist_dir=persist_path,
        collection_name=manifest["collection_name"],
        reset=False,
    )
    vector_store = li.ChromaVectorStore(chroma_collection=chroma_collection)
    vector_index = li.VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )
    nodes = _load_nodes_from_records(persist_path)
    hybrid_retriever = _build_hybrid_retriever(vector_index=vector_index, nodes=nodes)
    reranker = li.SentenceTransformerRerank(
        model=CONFIG.rerank_model,
        top_n=CONFIG.rerank_top_n,
        keep_retrieval_score=True,
        cross_encoder_kwargs={"max_length": CONFIG.rerank_max_length},
    )
    query_engine = li.RetrieverQueryEngine.from_args(
        retriever=hybrid_retriever,
        llm=llm,
        node_postprocessors=[reranker, _build_final_topk_postprocessor()],
        response_mode="compact",
    )
    return _attach_runtime_handles(
        query_engine=query_engine,
        llm=llm,
        embed_model=embed_model,
        persist_dir=persist_path,
    )


def _response_text(response: Any) -> str:
    if hasattr(response, "response"):
        return str(response.response)
    if hasattr(response, "text"):
        return str(response.text)
    return str(response)


def _source_node_texts(response: Any) -> list[str]:
    texts: list[str] = []
    for item in list(getattr(response, "source_nodes", []) or [])[: CONFIG.final_source_count]:
        node = getattr(item, "node", item)
        if hasattr(node, "get_content"):
            try:
                texts.append(str(node.get_content()))
                continue
            except Exception:
                pass
        texts.append(str(getattr(node, "text", "")))
    return texts


def _contains_refusal(text: str) -> bool:
    normalized = _normalize_text(text)
    return any(_normalize_text(marker) in normalized for marker in REFUSAL_MARKERS)


def _match_any_answer(prediction: str, answers: Sequence[str]) -> bool:
    pred_norm = _normalize_text(prediction)
    if not pred_norm:
        return False
    for answer in answers:
        answer_norm = _normalize_text(answer)
        if not answer_norm:
            continue
        if pred_norm == answer_norm:
            return True
        if answer_norm in pred_norm or pred_norm in answer_norm:
            return True
    return False


def _retrieval_hits_positive(source_texts: Sequence[str], positives: Sequence[dict[str, str]]) -> bool:
    normalized_sources = [_normalize_text(text) for text in source_texts if text]
    normalized_positives = [
        _normalize_text(item.get("text", "")) for item in positives if item.get("text")
    ]
    for source in normalized_sources:
        for positive in normalized_positives:
            if not source or not positive:
                continue
            if source in positive or positive in source:
                return True
    return False


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
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


def _rgb_outputs_dir(testset_path: str) -> Path:
    return _ensure_dir(Path(testset_path).expanduser().resolve().parent / "outputs")


def _ragas_outputs_dir(ragas_path: str) -> Path:
    return _ensure_dir(Path(ragas_path).expanduser().resolve().parent / "outputs")


def _build_rgb_query_engine(testset_path: str) -> Any:
    cases = load_rgb_testset(testset_path)
    documents = _build_documents_from_rgb_cases(cases)
    persist_dir = Path(testset_path).expanduser().resolve().parent / "rgb_index"
    li = _require_llamaindex()
    embed_model = _build_ollama_embedding()
    llm = _build_ollama_llm()
    splitter = li.SentenceSplitter(
        chunk_size=CONFIG.chunk_size,
        chunk_overlap=CONFIG.chunk_overlap,
    )
    _configure_global_settings(llm=llm, embed_model=embed_model, splitter=splitter)
    nodes = splitter.get_nodes_from_documents(documents)
    collection_name = f"{CONFIG.chroma_collection}_rgb"
    chroma_collection = _open_chroma_collection(
        persist_dir=persist_dir,
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
    hybrid_retriever = _build_hybrid_retriever(vector_index=vector_index, nodes=nodes)
    reranker = li.SentenceTransformerRerank(
        model=CONFIG.rerank_model,
        top_n=CONFIG.rerank_top_n,
        keep_retrieval_score=True,
        cross_encoder_kwargs={"max_length": CONFIG.rerank_max_length},
    )
    query_engine = li.RetrieverQueryEngine.from_args(
        retriever=hybrid_retriever,
        llm=llm,
        node_postprocessors=[reranker, _build_final_topk_postprocessor()],
        response_mode="compact",
    )
    return _attach_runtime_handles(
        query_engine=query_engine,
        llm=llm,
        embed_model=embed_model,
        persist_dir=persist_dir,
    )


def evaluate_rgb(query_engine: Any, testset_path: str) -> dict[str, Any]:
    cases = load_rgb_testset(testset_path)
    rgb_engine = _build_rgb_query_engine(testset_path)
    rows: list[dict[str, Any]] = []
    exact_or_soft_hits = 0
    refusal_count = 0
    retrieval_hit_count = 0
    total_latency = 0.0

    for case in cases:
        start = time.perf_counter()
        response = rgb_engine.query(case["question"])
        latency = time.perf_counter() - start
        answer_text = _response_text(response)
        source_texts = _source_node_texts(response)
        hit = _match_any_answer(answer_text, case["answers"])
        refusal = _contains_refusal(answer_text)
        retrieval_hit = _retrieval_hits_positive(source_texts, case["positive_ctxs"])
        exact_or_soft_hits += int(hit)
        refusal_count += int(refusal)
        retrieval_hit_count += int(retrieval_hit)
        total_latency += latency
        rows.append(
            {
                "id": case["id"],
                "question": case["question"],
                "prediction": answer_text,
                "gold_answers": " || ".join(case["answers"]),
                "hit": int(hit),
                "refusal": int(refusal),
                "retrieval_hit_at_3": int(retrieval_hit),
                "latency_sec": round(latency, 4),
            }
        )

    total = len(cases)
    summary = {
        "benchmark": "RGB",
        "sample_count": total,
        "accuracy": round(exact_or_soft_hits / total, 4) if total else 0.0,
        "refusal_rate": round(refusal_count / total, 4) if total else 0.0,
        "retrieval_hit_rate_at_3": round(retrieval_hit_count / total, 4) if total else 0.0,
        "avg_latency_sec": round(total_latency / total, 4) if total else 0.0,
    }
    outputs_dir = _rgb_outputs_dir(testset_path)
    _write_csv(outputs_dir / "rgb_detailed_results.csv", rows)
    _write_csv(outputs_dir / "rgb_summary.csv", [summary])
    return summary


def _load_ragas_testset(ragas_path: str) -> tuple[Any, Any]:
    ragas = _require_ragas()
    path = Path(ragas_path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        if "samples" in payload and isinstance(payload["samples"], list):
            payload = payload["samples"]
        else:
            payload = [payload]
    testset = ragas.Testset.from_list(payload)
    return testset, testset.to_evaluation_dataset()


def evaluate_ragas(query_engine: Any, testset: str | Sequence[dict[str, Any]]) -> dict[str, Any]:
    ragas = _require_ragas()
    if isinstance(testset, str):
        _, evaluation_dataset = _load_ragas_testset(testset)
        outputs_dir = _ragas_outputs_dir(testset)
    else:
        evaluation_dataset = ragas.EvaluationDataset.from_list(list(testset))
        outputs_dir = _ensure_dir(Path.cwd() / "ragas_outputs")

    metrics = [
        ragas.Faithfulness(),
        ragas.ContextPrecision(),
        ragas.AnswerRelevancy(),
        ragas.ContextRecall(),
    ]
    result = ragas.ragas_llamaindex_evaluate(
        query_engine=query_engine,
        dataset=evaluation_dataset,
        metrics=metrics,
        llm=query_engine._velo_llm,
        embeddings=query_engine._velo_embed_model,
        raise_exceptions=False,
        show_progress=True,
    )

    summary = {"benchmark": "RAGAS"}
    summary.update({key: round(float(value), 4) for key, value in result._repr_dict.items()})
    summary["sample_count"] = len(evaluation_dataset)

    rows = []
    for item in result.scores:
        row = {}
        row.update(item)
        rows.append(row)

    _write_csv(outputs_dir / "ragas_detailed_results.csv", rows)
    _write_csv(outputs_dir / "ragas_summary.csv", [summary])
    return summary


def prepare_datasets(data_dir: str, docs_dir: str) -> dict[str, str]:
    data_root = _ensure_dir(Path(data_dir).expanduser().resolve())
    rgb_dir = _ensure_dir(data_root / "rgb")
    ragas_dir = _ensure_dir(data_root / "ragas")

    rgb_path = download_rgb_dataset(str(rgb_dir))
    llm = _build_ollama_llm()
    embed_model = _build_ollama_embedding()
    ragas_samples = generate_ragas_testset(
        docs_dir=docs_dir,
        llm=llm,
        embed_model=embed_model,
        num_questions=CONFIG.ragas_default_questions,
    )
    ragas_path = ragas_dir / "ragas_synthetic_testset.json"
    ragas_path.write_text(
        json.dumps(ragas_samples, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"rgb_path": rgb_path, "ragas_path": str(ragas_path)}


def run_evaluation(query_engine: Any, rgb_path: str, ragas_path: str) -> dict[str, Any]:
    rgb_summary = evaluate_rgb(query_engine=query_engine, testset_path=rgb_path)
    ragas_summary = evaluate_ragas(query_engine=query_engine, testset=ragas_path)
    output_root = _ensure_dir(Path(ragas_path).expanduser().resolve().parent / "outputs")
    merged = [rgb_summary, ragas_summary]
    _write_csv(output_root / "evaluation_summary.csv", merged)
    return {"rgb": rgb_summary, "ragas": ragas_summary}


def smoke_test_query_engine(query_engine: Any, query: str = "这份文档主要讲了什么？") -> dict[str, Any]:
    response = query_engine.query(query)
    sources = _source_node_texts(response)
    return {"query": query, "response": _response_text(response), "source_nodes": sources}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LlamaIndex RAG 对比评估脚本")
    parser.add_argument(
        "--docs-dir",
        default="./docs",
        help="Markdown 文档目录，用于构建私有语料索引与生成 RAGAS 测试集。",
    )
    parser.add_argument(
        "--data-dir",
        default="./llamaindex_rag_eval/data",
        help="数据集缓存目录，用于保存 RGB 与 RAGAS 测试集。",
    )
    parser.add_argument(
        "--persist-dir",
        default="./llamaindex_rag_eval/private_index",
        help="LlamaIndex + Chroma 持久化目录。",
    )
    parser.add_argument(
        "--action",
        choices=("prepare", "build", "evaluate", "all", "smoke"),
        default="all",
        help="执行动作。",
    )
    parser.add_argument(
        "--rgb-path",
        default="",
        help="RGB 数据集路径；为空时走 prepare_datasets 自动下载。",
    )
    parser.add_argument(
        "--ragas-path",
        default="",
        help="RAGAS 测试集路径；为空时走 prepare_datasets 自动生成。",
    )
    parser.add_argument(
        "--smoke-query",
        default="这份文档主要讲了什么？",
        help="smoke 测试时使用的问题。",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    dataset_paths: dict[str, str] = {}
    if args.action in {"prepare", "all"}:
        dataset_paths = prepare_datasets(data_dir=args.data_dir, docs_dir=args.docs_dir)
        print(json.dumps({"prepared": dataset_paths}, ensure_ascii=False, indent=2))

    if args.action in {"build", "all"}:
        build_result = build_index(docs_dir=args.docs_dir, persist_dir=args.persist_dir)
        print(json.dumps({"build": build_result}, ensure_ascii=False, indent=2))

    if args.action in {"evaluate", "all", "smoke"}:
        query_engine = load_query_engine(persist_dir=args.persist_dir)
        smoke = smoke_test_query_engine(query_engine=query_engine, query=args.smoke_query)
        print(json.dumps({"smoke_test": smoke}, ensure_ascii=False, indent=2))

        if args.action in {"evaluate", "all"}:
            rgb_path = args.rgb_path or dataset_paths.get("rgb_path")
            ragas_path = args.ragas_path or dataset_paths.get("ragas_path")
            if not rgb_path or not ragas_path:
                raise ValueError("evaluate/all 模式下必须提供 rgb_path 和 ragas_path，或先执行 prepare。")
            summary = run_evaluation(
                query_engine=query_engine,
                rgb_path=rgb_path,
                ragas_path=ragas_path,
            )
            print(json.dumps({"evaluation": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
