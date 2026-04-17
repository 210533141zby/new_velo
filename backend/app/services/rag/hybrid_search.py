"""
=============================================================================
文件: hybrid_retrieval.py
描述: 将实验中的“自适应混合检索”落到后端 RAG 主链路

核心目标：
1. 复用主实验里的查询画像、自适应融合、RRF 和混合候选池思想。
2. 在 Chroma 向量召回之外，补一条文档级 BM25 词法检索链路。
3. 给 rerank 阶段提供更干净、更稳定的候选池，同时保留可解释分数。
=============================================================================
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from threading import Lock
from types import SimpleNamespace
from typing import Any, Sequence

from app.core.config import settings
from app.logger import logger

try:
    import jieba
except Exception:  # pragma: no cover - 依赖缺失时自动走正则分词兜底。
    jieba = None

try:
    from rank_bm25 import BM25Okapi
except Exception:  # pragma: no cover - 依赖缺失时自动走词法重叠兜底。
    BM25Okapi = None

IDENTIFIER_PATTERN = re.compile(r'[a-z]+(?:[-_][a-z]+)*-?\d+(?:\.\d+)?|\d+(?:\.\d+)?', re.IGNORECASE)

_index_lock = Lock()
_hybrid_index: 'HybridLexicalIndex | None' = None
_hybrid_index_signature: str | None = None
_hybrid_index_needs_refresh = True
_jieba_warning_emitted = False
_bm25_warning_emitted = False


@dataclass(frozen=True)
class IndexedDocument:
    doc_id: int
    title: str
    content_preview: str
    bm25_text: str
    tokens: list[str]
    identifier_tokens: set[str]


def clean_text(text: str, limit: int | None = None) -> str:
    normalized = re.sub(r'\s+', ' ', str(text or '').strip())
    if not normalized:
        normalized = '空白内容'
    if limit is not None:
        return normalized[:limit] or '空白内容'
    return normalized


def extract_identifiers(text: str) -> set[str]:
    identifiers: set[str] = set()
    for token in IDENTIFIER_PATTERN.findall(str(text or '').lower().replace('_', '-')):
        normalized = token.strip('-')
        if not normalized:
            continue
        identifiers.add(normalized)
        for number in re.findall(r'\d+(?:\.\d+)?', normalized):
            identifiers.add(number.lstrip('0') or '0')
    return identifiers


def has_identifier(text: str) -> bool:
    return bool(extract_identifiers(text))


def tokenize_for_bm25(text: str) -> list[str]:
    global _jieba_warning_emitted

    cleaned = clean_text(text).lower()
    if jieba is not None:
        raw_tokens = [token.strip() for token in jieba.lcut(cleaned) if token.strip()]
    else:
        if not _jieba_warning_emitted:
            logger.warning('jieba 未安装，Hybrid 检索将回退到正则分词')
            _jieba_warning_emitted = True
        raw_tokens = re.findall(r'[0-9a-z]+|[\u4e00-\u9fff]+', cleaned)

    normalized: list[str] = []
    for token in raw_tokens:
        if re.fullmatch(r'[\u4e00-\u9fff]+', token):
            normalized.append(token)
        else:
            parts = re.findall(r'[0-9a-z]+|[\u4e00-\u9fff]+', token)
            normalized.extend(parts if parts else [token])
    return [token for token in normalized if token]


def coverage_ratio(query_tokens: set[str], candidate_tokens: set[str]) -> float:
    if not query_tokens or not candidate_tokens:
        return 0.0
    return len(query_tokens & candidate_tokens) / len(query_tokens)


def normalize_scores(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}

    values = list(scores.values())
    low = min(values)
    high = max(values)
    if abs(high - low) <= 1e-12:
        return {key: 1.0 for key in scores}
    return {key: (value - low) / (high - low) for key, value in scores.items()}


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return fused


def compute_query_profile(query: str, idf_lookup: dict[str, float] | None = None) -> dict[str, float]:
    tokens = tokenize_for_bm25(query)
    token_set = set(tokens)
    idf_lookup = idf_lookup or {}
    idf_values = [float(idf_lookup.get(token, 0.0)) for token in token_set]
    avg_idf = (sum(idf_values) / len(idf_values)) if idf_values else 0.0
    max_idf = max(idf_values) if idf_values else 0.0

    lexical_weight = 0.36
    if has_identifier(query):
        lexical_weight += 0.18
    if len(tokens) <= 4:
        lexical_weight += 0.14
    if len(tokens) >= 10:
        lexical_weight -= 0.08
    if max_idf > max(avg_idf * 1.35, 2.5):
        lexical_weight += 0.10

    lexical_weight = max(0.24, min(0.78, lexical_weight))
    return {
        'lexical_weight': lexical_weight,
        'dense_weight': 1.0 - lexical_weight,
        'token_count': float(len(tokens)),
        'has_identifier': 1.0 if has_identifier(query) else 0.0,
    }


def build_index_signature(documents: Sequence[Any]) -> str:
    digest = hashlib.sha1()
    for document in documents:
        updated_at = getattr(document, 'updated_at', None)
        updated_value = updated_at.isoformat() if updated_at is not None else ''
        digest.update(
            f'{getattr(document, "id", "")}|{updated_value}|{len(str(getattr(document, "title", "") or ""))}|'
            f'{len(str(getattr(document, "content", "") or ""))}\n'.encode('utf-8')
        )
    return f'n{len(documents)}_{digest.hexdigest()[:16]}'


def _build_indexed_document(document: Any) -> IndexedDocument | None:
    doc_id = getattr(document, 'id', None)
    title = clean_text(getattr(document, 'title', ''), limit=240)
    content = clean_text(getattr(document, 'content', ''), limit=settings.RERANK_MAX_INPUT_CHARS)
    if doc_id is None or not content:
        return None

    # 标题重复一次，提升精确标题和文档名查询在 BM25 中的权重。
    bm25_text = clean_text(f'{title}\n{title}\n{content}', limit=settings.RERANK_MAX_INPUT_CHARS * 2)
    tokens = tokenize_for_bm25(bm25_text)
    return IndexedDocument(
        doc_id=int(doc_id),
        title=title,
        content_preview=content,
        bm25_text=bm25_text,
        tokens=tokens,
        identifier_tokens=extract_identifiers(f'{title} {content[:500]}'),
    )


class HybridLexicalIndex:
    def __init__(self, documents: Sequence[IndexedDocument]) -> None:
        global _bm25_warning_emitted

        self.documents = list(documents)
        self.by_id = {document.doc_id: document for document in self.documents}
        self._tokenized_corpus = [document.tokens or ['空白内容'] for document in self.documents]

        if self.documents and BM25Okapi is not None:
            self._bm25 = BM25Okapi(self._tokenized_corpus)
            self.idf_lookup = dict(getattr(self._bm25, 'idf', {}))
        else:
            self._bm25 = None
            self.idf_lookup = {}
            if self.documents and BM25Okapi is None and not _bm25_warning_emitted:
                logger.warning('rank_bm25 未安装，Hybrid 检索将回退到词法覆盖率排序')
                _bm25_warning_emitted = True

    def rank_bm25(self, query: str, top_k: int) -> tuple[list[int], dict[int, float], set[str]]:
        if not self.documents or top_k <= 0:
            return [], {}, set()

        query_tokens = tokenize_for_bm25(query)
        query_token_set = set(query_tokens)
        if not query_tokens:
            return [], {}, query_token_set

        raw_scores: dict[int, float] = {}
        if self._bm25 is not None:
            scores = self._bm25.get_scores(query_tokens)
            for index, score in enumerate(scores):
                raw_scores[self.documents[index].doc_id] = float(score)
        else:
            for document in self.documents:
                raw_scores[document.doc_id] = coverage_ratio(query_token_set, set(document.tokens))

        ranked_doc_ids = [
            doc_id
            for doc_id, _score in sorted(raw_scores.items(), key=lambda item: item[1], reverse=True)
            if _score > 0.0
        ][:top_k]
        return ranked_doc_ids, {doc_id: raw_scores[doc_id] for doc_id in ranked_doc_ids}, query_token_set


def hybrid_index_needs_refresh() -> bool:
    with _index_lock:
        return _hybrid_index is None or _hybrid_index_needs_refresh


def get_hybrid_index() -> HybridLexicalIndex:
    with _index_lock:
        return _hybrid_index or HybridLexicalIndex([])


def ensure_hybrid_index(documents: Sequence[Any]) -> HybridLexicalIndex:
    global _hybrid_index
    global _hybrid_index_signature
    global _hybrid_index_needs_refresh

    signature = build_index_signature(documents)
    with _index_lock:
        if _hybrid_index is not None and _hybrid_index_signature == signature and not _hybrid_index_needs_refresh:
            return _hybrid_index

        indexed_documents = []
        for document in documents:
            indexed = _build_indexed_document(document)
            if indexed is not None:
                indexed_documents.append(indexed)

        _hybrid_index = HybridLexicalIndex(indexed_documents)
        _hybrid_index_signature = signature
        _hybrid_index_needs_refresh = False
        logger.info(
            'Hybrid 词法索引已刷新',
            extra={
                'extra_data': {
                    'event': 'rag_hybrid_index_refreshed',
                    'document_count': len(indexed_documents),
                    'signature': signature,
                }
            },
        )
        return _hybrid_index


def invalidate_hybrid_index() -> None:
    global _hybrid_index_needs_refresh

    with _index_lock:
        _hybrid_index_needs_refresh = True


def _clone_doc(doc: Any) -> Any:
    return SimpleNamespace(
        metadata=dict(getattr(doc, 'metadata', {}) or {}),
        page_content=str(getattr(doc, 'page_content', '') or ''),
    )


def _build_synthetic_doc(indexed_document: IndexedDocument) -> Any:
    return SimpleNamespace(
        metadata={
            'source': indexed_document.title,
            'doc_id': indexed_document.doc_id,
        },
        page_content=indexed_document.content_preview,
    )


def _collapse_vector_matches(scored_matches: Sequence[tuple[Any, float]]) -> list[tuple[int, Any, float]]:
    best_by_doc: dict[int, tuple[Any, float]] = {}
    for doc, score in scored_matches:
        doc_id = doc.metadata.get('doc_id')
        if doc_id is None:
            continue
        numeric_doc_id = int(doc_id)
        best = best_by_doc.get(numeric_doc_id)
        if best is None or float(score) > float(best[1]):
            best_by_doc[numeric_doc_id] = (doc, float(score))

    ranked = sorted(best_by_doc.items(), key=lambda item: item[1][1], reverse=True)
    return [(doc_id, doc, score) for doc_id, (doc, score) in ranked]


def _fill_candidate_ids(target: list[int], source: Sequence[int], limit: int) -> None:
    for doc_id in source:
        if doc_id not in target:
            target.append(doc_id)
        if len(target) >= limit:
            return


def build_hybrid_candidates(
    query: str,
    vector_matches: Sequence[tuple[Any, float]],
    lexical_index: HybridLexicalIndex,
    bm25_query: str | None = None,
    vector_limit: int | None = None,
    bm25_limit: int | None = None,
    candidate_limit: int | None = None,
) -> list[tuple[Any, float]]:
    vector_limit = vector_limit or settings.RAG_VECTOR_SEARCH_LIMIT
    bm25_limit = bm25_limit or settings.RAG_BM25_SEARCH_LIMIT
    candidate_limit = candidate_limit or settings.RAG_HYBRID_CANDIDATE_LIMIT

    collapsed_vector = _collapse_vector_matches(vector_matches)[:vector_limit]
    vector_doc_ids = [doc_id for doc_id, _doc, _score in collapsed_vector]
    vector_score_map = {doc_id: float(score) for doc_id, _doc, score in collapsed_vector}
    vector_doc_map = {doc_id: doc for doc_id, doc, _score in collapsed_vector}

    bm25_doc_ids, bm25_score_map, _bm25_query_token_set = lexical_index.rank_bm25(bm25_query or query, bm25_limit)
    query_token_set = set(tokenize_for_bm25(bm25_query or query))
    fused_doc_ids = set(vector_doc_ids) | set(bm25_doc_ids)
    if not fused_doc_ids:
        return []

    normalized_dense = normalize_scores({doc_id: vector_score_map.get(doc_id, 0.0) for doc_id in fused_doc_ids})
    normalized_bm25 = normalize_scores({doc_id: bm25_score_map.get(doc_id, 0.0) for doc_id in fused_doc_ids})
    rrf_scores = reciprocal_rank_fusion([vector_doc_ids, bm25_doc_ids])
    normalized_rrf = normalize_scores({doc_id: rrf_scores.get(doc_id, 0.0) for doc_id in fused_doc_ids})

    profile = compute_query_profile(query, lexical_index.idf_lookup)
    query_identifiers = extract_identifiers(query)
    adaptive_scores: dict[int, float] = {}
    prepared_docs: dict[int, Any] = {}

    for doc_id in fused_doc_ids:
        indexed_document = lexical_index.by_id.get(doc_id)
        if indexed_document is None:
            continue

        doc = _clone_doc(vector_doc_map[doc_id]) if doc_id in vector_doc_map else _build_synthetic_doc(indexed_document)
        if doc_id in vector_doc_map and bm25_score_map.get(doc_id, 0.0) > 0.0 and len(query_token_set) >= 2:
            chunk_token_coverage = coverage_ratio(query_token_set, set(tokenize_for_bm25(getattr(doc, 'page_content', '') or '')))
            if chunk_token_coverage < 0.30:
                doc = _build_synthetic_doc(indexed_document)
                doc.metadata['candidate_source'] = 'hybrid_lexical_fallback'
        coverage = coverage_ratio(query_token_set, set(indexed_document.tokens))
        identifier_overlap = coverage_ratio(query_identifiers, indexed_document.identifier_tokens)
        adaptive_score = (
            profile['dense_weight'] * normalized_dense.get(doc_id, 0.0)
            + profile['lexical_weight'] * normalized_bm25.get(doc_id, 0.0)
            + normalized_rrf.get(doc_id, 0.0) * 0.26
            + coverage * 0.08
            + identifier_overlap * 0.03
        )

        doc.metadata.update(
            {
                'vector_score': vector_score_map.get(doc_id, 0.0),
                'bm25_score': bm25_score_map.get(doc_id, 0.0),
                'rrf_score': rrf_scores.get(doc_id, 0.0),
                'adaptive_score': adaptive_score,
                'coverage_score': coverage,
                'identifier_overlap': identifier_overlap,
                'lexical_weight': profile['lexical_weight'],
                'dense_weight': profile['dense_weight'],
                'candidate_source': 'hybrid',
            }
        )

        adaptive_scores[doc_id] = adaptive_score
        prepared_docs[doc_id] = doc

    adaptive_rank = [doc_id for doc_id, _score in sorted(adaptive_scores.items(), key=lambda item: item[1], reverse=True)]
    rrf_rank = [doc_id for doc_id, _score in sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)]

    mixed_candidates: list[int] = []
    seed_k = min(candidate_limit, max(1, candidate_limit // 2 + 4))
    _fill_candidate_ids(mixed_candidates, adaptive_rank[:seed_k], candidate_limit)
    _fill_candidate_ids(mixed_candidates, rrf_rank[:seed_k], candidate_limit)
    _fill_candidate_ids(mixed_candidates, adaptive_rank, candidate_limit)
    _fill_candidate_ids(mixed_candidates, vector_doc_ids, candidate_limit)
    _fill_candidate_ids(mixed_candidates, bm25_doc_ids, candidate_limit)

    return [(prepared_docs[doc_id], adaptive_scores[doc_id]) for doc_id in mixed_candidates if doc_id in prepared_docs]
