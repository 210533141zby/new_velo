"""
=============================================================================
文件: rerank_service.py
描述: RAG 检索结果重排服务

核心功能：
1. 统一管理检索候选的词法特征、重排分数和最终置信度。
2. 使用 CrossEncoder 对候选文档做二次排序，减少向量召回的误命中。
3. 生成前端直接可展示的引用排行和置信度信息。
=============================================================================
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from threading import Lock
from typing import Any, Sequence

from app.core.config import settings
from app.logger import logger

LEXICAL_STOP_TOKENS = {
    'a',
    'an',
    'and',
    'are',
    'is',
    'model',
    'the',
    'what',
    'which',
    '你',
    '它',
    '他',
    '她',
    '吗',
    '呢',
    '了',
    '什',
    '么',
    '和',
    '在',
    '是',
    '有',
    '的',
    '讲',
    '请',
    '问',
}

LOCAL_MODEL_WEIGHT_FILES = (
    'model.safetensors',
    'model.safetensors.index.json',
    'pytorch_model.bin',
)
IDENTIFIER_PATTERN = re.compile(r'[a-z]+(?:[-_][a-z]+)*-?\d+(?:\.\d+)?|\d+(?:\.\d+)?', re.IGNORECASE)

_reranker_instance: 'CrossEncoderReranker | None' = None


def _document_key(doc: Any) -> Any:
    return doc.metadata.get('doc_id') or str(doc.metadata.get('source') or '')


def _model_directory(model_name: str) -> Path:
    model_slug = model_name.replace('/', '--').replace(':', '-')
    model_dir = settings.rerank_cache_directory / model_slug
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def _has_local_model_files(model_dir: Path) -> bool:
    return all((model_dir / required_file).exists() for required_file in ('config.json',)) and any(
        (model_dir / filename).exists() for filename in LOCAL_MODEL_WEIGHT_FILES
    )


def normalize_lookup_text(value: str) -> str:
    normalized = re.sub(r'\s+', '', (value or '').strip().lower())
    return re.sub(r'[^0-9a-z\u4e00-\u9fff]+', '', normalized)


def tokenize_text(value: str) -> set[str]:
    tokens = set(re.findall(r'[0-9a-z]+|[\u4e00-\u9fff]', (value or '').lower()))
    return {token for token in tokens if token and token not in LEXICAL_STOP_TOKENS}


def coverage_ratio(query_tokens: set[str], candidate_tokens: set[str]) -> float:
    if not query_tokens or not candidate_tokens:
        return 0.0
    return len(query_tokens & candidate_tokens) / len(query_tokens)


def extract_identifier_tokens(value: str) -> set[str]:
    normalized = str(value or '').lower().replace('_', '-')
    identifiers: set[str] = set()
    for token in IDENTIFIER_PATTERN.findall(normalized):
        normalized_token = token.strip('-')
        if not normalized_token:
            continue
        identifiers.add(normalized_token)
        for number in re.findall(r'\d+(?:\.\d+)?', normalized_token):
            identifiers.add(number.lstrip('0') or '0')
    return identifiers


def has_identifier_mismatch(query: str, title: str) -> bool:
    query_identifiers = extract_identifier_tokens(query)
    title_identifiers = extract_identifier_tokens(title)
    if not query_identifiers or not title_identifiers:
        return False
    return query_identifiers.isdisjoint(title_identifiers)


def normalize_relevance_score(score: float | None) -> float:
    if score is None:
        return 0.0
    return max(0.0, min(float(score), 1.0))


def normalize_rerank_score(score: float | None) -> float:
    if score is None:
        return 0.0

    numeric_score = float(score)
    if 0.0 <= numeric_score <= 1.0:
        return numeric_score

    if numeric_score >= 18:
        return 1.0
    if numeric_score <= -18:
        return 0.0
    return 1.0 / (1.0 + math.exp(-numeric_score))


def normalize_rerank_scores_batch(scores: Sequence[float]) -> list[float]:
    if not scores:
        return []

    numeric_scores = [float(score) for score in scores]
    low = min(numeric_scores)
    high = max(numeric_scores)
    if math.isclose(low, high):
        if 0.0 <= numeric_scores[0] <= 1.0:
            fallback = numeric_scores[0]
        else:
            fallback = normalize_rerank_score(numeric_scores[0])
        return [fallback for _ in numeric_scores]

    # 无论模型输出的是 logits 还是一组非常贴近 0 的“伪概率”，
    # 后续筛选都更需要“这一批候选里谁相对更相关”。
    # 因此这里统一做按查询批次的 min-max 归一化。
    return [(score - low) / (high - low) for score in numeric_scores]


def build_rerank_input(doc: Any) -> str:
    title = str(doc.metadata.get('source') or '未命名文档').strip()
    content = re.sub(r'\s+', ' ', str(doc.page_content or '').strip())
    preview = content[: settings.RERANK_MAX_INPUT_CHARS]
    return f'标题：{title}\n内容：{preview}'


def build_retrieved_match(
    query: str,
    doc: Any,
    vector_score: float,
    rerank_score: float | None = None,
) -> dict[str, Any]:
    # Hybrid 检索接入后，候选的“粗排先验”优先使用自适应融合分数；
    # 若当前候选仍来自旧链路，则回退到向量相似度。
    retrieval_prior = doc.metadata.get('adaptive_score', vector_score)
    raw_score = normalize_relevance_score(retrieval_prior)
    rerank_confidence = normalize_rerank_score(rerank_score) if rerank_score is not None else raw_score
    title = str(doc.metadata.get('source') or 'Unknown')
    normalized_query = normalize_lookup_text(query)
    normalized_title = normalize_lookup_text(title)
    exact_title_hit = bool(normalized_title and normalized_title in normalized_query and not has_identifier_mismatch(query, title))

    query_tokens = tokenize_text(query)
    title_tokens = tokenize_text(title)
    content_tokens = tokenize_text(str(doc.page_content or '')[:500])
    query_identifiers = extract_identifier_tokens(query)
    title_identifiers = extract_identifier_tokens(title)
    title_overlap = coverage_ratio(query_tokens, title_tokens)
    metadata_coverage = normalize_relevance_score(doc.metadata.get('coverage_score'))
    content_overlap = max(coverage_ratio(query_tokens, content_tokens), metadata_coverage)
    lexical_anchor = max(title_overlap, content_overlap)
    metadata_identifier_overlap = normalize_relevance_score(doc.metadata.get('identifier_overlap'))
    identifier_overlap = max(coverage_ratio(query_identifiers, title_identifiers), metadata_identifier_overlap)
    identifier_mismatch = bool(query_identifiers and title_identifiers and identifier_overlap == 0.0)
    vector_prior = normalize_relevance_score(doc.metadata.get('vector_score'))
    hybrid_prior = max(raw_score, vector_prior)

    # 精排仍然重要，但当前系统接入的是“自适应 hybrid + rerank”而不是“只信 rerank”。
    # 对带强关键词、强编号的知识库查询，检索阶段的先验和词法证据要保留更高权重。
    final_score = rerank_confidence * 0.58 + hybrid_prior * 0.24 + lexical_anchor * 0.14 + identifier_overlap * 0.04
    if exact_title_hit:
        final_score = max(final_score + 0.06, 0.90)
    elif title_overlap >= 0.6:
        final_score += 0.04
    elif lexical_anchor == 0 and rerank_confidence < settings.RERANK_MIN_SCORE:
        final_score -= 0.10
    if identifier_mismatch:
        # 消融实验表明标识符约束不能做成硬规则，因此这里只保留轻量惩罚。
        final_score -= 0.08
    elif query_identifiers and identifier_overlap == 1.0:
        final_score += 0.02

    final_score = max(0.0, min(final_score, 0.99))

    return {
        'doc': doc,
        'title': title,
        'raw_score': raw_score,
        'rerank_score': rerank_confidence,
        'title_overlap': title_overlap,
        'content_overlap': content_overlap,
        'identifier_overlap': identifier_overlap,
        'identifier_mismatch': identifier_mismatch,
        'exact_title_hit': exact_title_hit,
        'candidate_source': doc.metadata.get('candidate_source') or 'vector',
        'final_score': final_score,
        'confidence': int(round(final_score * 100)),
    }


def rank_retrieved_matches(
    query: str,
    scored_matches: Sequence[tuple[Any, float]],
    rerank_scores: dict[Any, float] | None = None,
) -> list[dict[str, Any]]:
    rerank_scores = rerank_scores or {}
    best_match_by_source: dict[Any, dict[str, Any]] = {}

    for doc, score in scored_matches:
        source_key = _document_key(doc)
        ranked_match = build_retrieved_match(query, doc, score, rerank_scores.get(source_key))
        existing = best_match_by_source.get(source_key)
        if existing is None or ranked_match['final_score'] > existing['final_score']:
            best_match_by_source[source_key] = ranked_match

    return sorted(
        best_match_by_source.values(),
        key=lambda item: (item['final_score'], item['rerank_score'], item['raw_score']),
        reverse=True,
    )


class CrossEncoderReranker:
    def __init__(self) -> None:
        self._model = None
        self._load_lock = Lock()
        self._load_failed = False
        self._resolved_model_name: str | None = None
        self._resolved_model_dir: Path | None = None
        self._resolved_device: str | None = None

    def _resolve_device(self) -> str:
        if settings.RERANK_DEVICE != 'auto':
            return settings.RERANK_DEVICE

        try:
            import torch

            return 'cuda' if torch.cuda.is_available() else 'cpu'
        except Exception:
            return 'cpu'

    def _resolve_model_name(self, device: str) -> str:
        configured_model = settings.RERANK_MODEL.strip()
        if configured_model and configured_model.lower() != 'auto':
            return configured_model
        return settings.RERANK_GPU_MODEL if device == 'cuda' else settings.RERANK_CPU_MODEL

    def warmup(self) -> None:
        self._load_model()

    def _load_model(self):
        if self._model is not None:
            return self._model

        with self._load_lock:
            if self._model is not None:
                return self._model

            from sentence_transformers import CrossEncoder

            device = self._resolve_device()
            model_name = self._resolve_model_name(device)
            model_path = Path(model_name).expanduser()
            model_dir = _model_directory(model_name)
            has_complete_local_model = _has_local_model_files(model_path if model_path.exists() else model_dir)

            if model_path.exists() and _has_local_model_files(model_path):
                model_source = str(model_path)
                use_local_files_only = True
            elif has_complete_local_model:
                model_source = str(model_dir)
                use_local_files_only = True
            else:
                model_source = model_name
                use_local_files_only = False

            if use_local_files_only:
                self._model = CrossEncoder(
                    model_source,
                    device=device,
                    max_length=settings.RERANK_MAX_LENGTH,
                    local_files_only=True,
                )
            else:
                self._model = CrossEncoder(
                    model_source,
                    device=device,
                    max_length=settings.RERANK_MAX_LENGTH,
                    cache_folder=str(settings.rerank_cache_directory),
                )

            self._resolved_model_name = model_name
            self._resolved_model_dir = model_dir
            self._resolved_device = device
            logger.info(
                'Rerank 模型加载完成',
                extra={
                    'extra_data': {
                        'event': 'rerank_model_loaded',
                        'model': model_name,
                        'device': device,
                        'model_dir': str(model_dir),
                        'local_files_only': use_local_files_only,
                    }
                },
            )
            return self._model

    def score_documents(self, query: str, docs: Sequence[Any]) -> dict[Any, float]:
        if not settings.RERANK_ENABLED or not docs:
            return {}

        try:
            model = self._load_model()
        except Exception:
            if not self._load_failed:
                logger.exception(
                    'Rerank 模型加载失败，已回退到向量检索排序',
                    extra={
                        'extra_data': {
                            'event': 'rerank_model_load_failed',
                            'model': settings.RERANK_MODEL,
                            'cpu_model': settings.RERANK_CPU_MODEL,
                            'gpu_model': settings.RERANK_GPU_MODEL,
                            'cache_dir': str(settings.rerank_cache_directory),
                        }
                    },
                )
                self._load_failed = True
            return {}

        pairs = [(query, build_rerank_input(doc)) for doc in docs]

        try:
            raw_scores = model.predict(
                pairs,
                batch_size=settings.RERANK_BATCH_SIZE,
                show_progress_bar=False,
            )
        except Exception:
            logger.exception(
                'Rerank 推理失败，已回退到向量检索排序',
                extra={
                    'extra_data': {
                        'event': 'rerank_inference_failed',
                        'model': self._resolved_model_name or settings.RERANK_MODEL,
                        'device': self._resolved_device,
                    }
                },
            )
            return {}

        numeric_scores: list[float] = []
        for raw_score in raw_scores:
            if isinstance(raw_score, (list, tuple)):
                score_value = float(raw_score[0])
            else:
                try:
                    score_value = float(raw_score)
                except TypeError:
                    score_value = float(raw_score[0])

            numeric_scores.append(score_value)

        normalized_scores = normalize_rerank_scores_batch(numeric_scores)
        rerank_scores: dict[Any, float] = {}
        for doc, normalized_score in zip(docs, normalized_scores):
            rerank_scores[_document_key(doc)] = float(normalized_score)

        return rerank_scores


def get_reranker() -> CrossEncoderReranker:
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = CrossEncoderReranker()
    return _reranker_instance
