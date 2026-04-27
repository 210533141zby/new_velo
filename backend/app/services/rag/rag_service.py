from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace
from typing import Any, Sequence

from fastapi.concurrency import run_in_threadpool
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.cache import redis_manager
from app.core.config import settings
from app.logger import logger
from app.models import Document
from app.services.model_factory import get_chat_model
from app.services.rag.answer_generators import FallbackRequiredError, GeneratorFactory
from app.services.rag.answer_mode_router import AnswerModeRouter
from app.services.rag.evidence_scorer import UnifiedEvidenceScorer
from app.services.rag.hybrid_search import (
    build_hybrid_candidates,
    ensure_hybrid_index,
    extract_identifiers,
    get_hybrid_index,
    hybrid_index_needs_refresh,
    invalidate_hybrid_index,
    tokenize_for_bm25,
)
from app.services.rag.pipeline_models import AnswerMode, AnswerPlan, EvidenceAssessment, QueryIntent, RetrievedCandidate
from app.services.rag.prompt_templates import (
    build_assistant_identity_answer,
    build_no_context_answer,
    is_model_identity_query,
)
from app.services.rag.query_intent_builder import QueryIntentBuilder
from app.services.rag.vector_index_service import collection_name, delete_document_chunks, get_vector_store, index_document_chunks
from app.services.rag.rerank_service import get_reranker, normalize_lookup_text

RAG_CACHE_VERSION = 'v23'


def _document_key(doc: Any) -> Any:
    metadata = getattr(doc, 'metadata', {}) or {}
    return metadata.get('doc_id') or str(metadata.get('source') or '')


class RagService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self._vector_store = None
        self.evidence_scorer = UnifiedEvidenceScorer()
        self.answer_router = AnswerModeRouter()
        self.generator_factory = GeneratorFactory(get_chat_model)

    @property
    def vector_store(self):
        if self._vector_store is None:
            self._vector_store = get_vector_store()
        return self._vector_store

    def _cache_key(self, query: str) -> str:
        query_hash = hashlib.md5(query.encode('utf-8')).hexdigest()
        return f'rag:response:{RAG_CACHE_VERSION}:{query_hash}'

    async def _get_cached_result(self, query: str) -> dict[str, Any] | None:
        cached_payload = await redis_manager.get(self._cache_key(query))
        if not cached_payload:
            return None
        logger.info(
            'RAG 问答命中缓存',
            extra={'extra_data': {'event': 'rag_cache_hit', 'query': query}},
        )
        return json.loads(cached_payload)

    async def _cache_result(self, query: str, result: dict[str, Any]) -> dict[str, Any]:
        await redis_manager.set(self._cache_key(query), json.dumps(result, ensure_ascii=False), ex=3600)
        return result

    def _build_system_result(self, query: str) -> dict[str, Any] | None:
        if not is_model_identity_query(query):
            return None
        return {'response': build_assistant_identity_answer(for_rag=True), 'sources': []}

    def _collapse_scored_matches(self, query: str, scored_matches: Sequence[tuple[Any, float]]) -> list[tuple[Any, float]]:
        query_identifiers = extract_identifiers(query)
        query_tokens = {
            normalize_lookup_text(token)
            for token in tokenize_for_bm25(query)
            if normalize_lookup_text(token)
        }
        best_match_by_source: dict[Any, tuple[Any, float]] = {}
        for doc, score in scored_matches:
            source_key = _document_key(doc)
            content = normalize_lookup_text(getattr(doc, 'page_content', '') or '')
            identifier_hits = sum(1 for identifier in query_identifiers if identifier and identifier in content)
            token_hits = sum(1 for token in query_tokens if token and token in content)
            match_score = float(score) + identifier_hits * 10.0 + token_hits * 0.1
            existing = best_match_by_source.get(source_key)
            if existing is None or match_score > float(existing[1]):
                best_match_by_source[source_key] = (doc, match_score)
        return sorted(best_match_by_source.values(), key=lambda item: item[1], reverse=True)

    async def _load_active_documents(self) -> list[Document]:
        result = await self.db.execute(
            select(Document)
            .where(Document.is_active == True)
            .order_by(Document.updated_at.desc().nullslast(), Document.id.desc())
        )
        return [document for document in result.scalars().all() if document.content]

    def _build_retrieved_candidates(
        self,
        candidate_matches: Sequence[tuple[Any, float]],
        rerank_scores: dict[Any, float],
        document_by_id: dict[int, Document],
    ) -> list[RetrievedCandidate]:
        retrieved_candidates: list[RetrievedCandidate] = []
        for doc, adaptive_score in candidate_matches:
            metadata = dict(getattr(doc, 'metadata', {}) or {})
            raw_doc_id = metadata.get('doc_id')
            numeric_doc_id = int(raw_doc_id) if raw_doc_id is not None else None
            full_document = document_by_id.get(numeric_doc_id) if numeric_doc_id is not None else None
            full_title = str(getattr(full_document, 'title', '') or metadata.get('source') or '')
            chunk_text = str(getattr(doc, 'page_content', '') or '')
            full_content = str(getattr(full_document, 'content', '') or chunk_text or '')
            resolved_adaptive_score = float(metadata.get('adaptive_score') or adaptive_score or 0.0)
            metadata['adaptive_score'] = resolved_adaptive_score
            candidate_doc = SimpleNamespace(
                page_content=chunk_text,
                chunk_text=chunk_text,
                full_content=full_content,
                metadata={
                    **metadata,
                    'source': full_title,
                    'doc_id': numeric_doc_id,
                },
            )
            retrieved_candidates.append(
                RetrievedCandidate(
                    doc=candidate_doc,
                    doc_id=numeric_doc_id,
                    title=full_title,
                    adaptive_score=resolved_adaptive_score,
                    dense_score=float(metadata.get('vector_score') or 0.0),
                    bm25_score=float(metadata.get('bm25_score') or 0.0),
                    rrf_score=float(metadata.get('rrf_score') or 0.0),
                    rerank_score=float(rerank_scores.get(_document_key(doc), 0.0)),
                    coverage_score=float(metadata.get('coverage_score') or 0.0),
                    identifier_overlap=float(metadata.get('identifier_overlap') or 0.0),
                    chunk_text=chunk_text,
                    full_content=full_content,
                    metadata=metadata,
                )
            )
        return retrieved_candidates

    async def _retrieve_candidates(self, query: str, intent: QueryIntent) -> list[RetrievedCandidate]:
        vector_limit = max(intent.retrieval_depth, settings.RAG_VECTOR_SEARCH_LIMIT)
        bm25_limit = max(intent.retrieval_depth, settings.RAG_BM25_SEARCH_LIMIT)
        candidate_limit = max(intent.retrieval_depth, settings.RAG_HYBRID_CANDIDATE_LIMIT)
        correction_query = 'correction_query' in intent.trace_tags
        retrieval_query = intent.keyword_query if correction_query and intent.keyword_query.strip() else intent.normalized_query
        vector_matches = await run_in_threadpool(
            self.vector_store.similarity_search_with_relevance_scores,
            retrieval_query,
            vector_limit,
        )
        collapsed_matches = self._collapse_scored_matches(query, vector_matches)
        active_documents = await self._load_active_documents()
        lexical_index = ensure_hybrid_index(active_documents) if hybrid_index_needs_refresh() else get_hybrid_index()
        document_by_id = {int(document.id): document for document in active_documents}
        candidate_matches = build_hybrid_candidates(
            retrieval_query if correction_query else query,
            collapsed_matches,
            lexical_index,
            bm25_query=intent.keyword_query,
            vector_limit=vector_limit,
            bm25_limit=bm25_limit,
            candidate_limit=candidate_limit,
        )
        rerank_scores = await run_in_threadpool(
            get_reranker().score_documents,
            retrieval_query if correction_query else query,
            [doc for doc, _score in candidate_matches],
        )
        return self._build_retrieved_candidates(candidate_matches, rerank_scores, document_by_id)

    def _log_intent(self, query: str, intent: QueryIntent) -> None:
        logger.info(
            'RAG 意图解析完成',
            extra={
                'extra_data': {
                    'event': 'rag_intent_built',
                    'query': query,
                    'intent_type': intent.intent_type.value,
                    'retrieval_depth': intent.retrieval_depth,
                    'defense_profile': intent.defense_profile.value,
                    'evidence_requirement': intent.evidence_requirement.value,
                    'trace_tags': list(intent.trace_tags),
                }
            },
        )

    def _log_routing(
        self,
        query: str,
        intent: QueryIntent,
        candidates: Sequence[RetrievedCandidate],
        assessments: Sequence[EvidenceAssessment],
        plan: AnswerPlan,
    ) -> None:
        logger.info(
            'RAG 路由决策完成',
            extra={
                'extra_data': {
                    'event': 'rag_pipeline_routed',
                    'query': query,
                    'intent_type': intent.intent_type.value,
                    'candidate_count': len(candidates),
                    'assessment_count': len(assessments),
                    'usable_count': sum(1 for assessment in assessments if assessment.usable),
                    'answer_mode': plan.mode.value,
                    'reason': plan.reason,
                    'source_doc_ids': list(plan.source_doc_ids),
                }
            },
        )

    def _usable_assessments(self, assessments: Sequence[EvidenceAssessment]) -> list[EvidenceAssessment]:
        return [
            assessment
            for assessment in sorted(assessments, key=lambda item: float(item.final_score), reverse=True)
            if assessment.usable
        ]

    async def _execute_with_fallback(
        self,
        query: str,
        intent: QueryIntent,
        usable_assessments: Sequence[EvidenceAssessment],
        plan: AnswerPlan,
    ) -> tuple[str, list[dict]]:
        current_plan = plan
        while True:
            try:
                return await self.generator_factory.execute(current_plan, query, intent, usable_assessments)
            except FallbackRequiredError as exc:
                logger.info(
                    'RAG 生成器触发优雅降级',
                    extra={
                        'extra_data': {
                            'event': 'rag_generator_fallback',
                            'query': query,
                            'from_mode': current_plan.mode.value,
                            'reason': str(exc),
                        }
                    },
                )
                current_plan = self.generator_factory.downgrade(current_plan)
                if current_plan.mode is AnswerMode.NO_CONTEXT:
                    return build_no_context_answer(), []

    async def ensure_bootstrap_index(self):
        active_docs = await self._load_active_documents()
        if not active_docs:
            return

        collection_count = await run_in_threadpool(self.vector_store._collection.count)
        if collection_count > 0:
            return

        logger.info(
            f'RAG collection 为空，开始预热索引 {len(active_docs)} 篇文档',
            extra={
                'extra_data': {
                    'event': 'rag_bootstrap_start',
                    'document_count': len(active_docs),
                    'collection_name': collection_name(),
                }
            },
        )
        for doc in active_docs:
            await self.index_document(doc.id, doc.title, doc.content)
        logger.info(
            'RAG 预热索引完成',
            extra={
                'extra_data': {
                    'event': 'rag_bootstrap_complete',
                    'document_count': len(active_docs),
                    'collection_name': collection_name(),
                }
            },
        )

    async def index_document(self, doc_id: int, title: str, content: str):
        await index_document_chunks(doc_id, title, content)
        invalidate_hybrid_index()

    async def delete_document_index(self, doc_id: int):
        await delete_document_chunks(doc_id)
        invalidate_hybrid_index()

    async def rag_qa(self, query: str) -> dict:
        try:
            cached_result = await self._get_cached_result(query)
            if cached_result is not None:
                return cached_result

            system_result = self._build_system_result(query)
            if system_result is not None:
                return await self._cache_result(query, system_result)

            intent = await QueryIntentBuilder.build(query)
            self._log_intent(query, intent)
            candidates = await self._retrieve_candidates(query, intent)
            assessments = await self.evidence_scorer.assess_concurrently(candidates, intent)
            usable_assessments = self._usable_assessments(assessments)
            plan = self.answer_router.route(intent, usable_assessments)
            self._log_routing(query, intent, candidates, assessments, plan)
            response, sources = await self._execute_with_fallback(query, intent, usable_assessments, plan)
            result = {'response': response, 'sources': sources}
            await self._cache_result(query, result)
            logger.info(
                'RAG 问答完成',
                extra={
                    'extra_data': {
                        'event': 'rag_qa_success',
                        'query': query,
                        'source_count': len(sources),
                        'candidate_count': len(candidates),
                        'collection_name': collection_name(),
                    }
                },
            )
            return result
        except Exception:
            logger.exception(
                'RAG 问答失败',
                extra={'extra_data': {'event': 'rag_qa_failed', 'query': query, 'collection_name': collection_name()}},
            )
            return {'response': '抱歉，系统暂时无法回答您的请求。', 'sources': []}
