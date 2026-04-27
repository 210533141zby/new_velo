from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Sequence

from app.logger import logger
from app.services.rag.evidence_judge import judge_rag_document
from app.services.rag.hybrid_search import has_identifier, tokenize_for_bm25
from app.services.rag.pipeline_models import (
    DefenseProfile,
    EvidenceAssessment,
    EvidenceRequirement,
    JudgeDecision,
    QueryIntent,
    QueryIntentType,
    RetrievedCandidate,
    ScoreContribution,
)
from app.services.rag.rerank_service import normalize_lookup_text

JudgeCallable = Callable[[str, str, str], Awaitable[dict]]

BASE_RELEVANCE_WEIGHTS = {
    'adaptive_signal': 0.50,
    'rerank_signal': 0.50,
}
BASE_SIGNAL_WEIGHT = 0.50
TOPIC_SIGNAL_WEIGHTS = {
    DefenseProfile.STRICT: 0.18,
    DefenseProfile.MODERATE: 0.14,
    DefenseProfile.LOOSE: 0.10,
}
JUDGE_SIGNAL_WEIGHTS = {
    DefenseProfile.STRICT: 0.24,
    DefenseProfile.MODERATE: 0.18,
    DefenseProfile.LOOSE: 0.0,
}
MIN_BASE_RELEVANCE = {
    DefenseProfile.STRICT: 0.48,
    DefenseProfile.MODERATE: 0.42,
    DefenseProfile.LOOSE: 0.36,
}
MIN_TOPIC_ALIGNMENT = {
    DefenseProfile.STRICT: 0.20,
    DefenseProfile.MODERATE: 0.08,
    DefenseProfile.LOOSE: 0.05,
}
MIN_FINAL_SCORE = {
    DefenseProfile.STRICT: 0.58,
    DefenseProfile.MODERATE: 0.45,
    DefenseProfile.LOOSE: 0.46,
}
WEAK_EVIDENCE_THRESHOLD = 0.34
MAX_JUDGE_CANDIDATES = 3


@dataclass(frozen=True)
class CandidateSnapshot:
    candidate: RetrievedCandidate
    title: str
    content: str
    adaptive_score: float
    query_tokens: set[str]
    title_tokens: set[str]
    content_tokens: set[str]
    topic_alignment: float
    title_alignment: float
    base_relevance: float


def _clamp_score(value: float | int | None) -> float:
    if value is None:
        return 0.0
    return max(0.0, min(float(value), 1.0))


def _compact_text(value: str, limit: int = 2200) -> str:
    normalized = ' '.join(str(value or '').split()).strip()
    return normalized[:limit]


def _token_set(value: str) -> set[str]:
    return {
        normalize_lookup_text(token)
        for token in tokenize_for_bm25(value)
        if normalize_lookup_text(token)
    }


def _coverage_ratio(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left)


def _candidate_title(candidate: RetrievedCandidate) -> str:
    if candidate.title:
        return str(candidate.title)
    if candidate.doc is not None:
        metadata = getattr(candidate.doc, 'metadata', {}) or {}
        return str(metadata.get('source') or '')
    return ''


def _candidate_chunk_text(candidate: RetrievedCandidate) -> str:
    if candidate.doc is not None:
        page_content = str(getattr(candidate.doc, 'page_content', '') or '')
        if page_content.strip():
            return _compact_text(page_content, limit=2200)
    if candidate.chunk_text.strip():
        return _compact_text(candidate.chunk_text, limit=2200)
    if candidate.doc is not None:
        metadata = getattr(candidate.doc, 'metadata', {}) or {}
        for key in ('content', 'content_preview', 'summary'):
            text = str(metadata.get(key) or '')
            if text.strip():
                return _compact_text(text, limit=2200)
    for key in ('content', 'content_preview', 'summary'):
        text = str(candidate.metadata.get(key) or '')
        if text.strip():
            return _compact_text(text, limit=2200)
    return ''


def _build_contribution(name: str, raw_value: float, weight: float, reason: str) -> ScoreContribution:
    normalized_value = _clamp_score(raw_value)
    return ScoreContribution(
        name=name,
        raw_value=normalized_value,
        weight=weight,
        weighted_value=normalized_value * weight,
        reason=reason,
    )


def _average_contributions(contributions: Sequence[ScoreContribution]) -> float:
    total_weight = sum(contribution.weight for contribution in contributions)
    if total_weight <= 0:
        return 0.0
    return sum(contribution.weighted_value for contribution in contributions) / total_weight


def _normalize_judge_payload(payload: dict | JudgeDecision) -> JudgeDecision:
    if isinstance(payload, JudgeDecision):
        return payload

    topic_match = bool(payload.get('core_topic_match') or payload.get('topic_match'))
    direct_evidence = bool(payload.get('contains_direct_evidence') or payload.get('direct_evidence'))
    answerable = bool(payload.get('answerable'))
    passed = answerable and topic_match and direct_evidence
    return JudgeDecision(
        invoked=True,
        passed=passed,
        topic_match=topic_match,
        direct_evidence=direct_evidence,
        answerable=answerable,
        evidence_quote=str(payload.get('evidence_quote') or ''),
        answer_brief=str(payload.get('answer_brief') or ''),
        reason=str(payload.get('reason') or ''),
    )


class UnifiedEvidenceScorer:
    def __init__(
        self,
        *,
        judge_timeout_seconds: float = 6.0,
        judge_callable: JudgeCallable | None = None,
    ) -> None:
        self.judge_timeout_seconds = judge_timeout_seconds
        self.judge_callable = judge_callable or judge_rag_document

    async def assess_concurrently(
        self,
        candidates: Sequence[RetrievedCandidate],
        intent: QueryIntent,
    ) -> list[EvidenceAssessment]:
        snapshots = [self._build_snapshot(candidate, intent) for candidate in candidates]
        judge_decisions = await self._run_judge_batch(snapshots, intent)

        assessments: list[EvidenceAssessment] = []
        for snapshot, judge_decision in zip(snapshots, judge_decisions):
            assessment = self._build_assessment(snapshot, intent, judge_decision)
            self._emit_trace_log(assessment)
            assessments.append(assessment)
        return assessments

    def _build_snapshot(self, candidate: RetrievedCandidate, intent: QueryIntent) -> CandidateSnapshot:
        title = _candidate_title(candidate)
        content = _candidate_chunk_text(candidate)
        query_tokens = _token_set(intent.keyword_query or intent.normalized_query)
        title_tokens = _token_set(title)
        content_tokens = _token_set(content)
        topic_alignment = max(
            _coverage_ratio(query_tokens, title_tokens),
            _coverage_ratio(query_tokens, content_tokens),
        )
        title_alignment = _coverage_ratio(query_tokens, title_tokens)
        adaptive_score = _clamp_score(candidate.adaptive_score)
        base_relevance = self._compute_base_relevance(candidate)
        return CandidateSnapshot(
            candidate=candidate,
            title=title,
            content=content,
            adaptive_score=adaptive_score,
            query_tokens=query_tokens,
            title_tokens=title_tokens,
            content_tokens=content_tokens,
            topic_alignment=topic_alignment,
            title_alignment=title_alignment,
            base_relevance=base_relevance,
        )

    def _compute_base_relevance(self, candidate: RetrievedCandidate) -> float:
        contributions = (
            _build_contribution(
                'adaptive_signal',
                candidate.adaptive_score,
                BASE_RELEVANCE_WEIGHTS['adaptive_signal'],
                'hybrid_adaptive_score',
            ),
            _build_contribution(
                'rerank_signal',
                candidate.rerank_score,
                BASE_RELEVANCE_WEIGHTS['rerank_signal'],
                'cross_encoder_rerank',
            ),
        )
        return _average_contributions(contributions)

    async def _run_judge_batch(
        self,
        snapshots: Sequence[CandidateSnapshot],
        intent: QueryIntent,
    ) -> list[JudgeDecision]:
        decisions = [self._skipped_judge_decision() for _ in snapshots]
        if not self._should_invoke_judge(intent):
            return decisions

        judge_indexes = self._select_judge_indexes(snapshots, intent)
        tasks = [self._run_single_judge(snapshots[index], intent) for index in judge_indexes]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for index, result in zip(judge_indexes, results):
            if isinstance(result, JudgeDecision):
                decisions[index] = result
                continue
            if isinstance(result, Exception):
                decisions[index] = JudgeDecision(
                    invoked=True,
                    passed=False,
                    topic_match=False,
                    direct_evidence=False,
                    answerable=False,
                    reason='judge_exception',
                    latency_ms=0,
                )
                continue
            decisions[index] = _normalize_judge_payload(result)
        return decisions

    def _skipped_judge_decision(self, reason: str = 'judge_skipped') -> JudgeDecision:
        return JudgeDecision(
            invoked=False,
            passed=True,
            topic_match=True,
            direct_evidence=False,
            answerable=False,
            reason=reason,
            latency_ms=0,
        )

    def _select_judge_indexes(
        self,
        snapshots: Sequence[CandidateSnapshot],
        intent: QueryIntent,
    ) -> tuple[int, ...]:
        ranked_indexes = sorted(
            range(len(snapshots)),
            key=lambda index: snapshots[index].base_relevance,
            reverse=True,
        )
        relevance_floor = max(0.0, MIN_BASE_RELEVANCE[intent.defense_profile] - 0.08)
        selected: list[int] = []
        for index in ranked_indexes:
            if len(selected) >= MAX_JUDGE_CANDIDATES:
                break
            if snapshots[index].base_relevance < relevance_floor:
                continue
            selected.append(index)

        if not selected and ranked_indexes:
            selected.append(ranked_indexes[0])
        return tuple(sorted(selected))

    async def _run_single_judge(
        self,
        snapshot: CandidateSnapshot,
        intent: QueryIntent,
    ) -> JudgeDecision:
        if not self._should_invoke_judge(intent):
            return self._skipped_judge_decision()

        started_at = time.perf_counter()
        judge_content = snapshot.content
        if intent.intent_type is QueryIntentType.RELATION:
            full_content = ''
            if snapshot.candidate.doc is not None:
                full_content = str(getattr(snapshot.candidate.doc, 'full_content', '') or '')
            if not full_content.strip():
                full_content = snapshot.candidate.full_content
            if full_content.strip():
                judge_content = _compact_text(full_content, limit=6000)
        try:
            payload = await asyncio.wait_for(
                self.judge_callable(intent.normalized_query, snapshot.title, judge_content),
                timeout=self.judge_timeout_seconds,
            )
        except asyncio.TimeoutError:
            return JudgeDecision(
                invoked=True,
                passed=False,
                topic_match=False,
                direct_evidence=False,
                answerable=False,
                reason='judge_timeout',
                timed_out=True,
                latency_ms=int((time.perf_counter() - started_at) * 1000),
            )

        decision = _normalize_judge_payload(payload)
        return JudgeDecision(
            invoked=decision.invoked,
            passed=decision.passed,
            topic_match=decision.topic_match,
            direct_evidence=decision.direct_evidence,
            answerable=decision.answerable,
            evidence_quote=decision.evidence_quote,
            answer_brief=decision.answer_brief,
            reason=decision.reason,
            timed_out=decision.timed_out,
            latency_ms=int((time.perf_counter() - started_at) * 1000),
        )

    def _should_invoke_judge(self, intent: QueryIntent) -> bool:
        if not intent.needs_judge:
            return False
        return JUDGE_SIGNAL_WEIGHTS[intent.defense_profile] > 0.0

    def _build_assessment(
        self,
        snapshot: CandidateSnapshot,
        intent: QueryIntent,
        judge_decision: JudgeDecision,
    ) -> EvidenceAssessment:
        contributions = [
            _build_contribution(
                'base_signal',
                snapshot.base_relevance,
                BASE_SIGNAL_WEIGHT,
                'adaptive_plus_rerank',
            )
        ]
        contributions.append(
            _build_contribution(
                'topic_signal',
                snapshot.topic_alignment,
                TOPIC_SIGNAL_WEIGHTS[intent.defense_profile],
                'query_topic_alignment',
            )
        )
        if judge_decision.invoked:
            contributions.append(
                _build_contribution(
                    'judge_signal',
                    1.0 if judge_decision.passed else 0.0,
                    JUDGE_SIGNAL_WEIGHTS[intent.defense_profile],
                    'semantic_judge',
                )
            )

        base_relevance = snapshot.base_relevance
        final_score = _average_contributions(contributions)
        flags = list(self._build_flags(snapshot, intent, judge_decision, base_relevance, final_score))
        direct_evidence = self._determine_direct_evidence(snapshot, intent, judge_decision, base_relevance)
        supports_extractive = self._determine_extractive_support(intent, judge_decision, base_relevance, snapshot)
        usable, reject_reason = self._determine_usability(
            intent=intent,
            judge_decision=judge_decision,
            base_relevance=base_relevance,
            topic_alignment=snapshot.topic_alignment,
            final_score=final_score,
            direct_evidence=direct_evidence,
        )

        if usable:
            flags.append('USABLE')
        elif reject_reason:
            flags.append(reject_reason)

        trace_summary = {
            'doc_id': snapshot.candidate.doc_id,
            'title': snapshot.title[:80],
            'adaptive_score': round(snapshot.adaptive_score, 4),
            'base_relevance': round(base_relevance, 4),
            'topic_match': 'PASS' if snapshot.topic_alignment >= MIN_TOPIC_ALIGNMENT[intent.defense_profile] else 'FAIL',
            'judge_latency_ms': judge_decision.latency_ms,
            'flags': flags,
            'usable': usable,
            'final_score': round(final_score, 4),
            'judge_status': self._judge_status(judge_decision),
        }

        return EvidenceAssessment(
            candidate=snapshot.candidate,
            contributions=tuple(contributions),
            judge=judge_decision,
            final_score=final_score,
            usable=usable,
            reject_reason=reject_reason,
            direct_evidence=direct_evidence,
            supports_extractive=supports_extractive,
            evidence_quote=judge_decision.evidence_quote,
            answer_brief=judge_decision.answer_brief,
            flags=tuple(flags),
            trace_summary=trace_summary,
        )

    def _build_flags(
        self,
        snapshot: CandidateSnapshot,
        intent: QueryIntent,
        judge_decision: JudgeDecision,
        base_relevance: float,
        final_score: float,
    ) -> tuple[str, ...]:
        flags: list[str] = []
        if base_relevance < MIN_BASE_RELEVANCE[intent.defense_profile]:
            flags.append('LOW_BASE_RELEVANCE')
        if snapshot.topic_alignment < MIN_TOPIC_ALIGNMENT[intent.defense_profile]:
            flags.append('WEAK_TOPIC_ALIGNMENT')
        if has_identifier(intent.normalized_query) and snapshot.candidate.identifier_overlap <= 0.0:
            flags.append('IDENTIFIER_MISMATCH')
        if intent.intent_type is QueryIntentType.LOOKUP and snapshot.title_alignment < 0.25 and snapshot.topic_alignment < 0.30:
            flags.append('FAILED_SHORT_ENTITY_MATCH')
        if intent.evidence_requirement is EvidenceRequirement.ATOMIC_SPAN and snapshot.topic_alignment < WEAK_EVIDENCE_THRESHOLD:
            flags.append('WEAK_EVIDENCE')
        if judge_decision.invoked and judge_decision.timed_out:
            flags.append('LLM_JUDGE_TIMEOUT')
        elif judge_decision.invoked and judge_decision.passed:
            flags.append('PASSED_LLM_JUDGE')
        elif judge_decision.invoked:
            flags.append('FAILED_LLM_JUDGE')
        if final_score < MIN_FINAL_SCORE[intent.defense_profile]:
            flags.append('LOW_FINAL_SCORE')
        return tuple(flags)

    def _determine_direct_evidence(
        self,
        snapshot: CandidateSnapshot,
        intent: QueryIntent,
        judge_decision: JudgeDecision,
        base_relevance: float,
    ) -> bool:
        if judge_decision.invoked:
            return judge_decision.direct_evidence
        if intent.evidence_requirement is EvidenceRequirement.FULL_DOCUMENT:
            return False
        return snapshot.topic_alignment >= 0.35 and (
            snapshot.title_alignment >= 0.20 or base_relevance >= 0.52
        )

    def _determine_extractive_support(
        self,
        intent: QueryIntent,
        judge_decision: JudgeDecision,
        base_relevance: float,
        snapshot: CandidateSnapshot,
    ) -> bool:
        if intent.evidence_requirement is EvidenceRequirement.MULTI_SPAN:
            return (
                intent.intent_type is QueryIntentType.REASON
                and base_relevance >= 0.56
                and snapshot.topic_alignment >= 0.32
            )
        if intent.evidence_requirement is not EvidenceRequirement.ATOMIC_SPAN:
            return False
        if judge_decision.invoked and judge_decision.answerable:
            return True
        return base_relevance >= 0.56 and snapshot.topic_alignment >= 0.32

    def _determine_usability(
        self,
        *,
        intent: QueryIntent,
        judge_decision: JudgeDecision,
        base_relevance: float,
        topic_alignment: float,
        final_score: float,
        direct_evidence: bool,
    ) -> tuple[bool, str]:
        if topic_alignment < MIN_TOPIC_ALIGNMENT[intent.defense_profile]:
            return False, 'REJECT_TOPIC_MISMATCH'
        if final_score < MIN_FINAL_SCORE[intent.defense_profile]:
            return False, 'REJECT_LOW_FINAL_SCORE'
        if judge_decision.invoked and not judge_decision.passed:
            if judge_decision.timed_out:
                return False, 'REJECT_JUDGE_TIMEOUT'
            return False, 'REJECT_FAILED_LLM_JUDGE'
        if intent.defense_profile is DefenseProfile.STRICT and intent.evidence_requirement is EvidenceRequirement.ATOMIC_SPAN:
            if not direct_evidence:
                return False, 'REJECT_WEAK_DIRECT_EVIDENCE'
        return True, ''

    def _judge_status(self, judge_decision: JudgeDecision) -> str:
        if not judge_decision.invoked:
            return 'SKIP'
        if judge_decision.timed_out:
            return 'TIMEOUT'
        if judge_decision.passed:
            return 'PASS'
        return 'FAIL'

    def _emit_trace_log(self, assessment: EvidenceAssessment) -> None:
        logger.info(
            'RAG 统一证据评分完成',
            extra={
                'extra_data': {
                    'event': 'rag_evidence_assessed',
                    **assessment.trace_summary,
                }
            },
        )
