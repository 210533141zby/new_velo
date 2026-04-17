from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class QueryIntentType(str, Enum):
    LOOKUP = 'lookup'
    FACTOID = 'factoid'
    RELATION = 'relation'
    SUMMARY = 'summary'
    OVERVIEW = 'overview'
    REASON = 'reason'
    LOCATION = 'location'


class DefenseProfile(str, Enum):
    STRICT = 'strict'
    MODERATE = 'moderate'
    LOOSE = 'loose'


class EvidenceRequirement(str, Enum):
    ATOMIC_SPAN = 'atomic_span'
    MULTI_SPAN = 'multi_span'
    FULL_DOCUMENT = 'full_document'


class AnswerMode(str, Enum):
    NO_CONTEXT = 'no_context'
    EXTRACTIVE = 'extractive'
    STRUCTURED = 'structured'
    GENERATIVE = 'generative'


@dataclass(frozen=True)
class QueryIntent:
    original_query: str
    normalized_query: str
    keyword_query: str
    intent_type: QueryIntentType
    retrieval_depth: int
    defense_profile: DefenseProfile
    evidence_requirement: EvidenceRequirement
    wants_short_answer: bool
    needs_judge: bool
    trace_tags: tuple[str, ...] = ()

    @property
    def prefers_extractive(self) -> bool:
        return self.intent_type in {
            QueryIntentType.LOOKUP,
            QueryIntentType.FACTOID,
            QueryIntentType.LOCATION,
        } and self.evidence_requirement is EvidenceRequirement.ATOMIC_SPAN


@dataclass(frozen=True)
class RetrievedCandidate:
    doc: Any = None
    doc_id: int | None = None
    title: str = ''
    adaptive_score: float = 0.0
    dense_score: float = 0.0
    bm25_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: float = 0.0
    coverage_score: float = 0.0
    identifier_overlap: float = 0.0
    chunk_text: str = ''
    full_content: str = ''
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScoreContribution:
    name: str
    raw_value: float
    weight: float
    weighted_value: float
    reason: str


@dataclass(frozen=True)
class JudgeDecision:
    invoked: bool
    passed: bool
    topic_match: bool
    direct_evidence: bool
    answerable: bool
    evidence_quote: str = ''
    answer_brief: str = ''
    reason: str = ''
    timed_out: bool = False
    latency_ms: int = 0


@dataclass(frozen=True)
class EvidenceAssessment:
    candidate: RetrievedCandidate
    contributions: tuple[ScoreContribution, ...] = ()
    judge: JudgeDecision | None = None
    final_score: float = 0.0
    usable: bool = False
    reject_reason: str = ''
    direct_evidence: bool = False
    supports_extractive: bool = False
    evidence_quote: str = ''
    answer_brief: str = ''
    flags: tuple[str, ...] = ()
    trace_summary: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AnswerPlan:
    mode: AnswerMode
    reason: str
    primary_doc_id: int | None
    source_doc_ids: tuple[int, ...]
    generator_name: str
    trace_data: dict[str, Any] = field(default_factory=dict)
