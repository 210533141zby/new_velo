from __future__ import annotations

from typing import Sequence

from app.services.rag.pipeline_models import (
    AnswerMode,
    AnswerPlan,
    EvidenceAssessment,
    EvidenceRequirement,
    QueryIntent,
    QueryIntentType,
)

PRIMARY_EVIDENCE_GAP = 0.12


def _source_doc_ids(assessments: Sequence[EvidenceAssessment]) -> tuple[int, ...]:
    doc_ids: list[int] = []
    seen: set[int] = set()
    for assessment in assessments:
        doc_id = assessment.candidate.doc_id
        if doc_id is None or doc_id in seen:
            continue
        seen.add(doc_id)
        doc_ids.append(doc_id)
    return tuple(doc_ids)


def _rank_assessments(assessments: Sequence[EvidenceAssessment]) -> list[EvidenceAssessment]:
    return sorted(
        (assessment for assessment in assessments if assessment.usable),
        key=lambda assessment: (
            float(assessment.final_score),
            1.0 if assessment.direct_evidence else 0.0,
            1.0 if assessment.supports_extractive else 0.0,
        ),
        reverse=True,
    )


def _has_clear_primary_assessment(ranked_assessments: Sequence[EvidenceAssessment]) -> bool:
    if len(ranked_assessments) <= 1:
        return True
    return ranked_assessments[0].final_score >= ranked_assessments[1].final_score + PRIMARY_EVIDENCE_GAP


class AnswerModeRouter:
    @classmethod
    def route(cls, intent: QueryIntent, usable_assessments: Sequence[EvidenceAssessment]) -> AnswerPlan:
        ranked_assessments = _rank_assessments(usable_assessments)
        source_doc_ids = _source_doc_ids(ranked_assessments)
        primary = ranked_assessments[0] if ranked_assessments else None
        primary_doc_id = primary.candidate.doc_id if primary is not None else None

        if not ranked_assessments:
            return AnswerPlan(
                mode=AnswerMode.NO_CONTEXT,
                reason='no_usable_evidence',
                primary_doc_id=None,
                source_doc_ids=(),
                generator_name='no_context_generator',
                trace_data={'usable_doc_count': 0},
            )

        if intent.intent_type in {QueryIntentType.SUMMARY, QueryIntentType.OVERVIEW}:
            return AnswerPlan(
                mode=AnswerMode.GENERATIVE,
                reason='full_document_intent',
                primary_doc_id=primary_doc_id,
                source_doc_ids=source_doc_ids,
                generator_name='generative_generator',
                trace_data={'usable_doc_count': len(ranked_assessments), 'intent_type': intent.intent_type.value},
            )

        if intent.evidence_requirement is EvidenceRequirement.FULL_DOCUMENT:
            return AnswerPlan(
                mode=AnswerMode.GENERATIVE,
                reason='full_document_requirement',
                primary_doc_id=primary_doc_id,
                source_doc_ids=source_doc_ids,
                generator_name='generative_generator',
                trace_data={'usable_doc_count': len(ranked_assessments)},
            )

        if intent.intent_type is QueryIntentType.RELATION and (
            '共同点' in intent.normalized_query or '哪些人物' in intent.normalized_query
        ):
            return AnswerPlan(
                mode=AnswerMode.GENERATIVE,
                reason='relation_synthesis',
                primary_doc_id=primary_doc_id,
                source_doc_ids=source_doc_ids,
                generator_name='generative_generator',
                trace_data={'usable_doc_count': len(ranked_assessments)},
            )

        if len(ranked_assessments) > 1 and intent.intent_type in {
            QueryIntentType.RELATION,
            QueryIntentType.REASON,
        }:
            return AnswerPlan(
                mode=AnswerMode.GENERATIVE,
                reason='multi_document_synthesis',
                primary_doc_id=primary_doc_id,
                source_doc_ids=source_doc_ids,
                generator_name='generative_generator',
                trace_data={'usable_doc_count': len(ranked_assessments)},
            )

        if (
            primary is not None
            and intent.wants_short_answer
            and primary.direct_evidence
            and primary.supports_extractive
            and _has_clear_primary_assessment(ranked_assessments)
        ):
            return AnswerPlan(
                mode=AnswerMode.EXTRACTIVE,
                reason='atomic_evidence_available',
                primary_doc_id=primary_doc_id,
                source_doc_ids=source_doc_ids,
                generator_name='extractive_generator',
                trace_data={
                    'usable_doc_count': len(ranked_assessments),
                    'primary_score': primary.final_score,
                },
            )

        if intent.wants_short_answer:
            return AnswerPlan(
                mode=AnswerMode.STRUCTURED,
                reason='short_answer_without_precise_span',
                primary_doc_id=primary_doc_id,
                source_doc_ids=source_doc_ids,
                generator_name='structured_generator',
                trace_data={
                    'usable_doc_count': len(ranked_assessments),
                    'primary_score': primary.final_score if primary is not None else 0.0,
                },
            )

        return AnswerPlan(
            mode=AnswerMode.GENERATIVE,
            reason='default_generative_fallback',
            primary_doc_id=primary_doc_id,
            source_doc_ids=source_doc_ids,
            generator_name='generative_generator',
            trace_data={'usable_doc_count': len(ranked_assessments)},
        )
