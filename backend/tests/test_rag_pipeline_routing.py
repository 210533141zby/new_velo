import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.rag.answer_mode_router import AnswerModeRouter
from app.services.rag.pipeline_models import (
    AnswerMode,
    DefenseProfile,
    EvidenceAssessment,
    EvidenceRequirement,
    QueryIntent,
    QueryIntentType,
    RetrievedCandidate,
)
from app.services.rag.query_intent_builder import QueryIntentBuilder


def build_intent(
    intent_type: QueryIntentType,
    *,
    evidence_requirement: EvidenceRequirement,
    wants_short_answer: bool = True,
) -> QueryIntent:
    return QueryIntent(
        original_query='测试问题',
        normalized_query='测试问题',
        keyword_query='测试 问题',
        intent_type=intent_type,
        retrieval_depth=8,
        defense_profile=DefenseProfile.MODERATE,
        evidence_requirement=evidence_requirement,
        wants_short_answer=wants_short_answer,
        needs_judge=False,
        trace_tags=(),
    )


def build_assessment(
    doc_id: int,
    *,
    final_score: float = 0.9,
    usable: bool = True,
    direct_evidence: bool = True,
    supports_extractive: bool = True,
) -> EvidenceAssessment:
    return EvidenceAssessment(
        candidate=RetrievedCandidate(doc_id=doc_id, title=f'文档{doc_id}'),
        final_score=final_score,
        usable=usable,
        direct_evidence=direct_evidence,
        supports_extractive=supports_extractive,
    )


class QueryIntentBuilderTests(unittest.IsolatedAsyncioTestCase):
    async def test_builds_relation_intent_for_relation_query(self) -> None:
        intent = await QueryIntentBuilder.build('韩国与新加坡的关系')

        self.assertEqual(intent.intent_type, QueryIntentType.RELATION)
        self.assertEqual(intent.defense_profile, DefenseProfile.STRICT)
        self.assertEqual(intent.evidence_requirement, EvidenceRequirement.MULTI_SPAN)
        self.assertTrue(intent.needs_judge)

    async def test_builds_overview_intent_for_history_query(self) -> None:
        intent = await QueryIntentBuilder.build('雾潮镇的历史')

        self.assertEqual(intent.intent_type, QueryIntentType.OVERVIEW)
        self.assertEqual(intent.defense_profile, DefenseProfile.LOOSE)
        self.assertEqual(intent.evidence_requirement, EvidenceRequirement.FULL_DOCUMENT)
        self.assertFalse(intent.wants_short_answer)

    async def test_builds_lookup_intent_for_short_entity_query(self) -> None:
        intent = await QueryIntentBuilder.build('沈阳')

        self.assertEqual(intent.intent_type, QueryIntentType.LOOKUP)
        self.assertEqual(intent.defense_profile, DefenseProfile.STRICT)
        self.assertEqual(intent.evidence_requirement, EvidenceRequirement.ATOMIC_SPAN)
        self.assertIn('short_query', intent.trace_tags)

    async def test_builds_factoid_intent_for_specific_fact_question(self) -> None:
        intent = await QueryIntentBuilder.build('档案馆建于哪一年')

        self.assertEqual(intent.intent_type, QueryIntentType.FACTOID)
        self.assertEqual(intent.defense_profile, DefenseProfile.MODERATE)
        self.assertEqual(intent.evidence_requirement, EvidenceRequirement.ATOMIC_SPAN)
        self.assertIn('档案馆', intent.keyword_query)

    async def test_builds_strict_intent_for_attribute_style_query(self) -> None:
        intent = await QueryIntentBuilder.build('新加坡的档案馆')

        self.assertEqual(intent.intent_type, QueryIntentType.FACTOID)
        self.assertEqual(intent.defense_profile, DefenseProfile.STRICT)
        self.assertTrue(intent.needs_judge)
        self.assertIn('attribute_style_query', intent.trace_tags)

    async def test_reason_query_does_not_force_llm_judge(self) -> None:
        intent = await QueryIntentBuilder.build('为什么研究者重视编号为A-17-204的蓝布账册')

        self.assertEqual(intent.intent_type, QueryIntentType.REASON)
        self.assertEqual(intent.defense_profile, DefenseProfile.MODERATE)
        self.assertFalse(intent.needs_judge)
        self.assertNotIn('为什么', intent.keyword_query)


class AnswerModeRouterTests(unittest.TestCase):
    def test_routes_to_no_context_when_no_usable_docs(self) -> None:
        intent = build_intent(
            QueryIntentType.FACTOID,
            evidence_requirement=EvidenceRequirement.ATOMIC_SPAN,
        )

        plan = AnswerModeRouter.route(intent, [])

        self.assertEqual(plan.mode, AnswerMode.NO_CONTEXT)
        self.assertEqual(plan.reason, 'no_usable_evidence')

    def test_routes_factoid_with_direct_evidence_to_extractive(self) -> None:
        intent = build_intent(
            QueryIntentType.FACTOID,
            evidence_requirement=EvidenceRequirement.ATOMIC_SPAN,
        )

        plan = AnswerModeRouter.route(intent, [build_assessment(38, final_score=0.94)])

        self.assertEqual(plan.mode, AnswerMode.EXTRACTIVE)
        self.assertEqual(plan.generator_name, 'extractive_generator')
        self.assertEqual(plan.primary_doc_id, 38)

    def test_routes_short_answer_without_precise_span_to_structured(self) -> None:
        intent = build_intent(
            QueryIntentType.RELATION,
            evidence_requirement=EvidenceRequirement.MULTI_SPAN,
        )

        plan = AnswerModeRouter.route(
            intent,
            [build_assessment(38, final_score=0.81, direct_evidence=False, supports_extractive=False)],
        )

        self.assertEqual(plan.mode, AnswerMode.STRUCTURED)
        self.assertEqual(plan.generator_name, 'structured_generator')

    def test_routes_overview_to_generative(self) -> None:
        intent = build_intent(
            QueryIntentType.OVERVIEW,
            evidence_requirement=EvidenceRequirement.FULL_DOCUMENT,
            wants_short_answer=False,
        )

        plan = AnswerModeRouter.route(intent, [build_assessment(38, final_score=0.91)])

        self.assertEqual(plan.mode, AnswerMode.GENERATIVE)
        self.assertEqual(plan.reason, 'full_document_intent')

    def test_routes_multi_document_relation_to_generative(self) -> None:
        intent = build_intent(
            QueryIntentType.RELATION,
            evidence_requirement=EvidenceRequirement.MULTI_SPAN,
        )

        plan = AnswerModeRouter.route(
            intent,
            [
                build_assessment(38, final_score=0.88, direct_evidence=True, supports_extractive=False),
                build_assessment(41, final_score=0.76, direct_evidence=True, supports_extractive=False),
            ],
        )

        self.assertEqual(plan.mode, AnswerMode.GENERATIVE)
        self.assertEqual(plan.reason, 'multi_document_synthesis')
