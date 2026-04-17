import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.rag.answer_generators import ExtractiveGenerator, FallbackRequiredError, GeneratorFactory
from app.services.rag.pipeline_models import (
    AnswerMode,
    AnswerPlan,
    DefenseProfile,
    EvidenceAssessment,
    EvidenceRequirement,
    JudgeDecision,
    QueryIntent,
    QueryIntentType,
    RetrievedCandidate,
)
from app.services.rag.rag_service import RagService


def build_intent(
    intent_type: QueryIntentType,
    *,
    evidence_requirement: EvidenceRequirement = EvidenceRequirement.ATOMIC_SPAN,
) -> QueryIntent:
    return QueryIntent(
        original_query='测试问题',
        normalized_query='测试问题',
        keyword_query='测试 问题',
        intent_type=intent_type,
        retrieval_depth=8,
        defense_profile=DefenseProfile.MODERATE,
        evidence_requirement=evidence_requirement,
        wants_short_answer=True,
        needs_judge=False,
        trace_tags=(),
    )


def build_assessment(
    doc_id: int,
    *,
    title: str = '雾潮镇的档案馆',
    content: str = '档案馆的正式名称叫雾潮地方文书保藏中心，建于1958年。',
    final_score: float = 0.92,
    usable: bool = True,
    direct_evidence: bool = True,
    supports_extractive: bool = True,
    answer_brief: str = '',
    evidence_quote: str = '',
) -> EvidenceAssessment:
    doc = SimpleNamespace(page_content=content, metadata={'source': title, 'doc_id': doc_id})
    return EvidenceAssessment(
        candidate=RetrievedCandidate(doc=doc, doc_id=doc_id, title=title),
        judge=JudgeDecision(
            invoked=bool(answer_brief or evidence_quote),
            passed=bool(answer_brief or evidence_quote),
            topic_match=True,
            direct_evidence=direct_evidence,
            answerable=bool(answer_brief or evidence_quote),
            evidence_quote=evidence_quote,
            answer_brief=answer_brief,
        ),
        final_score=final_score,
        usable=usable,
        direct_evidence=direct_evidence,
        supports_extractive=supports_extractive,
        answer_brief=answer_brief,
        evidence_quote=evidence_quote,
    )


class ExtractiveGeneratorTests(unittest.IsolatedAsyncioTestCase):
    async def test_extracts_best_sentence_from_relevant_window(self) -> None:
        generator = ExtractiveGenerator()
        intent = build_intent(QueryIntentType.FACTOID)
        assessment = build_assessment(
            38,
            content='档案馆位于镇北的槐石路17号。档案馆的正式名称叫雾潮地方文书保藏中心，建于1958年。',
        )

        response = await generator.generate('雾潮镇档案馆的正式名称是什么', intent, [assessment])

        self.assertIn('雾潮地方文书保藏中心', response or '')

    async def test_returns_none_when_no_precise_sentence_exists(self) -> None:
        generator = ExtractiveGenerator()
        intent = build_intent(QueryIntentType.FACTOID)
        assessment = build_assessment(
            38,
            content='消息公布后，一位旅居新加坡的雾潮籍商人联系馆方，表示愿意资助下一阶段的海图专题整理。',
            direct_evidence=False,
            supports_extractive=False,
        )

        response = await generator.generate('新加坡的风景', intent, [assessment])

        self.assertIsNone(response)

    async def test_extracts_original_sentence_for_polite_factoid_query(self) -> None:
        generator = ExtractiveGenerator()
        intent = build_intent(QueryIntentType.FACTOID)
        assessment = build_assessment(
            33,
            title='量子计算与Shor算法',
            content='1994年，数学家Peter Shor提出了著名的Shor算法，这被视为量子计算领域的里程碑。\n该算法能在多项式时间内解决大整数质因数分解问题，而经典计算机对此问题需要指数时间。',
        )

        response = await generator.generate('请问一下，Shor主要解决什么问题吗？', intent, [assessment])

        self.assertEqual(response, '该算法能在多项式时间内解决大整数质因数分解问题，而经典计算机对此问题需要指数时间。')

    async def test_extracts_definition_line_from_list_style_document(self) -> None:
        generator = ExtractiveGenerator()
        intent = build_intent(QueryIntentType.FACTOID)
        assessment = build_assessment(
            35,
            title='北宋熙宁变法中的经济政策',
            content='1069年，宋神宗任用王安石推行全面改革。\n- **青苗法**：官府在青黄不接时向农民贷款，收获后加息20%偿还\n- **市易法**：设立市易务平价收购滞销货物，允许商户赊购',
        )

        response = await generator.generate('青苗法是什么', intent, [assessment])

        self.assertIn('青苗法：官府在青黄不接时向农民贷款', response or '')

    async def test_prefers_causal_sentence_for_reason_query(self) -> None:
        generator = ExtractiveGenerator()
        intent = build_intent(QueryIntentType.REASON, evidence_requirement=EvidenceRequirement.MULTI_SPAN)
        assessment = build_assessment(
            38,
            title='雾潮镇的档案馆',
            content='最受关注的藏品却不是年代最久的，而是一本编号为A-17-204的蓝布账册。\n那本账册记录了1949年至1952年间雾潮镇与邻近三处渔村的物资往来。\n研究者之所以重视它，是因为账册里频繁出现一个名字：沈见川。',
            direct_evidence=True,
            supports_extractive=True,
        )

        response = await generator.generate('为什么研究者重视编号为A-17-204的蓝布账册', intent, [assessment])

        self.assertIn('沈见川', response or '')


class GeneratorFactoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_downgrades_after_extractive_failure(self) -> None:
        mock_model = SimpleNamespace(ainvoke=AsyncMock(return_value=SimpleNamespace(content='结构化答案')))
        factory = GeneratorFactory(lambda: mock_model)
        intent = build_intent(QueryIntentType.FACTOID)
        plan = AnswerPlan(
            mode=AnswerMode.EXTRACTIVE,
            reason='atomic_evidence_available',
            primary_doc_id=38,
            source_doc_ids=(38,),
            generator_name='extractive_generator',
        )
        assessment = build_assessment(
            38,
            content='消息公布后，一位旅居新加坡的雾潮籍商人联系馆方，表示愿意资助下一阶段的海图专题整理。',
            direct_evidence=False,
            supports_extractive=False,
        )

        with self.assertRaises(FallbackRequiredError):
            await factory.execute(plan, '新加坡的风景', intent, [assessment])

        downgraded = factory.downgrade(plan)
        response, sources = await factory.execute(downgraded, '新加坡的风景', intent, [assessment])
        self.assertEqual(response, '结构化答案')
        self.assertEqual(sources[0]['doc_id'], 38)


class RagServicePipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_rag_qa_keeps_external_contract(self) -> None:
        service = RagService(AsyncMock())
        intent = build_intent(QueryIntentType.FACTOID)
        candidate = RetrievedCandidate(
            doc=SimpleNamespace(page_content='档案馆的正式名称叫雾潮地方文书保藏中心，建于1958年。', metadata={'source': '雾潮镇的档案馆', 'doc_id': 38}),
            doc_id=38,
            title='雾潮镇的档案馆',
        )
        assessment = build_assessment(38)
        plan = AnswerPlan(
            mode=AnswerMode.EXTRACTIVE,
            reason='atomic_evidence_available',
            primary_doc_id=38,
            source_doc_ids=(38,),
            generator_name='extractive_generator',
        )

        with patch('app.services.rag.rag_service.redis_manager.get', new=AsyncMock(return_value=None)), patch(
            'app.services.rag.rag_service.redis_manager.set', new=AsyncMock()
        ), patch(
            'app.services.rag.rag_service.QueryIntentBuilder.build',
            new=AsyncMock(return_value=intent),
        ), patch.object(
            service,
            '_retrieve_candidates',
            new=AsyncMock(return_value=[candidate]),
        ), patch.object(
            service.evidence_scorer,
            'assess_concurrently',
            new=AsyncMock(return_value=[assessment]),
        ), patch.object(
            service.answer_router,
            'route',
            return_value=plan,
        ):
            result = await service.rag_qa('雾潮镇档案馆的正式名称是什么')

        self.assertIsInstance(result, dict)
        self.assertEqual(set(result.keys()), {'response', 'sources'})
        self.assertIn('雾潮地方文书保藏中心', result['response'])
        self.assertEqual(result['sources'][0]['title'], '雾潮镇的档案馆')


if __name__ == '__main__':
    unittest.main()
