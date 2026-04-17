import asyncio
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.rag.evidence_scorer import UnifiedEvidenceScorer
from app.services.rag.pipeline_models import (
    DefenseProfile,
    EvidenceRequirement,
    QueryIntent,
    QueryIntentType,
    RetrievedCandidate,
)


def build_intent(
    intent_type: QueryIntentType,
    *,
    defense_profile: DefenseProfile,
    evidence_requirement: EvidenceRequirement,
    needs_judge: bool,
) -> QueryIntent:
    return QueryIntent(
        original_query='档案馆建于哪一年',
        normalized_query='档案馆建于哪一年',
        keyword_query='档案馆 建于 一年',
        intent_type=intent_type,
        retrieval_depth=8,
        defense_profile=defense_profile,
        evidence_requirement=evidence_requirement,
        wants_short_answer=True,
        needs_judge=needs_judge,
        trace_tags=(),
    )


def build_candidate(
    doc_id: int,
    *,
    title: str = '雾潮镇的档案馆',
    content: str = '档案馆的正式名称叫雾潮地方文书保藏中心，建于1958年。',
    rerank_score: float = 0.88,
    dense_score: float = 0.80,
    bm25_score: float = 0.76,
    rrf_score: float = 0.72,
    coverage_score: float = 0.64,
    identifier_overlap: float = 0.0,
) -> RetrievedCandidate:
    doc = type(
        'Doc',
        (),
        {
            'page_content': content,
            'metadata': {'source': title, 'doc_id': doc_id},
        },
    )()
    return RetrievedCandidate(
        doc=doc,
        doc_id=doc_id,
        title=title,
        dense_score=dense_score,
        bm25_score=bm25_score,
        rrf_score=rrf_score,
        rerank_score=rerank_score,
        coverage_score=coverage_score,
        identifier_overlap=identifier_overlap,
    )


class UnifiedEvidenceScorerTests(unittest.IsolatedAsyncioTestCase):
    async def test_preserves_candidate_count_without_silent_drop(self) -> None:
        async def fake_judge(_query: str, _title: str, content: str) -> dict:
            if '1958' in content:
                return {
                    'core_topic_match': True,
                    'contains_direct_evidence': True,
                    'answerable': True,
                    'evidence_quote': '建于1958年',
                    'answer_brief': '建于1958年。',
                }
            return {
                'core_topic_match': False,
                'contains_direct_evidence': False,
                'answerable': False,
                'reason': 'irrelevant',
            }

        scorer = UnifiedEvidenceScorer(judge_timeout_seconds=0.5, judge_callable=fake_judge)
        intent = build_intent(
            QueryIntentType.FACTOID,
            defense_profile=DefenseProfile.STRICT,
            evidence_requirement=EvidenceRequirement.ATOMIC_SPAN,
            needs_judge=True,
        )
        candidates = [
            build_candidate(1),
            build_candidate(2, title='无关文档', content='这是一段完全无关的说明文字。', coverage_score=0.05, rerank_score=0.12),
        ]

        assessments = await scorer.assess_concurrently(candidates, intent)

        self.assertEqual(len(assessments), 2)
        self.assertTrue(assessments[0].usable)
        self.assertFalse(assessments[1].usable)
        self.assertFalse(assessments[1].judge.invoked)

    async def test_runs_judge_in_parallel(self) -> None:
        async def slow_judge(_query: str, _title: str, _content: str) -> dict:
            await asyncio.sleep(0.12)
            return {
                'core_topic_match': True,
                'contains_direct_evidence': True,
                'answerable': True,
            }

        scorer = UnifiedEvidenceScorer(judge_timeout_seconds=1.0, judge_callable=slow_judge)
        intent = build_intent(
            QueryIntentType.RELATION,
            defense_profile=DefenseProfile.STRICT,
            evidence_requirement=EvidenceRequirement.MULTI_SPAN,
            needs_judge=True,
        )
        candidates = [build_candidate(1), build_candidate(2), build_candidate(3)]

        started_at = time.perf_counter()
        assessments = await scorer.assess_concurrently(candidates, intent)
        elapsed = time.perf_counter() - started_at

        self.assertEqual(len(assessments), 3)
        self.assertLess(elapsed, 0.24)
        self.assertTrue(all(assessment.judge is not None for assessment in assessments))
        self.assertTrue(all(assessment.judge.invoked for assessment in assessments))

    async def test_timeout_marks_candidate_and_does_not_block_batch(self) -> None:
        async def timeout_judge(_query: str, title: str, _content: str) -> dict:
            if title == '超时文档':
                await asyncio.sleep(0.18)
            else:
                await asyncio.sleep(0.02)
            return {
                'core_topic_match': True,
                'contains_direct_evidence': True,
                'answerable': True,
            }

        scorer = UnifiedEvidenceScorer(judge_timeout_seconds=0.05, judge_callable=timeout_judge)
        intent = build_intent(
            QueryIntentType.RELATION,
            defense_profile=DefenseProfile.STRICT,
            evidence_requirement=EvidenceRequirement.MULTI_SPAN,
            needs_judge=True,
        )
        candidates = [
            build_candidate(1, title='正常文档'),
            build_candidate(2, title='超时文档'),
        ]

        started_at = time.perf_counter()
        assessments = await scorer.assess_concurrently(candidates, intent)
        elapsed = time.perf_counter() - started_at

        self.assertEqual(len(assessments), 2)
        self.assertLess(elapsed, 0.14)
        self.assertTrue(assessments[1].judge.timed_out)
        self.assertFalse(assessments[1].usable)
        self.assertIn('LLM_JUDGE_TIMEOUT', assessments[1].flags)

    async def test_emits_flat_trace_logs(self) -> None:
        async def fake_judge(_query: str, _title: str, _content: str) -> dict:
            return {
                'core_topic_match': True,
                'contains_direct_evidence': True,
                'answerable': True,
                'evidence_quote': '建于1958年',
                'answer_brief': '建于1958年。',
            }

        scorer = UnifiedEvidenceScorer(judge_timeout_seconds=0.5, judge_callable=fake_judge)
        intent = build_intent(
            QueryIntentType.FACTOID,
            defense_profile=DefenseProfile.STRICT,
            evidence_requirement=EvidenceRequirement.ATOMIC_SPAN,
            needs_judge=True,
        )

        with patch('app.services.rag.evidence_scorer.logger.info') as log_info:
            assessments = await scorer.assess_concurrently([build_candidate(1)], intent)

        self.assertEqual(len(assessments), 1)
        self.assertEqual(log_info.call_count, 1)
        trace_payload = log_info.call_args.kwargs['extra']['extra_data']
        self.assertEqual(trace_payload['event'], 'rag_evidence_assessed')
        self.assertEqual(trace_payload['doc_id'], 1)
        self.assertIn('base_relevance', trace_payload)
        self.assertIn('topic_match', trace_payload)
        self.assertIn('judge_latency_ms', trace_payload)
        self.assertIn('flags', trace_payload)
        self.assertIn('usable', trace_payload)
