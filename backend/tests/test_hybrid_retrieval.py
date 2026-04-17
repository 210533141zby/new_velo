import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.rag.hybrid_search import build_hybrid_candidates, compute_query_profile, ensure_hybrid_index


class FakeDoc:
    def __init__(self, title: str, doc_id: int, page_content: str, metadata: dict | None = None) -> None:
        self.metadata = {'source': title, 'doc_id': doc_id, **(metadata or {})}
        self.page_content = page_content


class HybridRetrievalTests(unittest.TestCase):
    def test_query_profile_increases_lexical_weight_for_identifier_query(self) -> None:
        indexed = ensure_hybrid_index(
            [
                SimpleNamespace(id=1, title='CSAPP 第3章', content='讲解链接、装载与共享库', updated_at=None),
                SimpleNamespace(id=2, title='数据库事务', content='介绍隔离级别与并发控制', updated_at=None),
            ]
        )

        profile = compute_query_profile('CSAPP 第3章讲了什么', indexed.idf_lookup)

        self.assertGreater(profile['lexical_weight'], 0.5)
        self.assertLess(profile['dense_weight'], 0.5)

    def test_hybrid_candidates_can_recover_bm25_only_document(self) -> None:
        indexed = ensure_hybrid_index(
            [
                SimpleNamespace(id=1, title='CSAPP 第3章', content='这一章主要讲链接、装载与共享库。', updated_at=None),
                SimpleNamespace(id=2, title='Python 教程', content='主要介绍基础语法。', updated_at=None),
                SimpleNamespace(id=3, title='CSAPP 第4章', content='这一章主要讲处理器体系结构。', updated_at=None),
            ]
        )
        vector_matches = [
            (FakeDoc('Python 教程', 2, '主要介绍基础语法。'), 0.91),
        ]

        candidates = build_hybrid_candidates(
            'CSAPP 第3章讲了什么',
            vector_matches,
            indexed,
            vector_limit=10,
            bm25_limit=10,
            candidate_limit=5,
        )

        candidate_ids = [item[0].metadata['doc_id'] for item in candidates]
        self.assertIn(1, candidate_ids)
        recovered = next(doc for doc, _score in candidates if doc.metadata['doc_id'] == 1)
        self.assertEqual(recovered.metadata['candidate_source'], 'hybrid')
        self.assertGreater(recovered.metadata['bm25_score'], 0.0)
        self.assertGreater(recovered.metadata['adaptive_score'], 0.0)


if __name__ == '__main__':
    unittest.main()
