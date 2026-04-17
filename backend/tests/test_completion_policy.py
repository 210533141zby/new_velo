import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.api.completion import CompletionRequest, generate_completion
from app.services.completion.completion_policy import (
    build_context_instruction,
    get_profile,
    infer_completion_context,
    post_process_completion,
    truncate_context,
    violates_context_constraints,
)
from app.services.rag.prompt_templates import build_assistant_identity_answer, is_model_identity_query


class CompletionPolicyTests(unittest.TestCase):
    def test_standardized_profile_values(self) -> None:
        self.assertEqual(get_profile('auto').prefix_window, 512)
        self.assertEqual(get_profile('auto').suffix_window, 96)
        self.assertEqual(get_profile('auto').max_chars, 24)
        self.assertEqual(get_profile('manual').prefix_window, 896)
        self.assertEqual(get_profile('manual').suffix_window, 160)
        self.assertEqual(get_profile('manual').max_chars, 56)

    def test_truncate_context_uses_standardized_windows(self) -> None:
        prefix = 'a' * 2000
        suffix = 'b' * 500
        manual_prefix, manual_suffix = truncate_context(prefix, suffix, 'manual')
        auto_prefix, auto_suffix = truncate_context(prefix, suffix, 'auto')
        self.assertEqual(len(manual_prefix), 896)
        self.assertEqual(len(manual_suffix), 160)
        self.assertEqual(len(auto_prefix), 512)
        self.assertEqual(len(auto_suffix), 96)

    def test_infer_list_item_context(self) -> None:
        context = infer_completion_context('## 主要措施\n\n- ', 'markdown')
        self.assertEqual(context.block_kind, 'list_item_body')

    def test_rejects_unsignaled_markdown_list_lead_in(self) -> None:
        prefix = (
            '在我的文档中**数据记录**：\n'
            '变法期间国库岁入从熙宁初年的6000万贯增至8000万贯，但地方执行出现强制配贷现象。'
        )
        context = infer_completion_context(prefix, 'markdown')
        candidate = '- 王安石变法的经济政策体系还包括：'
        self.assertTrue(violates_context_constraints(candidate, prefix, '', context))
        self.assertEqual(post_process_completion(candidate, prefix, '', context, 'manual'), '')

    def test_allows_list_item_body_after_explicit_marker(self) -> None:
        prefix = '## 主要措施\n\n- '
        context = infer_completion_context(prefix, 'markdown')
        candidate = '青苗法试图缓解农民在青黄不接时的借贷压力'
        self.assertFalse(violates_context_constraints(candidate, prefix, '', context))
        self.assertEqual(post_process_completion(candidate, prefix, '', context, 'manual'), candidate)

    def test_paragraph_context_instruction_forbids_new_markdown_blocks(self) -> None:
        context = infer_completion_context('最新考古发现显示部分州县商业税增长35%', 'markdown')
        instruction = build_context_instruction(context)
        self.assertIn('不要新起列表', instruction)
        self.assertIn('还包括', instruction)


class RagPromptTests(unittest.TestCase):
    def test_detects_model_identity_query(self) -> None:
        self.assertTrue(is_model_identity_query('你是什么模型'))
        self.assertTrue(is_model_identity_query('what model are you'))
        self.assertIn('当前对话使用的模型配置是', build_assistant_identity_answer(for_rag=True))


class CompletionApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_api_passes_language_into_completion_service(self) -> None:
        request = CompletionRequest(prefix='前文', suffix='', language='markdown', trigger_mode='manual')
        with patch('app.api.completion.complete_text', new=AsyncMock(return_value='补文')) as mocked_complete:
            result = await generate_completion(request)
        mocked_complete.assert_awaited_once_with('前文', '', language='markdown', trigger_mode='manual')
        self.assertEqual(result, {'completion': '补文'})

    async def test_api_preserves_http_exception_status(self) -> None:
        request = CompletionRequest(prefix='前文', suffix='', language='markdown', trigger_mode='manual')
        with patch('app.api.completion.complete_text', new=AsyncMock(return_value=None)):
            with self.assertRaises(HTTPException) as captured:
                await generate_completion(request)
        self.assertEqual(captured.exception.status_code, 503)


if __name__ == '__main__':
    unittest.main()
