import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.api.completion import CompletionRequest, generate_completion
from app.services.completion.completion_policy import (
    build_retry_hint,
    build_context_instruction,
    get_profile,
    infer_completion_context,
    post_process_completion_with_reason,
    post_process_completion,
    remove_leading_suffix_overlap,
    trim_completion,
    truncate_context,
    validate_completion_candidate,
)
from app.services.rag.prompt_templates import build_assistant_identity_answer, is_model_identity_query


class CompletionPolicyTests(unittest.TestCase):
    def test_standardized_profile_values(self) -> None:
        self.assertEqual(get_profile('auto').prefix_window, 512)
        self.assertEqual(get_profile('auto').suffix_window, 96)
        self.assertEqual(get_profile('auto').max_tokens, 24)
        self.assertEqual(get_profile('manual').prefix_window, 768)
        self.assertEqual(get_profile('manual').suffix_window, 128)
        self.assertEqual(get_profile('manual').max_tokens, 48)

    def test_truncate_context_uses_standardized_windows(self) -> None:
        prefix = 'a' * 2000
        suffix = 'b' * 500
        manual_prefix, manual_suffix = truncate_context(prefix, suffix, 'manual')
        auto_prefix, auto_suffix = truncate_context(prefix, suffix, 'auto')
        self.assertEqual(len(manual_prefix), 768)
        self.assertEqual(len(manual_suffix), 128)
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
        self.assertEqual(validate_completion_candidate(candidate, prefix, '', context), 'unexpected_markdown_block')
        self.assertEqual(post_process_completion(candidate, prefix, '', context, 'manual'), '')

    def test_allows_list_item_body_after_explicit_marker(self) -> None:
        prefix = '## 主要措施\n\n- '
        context = infer_completion_context(prefix, 'markdown')
        candidate = '青苗法试图缓解农民在青黄不接时的借贷压力'
        self.assertIsNone(validate_completion_candidate(candidate, prefix, '', context))
        self.assertEqual(post_process_completion(candidate, prefix, '', context, 'manual'), candidate)

    def test_paragraph_context_instruction_forbids_new_markdown_blocks(self) -> None:
        context = infer_completion_context('最新考古发现显示部分州县商业税增长35%', 'markdown')
        instruction = build_context_instruction(context)
        self.assertIn('不要新起列表', instruction)
        self.assertIn('还包括', instruction)

    def test_trim_completion_returns_empty_for_blank_text(self) -> None:
        self.assertEqual(trim_completion('', 'manual'), '')
        self.assertEqual(trim_completion('\n', 'manual'), '')

    def test_trim_completion_keeps_full_first_non_empty_line(self) -> None:
        text = '\n  第一行补全文本，且不应该因为字符上限被硬截断。  \n第二行'
        self.assertEqual(
            trim_completion(text, 'manual'),
            '第一行补全文本，且不应该因为字符上限被硬截断。',
        )

    def test_post_process_reports_reason_for_empty_model_response(self) -> None:
        prefix = '北宋时期，王安石希望'
        suffix = ''
        context = infer_completion_context(prefix, 'markdown')
        self.assertEqual(
            post_process_completion_with_reason('', prefix, suffix, context, 'manual'),
            ('', 'empty_model_response'),
        )

    def test_post_process_returns_empty_when_suffix_overlap_consumes_candidate(self) -> None:
        prefix = '人类线粒体DNA会'
        suffix = '影响能量代谢\n争议点：后续研究认为这属于极罕见例外。'
        context = infer_completion_context(prefix, 'markdown')
        candidate = '影响能量代谢'
        self.assertEqual(post_process_completion(candidate, prefix, suffix, context, 'manual'), '')
        self.assertEqual(
            post_process_completion_with_reason(candidate, prefix, suffix, context, 'manual'),
            ('', 'covered_by_suffix'),
        )

    def test_rejects_fragmented_sentence_start_between_complete_sentences(self) -> None:
        prefix = '研究者之所以重视它，是因为账册里频繁出现一个名字：沈见川。'
        suffix = '\n沈见川原本只是地方口述史里反复出现的“修船匠”。'
        context = infer_completion_context(prefix, 'markdown')
        candidate = '为动力帆船。这艘船后来成为雾潮镇渔村的一艘重要船只，不仅提高了捕鱼效率，还促进了当地'
        self.assertEqual(
            post_process_completion_with_reason(candidate, prefix, suffix, context, 'manual'),
            ('', 'fragmented_sentence_start'),
        )

    def test_rejects_unterminated_sentence_between_complete_sentences(self) -> None:
        prefix = '这本账册后来成为研究雾潮镇航运史的重要线索。'
        suffix = '沈见川原本只是地方口述史里反复出现的“修船匠”。'
        context = infer_completion_context(prefix, 'markdown')
        candidate = '这也解释了为什么馆员后来持续追查与他相关的改装记录'
        self.assertEqual(
            post_process_completion_with_reason(candidate, prefix, suffix, context, 'manual'),
            ('', 'unterminated_sentence'),
        )

    def test_remove_leading_suffix_overlap_strips_repeated_suffix_prefix(self) -> None:
        suffix = '目前法医鉴定和祖先追溯仍基于母系遗传假设。'
        candidate = '目前法医鉴定和祖先追溯仍基于母系遗传假设，因此mtDNA在这些领域中仍然发挥着核心作用。'
        self.assertEqual(
            remove_leading_suffix_overlap(candidate, suffix),
            '，因此mtDNA在这些领域中仍然发挥着核心作用。',
        )

    def test_post_process_keeps_only_non_overlapping_tail_when_candidate_repeats_suffix_prefix(self) -> None:
        prefix = '争议点：2018年《PNAS》报道了首例父系mtDNA传递病例。'
        suffix = '目前法医鉴定和祖先追溯仍基于母系遗传假设。'
        context = infer_completion_context(prefix, 'markdown')
        candidate = '目前法医鉴定和祖先追溯仍基于母系遗传假设，因此mtDNA在这些领域中仍然发挥着核心作用。'
        self.assertEqual(
            post_process_completion(candidate, prefix, suffix, context, 'manual'),
            '，因此mtDNA在这些领域中仍然发挥着核心作用。',
        )

    def test_rejects_unsupported_quantitative_detail_not_present_in_context(self) -> None:
        prefix = '研究者之所以重视它，是因为账册里频繁出现一个名字：沈见川。'
        suffix = '沈见川原本只是地方口述史里反复出现的“修船匠”，没有留下正式传记。'
        context = infer_completion_context(prefix, 'markdown')
        candidate = '这个名字在账册中出现了数十次，时间跨度覆盖了账册记录的整个时期。'
        self.assertEqual(
            post_process_completion_with_reason(candidate, prefix, suffix, context, 'manual'),
            ('', 'unsupported_quant_detail'),
        )

    def test_rejects_unfinished_bridge_tail(self) -> None:
        prefix = '1994年，'
        suffix = '数学家Peter Shor提出了著名的Shor算法，这被视为量子计算领域的里程碑。'
        context = infer_completion_context(prefix, 'markdown')
        candidate = '量子计算仍处于理论探索阶段，'
        self.assertEqual(
            post_process_completion_with_reason(candidate, prefix, suffix, context, 'manual'),
            ('', 'unfinished_bridge_tail'),
        )

    def test_allows_inline_enumeration_bridge_fragment(self) -> None:
        prefix = '2014年9月，镇财政拨款180万元，用于屋顶防水、'
        suffix = '恒温库房和基础数字化设备采购；2016年又追加73万元，建成了第一间影像修复室。'
        context = infer_completion_context(prefix, 'markdown')
        candidate = '墙面防潮改造、'
        self.assertEqual(
            post_process_completion_with_reason(candidate, prefix, suffix, context, 'manual'),
            ('墙面防潮改造、', None),
        )

    def test_rejects_short_non_terminal_bridge_fragment(self) -> None:
        prefix = '2014年9月，镇财政拨款180万元，用于屋顶防水、'
        suffix = '恒温库房和基础数字化设备采购；2016年又追加73万元，建成了第一间影像修复室。'
        context = infer_completion_context(prefix, 'markdown')
        candidate = '并增设了'
        self.assertEqual(
            post_process_completion_with_reason(candidate, prefix, suffix, context, 'manual'),
            ('', 'short_bridge_fragment'),
        )

    def test_salvages_dangling_purpose_tail_into_enumeration_item(self) -> None:
        prefix = '2014年9月，镇财政拨款180万元，用于屋顶防水、'
        suffix = '恒温库房和基础数字化设备采购；2016年又追加73万元，建成了第一间影像修复室。'
        context = infer_completion_context(prefix, 'markdown')
        candidate = '墙面翻新、电路改造，并采购了一批恒温恒湿设备，用于'
        self.assertEqual(
            post_process_completion_with_reason(candidate, prefix, suffix, context, 'manual'),
            ('墙面翻新、电路改造、', None),
        )

    def test_salvages_dangling_enumeration_conjunction(self) -> None:
        prefix = '2014年9月，镇财政拨款180万元，用于屋顶防水、'
        suffix = '恒温库房和基础数字化设备采购；2016年又追加73万元，建成了第一间影像修复室。'
        context = infer_completion_context(prefix, 'markdown')
        candidate = '墙面翻新、电路改造，以及'
        self.assertEqual(
            post_process_completion_with_reason(candidate, prefix, suffix, context, 'manual'),
            ('墙面翻新、电路改造、', None),
        )

    def test_post_process_reports_reason_for_too_short_candidate(self) -> None:
        prefix = '北宋时期，王安石希望'
        suffix = ''
        context = infer_completion_context(prefix, 'markdown')
        self.assertEqual(
            post_process_completion_with_reason('是', prefix, suffix, context, 'manual'),
            ('', 'too_short'),
        )

    def test_retry_hint_for_fragmented_sentence_start(self) -> None:
        self.assertIn('完整自然的句子开头', build_retry_hint('fragmented_sentence_start'))


class RagPromptTests(unittest.TestCase):
    def test_detects_model_identity_query(self) -> None:
        self.assertTrue(is_model_identity_query('你是什么模型'))
        self.assertTrue(is_model_identity_query('what model are you'))
        self.assertIn('当前对话使用的模型配置是', build_assistant_identity_answer(for_rag=True))


class CompletionApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_api_returns_reason_for_empty_prefix(self) -> None:
        request = CompletionRequest(prefix='', suffix='后文', language='markdown', trigger_mode='manual')
        result = await generate_completion(request)
        self.assertEqual(result, {'completion': '', 'reason': 'empty_prefix'})

    async def test_api_passes_language_into_completion_service(self) -> None:
        request = CompletionRequest(prefix='前文', suffix='', language='markdown', trigger_mode='manual')
        with patch('app.api.completion.complete_text_detailed', new=AsyncMock(return_value={'completion': '补文', 'reason': None})) as mocked_complete:
            result = await generate_completion(request)
        mocked_complete.assert_awaited_once_with('前文', '', language='markdown', trigger_mode='manual')
        self.assertEqual(result, {'completion': '补文', 'reason': None})

    async def test_api_preserves_http_exception_status(self) -> None:
        request = CompletionRequest(prefix='前文', suffix='', language='markdown', trigger_mode='manual')
        with patch('app.api.completion.complete_text_detailed', new=AsyncMock(return_value={'completion': None, 'reason': 'internal_error'})):
            with self.assertRaises(HTTPException) as captured:
                await generate_completion(request)
        self.assertEqual(captured.exception.status_code, 503)


if __name__ == '__main__':
    unittest.main()
