"""
=============================================================================
文件: completion_policy.py
描述: 文本补全策略层

设计目标：
1. 将“如何约束补全”与“如何调用模型”分离，避免在 Prompt 和后处理上持续叠补丁。
2. 为中文 Markdown 文档补全建立可解释、可测试的上下文分类与后验审核机制。
3. 在保持质量的前提下，标准化上下文窗口和最大输出长度，兼顾延迟与稳定性。
=============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal

CompletionTriggerMode = Literal["auto", "manual"]
MARKDOWN_LANGUAGES = {"markdown", "md", "mdx"}
MARKDOWN_LIST_MARKER_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s*$")
MARKDOWN_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)")
MARKDOWN_HEADING_MARKER_RE = re.compile(r"^\s{0,3}#{1,6}\s*$")
MARKDOWN_QUOTE_MARKER_RE = re.compile(r"^\s*>\s*$")
MARKDOWN_BLOCK_START_RE = re.compile(r"^\s{0,3}(?:[-*+]\s+|\d+[.)]\s+|#{1,6}\s+|>\s+|```|~~~)")
SENTENCE_END_RE = re.compile(r"[。！？!?；;]$")
SENTENCE_SPLIT_RE = re.compile(r"[。！？!?；;]")
ARABIC_DETAIL_RE = re.compile(r"\d+")
CN_QUANT_DETAIL_RE = re.compile(
    r"(?:数十|数百|数千|几十|几百|几千|[一二两三四五六七八九十百千万几]+)"
    r"(?:年|月|日|天|周|次|册|份|个|项|页|篇|公里|分钟|小时)"
)
DETAIL_SPAN_MARKERS = ("长达", "历时", "跨度", "整个时期")
UNFINISHED_TAIL_MARKERS = ("直到", "例如", "比如", "其中", "使得", "从而", "以及", "用于", "以便")


@dataclass(frozen=True)
class CompletionProfile:
    prefix_window: int
    suffix_window: int
    prompt_prefix_window: int
    prompt_suffix_window: int
    max_tokens: int
    timeout: float
    temperature: float
    system_prompt: str


@dataclass(frozen=True)
class CompletionContext:
    language: str
    block_kind: str
    prefer_sentence_continuation: bool


COMPLETION_PROFILES = {
    "auto": CompletionProfile(
        prefix_window=512,
        suffix_window=96,
        prompt_prefix_window=288,
        prompt_suffix_window=72,
        max_tokens=24,
        timeout=7.0,
        temperature=0.0,
        system_prompt=(
            "你是中文写作补全助手。"
            "只输出可直接插入光标处的中文文本，不要解释，不要复述题目，不要编造前后文未提供的新事实。"
        ),
    ),
    "manual": CompletionProfile(
        prefix_window=768,
        suffix_window=128,
        prompt_prefix_window=384,
        prompt_suffix_window=80,
        max_tokens=48,
        timeout=10.0,
        temperature=0.0,
        system_prompt=(
            "你是中文写作补全助手。"
            "只输出可直接插入光标处的中文文本，不要解释，不要复述题目，不要编造前后文未提供的新事实。"
        ),
    ),
}


def normalize_trigger_mode(trigger_mode: str | None) -> CompletionTriggerMode:
    return "auto" if trigger_mode == "auto" else "manual"


def normalize_language(language: str | None) -> str:
    return (language or "").strip().lower()


def get_profile(trigger_mode: str | None) -> CompletionProfile:
    return COMPLETION_PROFILES[normalize_trigger_mode(trigger_mode)]


def is_markdown_language(language: str | None) -> bool:
    return normalize_language(language) in MARKDOWN_LANGUAGES


def truncate_context(prefix: str, suffix: str, trigger_mode: str | None) -> tuple[str, str]:
    profile = get_profile(trigger_mode)
    return prefix[-profile.prefix_window:], suffix[: profile.suffix_window]


def infer_completion_context(prefix: str, language: str | None) -> CompletionContext:
    current_line = prefix.rsplit("\n", 1)[-1] if prefix else ""
    normalized_language = normalize_language(language)

    if not is_markdown_language(normalized_language):
        block_kind = "plain_text"
    elif MARKDOWN_LIST_MARKER_RE.match(current_line):
        block_kind = "list_item_body"
    elif MARKDOWN_HEADING_MARKER_RE.match(current_line):
        block_kind = "heading_text"
    elif MARKDOWN_QUOTE_MARKER_RE.match(current_line):
        block_kind = "quote_body"
    else:
        block_kind = "paragraph"

    return CompletionContext(
        language=normalized_language,
        block_kind=block_kind,
        prefer_sentence_continuation=bool(current_line.strip()) and block_kind in {"plain_text", "paragraph"},
    )


def build_context_instruction(context: CompletionContext) -> str:
    if context.block_kind == "list_item_body":
        return (
            "当前光标位于 Markdown 列表项标记之后。只补全该列表项的正文，不要再输出新的 "
            "`-`、`*` 或编号，也不要写“还包括：”“如下：”这类总括性引导句。"
        )
    if context.block_kind == "heading_text":
        return "当前光标位于 Markdown 标题标记之后。只补全标题文本，不要输出正文、列表或解释。"
    if context.block_kind == "quote_body":
        return "当前光标位于 Markdown 引用标记之后。只补全这一行引用正文，不要新起其他 Markdown 结构。"
    if context.block_kind == "paragraph":
        return (
            "当前光标位于 Markdown 正文中。只续写当前句子或当前段落，不要新起列表、标题、引用或代码块；"
            "不要输出以 `-`、`*`、`#`、`>` 或编号列表开头的文本；不要写“还包括：”“如下：”“主要有：”这类引导句。"
        )
    return (
        "当前是在自然语言正文中补全。优先延续当前句子或当前段落，不要突然转入新主题，"
        "也不要写空泛总括句。"
    )


def build_retry_hint(previous_reject_reason: str | None) -> str:
    if previous_reject_reason == "unfinished_bridge_tail":
        return (
            "上一次结果停在明显未说完的尾巴上。"
            "这一次请输出一个可以直接插入的完整短句，不要以逗号、冒号或未完连接词收尾。"
        )
    if previous_reject_reason == "unsupported_quant_detail":
        return (
            "上一次结果补入了前后文没有明示的数量、次数或时间跨度。"
            "这一次请改为短而中性的承接句，不要新增任何数值、频率或跨度细节。"
        )
    if previous_reject_reason == "fragmented_sentence_start":
        return (
            "上一次结果像从半句话中间截断的残片。"
            "这一次必须从完整自然的句子开头开始，不要以上文缺失的谓语、连词或残句开头。"
        )
    if previous_reject_reason == "unterminated_sentence":
        return (
            "上一次结果停在未完成的句尾。"
            "这一次请输出一个自然收束的完整句子，不要停在未完成短语上。"
        )
    if previous_reject_reason == "covered_by_suffix":
        return (
            "上一次结果和后文重复。"
            "这一次不要重复后文已出现的信息，只有在确实需要桥接时才补一小段完整文本。"
        )
    if previous_reject_reason == "empty_model_response":
        return "上一次没有生成任何可用内容。这一次请直接给出一小段完整、具体、可插入的文本。"
    return (
        "上一次结果为空、重复、结构不对或过于空泛。"
        "这一次请务必给出具体、贴合上下文的补文，不要输出套话，不要生成新的 Markdown 结构。"
    )


def build_length_instruction(trigger_mode: str | None, has_suffix: bool) -> str:
    mode = normalize_trigger_mode(trigger_mode)
    if mode == "auto":
        return "长度控制在半句话到一句话，通常不超过25个中文字符。"
    if has_suffix:
        return "长度控制在一句到两句话内，通常不超过80个中文字符。"
    return "长度控制在一句话内，通常不超过60个中文字符。"


def build_chat_messages(
    prefix: str,
    suffix: str,
    context: CompletionContext,
    trigger_mode: str | None,
    attempt: int,
    previous_reject_reason: str | None = None,
) -> list[dict[str, str]]:
    profile = get_profile(trigger_mode)
    prefix_tail = prefix[-profile.prompt_prefix_window:]
    suffix_head = suffix[: profile.prompt_suffix_window]
    context_instruction = build_context_instruction(context)
    retry_hint = build_retry_hint(previous_reject_reason) if attempt > 0 else ""
    requirements = [
        "只输出要插入光标处的文本，不要解释，不要加引号，不要加“补全文本：”之类标签。",
        "必须同时和前文末尾、后文开头自然衔接。",
        "不要重复前文或后文里已经出现的完整短语或句子。",
        "不要引入前后文没有出现的新人物、新出处、新数据、新设定。",
        "避免主观推断、评价或升华，优先只写前后文能够直接支持的承接句。",
        "不得补充前后文未明示的数量、年份、时间跨度、范围或统计判断。",
        "优先补事实承接、限定或转折，不要写空泛过渡套话，如“随着时间的推移”“值得注意的是”“这说明了”。",
        context_instruction,
        build_length_instruction(trigger_mode, bool(suffix_head)),
    ]
    if retry_hint:
        requirements.append(retry_hint)
    if suffix_head:
        task = "在前文和后文之间补出一小段自然、具体、语义连贯的中文文本。"
        sections = [
            f"前文：\n{prefix_tail}",
            f"后文：\n{suffix_head}",
            f"任务：{task}",
            "要求：\n" + "\n".join(f"{index}. {item}" for index, item in enumerate(requirements, start=1)),
        ]
    else:
        task = "沿着前文继续写一小段自然、具体、语义连贯的中文文本。"
        sections = [
            f"前文：\n{prefix_tail}",
            f"任务：{task}",
            "要求：\n" + "\n".join(f"{index}. {item}" for index, item in enumerate(requirements, start=1)),
        ]

    return [
        {'role': 'system', 'content': profile.system_prompt},
        {'role': 'user', 'content': "\n\n".join(sections)},
    ]


def contains_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def remove_prefix_overlap(text: str, prefix: str) -> str:
    if not text or not prefix:
        return text
    max_overlap = min(len(text), len(prefix))
    for index in range(max_overlap, 0, -1):
        if text[:index] == prefix[-index:]:
            return text[index:]
    return text


def remove_suffix_overlap(text: str, suffix: str) -> str:
    if not text or not suffix:
        return text
    safe_suffix = suffix.lstrip()
    if not safe_suffix:
        return text
    max_overlap = min(len(text), len(safe_suffix))
    for index in range(max_overlap, 0, -1):
        if text[-index:] == safe_suffix[:index]:
            return text[:-index]
    return text


def remove_leading_suffix_overlap(text: str, suffix: str) -> str:
    if not text or not suffix:
        return text
    safe_suffix = suffix.lstrip()
    if not safe_suffix:
        return text
    max_overlap = min(len(text), len(safe_suffix))
    min_overlap = 4 if contains_cjk(text[:max_overlap] + safe_suffix[:max_overlap]) else 6
    for index in range(max_overlap, min_overlap - 1, -1):
        if text[:index] == safe_suffix[:index]:
            return text[index:]
    return text


def is_suffix_echo(text: str, suffix: str) -> bool:
    candidate = text.strip()
    safe_suffix = suffix.lstrip()
    if not candidate or not safe_suffix:
        return False
    min_echo_length = 4 if contains_cjk(candidate) else 6
    return len(candidate) >= min_echo_length and safe_suffix.startswith(candidate)


def is_sentence_boundary_insertion(prefix: str, suffix: str) -> bool:
    return bool(prefix.strip()) and bool(suffix.strip()) and bool(SENTENCE_END_RE.search(prefix.rstrip()))


def ends_with_terminal_punctuation(text: str) -> bool:
    stripped = text.rstrip()
    while stripped and stripped[-1] in '”’」』）》】)]':
        stripped = stripped[:-1].rstrip()
    return bool(stripped) and bool(SENTENCE_END_RE.search(stripped))


def has_fragmented_sentence_start(text: str, prefix: str, suffix: str) -> bool:
    if not is_sentence_boundary_insertion(prefix, suffix):
        return False
    first_match = SENTENCE_SPLIT_RE.search(text)
    if not first_match:
        return False
    first_sentence = text[:first_match.start()].strip(" ，,、：:")
    remaining = text[first_match.end():].strip()
    return bool(remaining) and len(first_sentence) < 7


def has_unterminated_sentence(text: str, prefix: str, suffix: str) -> bool:
    if not is_sentence_boundary_insertion(prefix, suffix):
        return False
    cleaned = text.strip(" ，,、：:")
    return len(cleaned) >= 12 and not ends_with_terminal_punctuation(text)


def has_unsupported_quant_detail(text: str, prefix: str, suffix: str) -> bool:
    combined = f"{prefix}\n{suffix}"
    for match in ARABIC_DETAIL_RE.findall(text):
        if match not in combined:
            return True
    for match in CN_QUANT_DETAIL_RE.findall(text):
        if match not in combined:
            return True
    return any(marker in text and marker not in combined for marker in DETAIL_SPAN_MARKERS)


def is_inline_enumeration_bridge(prefix: str, suffix: str, text: str) -> bool:
    if not suffix.strip():
        return False
    candidate = text.strip()
    prefix_tail = prefix.rstrip()
    return bool(prefix_tail) and prefix_tail.endswith("、") and candidate.endswith("、")


def has_short_non_terminal_bridge(text: str, prefix: str, suffix: str) -> bool:
    if not suffix.strip():
        return False
    candidate = text.strip()
    if not candidate or ends_with_terminal_punctuation(candidate):
        return False
    if is_inline_enumeration_bridge(prefix, suffix, candidate):
        return False
    return len(candidate.strip(" ，,、：:；;")) < 6


def salvage_inline_enumeration_candidate(text: str, prefix: str, suffix: str) -> str:
    candidate = text.strip()
    if not candidate or not suffix.strip():
        return candidate
    if not prefix.rstrip().endswith("、"):
        return candidate
    if candidate.endswith("、") or ends_with_terminal_punctuation(candidate) or "，" not in candidate:
        return candidate
    head, tail = candidate.split("，", 1)
    head = head.rstrip("、，").strip()
    tail = tail.strip()
    if not head or not tail:
        return candidate
    if not any(marker in tail for marker in UNFINISHED_TAIL_MARKERS) and len(tail.strip(" ，,、：:；;")) >= 8:
        return candidate
    return f"{head}、"


def has_unfinished_bridge_tail(text: str, prefix: str, suffix: str) -> bool:
    if not suffix.strip():
        return False
    candidate = text.strip()
    if not candidate or ends_with_terminal_punctuation(candidate):
        return False
    if is_inline_enumeration_bridge(prefix, suffix, candidate):
        return False
    if candidate.endswith(("，", "、", "：", ":", "（", "(")):
        return True
    return any(candidate.endswith(marker) for marker in UNFINISHED_TAIL_MARKERS)


def trim_completion(text: str, trigger_mode: str | None) -> str:
    if not text:
        return ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    return lines[0]


def normalize_completion_candidate(
    text: str,
    prefix: str,
    suffix: str,
    trigger_mode: str | None,
) -> str:
    candidate = trim_completion((text or "").strip(), trigger_mode)
    if not candidate:
        return ""
    candidate = remove_prefix_overlap(candidate, prefix).lstrip()
    candidate = remove_leading_suffix_overlap(candidate, suffix).lstrip()
    candidate = remove_suffix_overlap(candidate, suffix).rstrip()
    candidate = salvage_inline_enumeration_candidate(candidate, prefix, suffix)
    return candidate


def validate_completion_candidate(
    text: str,
    prefix: str,
    suffix: str,
    context: CompletionContext,
) -> str | None:
    candidate = text.strip()
    if not candidate:
        return "empty"
    if has_short_non_terminal_bridge(candidate, prefix, suffix):
        return "short_bridge_fragment"
    if has_unfinished_bridge_tail(candidate, prefix, suffix):
        return "unfinished_bridge_tail"
    if has_unsupported_quant_detail(candidate, prefix, suffix):
        return "unsupported_quant_detail"
    if has_fragmented_sentence_start(candidate, prefix, suffix):
        return "fragmented_sentence_start"
    if has_unterminated_sentence(candidate, prefix, suffix):
        return "unterminated_sentence"
    if prefix.rstrip().endswith(candidate):
        return "prefix_echo"
    if is_suffix_echo(candidate, suffix):
        return "suffix_echo"
    if not candidate.strip("，。；：,:!?！？、（）()《》“”‘’【】[]-— "):
        return "punctuation_only"
    if context.block_kind in {"paragraph", "plain_text"} and MARKDOWN_BLOCK_START_RE.match(candidate):
        return "unexpected_markdown_block"
    if context.block_kind == "list_item_body" and MARKDOWN_LIST_ITEM_RE.match(candidate):
        return "unexpected_list_marker"
    if context.block_kind == "heading_text" and (candidate.startswith("#") or MARKDOWN_BLOCK_START_RE.match(candidate)):
        return "unexpected_heading_block"
    if context.block_kind == "quote_body" and candidate.startswith(">"):
        return "unexpected_quote_marker"
    if context.prefer_sentence_continuation and candidate.endswith(("：", ":")):
        return "trailing_colon"
    return None


def post_process_completion_with_reason(
    text: str,
    prefix: str,
    suffix: str,
    context: CompletionContext,
    trigger_mode: str | None,
) -> tuple[str, str | None]:
    candidate = trim_completion((text or "").strip(), trigger_mode)
    if not candidate:
        return "", "empty_model_response"

    candidate = remove_prefix_overlap(candidate, prefix).lstrip()
    if not candidate:
        return "", "prefix_echo"

    candidate = remove_leading_suffix_overlap(candidate, suffix).lstrip()
    if not candidate:
        return "", "covered_by_suffix"

    candidate = remove_suffix_overlap(candidate, suffix).rstrip()
    if not candidate:
        return "", "covered_by_suffix"

    candidate = salvage_inline_enumeration_candidate(candidate, prefix, suffix)
    reason = validate_completion_candidate(candidate, prefix, suffix, context)
    if reason:
        return "", reason

    mode = normalize_trigger_mode(trigger_mode)
    if len(candidate.strip(" ，,。；;")) < (4 if mode == "manual" else 2):
        return "", "too_short"
    return candidate, None


def post_process_completion(
    text: str,
    prefix: str,
    suffix: str,
    context: CompletionContext,
    trigger_mode: str | None,
) -> str:
    candidate, _reason = post_process_completion_with_reason(text, prefix, suffix, context, trigger_mode)
    return candidate
