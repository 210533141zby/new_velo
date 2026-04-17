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
GENERIC_COMPLETION_PATTERNS = (
    "在当今社会",
    "促进社会经济的发展",
    "推动社会经济的发展",
    "具有重要意义",
    "起到重要作用",
    "今天",
    "进一步",
    "值得注意的是",
)
GENERIC_LEAD_IN_PATTERNS = (
    re.compile(r"^(?:[-*+]\s*)?.{0,24}(?:还包括|包括以下|如下|主要有|主要包括|可分为|分为以下).{0,12}[：:]$"),
    re.compile(r"^(?:[-*+]\s*)?(?:以下|如下)[：:]$"),
)
MARKDOWN_LIST_MARKER_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s*$")
MARKDOWN_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)")
MARKDOWN_HEADING_MARKER_RE = re.compile(r"^\s{0,3}#{1,6}\s*$")
MARKDOWN_QUOTE_MARKER_RE = re.compile(r"^\s*>\s*$")
MARKDOWN_BLOCK_START_RE = re.compile(r"^\s{0,3}(?:[-*+]\s+|\d+[.)]\s+|#{1,6}\s+|>\s+|```|~~~)")


@dataclass(frozen=True)
class CompletionProfile:
    prefix_window: int
    suffix_window: int
    prompt_prefix_window: int
    prompt_suffix_window: int
    max_tokens: int
    timeout: float
    max_chars: int
    temperature: float
    system_prompt: str


@dataclass(frozen=True)
class CompletionContext:
    language: str
    block_kind: str
    prefer_sentence_continuation: bool
    recent_lines: tuple[str, ...]


COMPLETION_PROFILES = {
    "auto": CompletionProfile(
        prefix_window=512,
        suffix_window=96,
        prompt_prefix_window=288,
        prompt_suffix_window=72,
        max_tokens=20,
        timeout=7.0,
        max_chars=24,
        temperature=0.10,
        system_prompt=(
            "你是中文写作补全器。请根据前文自然续写一小段具体、贴合语境的文本。"
            "只输出要插入光标处的内容，不要解释，不要重复上下文，避免空泛套话。"
        ),
    ),
    "manual": CompletionProfile(
        prefix_window=896,
        suffix_window=160,
        prompt_prefix_window=512,
        prompt_suffix_window=96,
        max_tokens=28,
        timeout=8.5,
        max_chars=56,
        temperature=0.15,
        system_prompt=(
            "你是中文写作补全器。任务是在前文和后文之间补出自然、具体、衔接紧密的文本。"
            "只输出要插入光标处的内容，不要解释，不要重复上下文，避免空泛套话。"
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


def recent_nonempty_lines(prefix: str, limit: int = 6) -> tuple[str, ...]:
    lines = [line.strip() for line in prefix.splitlines() if line.strip()]
    return tuple(lines[-limit:])


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
        recent_lines=recent_nonempty_lines(prefix),
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


def build_examples(context: CompletionContext) -> str:
    if context.block_kind == "list_item_body":
        return (
            "示例：\n"
            "前文：\n## 主要措施\n\n- \n\n"
            "补全文本：\n青苗法试图缓解农民在青黄不接时的借贷压力\n\n"
        )
    if context.block_kind == "paragraph":
        return (
            "示例（正确）：\n"
            "前文：\n最新考古发现（郑州商税碑）显示部分州县商业税增长35%\n\n"
            "续写文本：\n，说明部分地区的商业活力确有提升。\n\n"
            "反例（不要这样写）：\n"
            "续写文本：\n- 某项政策体系还包括：\n\n"
        )
    return (
        "示例：\n"
        "前文：\n北宋时期，王安石推行新法的核心目标，是希望通过制度改革来\n\n"
        "续写文本：\n提高财政收入和资源调配效率，\n\n"
    )


def build_prompt(
    prefix: str,
    suffix: str,
    context: CompletionContext,
    trigger_mode: str | None,
    attempt: int,
) -> str:
    profile = get_profile(trigger_mode)
    prefix_tail = prefix[-profile.prompt_prefix_window:]
    suffix_head = suffix[: profile.prompt_suffix_window]

    retry_hint = ""
    if attempt > 0:
        retry_hint = (
            "上一次结果为空、重复、结构不对或过于空泛。"
            "这一次请务必给出具体、贴合上下文的补文，不要输出套话，不要生成新的 Markdown 结构。\n"
        )

    examples = build_examples(context)
    context_instruction = build_context_instruction(context)

    if suffix_head:
        user_prompt = (
            f"{retry_hint}"
            f"{examples}"
            f"前文：\n{prefix_tail}\n\n"
            f"后文：\n{suffix_head}\n\n"
            f"要求：{context_instruction} 优先延续文中的关键词、人物、概念和语气；"
            "不要重复前后文；不要写套话，如“在当今社会”“具有重要意义”。\n\n"
            "补全文本：\n"
        )
    else:
        user_prompt = (
            f"{retry_hint}"
            f"{examples}"
            f"前文：\n{prefix_tail}\n\n"
            f"要求：{context_instruction} 续写要自然、具体，直接接着前文往下写；"
            "不要重复前文；不要用“今天”“在当今社会”这种空泛开头。\n\n"
            "续写文本：\n"
        )

    return (
        "<|im_start|>system\n"
        f"{profile.system_prompt}\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{user_prompt}"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


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


def is_suffix_echo(text: str, suffix: str) -> bool:
    candidate = text.strip()
    safe_suffix = suffix.lstrip()
    if not candidate or not safe_suffix:
        return False
    min_echo_length = 4 if contains_cjk(candidate) else 6
    return len(candidate) >= min_echo_length and safe_suffix.startswith(candidate)


def build_stop_sequences(suffix: str, trigger_mode: str | None, attempt: int) -> list[str]:
    stops = ["<|endoftext|>", "<|file_sep|>", "<|im_end|>", "\n"]
    safe_suffix = suffix.lstrip()
    if not safe_suffix or attempt > 0:
        return stops

    mode = normalize_trigger_mode(trigger_mode)
    if contains_cjk(safe_suffix):
        dynamic_stop = safe_suffix[:14] if mode == "manual" else safe_suffix[:8]
        min_stop_length = 6 if mode == "manual" else 4
    else:
        dynamic_stop = safe_suffix[:24] if mode == "manual" else safe_suffix[:14]
        min_stop_length = 10 if mode == "manual" else 6

    if len(dynamic_stop) >= min_stop_length:
        stops.append(dynamic_stop)
    return stops


def normalize_compare_text(text: str) -> str:
    return re.sub(r"\s+", "", text)


def looks_like_generic_lead_in(text: str) -> bool:
    candidate = text.strip()
    return any(pattern.match(candidate) for pattern in GENERIC_LEAD_IN_PATTERNS)


def is_recent_line_echo(text: str, recent_lines: tuple[str, ...]) -> bool:
    candidate = normalize_compare_text(text.strip(" ，,。；;"))
    if len(candidate) < 8:
        return False
    for line in recent_lines:
        normalized_line = normalize_compare_text(line)
        if not normalized_line:
            continue
        if candidate == normalized_line or candidate in normalized_line or normalized_line in candidate:
            return True
    return False


def trim_completion(text: str, trigger_mode: str | None) -> str:
    profile = get_profile(trigger_mode)
    candidate = text.splitlines()[0].strip()
    if len(candidate) > profile.max_chars:
        candidate = candidate[: profile.max_chars].rstrip(" ，,。；;")
    return candidate


def violates_context_constraints(
    text: str,
    prefix: str,
    suffix: str,
    context: CompletionContext,
) -> bool:
    candidate = text.strip()
    if not candidate:
        return True
    if any(pattern in candidate for pattern in GENERIC_COMPLETION_PATTERNS):
        return True
    if looks_like_generic_lead_in(candidate):
        return True
    if is_recent_line_echo(candidate, context.recent_lines):
        return True
    if prefix.rstrip().endswith(candidate):
        return True
    if is_suffix_echo(candidate, suffix):
        return True
    if not candidate.strip("，。；：,:!?！？、（）()《》“”‘’【】[]-— "):
        return True
    if context.block_kind in {"paragraph", "plain_text"} and MARKDOWN_BLOCK_START_RE.match(candidate):
        return True
    if context.block_kind == "list_item_body" and MARKDOWN_LIST_ITEM_RE.match(candidate):
        return True
    if context.block_kind == "heading_text" and (candidate.startswith("#") or MARKDOWN_BLOCK_START_RE.match(candidate)):
        return True
    if context.block_kind == "quote_body" and candidate.startswith(">"):
        return True
    if context.prefer_sentence_continuation and candidate.endswith(("：", ":")):
        return True
    return False


def post_process_completion(
    text: str,
    prefix: str,
    suffix: str,
    context: CompletionContext,
    trigger_mode: str | None,
) -> str:
    candidate = (text or "").strip()
    if not candidate:
        return ""
    candidate = remove_prefix_overlap(candidate, prefix).lstrip()
    candidate = remove_suffix_overlap(candidate, suffix).rstrip()
    if is_suffix_echo(candidate, suffix):
        return ""
    candidate = trim_completion(candidate, trigger_mode)
    candidate = remove_suffix_overlap(candidate, suffix).rstrip()
    if is_suffix_echo(candidate, suffix):
        return ""
    if violates_context_constraints(candidate, prefix, suffix, context):
        return ""
    return candidate


def is_low_quality_completion(
    text: str,
    prefix: str,
    suffix: str,
    context: CompletionContext,
    trigger_mode: str | None,
) -> bool:
    candidate = text.strip(" ，,。；;")
    if not candidate:
        return True
    if violates_context_constraints(candidate, prefix, suffix, context):
        return True
    mode = normalize_trigger_mode(trigger_mode)
    return len(candidate) < (4 if mode == "manual" else 2)
