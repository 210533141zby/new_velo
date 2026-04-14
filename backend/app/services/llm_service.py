"""
=============================================================================
文件: llm_service.py
描述: LLM 底层服务调用 (重构版 - 针对中文幽灵文本优化)

核心功能：
1. 幽灵文本补全：封装对本机 OpenAI 兼容模型服务的调用，专为中文编辑器体验优化。
2. Prompt 构建：使用指令格式组合前后文。
3. 上下文裁剪：精细控制上下文长度，降低延迟。
=============================================================================
"""

import logging
import re
from typing import Literal

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

HTTP_TIMEOUT_SECONDS = 8.5
_http_client: httpx.AsyncClient | None = None
CompletionTriggerMode = Literal["auto", "manual"]
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
COMPLETION_PROFILES = {
    "auto": {
        "prefix_window": 640,
        "suffix_window": 128,
        "prompt_prefix_window": 360,
        "prompt_suffix_window": 96,
        "max_tokens": 24,
        "timeout": 7.0,
        "max_chars": 28,
        "split_pattern": r'[\n。！？!?]',
        "system_prompt": "你是中文写作补全器。请根据前文自然续写一小段具体、贴合语境的文本。只输出要插入光标处的内容，不要解释，不要重复上下文，避免空泛套话。",
    },
    "manual": {
        "prefix_window": 1200,
        "suffix_window": 240,
        "prompt_prefix_window": 720,
        "prompt_suffix_window": 160,
        "max_tokens": 40,
        "timeout": 8.5,
        "max_chars": 80,
        "split_pattern": r'[\n]',
        "system_prompt": "你是中文写作补全器。任务是在前文和后文之间补出自然、具体、衔接紧密的文本。只输出要插入光标处的内容，不要解释，不要重复上下文，避免空泛套话。",
    },
}


def get_completion_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(
            timeout=HTTP_TIMEOUT_SECONDS,
            limits=httpx.Limits(max_keepalive_connections=4, max_connections=10),
        )
    return _http_client


async def close_completion_http_client() -> None:
    global _http_client
    if _http_client is not None:
        await _http_client.aclose()
        _http_client = None


def _normalize_trigger_mode(trigger_mode: str | None) -> CompletionTriggerMode:
    return "auto" if trigger_mode == "auto" else "manual"


def _get_profile(trigger_mode: str | None) -> dict:
    return COMPLETION_PROFILES[_normalize_trigger_mode(trigger_mode)]


def _truncate_context(prefix: str, suffix: str, trigger_mode: str | None) -> tuple[str, str]:
    """上下文精细裁剪。"""
    profile = _get_profile(trigger_mode)
    return prefix[-profile["prefix_window"]:], suffix[:profile["suffix_window"]]


def _contains_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def _remove_prefix_overlap(text: str, prefix: str) -> str:
    if not text or not prefix:
        return text

    max_overlap = min(len(text), len(prefix))
    for i in range(max_overlap, 0, -1):
        if text[:i] == prefix[-i:]:
            return text[i:]
    return text


def _remove_suffix_overlap(text: str, suffix: str) -> str:
    if not text or not suffix:
        return text

    safe_suffix = suffix.lstrip()
    if not safe_suffix:
        return text

    max_overlap = min(len(text), len(safe_suffix))
    for i in range(max_overlap, 0, -1):
        if text[-i:] == safe_suffix[:i]:
            return text[:-i]
    return text


def _is_suffix_echo(text: str, suffix: str) -> bool:
    candidate = text.strip()
    safe_suffix = suffix.lstrip()
    if not candidate or not safe_suffix:
        return False

    min_echo_length = 4 if _contains_cjk(candidate) else 6
    return len(candidate) >= min_echo_length and safe_suffix.startswith(candidate)


def _build_stop_sequences(suffix: str, trigger_mode: str | None, attempt: int) -> list[str]:
    stops = ["<|endoftext|>", "<|file_sep|>", "<|im_end|>"]

    safe_suffix = suffix.lstrip()
    if not safe_suffix or attempt > 0:
        return stops

    mode = _normalize_trigger_mode(trigger_mode)
    if _contains_cjk(safe_suffix):
        dynamic_stop = safe_suffix[:14] if mode == "manual" else safe_suffix[:8]
        min_stop_length = 6 if mode == "manual" else 4
    else:
        dynamic_stop = safe_suffix[:24] if mode == "manual" else safe_suffix[:14]
        min_stop_length = 10 if mode == "manual" else 6

    if len(dynamic_stop) >= min_stop_length:
        stops.append(dynamic_stop)

    return stops


def _build_prompt(prefix: str, suffix: str, trigger_mode: str | None, attempt: int) -> str:
    mode = _normalize_trigger_mode(trigger_mode)
    profile = _get_profile(mode)
    prefix_tail = prefix[-profile["prompt_prefix_window"]:]
    suffix_head = suffix[:profile["prompt_suffix_window"]]

    retry_hint = ""
    if attempt > 0:
        retry_hint = "上一次结果为空、重复或过于空泛。这一次请务必给出具体、贴合上下文的补文，不要输出套话。\n"

    examples = ""
    if mode == "manual":
        examples = (
            "示例：\n"
            "前文：\n北宋时期，王安石推行新法的核心目标，是希望通过制度改革来\n\n"
            "后文：\n缓解财政困境，并增强国家治理能力。\n\n"
            "补全文本：\n提高财政收入和资源调配效率，\n\n"
        )
    else:
        examples = (
            "示例：\n"
            "前文：\n今天我们讨论人工智能在教育中的真实应用\n\n"
            "续写文本：\n，重点看备课、批改与个性化辅导。\n\n"
        )

    if suffix_head:
        user_prompt = (
            f"{retry_hint}"
            f"{examples}"
            f"前文：\n{prefix_tail}\n\n"
            f"后文：\n{suffix_head}\n\n"
            "要求：补文要紧贴上下文，优先延续文中的关键词、人物、概念和语气；不要重复前后文；不要写套话，如“在当今社会”“具有重要意义”。\n\n"
            "补全文本：\n"
        )
    else:
        user_prompt = (
            f"{retry_hint}"
            f"{examples}"
            f"前文：\n{prefix_tail}\n\n"
            "要求：续写要自然、具体，直接接着前文往下写；不要重复前文；不要用“今天”“在当今社会”这种空泛开头。\n\n"
            "续写文本：\n"
        )

    return (
        "<|im_start|>system\n"
        f"{profile['system_prompt']}\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{user_prompt}"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _trim_completion(text: str, trigger_mode: str | None) -> str:
    mode = _normalize_trigger_mode(trigger_mode)
    profile = _get_profile(mode)
    text = re.split(profile["split_pattern"], text)[0].strip()
    if len(text) > profile["max_chars"]:
        text = text[:profile["max_chars"]].rstrip(" ，,。；;")
    return text


def _is_low_quality_completion(text: str, prefix: str, suffix: str, trigger_mode: str | None) -> bool:
    mode = _normalize_trigger_mode(trigger_mode)
    candidate = text.strip(" ，,。；;")
    if not candidate:
        return True

    if any(pattern in candidate for pattern in GENERIC_COMPLETION_PATTERNS):
        return True

    if prefix.rstrip().endswith(candidate):
        return True

    if _is_suffix_echo(candidate, suffix):
        return True

    if mode == "manual":
        return len(candidate) < 4

    return len(candidate) < 2


def _post_process_completion(text: str, prefix: str, suffix: str, trigger_mode: str | None) -> str:
    text = (text or "").strip()
    if not text:
        return ""

    text = _remove_prefix_overlap(text, prefix).lstrip()
    text = _remove_suffix_overlap(text, suffix).rstrip()

    if _is_suffix_echo(text, suffix):
        return ""

    text = _trim_completion(text, trigger_mode)
    text = _remove_suffix_overlap(text, suffix).rstrip()

    if _is_suffix_echo(text, suffix):
        return ""

    return text


async def _request_completion(prefix: str, suffix: str, trigger_mode: str | None, attempt: int) -> str:
    mode = _normalize_trigger_mode(trigger_mode)
    profile = _get_profile(mode)
    payload = {
        "model": settings.COMPLETION_MODEL,
        "prompt": _build_prompt(prefix, suffix, mode, attempt),
        "max_tokens": profile["max_tokens"],
        "temperature": 0.15 if mode == "manual" else 0.1,
        "repetition_penalty": 1.05,
        "stop": _build_stop_sequences(suffix, mode, attempt),
        "stream": False,
    }

    headers = {}
    if settings.OPENAI_API_KEY:
        headers["Authorization"] = f"Bearer {settings.OPENAI_API_KEY}"

    url = f"{settings.completion_api_base.rstrip('/')}/completions"
    client = get_completion_http_client()
    response = await client.post(url, json=payload, headers=headers, timeout=profile["timeout"])
    response.raise_for_status()
    result = response.json()

    if "choices" in result and result["choices"]:
        choice = result["choices"][0]
        return choice.get("text") or choice.get("message", {}).get("content", "")
    return result.get("content", "")


async def complete_text(prefix: str, suffix: str, trigger_mode: str | None = None) -> str:
    mode = _normalize_trigger_mode(trigger_mode)
    logger.info(
        f"[GhostBackend] 收到请求 - mode={mode}, Prefix长度: {len(prefix)}, Suffix长度: {len(suffix)}"
    )

    safe_prefix, safe_suffix = _truncate_context(prefix, suffix, mode)
    best_candidate = ""

    for attempt in range(2):
        try:
            generated_text = await _request_completion(safe_prefix, safe_suffix, mode, attempt)
            candidate = _post_process_completion(generated_text, safe_prefix, safe_suffix, mode)
            if candidate and not best_candidate:
                best_candidate = candidate

            if candidate and not _is_low_quality_completion(candidate, safe_prefix, safe_suffix, mode):
                return candidate
        except httpx.RequestError as e:
            logger.error(f"LLM 请求失败: {e}")
            if attempt == 0:
                continue
            return best_candidate
        except Exception as e:
            logger.exception(f"[GhostBackend] 代码补全服务异常: {e}")
            if attempt == 0:
                continue
            return best_candidate

    return best_candidate
