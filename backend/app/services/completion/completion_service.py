"""
=============================================================================
文件: completion_service.py
描述: 补全模型调用编排层

架构职责：
1. 只负责模型请求、有限次重试与结果返回。
2. 不承担补全策略、上下文分类和硬校验，这些职责统一下沉到 completion_policy.py。
=============================================================================
"""

import logging

import httpx

from app.core.config import settings
from app.services.completion.completion_policy import (
    build_prompt,
    build_stop_sequences,
    get_profile,
    infer_completion_context,
    is_low_quality_completion,
    normalize_trigger_mode,
    post_process_completion,
    truncate_context,
)

logger = logging.getLogger(__name__)

HTTP_TIMEOUT_SECONDS = 8.5
_http_client: httpx.AsyncClient | None = None


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


async def _request_completion(
    prefix: str,
    suffix: str,
    language: str | None,
    trigger_mode: str | None,
    attempt: int,
) -> str:
    context = infer_completion_context(prefix, language)
    profile = get_profile(trigger_mode)
    payload = {
        'model': settings.COMPLETION_MODEL,
        'prompt': build_prompt(prefix, suffix, context, trigger_mode, attempt),
        'max_tokens': profile.max_tokens,
        'temperature': profile.temperature,
        'repetition_penalty': 1.05,
        'stop': build_stop_sequences(suffix, trigger_mode, attempt),
        'stream': False,
    }

    headers = {}
    if settings.OPENAI_API_KEY:
        headers['Authorization'] = f'Bearer {settings.OPENAI_API_KEY}'

    url = f"{settings.completion_api_base.rstrip('/')}/completions"
    client = get_completion_http_client()
    response = await client.post(url, json=payload, headers=headers, timeout=profile.timeout)
    response.raise_for_status()
    result = response.json()

    if 'choices' in result and result['choices']:
        choice = result['choices'][0]
        return choice.get('text') or choice.get('message', {}).get('content', '')
    return result.get('content', '')


async def complete_text(
    prefix: str,
    suffix: str,
    language: str | None = None,
    trigger_mode: str | None = None,
) -> str:
    mode = normalize_trigger_mode(trigger_mode)
    safe_prefix, safe_suffix = truncate_context(prefix, suffix, mode)
    context = infer_completion_context(safe_prefix, language)

    logger.info(
        '[GhostBackend] 收到请求 - mode=%s, language=%s, block_kind=%s, prefix_len=%s, suffix_len=%s',
        mode,
        language,
        context.block_kind,
        len(prefix),
        len(suffix),
    )

    best_candidate = ''
    for attempt in range(2):
        try:
            generated_text = await _request_completion(safe_prefix, safe_suffix, language, mode, attempt)
            candidate = post_process_completion(generated_text, safe_prefix, safe_suffix, context, mode)
            if candidate and not best_candidate:
                best_candidate = candidate
            if candidate and not is_low_quality_completion(candidate, safe_prefix, safe_suffix, context, mode):
                return candidate
        except httpx.RequestError as exc:
            logger.error('LLM 请求失败: %s', exc)
            if attempt == 0:
                continue
            return best_candidate
        except Exception as exc:
            logger.exception('[GhostBackend] 代码补全服务异常: %s', exc)
            if attempt == 0:
                continue
            return best_candidate
    return best_candidate
