from __future__ import annotations

import json
import re
from typing import Any

from app.core.config import settings
from app.logger import logger
from app.services.model_factory import get_rag_judge_model
from app.services.rag.prompt_templates import build_document_judge_prompt
from app.services.rag.text_utils import compact_text

JUDGE_EVIDENCE_LIMIT = 160
JUDGE_ANSWER_LIMIT = 120


def _strip_json_fence(value: str) -> str:
    stripped = str(value or '').strip()
    if stripped.startswith('```'):
        stripped = re.sub(r'^```(?:json)?\s*', '', stripped)
        stripped = re.sub(r'\s*```$', '', stripped)
    return stripped.strip()


def _extract_json_object(value: str) -> dict[str, Any]:
    stripped = _strip_json_fence(value)
    candidates = [stripped]
    start = stripped.find('{')
    end = stripped.rfind('}')
    if start != -1 and end != -1 and end > start:
        candidates.append(stripped[start : end + 1])

    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def _normalize_yes_no_flag(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value

    normalized = re.sub(r'[^0-9a-z\u4e00-\u9fff]+', '', str(value or '').strip().lower())
    if normalized in {'yes', 'true', '1', '是', '能', '可以', '可回答', '通过', 'accept'}:
        return True
    if normalized in {'no', 'false', '0', '否', '不能', '不可以', '不可回答', '拒绝', 'reject'}:
        return False
    return None


def _normalize_document_judge_result(payload: dict[str, Any]) -> dict[str, Any]:
    core_topic_match = _normalize_yes_no_flag(payload.get('core_topic_match') or payload.get('topic_match'))
    contains_direct_evidence = _normalize_yes_no_flag(
        payload.get('contains_direct_evidence') or payload.get('direct_evidence') or payload.get('supported')
    )
    answerable = _normalize_yes_no_flag(payload.get('answerable') or payload.get('can_answer'))
    if answerable is None:
        answerable = bool(core_topic_match) and bool(contains_direct_evidence)
    elif core_topic_match is False or contains_direct_evidence is False:
        answerable = False

    evidence_quote = compact_text(str(payload.get('evidence_quote') or payload.get('evidence') or ''), JUDGE_EVIDENCE_LIMIT)
    answer_brief = compact_text(str(payload.get('answer_brief') or payload.get('answer') or ''), JUDGE_ANSWER_LIMIT)
    reason = compact_text(str(payload.get('reason') or payload.get('diagnosis') or ''), 24)

    if not answerable:
        evidence_quote = ''
        answer_brief = ''

    return {
        'core_topic_match': bool(core_topic_match) if core_topic_match is not None else False,
        'contains_direct_evidence': bool(contains_direct_evidence) if contains_direct_evidence is not None else False,
        'answerable': bool(answerable),
        'evidence_quote': evidence_quote,
        'answer_brief': answer_brief,
        'reason': reason,
    }


async def judge_rag_document(query: str, title: str, content: str) -> dict[str, Any]:
    if not settings.RAG_JUDGE_ENABLED:
        return {
            'core_topic_match': True,
            'contains_direct_evidence': True,
            'answerable': True,
            'evidence_quote': '',
            'answer_brief': '',
            'reason': '',
        }

    payload: dict[str, Any] = {}
    compact_content = compact_text(content, settings.RAG_JUDGE_CONTEXT_CHARS)
    try:
        response = await get_rag_judge_model().ainvoke(build_document_judge_prompt(query, title, compact_content))
        payload = _extract_json_object(getattr(response, 'content', ''))
    except Exception:
        logger.exception(
            'RAG 文档判别失败，回退到规则防御',
            extra={'extra_data': {'event': 'rag_document_judge_failed', 'query': query, 'title': title}},
        )
        return {
            'core_topic_match': True,
            'contains_direct_evidence': True,
            'answerable': True,
            'evidence_quote': '',
            'answer_brief': '',
            'reason': 'judge_fallback',
        }

    return _normalize_document_judge_result(payload)
