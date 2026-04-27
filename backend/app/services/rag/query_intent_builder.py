from __future__ import annotations

import re

from app.services.rag.hybrid_search import has_identifier, tokenize_for_bm25
from app.services.rag.pipeline_models import (
    DefenseProfile,
    EvidenceRequirement,
    QueryIntent,
    QueryIntentType,
)
from app.services.rag.rerank_service import normalize_lookup_text

POLITE_PREFIXES = (
    '请问一下',
    '请问',
    '我想知道',
    '我想了解',
    '麻烦问下',
    '麻烦问一下',
    '帮我看看',
    '帮我看下',
    '帮我查下',
    '帮我',
)
QUERY_STOP_TOKENS = {
    '请',
    '请问',
    '一下',
    '我想',
    '我想知道',
    '我想了解',
    '帮我',
    '麻烦',
    '什么',
    '怎么',
    '如何',
    '吗',
    '呢',
    '呀',
    '啊',
    '的',
    '了',
    '是',
    '为',
    '根据',
    '知识库',
    '判断',
    '下面',
    '只',
    '输出',
    '文本',
    '这段',
    '新闻',
    '开头',
    '续写',
    '错误',
    '纠正',
    '修正',
    '改正',
    '更正',
    '和',
    '与',
    '为什么',
    '为何',
    '原因',
    '为啥',
    '因何',
}
SUMMARY_MARKERS = ('讲了什么', '写了什么', '主要内容', '内容是什么', '总结', '概括')
OVERVIEW_MARKERS = ('历史', '概况', '背景', '介绍', '沿革', '发展', '情况')
REASON_MARKERS = ('为什么', '为何', '原因', '为啥', '因何')
LOCATION_MARKERS = ('在哪', '哪里', '地址', '位置', '位于', '坐落')
RELATION_MARKERS = ('关系', '联系', '区别', '对比', '比较', '共同点')
RELATION_LINKERS = ('与', '和', '跟', '及', '同')
MULTI_INFO_MARKERS = ('哪些', '哪些方面', '哪些领域', '有哪些', '包括', '包含', '以及', '同时', '并且', '分别')
MULTI_INFO_SLOT_MARKERS = ('哪些', '多少', '几个', '哪位', '是谁', '什么', '何时', '什么时候', '哪年', '哪一年', '立场')
LOCATION_FALSE_POSITIVE_MARKERS = (
    '在哪些方面',
    '在哪些领域',
    '在哪些环节',
    '在哪些层面',
    '体现在哪些',
    '邮箱地址',
    '电子邮箱地址',
    '邮件地址',
    '网址地址',
)
EXACT_FACT_MARKERS = ('什么时候', '何时', '哪年', '哪一年', '几年', '日期', '时间')
CORRECTION_MARKERS = ('纠正', '修正', '改正', '更正')
CORRECTION_QUERY_MARKERS = ('续写', '错误', '原文', '新闻开头', '只输出纠正后的文本', '只输出修正后的文本', '只输出更正后的文本')
CORRECTION_STOP_TOKENS = {
    '新华社',
    '日电',
    '记者',
    '新闻',
    '开头',
    '续写',
    '错误',
    '纠正',
    '修正',
    '改正',
    '更正',
    '文本',
    '输出',
}
QUESTION_MARKERS = SUMMARY_MARKERS + OVERVIEW_MARKERS + REASON_MARKERS + LOCATION_MARKERS + RELATION_MARKERS
TRAILING_NOISE = '，,。.!！？?：:；; '
TRAILING_PARTICLES = ('吗', '呢', '呀', '啊', '嘛')
DEFAULT_RETRIEVAL_DEPTH = {
    QueryIntentType.LOOKUP: 6,
    QueryIntentType.FACTOID: 8,
    QueryIntentType.LOCATION: 8,
    QueryIntentType.REASON: 10,
    QueryIntentType.RELATION: 12,
    QueryIntentType.SUMMARY: 12,
    QueryIntentType.OVERVIEW: 14,
}


def _collapse_spaces(value: str) -> str:
    return ' '.join(str(value or '').split()).strip()


def _strip_polite_prefix(query: str) -> tuple[str, bool]:
    stripped = query
    removed = False
    while stripped:
        matched_prefix = next((prefix for prefix in POLITE_PREFIXES if stripped.startswith(prefix)), '')
        if not matched_prefix:
            break
        stripped = stripped[len(matched_prefix) :].lstrip()
        removed = True
    return stripped, removed


def _strip_trailing_noise(query: str) -> str:
    stripped = query.strip(TRAILING_NOISE)
    removed = True
    while stripped and removed:
        removed = False
        for particle in TRAILING_PARTICLES:
            if stripped.endswith(particle):
                stripped = stripped[: -len(particle)].rstrip(TRAILING_NOISE)
                removed = True
    return stripped.strip(TRAILING_NOISE)


def normalize_query(query: str) -> tuple[str, tuple[str, ...]]:
    original = _collapse_spaces(query)
    stripped_prefix, prefix_removed = _strip_polite_prefix(original)
    normalized = _strip_trailing_noise(stripped_prefix)

    trace_tags: list[str] = []
    if prefix_removed:
        trace_tags.append('polite_prefix_removed')
    if normalized != original:
        trace_tags.append('query_normalized')
    return normalized or original, tuple(trace_tags)


def _ranked_query_terms(text: str, *, stop_tokens: set[str], limit: int) -> list[str]:
    tokens: list[str] = []
    seen: set[str] = set()
    ranked_tokens: list[tuple[int, int, str]] = []
    for index, token in enumerate(tokenize_for_bm25(text)):
        normalized = normalize_lookup_text(token)
        if (
            not normalized
            or normalized in stop_tokens
            or normalized in seen
            or (len(normalized) <= 1 and not has_identifier(token))
        ):
            continue
        seen.add(normalized)
        priority = 2
        if has_identifier(token):
            priority = 0
        elif len(normalized) >= 3:
            priority = 1
        ranked_tokens.append((priority, index, str(token)))

    ranked_tokens.sort(key=lambda item: (item[0], item[1]))
    for _priority, _index, token in ranked_tokens[:limit]:
        tokens.append(token)
    return tokens


def build_keyword_query(query: str) -> str:
    terms = _ranked_query_terms(query, stop_tokens=QUERY_STOP_TOKENS, limit=8)
    return ' '.join(terms).strip() or query


def _extract_correction_anchor(query: str) -> str:
    patterns = (
        r'新闻开头[:：]\s*(.*?)(?:\s*续写[:：]|$)',
        r'原文[:：]\s*(.*?)(?:\s*待纠正[:：]|\s*续写[:：]|\s*错误[:：]|$)',
    )
    for pattern in patterns:
        match = re.search(pattern, query, flags=re.S)
        if match:
            return _collapse_spaces(match.group(1))
    return ''


def _looks_like_correction_query(query: str, normalized_query: str) -> bool:
    return any(marker in normalized_query for marker in CORRECTION_MARKERS) and any(
        marker in query or marker in normalized_query
        for marker in CORRECTION_QUERY_MARKERS
    )


def _build_correction_keyword_query(query: str) -> str:
    anchor = _extract_correction_anchor(query) or query
    correction_terms = _ranked_query_terms(
        anchor,
        stop_tokens=QUERY_STOP_TOKENS | CORRECTION_STOP_TOKENS,
        limit=12,
    )
    if correction_terms:
        non_numeric_terms = [token for token in correction_terms if not normalize_lookup_text(token).isdigit()]
        numeric_terms = [token for token in correction_terms if normalize_lookup_text(token).isdigit()]
        correction_terms = (non_numeric_terms[:10] + numeric_terms[:2])[:12]
    if correction_terms:
        return ' '.join(correction_terms)
    return build_keyword_query(anchor)


def _contains_any_marker(normalized_query: str, markers: tuple[str, ...]) -> bool:
    return any(marker in normalized_query for marker in markers)


def _looks_like_relation_query(query: str, normalized_query: str) -> bool:
    if '共同点' in normalized_query and any(linker in query for linker in RELATION_LINKERS):
        return True
    if not _contains_any_marker(normalized_query, RELATION_MARKERS):
        return False
    return any(linker in query for linker in RELATION_LINKERS)


def _looks_like_multi_info_query(query: str, normalized_query: str) -> bool:
    if any(marker in normalized_query for marker in MULTI_INFO_MARKERS):
        return True
    slot_count = sum(1 for marker in MULTI_INFO_SLOT_MARKERS if marker in normalized_query)
    has_clause_separator = any(marker in query for marker in ('，', ',', '；', ';', '以及', '同时', '并且', '而'))
    return has_clause_separator and slot_count >= 2


def _looks_like_location_query(normalized_query: str) -> bool:
    if any(marker in normalized_query for marker in LOCATION_FALSE_POSITIVE_MARKERS):
        return False
    if '邮箱' in normalized_query and '地址' in normalized_query:
        return False
    if '网址' in normalized_query and '地址' in normalized_query:
        return False
    return _contains_any_marker(normalized_query, LOCATION_MARKERS)


def _looks_like_lookup_query(query: str, normalized_query: str) -> bool:
    if not normalized_query or len(normalized_query) > 4:
        return False
    if any(symbol in query for symbol in ('?', '？')):
        return False
    return not _contains_any_marker(normalized_query, QUESTION_MARKERS)


def _has_meaningful_side(value: str) -> bool:
    for token in tokenize_for_bm25(value):
        normalized = normalize_lookup_text(token)
        if not normalized or normalized in QUERY_STOP_TOKENS:
            continue
        if len(normalized) > 1 or has_identifier(token):
            return True
    return False


def _looks_like_attribute_query(query: str, intent_type: QueryIntentType) -> bool:
    if intent_type in {
        QueryIntentType.RELATION,
        QueryIntentType.REASON,
        QueryIntentType.SUMMARY,
        QueryIntentType.OVERVIEW,
    }:
        return False
    if '的' not in query:
        return False
    subject, attribute = query.split('的', 1)
    return _has_meaningful_side(subject) and _has_meaningful_side(attribute)


def infer_intent_type(query: str) -> tuple[QueryIntentType, tuple[str, ...]]:
    normalized_query = normalize_lookup_text(query)
    trace_tags: list[str] = []

    if _looks_like_relation_query(query, normalized_query):
        trace_tags.append('intent_relation')
        return QueryIntentType.RELATION, tuple(trace_tags)
    if _contains_any_marker(normalized_query, SUMMARY_MARKERS):
        trace_tags.append('intent_summary')
        return QueryIntentType.SUMMARY, tuple(trace_tags)
    if _contains_any_marker(normalized_query, OVERVIEW_MARKERS):
        trace_tags.append('intent_overview')
        return QueryIntentType.OVERVIEW, tuple(trace_tags)
    if _contains_any_marker(normalized_query, REASON_MARKERS):
        trace_tags.append('intent_reason')
        return QueryIntentType.REASON, tuple(trace_tags)
    if _looks_like_location_query(normalized_query):
        trace_tags.append('intent_location')
        return QueryIntentType.LOCATION, tuple(trace_tags)
    if _looks_like_lookup_query(query, normalized_query):
        trace_tags.append('intent_lookup')
        return QueryIntentType.LOOKUP, tuple(trace_tags)

    trace_tags.append('intent_factoid')
    return QueryIntentType.FACTOID, tuple(trace_tags)


def build_retrieval_depth(query: str, intent_type: QueryIntentType) -> int:
    depth = DEFAULT_RETRIEVAL_DEPTH[intent_type]
    token_count = len(tokenize_for_bm25(query))
    if has_identifier(query):
        depth += 2
    if token_count >= 10:
        depth += 2
    return min(depth, 16)


def infer_defense_profile(
    intent_type: QueryIntentType,
    query: str,
    normalized_query: str,
    *,
    attribute_style: bool,
    correction_query: bool = False,
) -> DefenseProfile:
    if correction_query:
        return DefenseProfile.MODERATE
    if intent_type is QueryIntentType.RELATION and '共同点' in normalized_query:
        return DefenseProfile.LOOSE
    if intent_type is QueryIntentType.FACTOID and _looks_like_multi_info_query(query, normalized_query):
        return DefenseProfile.MODERATE
    if intent_type in {QueryIntentType.LOOKUP, QueryIntentType.RELATION} or attribute_style:
        return DefenseProfile.STRICT
    if intent_type in {QueryIntentType.SUMMARY, QueryIntentType.OVERVIEW}:
        return DefenseProfile.LOOSE
    return DefenseProfile.MODERATE


def infer_evidence_requirement(
    intent_type: QueryIntentType,
    query: str,
    normalized_query: str,
    *,
    correction_query: bool = False,
) -> EvidenceRequirement:
    if correction_query:
        return EvidenceRequirement.FULL_DOCUMENT
    if intent_type is QueryIntentType.FACTOID and _looks_like_multi_info_query(query, normalized_query):
        return EvidenceRequirement.MULTI_SPAN
    if intent_type in {QueryIntentType.LOOKUP, QueryIntentType.FACTOID, QueryIntentType.LOCATION}:
        return EvidenceRequirement.ATOMIC_SPAN
    if intent_type in {QueryIntentType.RELATION, QueryIntentType.REASON}:
        return EvidenceRequirement.MULTI_SPAN
    return EvidenceRequirement.FULL_DOCUMENT


def should_enable_judge(
    intent_type: QueryIntentType,
    normalized_query: str,
    *,
    attribute_style: bool,
    evidence_requirement: EvidenceRequirement,
    defense_profile: DefenseProfile,
) -> bool:
    if defense_profile is not DefenseProfile.STRICT:
        return False
    if evidence_requirement is EvidenceRequirement.MULTI_SPAN:
        return False
    if any(marker in normalized_query for marker in EXACT_FACT_MARKERS):
        return False
    return intent_type in {QueryIntentType.LOOKUP, QueryIntentType.LOCATION} or attribute_style


class QueryIntentBuilder:
    @classmethod
    async def build(cls, query: str) -> QueryIntent:
        original_query = _collapse_spaces(query)
        normalized_query, normalization_tags = normalize_query(original_query)
        normalized_lookup = normalize_lookup_text(normalized_query)
        correction_query = _looks_like_correction_query(original_query, normalized_lookup)
        keyword_query = _build_correction_keyword_query(original_query) if correction_query else build_keyword_query(normalized_query)
        intent_type, intent_tags = infer_intent_type(normalized_query)
        attribute_style = _looks_like_attribute_query(normalized_query, intent_type)
        retrieval_depth = build_retrieval_depth(normalized_query, intent_type)
        defense_profile = infer_defense_profile(
            intent_type,
            original_query,
            normalized_lookup,
            attribute_style=attribute_style,
            correction_query=correction_query,
        )
        evidence_requirement = infer_evidence_requirement(
            intent_type,
            original_query,
            normalized_lookup,
            correction_query=correction_query,
        )
        wants_short_answer = (intent_type not in {QueryIntentType.SUMMARY, QueryIntentType.OVERVIEW}) and not correction_query
        needs_judge = should_enable_judge(
            intent_type,
            normalized_lookup,
            attribute_style=attribute_style,
            evidence_requirement=evidence_requirement,
            defense_profile=defense_profile,
        )

        trace_tags: list[str] = list(normalization_tags) + list(intent_tags)
        if attribute_style:
            trace_tags.append('attribute_style_query')
        if _looks_like_multi_info_query(original_query, normalized_query):
            trace_tags.append('multi_info_query')
        if correction_query:
            trace_tags.append('correction_query')
        if has_identifier(normalized_query):
            trace_tags.append('has_identifier')
        if len(normalize_lookup_text(normalized_query)) <= 6:
            trace_tags.append('short_query')
        if keyword_query != normalized_query:
            trace_tags.append('keyword_query_compacted')

        ordered_trace_tags: list[str] = []
        seen_tags: set[str] = set()
        for tag in trace_tags:
            if not tag or tag in seen_tags:
                continue
            seen_tags.add(tag)
            ordered_trace_tags.append(tag)

        return QueryIntent(
            original_query=original_query,
            normalized_query=normalized_query,
            keyword_query=keyword_query,
            intent_type=intent_type,
            retrieval_depth=retrieval_depth,
            defense_profile=defense_profile,
            evidence_requirement=evidence_requirement,
            wants_short_answer=wants_short_answer,
            needs_judge=needs_judge,
            trace_tags=tuple(ordered_trace_tags),
        )
