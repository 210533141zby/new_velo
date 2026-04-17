from __future__ import annotations

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


def build_keyword_query(query: str) -> str:
    tokens: list[str] = []
    seen: set[str] = set()
    for token in tokenize_for_bm25(query):
        normalized = normalize_lookup_text(token)
        if (
            not normalized
            or normalized in QUERY_STOP_TOKENS
            or normalized in seen
            or (len(normalized) <= 1 and not has_identifier(token))
        ):
            continue
        seen.add(normalized)
        tokens.append(token)
    return ' '.join(tokens[:8]).strip() or query


def _contains_any_marker(normalized_query: str, markers: tuple[str, ...]) -> bool:
    return any(marker in normalized_query for marker in markers)


def _looks_like_relation_query(query: str, normalized_query: str) -> bool:
    if '共同点' in normalized_query and any(linker in query for linker in RELATION_LINKERS):
        return True
    if not _contains_any_marker(normalized_query, RELATION_MARKERS):
        return False
    return any(linker in query for linker in RELATION_LINKERS)


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
    if _contains_any_marker(normalized_query, LOCATION_MARKERS):
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


def infer_defense_profile(intent_type: QueryIntentType, normalized_query: str, *, attribute_style: bool) -> DefenseProfile:
    if intent_type is QueryIntentType.RELATION and '共同点' in normalized_query:
        return DefenseProfile.LOOSE
    if intent_type in {QueryIntentType.LOOKUP, QueryIntentType.RELATION} or attribute_style:
        return DefenseProfile.STRICT
    if intent_type in {QueryIntentType.SUMMARY, QueryIntentType.OVERVIEW}:
        return DefenseProfile.LOOSE
    return DefenseProfile.MODERATE


def infer_evidence_requirement(intent_type: QueryIntentType) -> EvidenceRequirement:
    if intent_type in {QueryIntentType.LOOKUP, QueryIntentType.FACTOID, QueryIntentType.LOCATION}:
        return EvidenceRequirement.ATOMIC_SPAN
    if intent_type in {QueryIntentType.RELATION, QueryIntentType.REASON}:
        return EvidenceRequirement.MULTI_SPAN
    return EvidenceRequirement.FULL_DOCUMENT


class QueryIntentBuilder:
    @classmethod
    async def build(cls, query: str) -> QueryIntent:
        original_query = _collapse_spaces(query)
        normalized_query, normalization_tags = normalize_query(original_query)
        keyword_query = build_keyword_query(normalized_query)
        intent_type, intent_tags = infer_intent_type(normalized_query)
        attribute_style = _looks_like_attribute_query(normalized_query, intent_type)
        retrieval_depth = build_retrieval_depth(normalized_query, intent_type)
        defense_profile = infer_defense_profile(intent_type, normalized_query, attribute_style=attribute_style)
        evidence_requirement = infer_evidence_requirement(intent_type)
        wants_short_answer = intent_type not in {QueryIntentType.SUMMARY, QueryIntentType.OVERVIEW}
        needs_judge = defense_profile is DefenseProfile.STRICT

        trace_tags: list[str] = list(normalization_tags) + list(intent_tags)
        if attribute_style:
            trace_tags.append('attribute_style_query')
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
