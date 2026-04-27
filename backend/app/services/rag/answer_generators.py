from __future__ import annotations

import re
from dataclasses import replace
from typing import Callable, Sequence

from app.services.rag.hybrid_search import has_identifier, tokenize_for_bm25
from app.services.rag.pipeline_models import (
    AnswerMode,
    AnswerPlan,
    EvidenceRequirement,
    EvidenceAssessment,
    QueryIntent,
    QueryIntentType,
)
from app.services.rag.prompt_templates import (
    build_general_rag_prompt,
    build_no_context_answer,
    build_structured_rag_prompt,
)
from app.services.rag.rerank_service import normalize_lookup_text
from app.services.rag.text_utils import compact_text, split_paragraphs, split_text_segments

QUESTION_STOP_TOKENS = {
    '请问',
    '请问一下',
    '一下',
    '什么',
    '多少',
    '几',
    '几岁',
    '多少岁',
    '哪',
    '哪年',
    '哪一年',
    '几年',
    '吗',
    '呢',
    '啊',
    '呀',
    '何时',
    '什么时候',
    '为何',
    '为什么',
    '是否',
    '是什么',
    '关系',
    '主要',
    '是',
    '问题',
}
REASON_EVIDENCE_MARKERS = ('因为', '原因', '之所以')
MULTI_INFO_MARKERS = ('哪些', '什么角色', '包括', '包含', '有哪些', '几个', '多少', '分别')
GENERIC_ANSWER_PREFIXES = ('核心内容如下：', '核心内容：', '回答如下：', '答案如下：')
GENERIC_META_MARKERS = ('来源：', '根据参考文档', '根据参考资料', '根据参考上下文')
SUMMARY_ADVICE_MARKERS = ('建议', '提醒', '请注意', '需注意', '做好', '避免')
CORRECTION_META_MARKERS = ('纠正说明', '更正说明', '修正说明', '无需纠正')
REFUSAL_SENTENCE_MARKERS = (
    '无法确定',
    '无法回答',
    '无法给出可靠回答',
    '没有找到足够相关',
    '资料不足',
    '没有相关资料',
    '未提及',
    '未涉及',
)
LOW_RELEVANCE_EXTENSIONS = ('此外', '另外', '同时', '并且', '而且', '再者', '还有')


class FallbackRequiredError(RuntimeError):
    pass


def _assessment_rank(assessment: EvidenceAssessment) -> tuple[float, float, float]:
    return (
        float(assessment.final_score),
        1.0 if assessment.direct_evidence else 0.0,
        1.0 if assessment.supports_extractive else 0.0,
    )


def _select_assessments(
    plan: AnswerPlan,
    assessments: Sequence[EvidenceAssessment],
) -> list[EvidenceAssessment]:
    ranked = sorted(assessments, key=_assessment_rank, reverse=True)
    if not ranked:
        return []
    if not plan.source_doc_ids:
        return ranked[:3]

    selected: list[EvidenceAssessment] = []
    selected_ids = set(plan.source_doc_ids)
    for assessment in ranked:
        if assessment.candidate.doc_id in selected_ids:
            selected.append(assessment)
    return selected or ranked[:3]


def build_sources(assessments: Sequence[EvidenceAssessment]) -> list[dict]:
    sources: list[dict] = []
    for index, assessment in enumerate(assessments, start=1):
        sources.append(
            {
                'title': assessment.candidate.title,
                'doc_id': assessment.candidate.doc_id,
                'rank': index,
                'confidence': int(round(float(assessment.final_score) * 100)),
            }
        )
    return sources


def _candidate_chunk_text(assessment: EvidenceAssessment) -> str:
    candidate = assessment.candidate
    page_content = str(getattr(candidate.doc, 'page_content', '') or '')
    if page_content.strip():
        return page_content
    if candidate.chunk_text.strip():
        return candidate.chunk_text
    return ''


def _candidate_full_content(assessment: EvidenceAssessment) -> str:
    candidate = assessment.candidate
    if candidate.full_content.strip():
        return candidate.full_content
    if getattr(candidate.doc, 'full_content', ''):
        return str(getattr(candidate.doc, 'full_content', '') or '')
    chunk_text = _candidate_chunk_text(assessment)
    return chunk_text


def _history_paragraph_score(query: str, paragraph: str, query_tokens: set[str]) -> float:
    score = _window_score(query, paragraph, query_tokens)
    if re.search(r'\d{4}年', paragraph):
        score += 0.10
    if any(marker in paragraph for marker in ('最早', '建于', '创立', '成立', '起步', '发展', '后来', '直到', '最初', '早期')):
        score += 0.10
    return score


def _build_context_excerpt(
    query: str,
    assessment: EvidenceAssessment,
    *,
    overview_mode: bool,
    relation_mode: bool,
    correction_mode: bool,
    use_full_content: bool,
) -> str:
    raw_content = _candidate_full_content(assessment) if use_full_content else _candidate_chunk_text(assessment)
    paragraphs = split_paragraphs(raw_content)
    if not paragraphs:
        return compact_text(raw_content, 2200)

    effective_query = _correction_anchor_text(query) if correction_mode else query
    query_tokens = _query_tokens(effective_query if overview_mode or correction_mode else query, '' if overview_mode else assessment.candidate.title)
    if not query_tokens:
        query_tokens = _query_tokens(effective_query)

    selected_indexes: list[int] = [0] if overview_mode else []
    ranked_paragraphs = sorted(
        range(len(paragraphs)),
        key=lambda index: (
            _history_paragraph_score(effective_query, paragraphs[index], query_tokens)
            if overview_mode
            else _window_score(effective_query, paragraphs[index], query_tokens),
            len(paragraphs[index]),
        ),
        reverse=True,
    )
    target_count = 5 if overview_mode else 4 if relation_mode or correction_mode else 2
    for index in ranked_paragraphs:
        if len(selected_indexes) >= target_count:
            break
        if index not in selected_indexes:
            selected_indexes.append(index)

    excerpt = '\n'.join(compact_text(paragraphs[index], 700) for index in sorted(selected_indexes))
    return compact_text(excerpt, 2200)


def _context_blocks(
    query: str,
    intent: QueryIntent,
    assessments: Sequence[EvidenceAssessment],
    *,
    use_full_content: bool,
) -> str:
    blocks: list[str] = []
    overview_mode = intent.intent_type.name.lower() in {'summary', 'overview'}
    relation_mode = intent.intent_type.name.lower() == 'relation'
    correction_mode = _is_correction_query(intent)
    for index, assessment in enumerate(assessments, start=1):
        content = _build_context_excerpt(
            query,
            assessment,
            overview_mode=overview_mode,
            relation_mode=relation_mode,
            correction_mode=correction_mode,
            use_full_content=use_full_content,
        )
        blocks.append(
            f'参考文档 {index}\n标题：{assessment.candidate.title}\n置信度：{int(round(assessment.final_score * 100))}%\n内容：\n{content}'
        )
    return '\n\n'.join(blocks)


def _query_tokens(query: str, title: str = '') -> set[str]:
    title_tokens = {
        normalize_lookup_text(token)
        for token in tokenize_for_bm25(title)
        if normalize_lookup_text(token)
    }
    raw_tokens = {
        normalize_lookup_text(token)
        for token in tokenize_for_bm25(query)
        if normalize_lookup_text(token)
    }
    trimmed_tokens = raw_tokens - title_tokens if raw_tokens - title_tokens else raw_tokens
    normalized_query = normalize_lookup_text(query)
    stop_tokens = set(QUESTION_STOP_TOKENS)
    if any(marker in normalized_query for marker in ('哪一年', '哪年', '几年', '何时', '什么时候')):
        stop_tokens.update({'年', '一年'})
    return {token for token in trimmed_tokens if token and token not in stop_tokens}


def _coverage_ratio(query_tokens: set[str], candidate_tokens: set[str]) -> float:
    if not query_tokens or not candidate_tokens:
        return 0.0
    return len(query_tokens & candidate_tokens) / len(query_tokens)


def _window_tokens(value: str) -> set[str]:
    return {
        normalize_lookup_text(token)
        for token in tokenize_for_bm25(value)
        if normalize_lookup_text(token)
    }


def _window_score(query: str, window: str, query_tokens: set[str]) -> float:
    window_tokens = _window_tokens(window)
    score = _coverage_ratio(query_tokens, window_tokens)
    if has_identifier(query):
        normalized_window = normalize_lookup_text(window)
        identifier_tokens = [normalize_lookup_text(token) for token in tokenize_for_bm25(query) if has_identifier(token)]
        if any(token and token in normalized_window for token in identifier_tokens):
            score += 0.12
    return score


def _matched_token_count(window: str, query_tokens: set[str]) -> int:
    return len(query_tokens & _window_tokens(window))


def _candidate_windows(content: str) -> list[str]:
    paragraphs = split_paragraphs(content)
    if len(paragraphs) > 1:
        return paragraphs

    sentences = split_text_segments(content)
    if len(sentences) <= 3:
        return [''.join(sentences)] if sentences else []

    windows: list[str] = []
    for index in range(len(sentences)):
        windows.append(''.join(sentences[index : index + 3]))
    return windows


def _clean_extractive_sentence(value: str) -> str:
    cleaned = str(value or '').strip()
    cleaned = re.sub(r'^\s*[-*•]+\s*', '', cleaned)
    cleaned = cleaned.replace('**', '')
    cleaned = cleaned.replace('\\', '')
    cleaned = re.split(r'\s[-*•]\s+', cleaned, maxsplit=1)[0]
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned.strip()


def _extractive_score(intent: QueryIntent, query: str, text: str, query_tokens: set[str]) -> tuple[float, int, int]:
    score = _window_score(query, text, query_tokens)
    if intent.intent_type.value == 'reason' and any(marker in text for marker in REASON_EVIDENCE_MARKERS):
        score += 0.8
    return (score, _matched_token_count(text, query_tokens), len(text))


def _is_multi_info_query(query: str, intent: QueryIntent) -> bool:
    normalized_query = normalize_lookup_text(query)
    return intent.intent_type in {QueryIntentType.RELATION, QueryIntentType.REASON} or (
        intent.intent_type is QueryIntentType.FACTOID
        and any(marker in normalized_query for marker in MULTI_INFO_MARKERS)
    )


def _is_correction_query(intent: QueryIntent) -> bool:
    return 'correction_query' in intent.trace_tags


def _extract_query_block(query: str, marker: str, end_markers: tuple[str, ...]) -> str:
    boundary = f"(?:{'|'.join(end_markers)}|$)" if end_markers else '$'
    pattern = rf'{re.escape(marker)}[:：]\s*(.*?){boundary}'
    match = re.search(pattern, query, flags=re.S)
    if not match:
        return ''
    return compact_text(match.group(1), 900)


def _correction_anchor_text(query: str) -> str:
    return _extract_query_block(query, '新闻开头', ('续写[:：]',)) or _extract_query_block(query, '原文', ('续写[:：]', '错误[:：]', '待纠正[:：]')) or query


def _correction_continuation_text(query: str) -> str:
    return _extract_query_block(query, '续写', tuple())


def _structured_context_limit(query: str, intent: QueryIntent) -> int:
    if intent.evidence_requirement is EvidenceRequirement.MULTI_SPAN:
        return 5
    if intent.intent_type is QueryIntentType.RELATION:
        return 4
    if intent.intent_type is QueryIntentType.REASON:
        return 3
    if _is_multi_info_query(query, intent):
        return 4
    return 2


def _structured_use_full_content(query: str, intent: QueryIntent) -> bool:
    return _is_multi_info_query(query, intent)


def _support_windows(assessments: Sequence[EvidenceAssessment], *, use_full_content: bool) -> list[str]:
    windows: list[str] = []
    for assessment in assessments:
        raw_content = _candidate_full_content(assessment) if use_full_content else _candidate_chunk_text(assessment)
        for paragraph in split_paragraphs(raw_content):
            compacted = compact_text(paragraph, 700)
            if compacted:
                windows.append(compacted)
            if len(windows) >= 24:
                return windows
    return windows


def _sentence_support_ratio(sentence: str, windows: Sequence[str]) -> float:
    sentence_tokens = _window_tokens(sentence) - QUESTION_STOP_TOKENS
    if not sentence_tokens:
        return 1.0

    normalized_sentence = normalize_lookup_text(sentence)
    best_score = 0.0
    for window in windows:
        normalized_window = normalize_lookup_text(window)
        if normalized_sentence and normalized_sentence in normalized_window:
            return 1.0
        best_score = max(best_score, _coverage_ratio(sentence_tokens, _window_tokens(window)))
    return best_score


def _strip_answer_prefixes(content: str) -> str:
    cleaned = str(content or '').strip()
    updated = True
    while cleaned and updated:
        updated = False
        for prefix in GENERIC_ANSWER_PREFIXES:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix) :].lstrip()
                updated = True
    return cleaned


def _is_meta_sentence(sentence: str, intent: QueryIntent) -> bool:
    normalized_sentence = normalize_lookup_text(sentence)
    if any(marker in normalized_sentence for marker in (normalize_lookup_text(item) for item in GENERIC_META_MARKERS)):
        return True
    if _is_correction_query(intent) and any(marker in normalized_sentence for marker in (normalize_lookup_text(item) for item in CORRECTION_META_MARKERS)):
        return True
    if intent.intent_type in {QueryIntentType.SUMMARY, QueryIntentType.OVERVIEW} and any(marker in sentence for marker in SUMMARY_ADVICE_MARKERS):
        return True
    return False


def _is_refusal_sentence(sentence: str) -> bool:
    normalized_sentence = normalize_lookup_text(sentence)
    return any(marker in normalized_sentence for marker in (normalize_lookup_text(item) for item in REFUSAL_SENTENCE_MARKERS))


def _looks_like_low_relevance_extension(sentence: str, *, query_score: float, matched_count: int) -> bool:
    normalized_sentence = normalize_lookup_text(sentence)
    return (
        any(normalized_sentence.startswith(normalize_lookup_text(marker)) for marker in LOW_RELEVANCE_EXTENSIONS)
        and query_score < 0.16
        and matched_count < 2
    )


def _correction_selected_sentences(query: str, sentences: Sequence[str], windows: Sequence[str]) -> list[str]:
    if not sentences:
        return []

    anchor_text = _correction_anchor_text(query)
    continuation_text = _correction_continuation_text(query)
    normalized_anchor = normalize_lookup_text(anchor_text)
    continuation_tokens = _window_tokens(continuation_text)

    filtered_sentences = [
        sentence
        for sentence in sentences
        if normalize_lookup_text(sentence) and normalize_lookup_text(sentence) not in normalized_anchor
    ]
    if not filtered_sentences:
        filtered_sentences = list(sentences)

    scored: list[tuple[float, float, int, str]] = []
    for index, sentence in enumerate(filtered_sentences):
        continuation_score = _coverage_ratio(continuation_tokens, _window_tokens(sentence)) if continuation_tokens else 0.0
        support_score = _sentence_support_ratio(sentence, windows)
        scored.append((continuation_score, support_score, index, sentence))

    if not scored:
        return []

    best_continuation = max(item[0] for item in scored)
    selected = [
        sentence
        for continuation_score, support_score, _index, sentence in scored
        if support_score >= 0.22 and continuation_score >= max(0.12, best_continuation - 0.10)
    ]
    if selected:
        return selected[:2]

    best_sentence = max(scored, key=lambda item: (item[1], item[0]))
    return [best_sentence[3]]


def _post_process_generative_answer(
    query: str,
    intent: QueryIntent,
    content: str,
    assessments: Sequence[EvidenceAssessment],
    *,
    use_full_content: bool,
) -> str:
    cleaned = _strip_answer_prefixes(content)
    if _is_correction_query(intent):
        for marker in CORRECTION_META_MARKERS:
            position = cleaned.find(marker)
            if position > 0:
                cleaned = cleaned[:position].rstrip()
                break

    windows = _support_windows(assessments, use_full_content=use_full_content)
    sentences = [_clean_extractive_sentence(sentence) for sentence in split_text_segments(cleaned)]
    sentences = [sentence for sentence in sentences if sentence]
    if not sentences:
        return compact_text(cleaned, 1200)

    if _is_correction_query(intent):
        selected = _correction_selected_sentences(query, sentences, windows)
        return compact_text(' '.join(selected) or cleaned, 320)

    if len(sentences) == 1:
        return compact_text(sentences[0], 1200)

    query_tokens = _query_tokens(query)
    scored_sentences: list[tuple[int, str, float, float, int, bool]] = []
    for index, sentence in enumerate(sentences):
        if _is_meta_sentence(sentence, intent):
            continue
        support_score = _sentence_support_ratio(sentence, windows)
        query_score = _window_score(query, sentence, query_tokens) if query_tokens else 0.0
        matched_count = _matched_token_count(sentence, query_tokens) if query_tokens else 0
        scored_sentences.append(
            (index, sentence, support_score, query_score, matched_count, _is_refusal_sentence(sentence))
        )

    kept: list[str] = []
    refusal_candidates: list[str] = []
    for index, sentence, support_score, query_score, matched_count, is_refusal in scored_sentences:
        if is_refusal:
            refusal_candidates.append(sentence)
            continue
        if _looks_like_low_relevance_extension(sentence, query_score=query_score, matched_count=matched_count):
            continue
        if support_score < (0.18 if index == 0 else 0.34):
            continue
        if intent.intent_type not in {QueryIntentType.SUMMARY, QueryIntentType.OVERVIEW}:
            if query_tokens and not (
                query_score >= (0.08 if index == 0 else 0.14)
                or matched_count >= (1 if index == 0 else 2)
            ):
                continue
        kept.append(sentence)

    if kept:
        return compact_text(' '.join(kept), 1200)
    if refusal_candidates:
        return compact_text(refusal_candidates[0], 240)
    if scored_sentences:
        fallback_sentence = max(scored_sentences, key=lambda item: (item[2], item[3], item[4], -item[0]))[1]
        return compact_text(fallback_sentence, 240)
    return compact_text(cleaned, 1200)


def _usable_evidence_quote(value: str) -> str:
    cleaned = compact_text(value, 240)
    if not cleaned:
        return ''
    if len(cleaned) > 120:
        return ''
    if cleaned.count('：') > 1:
        return ''
    if ' - ' in cleaned or '•' in cleaned:
        return ''
    return cleaned


def _meaningful_query_terms(query: str) -> list[tuple[int, str]]:
    terms: list[tuple[int, str]] = []
    seen: set[str] = set()
    for token in tokenize_for_bm25(query):
        normalized = normalize_lookup_text(token)
        if (
            not normalized
            or normalized in QUESTION_STOP_TOKENS
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
        terms.append((priority, str(token)))
    return terms


def _trim_to_relevant_start(query: str, value: str) -> str:
    cleaned = _clean_extractive_sentence(value)
    if not cleaned:
        return ''
    for priority in (0, 1, 2):
        positions = [
            cleaned.find(term)
            for term_priority, term in _meaningful_query_terms(query)
            if term_priority == priority and term and cleaned.find(term) >= 0
        ]
        if not positions:
            continue
        trimmed = cleaned[min(positions) :].lstrip('：:，,；; ')
        return _clean_extractive_sentence(trimmed)
    trimmed = cleaned.lstrip('：:，,；; ')
    return _clean_extractive_sentence(trimmed)


class ExtractiveGenerator:
    async def generate(self, query: str, intent: QueryIntent, assessments: Sequence[EvidenceAssessment]) -> str | None:
        primary = assessments[0] if assessments else None
        if primary is None:
            return None

        evidence_quote = _usable_evidence_quote(primary.evidence_quote)
        if evidence_quote:
            return evidence_quote

        content = _candidate_chunk_text(primary)
        if not content.strip():
            return None

        query_tokens = _query_tokens(query, primary.candidate.title)
        if not query_tokens:
            query_tokens = _query_tokens(query)
        windows = _candidate_windows(content)
        if not windows:
            return None

        scored_windows = sorted(windows, key=lambda window: _extractive_score(intent, query, window, query_tokens), reverse=True)
        best_window = scored_windows[0]
        if _window_score(query, best_window, query_tokens) < 0.32:
            return None

        sentences = split_text_segments(best_window)
        if not sentences:
            sentences = [best_window]

        scored_sentences = sorted(
            ((sentence, _extractive_score(intent, query, sentence, query_tokens)) for sentence in sentences),
            key=lambda item: item[1],
            reverse=True,
        )
        best_sentence = scored_sentences[0][0]
        candidate_answer = _trim_to_relevant_start(query, best_sentence)
        if len(scored_sentences) >= 2:
            top1_score = scored_sentences[0][1][0]
            top2_score = scored_sentences[1][1][0]
            combined = _trim_to_relevant_start(
                query,
                f'{scored_sentences[0][0]} {scored_sentences[1][0]}'
            )
            if top1_score - top2_score < 0.15 and len(combined) <= 240:
                candidate_answer = combined

        if len(candidate_answer) < 6:
            return None
        required_hits = 1 if has_identifier(query) or len(query_tokens) <= 1 else min(2, len(query_tokens))
        if _matched_token_count(candidate_answer, query_tokens) < required_hits:
            return None
        minimum_score = 0.24 if intent.intent_type.value == 'reason' and any(marker in candidate_answer for marker in REASON_EVIDENCE_MARKERS) else 0.58
        if _window_score(query, candidate_answer, query_tokens) < minimum_score:
            return None
        return compact_text(candidate_answer, 240)


class StructuredGenerator:
    def __init__(self, model_getter: Callable):
        self.model_getter = model_getter

    async def generate(self, query: str, intent: QueryIntent, assessments: Sequence[EvidenceAssessment]) -> str:
        primary = assessments[0] if assessments else None
        if (
            primary is not None
            and primary.answer_brief
            and intent.evidence_requirement is EvidenceRequirement.ATOMIC_SPAN
            and not _is_multi_info_query(query, intent)
        ):
            return compact_text(primary.answer_brief, 160)

        context_limit = _structured_context_limit(query, intent)
        use_full_content = _structured_use_full_content(query, intent)
        prompt = build_structured_rag_prompt(
            query,
            _context_blocks(query, intent, assessments[:context_limit], use_full_content=use_full_content),
        )
        response = await self.model_getter().ainvoke(prompt)
        content = compact_text(getattr(response, 'content', ''), 240)
        content = _post_process_generative_answer(
            query,
            intent,
            content,
            assessments[:context_limit],
            use_full_content=use_full_content,
        )
        if not content:
            raise FallbackRequiredError('structured_empty')
        return content


class GenerativeGenerator:
    def __init__(self, model_getter: Callable):
        self.model_getter = model_getter

    async def generate(self, query: str, intent: QueryIntent, assessments: Sequence[EvidenceAssessment]) -> str:
        correction_mode = _is_correction_query(intent)
        use_full_content = intent.intent_type in {QueryIntentType.SUMMARY, QueryIntentType.OVERVIEW} or correction_mode
        context_limit = 2 if correction_mode else 5 if use_full_content or intent.evidence_requirement is EvidenceRequirement.MULTI_SPAN else 3
        warning = ''
        if correction_mode:
            warning = (
                '这是纠错任务。请只输出应替换“续写”部分的修正后文本，不要重复新闻开头，'
                '不要追加纠正说明、解释、来源或评价。'
            )
        elif intent.intent_type in {QueryIntentType.SUMMARY, QueryIntentType.OVERVIEW}:
            warning = '这是概括任务。只保留与问题直接相关的核心事实，不要补充建议、评论、来源说明或未被问题要求的延伸信息。'
        prompt = build_general_rag_prompt(
            query,
            _context_blocks(query, intent, assessments[:context_limit], use_full_content=use_full_content),
            warning=warning,
        )
        response = await self.model_getter().ainvoke(prompt)
        content = compact_text(getattr(response, 'content', ''), 1200)
        content = _post_process_generative_answer(
            query,
            intent,
            content,
            assessments[:context_limit],
            use_full_content=use_full_content,
        )
        if not content:
            raise FallbackRequiredError('generative_empty')
        return content


class GeneratorFactory:
    def __init__(self, model_getter: Callable):
        self.extractive = ExtractiveGenerator()
        self.structured = StructuredGenerator(model_getter)
        self.generative = GenerativeGenerator(model_getter)

    async def execute(
        self,
        plan: AnswerPlan,
        query: str,
        intent: QueryIntent,
        assessments: Sequence[EvidenceAssessment],
    ) -> tuple[str, list[dict]]:
        selected = _select_assessments(plan, assessments)
        if plan.mode is AnswerMode.NO_CONTEXT or not selected:
            return build_no_context_answer(), []

        if plan.mode is AnswerMode.EXTRACTIVE:
            extracted = await self.extractive.generate(query, intent, selected)
            if extracted is None:
                raise FallbackRequiredError('extractive_failed')
            return extracted, build_sources(selected[:1])

        if plan.mode is AnswerMode.STRUCTURED:
            return await self.structured.generate(query, intent, selected), build_sources(selected[:5])

        if plan.mode is AnswerMode.GENERATIVE:
            return await self.generative.generate(query, intent, selected), build_sources(selected[:5])

        return build_no_context_answer(), []

    @staticmethod
    def downgrade(plan: AnswerPlan) -> AnswerPlan:
        if plan.mode is AnswerMode.EXTRACTIVE:
            return replace(
                plan,
                mode=AnswerMode.STRUCTURED,
                reason='extractive_fallback',
                generator_name='structured_generator',
            )
        if plan.mode is AnswerMode.STRUCTURED:
            return replace(
                plan,
                mode=AnswerMode.GENERATIVE,
                reason='structured_fallback',
                generator_name='generative_generator',
            )
        return replace(
            plan,
            mode=AnswerMode.NO_CONTEXT,
            reason='generation_exhausted',
            generator_name='no_context_generator',
            source_doc_ids=(),
            primary_doc_id=None,
        )
