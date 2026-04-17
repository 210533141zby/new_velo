from __future__ import annotations

import re
from dataclasses import replace
from typing import Callable, Sequence

from app.services.rag.hybrid_search import has_identifier, tokenize_for_bm25
from app.services.rag.pipeline_models import (
    AnswerMode,
    AnswerPlan,
    EvidenceAssessment,
    QueryIntent,
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
    if re.search(r'(18|19|20)\d{2}', paragraph):
        score += 0.12
    if any(marker in paragraph for marker in ('最早', '建于', '起步', '早年', '后来', '直到', '沿海', '渔业', '捕鱼')):
        score += 0.12
    return score


def _build_context_excerpt(
    query: str,
    assessment: EvidenceAssessment,
    *,
    overview_mode: bool,
    relation_mode: bool,
    use_full_content: bool,
) -> str:
    raw_content = _candidate_full_content(assessment) if use_full_content else _candidate_chunk_text(assessment)
    paragraphs = split_paragraphs(raw_content)
    if not paragraphs:
        return compact_text(raw_content, 2200)

    query_tokens = _query_tokens(query if overview_mode else query, '' if overview_mode else assessment.candidate.title)
    if not query_tokens:
        query_tokens = _query_tokens(query)

    selected_indexes: list[int] = [0] if overview_mode else []
    ranked_paragraphs = sorted(
        range(len(paragraphs)),
        key=lambda index: (
            _history_paragraph_score(query, paragraphs[index], query_tokens)
            if overview_mode
            else _window_score(query, paragraphs[index], query_tokens),
            len(paragraphs[index]),
        ),
        reverse=True,
    )
    target_count = 5 if overview_mode else 4 if relation_mode else 2
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
    for index, assessment in enumerate(assessments, start=1):
        content = _build_context_excerpt(
            query,
            assessment,
            overview_mode=overview_mode,
            relation_mode=relation_mode,
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


class ExtractiveGenerator:
    async def generate(self, query: str, intent: QueryIntent, assessments: Sequence[EvidenceAssessment]) -> str | None:
        primary = assessments[0] if assessments else None
        if primary is None:
            return None

        evidence_quote = compact_text(primary.evidence_quote)
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

        best_sentence = sorted(sentences, key=lambda sentence: _extractive_score(intent, query, sentence, query_tokens), reverse=True)[0]
        cleaned_sentence = _clean_extractive_sentence(best_sentence)
        if len(cleaned_sentence) < 6:
            return None
        required_hits = 1 if has_identifier(query) or len(query_tokens) <= 1 else min(2, len(query_tokens))
        if _matched_token_count(cleaned_sentence, query_tokens) < required_hits:
            return None
        minimum_score = 0.24 if intent.intent_type.value == 'reason' and any(marker in cleaned_sentence for marker in REASON_EVIDENCE_MARKERS) else 0.58
        if _window_score(query, cleaned_sentence, query_tokens) < minimum_score:
            return None
        return compact_text(cleaned_sentence, 240)


class StructuredGenerator:
    def __init__(self, model_getter: Callable):
        self.model_getter = model_getter

    async def generate(self, query: str, intent: QueryIntent, assessments: Sequence[EvidenceAssessment]) -> str:
        primary = assessments[0] if assessments else None
        if primary is not None and primary.answer_brief:
            return compact_text(primary.answer_brief, 160)

        prompt = build_structured_rag_prompt(
            query,
            _context_blocks(query, intent, assessments[:2], use_full_content=False),
        )
        response = await self.model_getter().ainvoke(prompt)
        content = compact_text(getattr(response, 'content', ''), 240)
        if not content:
            raise FallbackRequiredError('structured_empty')
        return content


class GenerativeGenerator:
    def __init__(self, model_getter: Callable):
        self.model_getter = model_getter

    async def generate(self, query: str, intent: QueryIntent, assessments: Sequence[EvidenceAssessment]) -> str:
        context_limit = 5 if intent.intent_type.name.lower() in {'summary', 'overview'} else 3
        prompt = build_general_rag_prompt(
            query,
            _context_blocks(query, intent, assessments[:context_limit], use_full_content=True),
        )
        response = await self.model_getter().ainvoke(prompt)
        content = compact_text(getattr(response, 'content', ''), 1200)
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
            return await self.structured.generate(query, intent, selected), build_sources(selected[:2])

        if plan.mode is AnswerMode.GENERATIVE:
            return await self.generative.generate(query, intent, selected), build_sources(selected[:3])

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
