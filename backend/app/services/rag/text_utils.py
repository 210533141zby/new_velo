from __future__ import annotations

import re


def compact_text(value: str, limit: int | None = None) -> str:
    normalized = re.sub(r'\s+', ' ', str(value or '').strip())
    if limit is not None:
        return normalized[:limit].strip()
    return normalized


def split_text_segments(content: str) -> list[str]:
    normalized = re.sub(
        r'(?<=[\u4e00-\u9fff）】])\s*[-•]\s*(?=[\u4e00-\u9fff【（])',
        '\n',
        content or '',
    )
    return [segment.strip() for segment in re.split(r'(?<=[。！？；])|\n+', normalized) if segment.strip()]


def split_paragraphs(content: str) -> list[str]:
    paragraphs = [paragraph.strip() for paragraph in re.split(r'\n{2,}', content or '') if paragraph.strip()]
    return paragraphs or split_text_segments(content)
