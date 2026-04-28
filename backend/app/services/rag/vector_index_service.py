from __future__ import annotations

import asyncio
import re
from pathlib import Path
from collections import Counter

from fastapi.concurrency import run_in_threadpool
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from app.core.config import settings
from app.logger import logger

try:
    import jieba.posseg as pseg
except Exception:  # pragma: no cover - 依赖缺失时跳过实体注入。
    pseg = None

_embeddings_instance = None
_vector_store_instance = None
CORE_ENTITY_STOPWORDS = {
    '文档',
    '资料',
    '介绍',
    '内容',
    '情况',
    '背景',
    '历史',
    '概况',
    '总结',
    '概述',
    '说明',
    '方案',
    '测试',
}


def _sanitize_collection_part(value: str) -> str:
    cleaned = re.sub(r'[^a-zA-Z0-9_-]+', '_', value or '')
    return cleaned.strip('_').lower() or 'default'


def collection_name() -> str:
    provider = _sanitize_collection_part(settings.EMBEDDING_PROVIDER)
    model = _sanitize_collection_part(settings.EMBEDDING_MODEL)
    return f'velo_{provider}_{model}'[:63]


class CpuEmbeddingClient:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device='cpu')

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        vector = self.model.encode(text, normalize_embeddings=True)
        return vector.tolist()


def get_embeddings():
    global _embeddings_instance
    if _embeddings_instance is None:
        provider = (settings.EMBEDDING_PROVIDER or '').lower()
        _embeddings_instance = (
            CpuEmbeddingClient(settings.EMBEDDING_MODEL)
            if provider == 'huggingface'
            else OpenAIEmbeddings(
                api_key=settings.llm_api_key,
                base_url=settings.embedding_api_base,
                model=settings.EMBEDDING_MODEL,
                check_embedding_ctx_length=False,
            )
        )
    return _embeddings_instance


def get_vector_store():
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = Chroma(
            collection_name=collection_name(),
            persist_directory=str(settings.chroma_persist_directory),
            embedding_function=get_embeddings(),
        )
    return _vector_store_instance


def _first_paragraph(content: str) -> str:
    for block in re.split(r'\n\s*\n', str(content or '')):
        cleaned = re.sub(r'^\s*#+\s*', '', block).strip()
        if cleaned:
            return cleaned[:500]
    return ''


def _extract_core_entities(title: str, first_paragraph: str) -> str:
    if pseg is None:
        return ''

    counter: Counter[str] = Counter()
    for part in re.split(r'[的:：\-\s()\[\]（）《》]+', str(title or '')):
        word = str(part or '').strip()
        if len(word) < 2:
            continue
        if word in CORE_ENTITY_STOPWORDS:
            continue
        if re.fullmatch(r'[\W_]+', word):
            continue
        counter[word] += 4

    for text, weight in ((title, 2), (first_paragraph, 1)):
        for token, flag in pseg.cut(str(text or '')):
            word = str(token or '').strip()
            pos = str(flag or '').strip().lower()
            if len(word) < 2:
                continue
            if word in CORE_ENTITY_STOPWORDS:
                continue
            if not pos.startswith('n'):
                continue
            if re.fullmatch(r'[\W_]+', word):
                continue
            counter[word] += weight

    if not counter:
        return ''
    entities = [word for word, _count in counter.most_common(3)]
    return f'【核心主题：{"、".join(entities)}】'


async def index_document_chunks(doc_id: int, title: str, content: str):
    if not content:
        return
    try:
        headers_to_split_on = [('#', 'Header 1'), ('##', 'Header 2'), ('###', 'Header 3')]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        md_header_splits = markdown_splitter.split_text(content)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_splits = text_splitter.split_documents(md_header_splits)
        if not final_splits and content:
            final_splits = text_splitter.create_documents([content])

        entity_prefix = _extract_core_entities(title, _first_paragraph(content))
        docs = []
        for split in final_splits:
            if entity_prefix:
                split.page_content = f'{entity_prefix}\n{split.page_content}'
            if not split.metadata:
                split.metadata = {}
            split.metadata['source'] = title
            split.metadata['doc_id'] = doc_id
            docs.append(split)
        if not docs:
            return

        retry_delay = 2
        vector_store = get_vector_store()
        for retry in range(3):
            try:
                await run_in_threadpool(vector_store.delete, where={'doc_id': doc_id})
                await run_in_threadpool(vector_store.add_documents, docs)
                logger.info(
                    f'已索引文档 {doc_id}: {title}',
                    extra={
                        'extra_data': {
                            'event': 'rag_index_success',
                            'document_id': doc_id,
                            'chunk_count': len(docs),
                            'collection_name': collection_name(),
                        }
                    },
                )
                break
            except Exception as exc:
                if retry < 2:
                    logger.warning(
                        f'索引文档失败 (尝试 {retry + 1}/3): {exc}，将在 {retry_delay} 秒后重试...',
                        extra={
                            'extra_data': {
                                'event': 'rag_index_retry',
                                'document_id': doc_id,
                                'retry': retry + 1,
                                'collection_name': collection_name(),
                            }
                        },
                    )
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise
    except Exception:
        logger.exception(
            '索引文档最终失败',
            extra={
                'extra_data': {
                    'event': 'rag_index_failed',
                    'document_id': doc_id,
                    'collection_name': collection_name(),
                }
            },
        )


async def delete_document_chunks(doc_id: int):
    try:
        await run_in_threadpool(get_vector_store().delete, where={'doc_id': doc_id})
        logger.info(
            f'已删除文档索引: {doc_id}',
            extra={
                'extra_data': {
                    'event': 'rag_delete_success',
                    'document_id': doc_id,
                    'collection_name': collection_name(),
                }
            },
        )
    except Exception:
        logger.exception(
            '删除文档索引失败',
            extra={
                'extra_data': {
                    'event': 'rag_delete_failed',
                    'document_id': doc_id,
                    'collection_name': collection_name(),
                }
            },
        )
