"""
=============================================================================
文件: agent_service.py
描述: AI 智能体核心服务

主要功能：
1. 文档索引 (Indexing)：将 Markdown 文档切分并存储到 ChromaDB 向量数据库中，用于后续检索。
2. RAG 问答 (Retrieval-Augmented Generation)：基于用户问题检索相关文档，结合上下文生成回答。
=============================================================================
"""

import asyncio
import hashlib
import json
import os
import re

from fastapi.concurrency import run_in_threadpool
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from openai import APITimeoutError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.cache import redis_manager
from app.core.config import settings
from app.logger import logger
from app.models import Document
from app.services.base import BaseService

PERSIST_DIRECTORY = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "chroma_db")
_chat_llm_instance = None
_embeddings_instance = None
_vector_store_instance = None


def _sanitize_collection_part(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", value or "")
    return cleaned.strip("_").lower() or "default"


def _collection_name() -> str:
    provider = _sanitize_collection_part(settings.EMBEDDING_PROVIDER)
    model = _sanitize_collection_part(settings.EMBEDDING_MODEL)
    return f"velo_{provider}_{model}"[:63]


def _dedupe_scored_matches(scored_matches):
    best_match_by_source = {}

    for doc, score in scored_matches:
        source_key = doc.metadata.get("doc_id") or doc.metadata.get("source", "Unknown")
        existing = best_match_by_source.get(source_key)
        if existing is None or float(score) > existing["score"]:
            best_match_by_source[source_key] = {
                "doc": doc,
                "score": float(score),
            }

    return sorted(best_match_by_source.values(), key=lambda item: item["score"], reverse=True)


def _select_relevant_matches(scored_matches):
    unique_matches = _dedupe_scored_matches(scored_matches)
    if not unique_matches:
        return []

    best_score = unique_matches[0]["score"]
    score_threshold = max(0.35, best_score * 0.85)
    filtered_matches = [item for item in unique_matches if item["score"] >= score_threshold]

    if not filtered_matches:
        return unique_matches[:1]

    return filtered_matches[:3]


def _build_chat_llm() -> ChatOpenAI:
    return ChatOpenAI(
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.chat_api_base,
        model=settings.CHAT_MODEL,
        temperature=0.3,
    )


class CPUEmbeddingFunction:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device="cpu")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        vector = self.model.encode(text, normalize_embeddings=True)
        return vector.tolist()


def get_chat_llm() -> ChatOpenAI:
    global _chat_llm_instance
    if _chat_llm_instance is None:
        _chat_llm_instance = _build_chat_llm()
    return _chat_llm_instance


def get_embeddings():
    global _embeddings_instance
    if _embeddings_instance is None:
        provider = (settings.EMBEDDING_PROVIDER or "").lower()
        _embeddings_instance = (
            CPUEmbeddingFunction(settings.EMBEDDING_MODEL)
            if provider == "huggingface"
            else OpenAIEmbeddings(
                api_key=settings.OPENAI_API_KEY,
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
            collection_name=_collection_name(),
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=get_embeddings(),
        )
    return _vector_store_instance


class AgentService(BaseService):
    def __init__(self, db: AsyncSession):
        super().__init__(db)
        self._vector_store = None

    @property
    def vector_store(self):
        if self._vector_store is None:
            self._vector_store = get_vector_store()
        return self._vector_store

    async def ensure_bootstrap_index(self):
        active_docs_result = await self.db.execute(
            select(Document)
            .where(Document.is_active == True)
            .order_by(Document.updated_at.desc().nullslast(), Document.id.desc())
        )
        active_docs = [doc for doc in active_docs_result.scalars().all() if doc.content]
        if not active_docs:
            return

        collection_count = await run_in_threadpool(self.vector_store._collection.count)
        if collection_count > 0:
            return

        logger.info(
            f"RAG collection 为空，开始预热索引 {len(active_docs)} 篇文档",
            extra={
                "extra_data": {
                    "event": "rag_bootstrap_start",
                    "document_count": len(active_docs),
                    "collection_name": _collection_name(),
                }
            },
        )

        for doc in active_docs:
            await self.index_document(doc.id, doc.title, doc.content)

        logger.info(
            "RAG 预热索引完成",
            extra={
                "extra_data": {
                    "event": "rag_bootstrap_complete",
                    "document_count": len(active_docs),
                    "collection_name": _collection_name(),
                }
            },
        )

    async def index_document(self, doc_id: int, title: str, content: str):
        if not content:
            return

        try:
            headers_to_split_on = [
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ]
            markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
            md_header_splits = markdown_splitter.split_text(content)

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            final_splits = text_splitter.split_documents(md_header_splits)
            if not final_splits and content:
                final_splits = text_splitter.create_documents([content])

            docs = []
            for split in final_splits:
                if not split.metadata:
                    split.metadata = {}
                split.metadata["source"] = title
                split.metadata["doc_id"] = doc_id
                docs.append(split)

            if not docs:
                return

            max_retries = 3
            retry_delay = 2
            for retry in range(max_retries):
                try:
                    await run_in_threadpool(self.vector_store.add_documents, docs)
                    logger.info(f"已索引文档 {doc_id}: {title}", extra={
                        "extra_data": {
                            "event": "rag_index_success",
                            "document_id": doc_id,
                            "chunk_count": len(docs),
                            "collection_name": _collection_name(),
                        }
                    })
                    break
                except (APITimeoutError, Exception) as e:
                    if retry < max_retries - 1:
                        logger.warning(
                            f"索引文档失败 (尝试 {retry + 1}/{max_retries}): {e}，将在 {retry_delay} 秒后重试...",
                            extra={
                                "extra_data": {
                                    "event": "rag_index_retry",
                                    "document_id": doc_id,
                                    "retry": retry + 1,
                                    "collection_name": _collection_name(),
                                }
                            },
                        )
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2
                    else:
                        raise
        except Exception:
            logger.exception("索引文档最终失败", extra={
                "extra_data": {
                    "event": "rag_index_failed",
                    "document_id": doc_id,
                    "collection_name": _collection_name(),
                }
            })

    async def delete_document_index(self, doc_id: int):
        try:
            await run_in_threadpool(self.vector_store.delete, where={"doc_id": doc_id})
            logger.info(f"已删除文档索引: {doc_id}", extra={
                "extra_data": {
                    "event": "rag_delete_success",
                    "document_id": doc_id,
                    "collection_name": _collection_name(),
                }
            })
        except Exception:
            logger.exception("删除文档索引失败", extra={
                "extra_data": {
                    "event": "rag_delete_failed",
                    "document_id": doc_id,
                    "collection_name": _collection_name(),
                }
            })

    async def polish_text(self, text: str) -> str:
        prompt = f"""
        You are a professional editor. Please polish the following text to make it more concise, clear, and professional, while maintaining the original meaning.

        Original Text:
        {text}

        Polished Text:
        """
        response = await get_chat_llm().ainvoke(prompt)
        logger.info("文本润色完成", extra={
            "extra_data": {
                "event": "ai_polish_success",
                "text_length": len(text),
            }
        })
        return response.content

    async def complete_text(self, text: str) -> str:
        prompt = f"""
        You are a helpful writing assistant. Please continue writing the following text naturally.

        Context:
        {text}

        Continuation:
        """
        response = await get_chat_llm().ainvoke(prompt)
        logger.info("文本续写完成", extra={
            "extra_data": {
                "event": "ai_complete_success",
                "text_length": len(text),
            }
        })
        return response.content

    async def rag_qa(self, query: str) -> dict:
        try:
            query_hash = hashlib.md5(query.encode("utf-8")).hexdigest()
            cache_key = f"rag:response:v4:{query_hash}"

            cached_result = await redis_manager.get(cache_key)
            if cached_result:
                logger.info(f"[CACHE HIT] RAG 问答命中缓存: {query}", extra={
                    "extra_data": {
                        "event": "rag_cache_hit",
                        "query": query,
                    }
                })
                return json.loads(cached_result)

            scored_matches = await run_in_threadpool(
                self.vector_store.similarity_search_with_relevance_scores,
                query,
                6,
            )
            relevant_matches = _select_relevant_matches(scored_matches)
            docs = [item["doc"] for item in relevant_matches]
            context = "\n\n".join([d.page_content for d in docs])

            sources = []
            for item in relevant_matches:
                doc = item["doc"]
                sources.append({
                    "title": doc.metadata.get("source", "Unknown"),
                    "doc_id": doc.metadata.get("doc_id"),
                })

            prompt = f"""
            You are a knowledgeable assistant. Answer the question based on the following context.
            If the answer is not in the context, say \"I don't know based on the provided information\".

            Context:
            {context}

            Question:
            {query}

            Answer:
            """
            response = await get_chat_llm().ainvoke(prompt)
            result = {
                "response": response.content,
                "sources": sources,
            }
            await redis_manager.set(cache_key, json.dumps(result), ex=3600)

            logger.info(f"RAG 问答完成: {query}", extra={
                "extra_data": {
                    "event": "rag_qa_success",
                    "query": query,
                    "source_count": len(sources),
                    "best_score": relevant_matches[0]["score"] if relevant_matches else None,
                    "collection_name": _collection_name(),
                }
            })
            return result
        except Exception:
            logger.exception("RAG 问答失败", extra={
                "extra_data": {
                    "event": "rag_qa_failed",
                    "query": query,
                    "collection_name": _collection_name(),
                }
            })
            return {
                "response": "抱歉，系统暂时无法回答您的请求。",
                "sources": [],
            }

    async def ask_ai(self, query: str) -> str:
        prompt = f"You are a helpful assistant. Please answer the following question:\n\n{query}"
        response = await get_chat_llm().ainvoke(prompt)
        return response.content
