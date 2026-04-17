from __future__ import annotations

from langchain_openai import ChatOpenAI

from app.core.config import settings

_chat_model_instance = None
_rag_judge_model_instance = None


def _build_chat_model(temperature: float = 0.3, max_tokens: int | None = None) -> ChatOpenAI:
    return ChatOpenAI(
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.chat_api_base,
        model=settings.CHAT_MODEL,
        temperature=temperature,
        timeout=120,
        max_retries=1,
        max_tokens=max_tokens,
    )


def get_chat_model() -> ChatOpenAI:
    global _chat_model_instance
    if _chat_model_instance is None:
        _chat_model_instance = _build_chat_model()
    return _chat_model_instance


def get_rag_judge_model() -> ChatOpenAI:
    global _rag_judge_model_instance
    if _rag_judge_model_instance is None:
        _rag_judge_model_instance = _build_chat_model(
            temperature=0.0,
            max_tokens=settings.RAG_JUDGE_MAX_TOKENS,
        )
    return _rag_judge_model_instance
