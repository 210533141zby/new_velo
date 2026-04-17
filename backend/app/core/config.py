"""
=============================================================================
文件: config.py
描述: 全局配置管理

核心功能：
1. 环境变量加载：从 .env 文件或系统环境变量中读取配置。
2. 统一管理：集中管理数据库、Redis、AI 模型等所有关键参数。
3. 路径规范化：将数据库、Chroma、Rerank 模型缓存等持久化目录统一收敛到 DATA_DIR 之下。
=============================================================================
"""

import os
from pathlib import Path
from typing import List, Optional

try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {'1', 'true', 'yes', 'on'}


class Settings(BaseSettings):
    PROJECT_NAME: str = "Wiki AI"
    API_V1_STR: str = "/api/v1"

    POSTGRES_SERVER: str = os.getenv("POSTGRES_SERVER", "sqlite")
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "postgres")
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "wiki_db")
    POSTGRES_PORT: str = os.getenv("POSTGRES_PORT", "5432")

    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", 6379))

    BACKEND_CORS_ORIGINS: List[str] = ["*"]

    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "ollama")
    OPENAI_API_BASE: Optional[str] = os.getenv("OPENAI_API_BASE", None)
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama")
    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "http://127.0.0.1:11434/v1")
    CHAT_MODEL: str = os.getenv("CHAT_MODEL", "qwen2.5:7b-instruct")
    COMPLETION_MODEL: str = os.getenv("COMPLETION_MODEL", "qwen2.5-coder:14b")
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "ollama")
    EMBEDDING_BASE_URL: str = os.getenv("EMBEDDING_BASE_URL", "http://127.0.0.1:11434/v1")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest")
    AI_WARMUP_ON_STARTUP: bool = _env_bool("AI_WARMUP_ON_STARTUP", True)
    RAG_RESULT_LIMIT: int = int(os.getenv("RAG_RESULT_LIMIT", 3))
    RAG_VECTOR_SEARCH_LIMIT: int = int(os.getenv("RAG_VECTOR_SEARCH_LIMIT", 50))
    RAG_BM25_SEARCH_LIMIT: int = int(os.getenv("RAG_BM25_SEARCH_LIMIT", 50))
    RAG_HYBRID_CANDIDATE_LIMIT: int = int(os.getenv("RAG_HYBRID_CANDIDATE_LIMIT", 30))
    RAG_JUDGE_ENABLED: bool = _env_bool("RAG_JUDGE_ENABLED", True)
    RAG_JUDGE_MAX_TOKENS: int = int(os.getenv("RAG_JUDGE_MAX_TOKENS", 220))
    RAG_JUDGE_CONTEXT_CHARS: int = int(os.getenv("RAG_JUDGE_CONTEXT_CHARS", 2200))
    RERANK_ENABLED: bool = _env_bool("RERANK_ENABLED", True)
    RERANK_MODEL: str = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
    RERANK_CPU_MODEL: str = os.getenv("RERANK_CPU_MODEL", "BAAI/bge-reranker-base")
    RERANK_GPU_MODEL: str = os.getenv("RERANK_GPU_MODEL", "BAAI/bge-reranker-v2-m3")
    RERANK_DEVICE: str = os.getenv("RERANK_DEVICE", "auto")
    RERANK_BATCH_SIZE: int = int(os.getenv("RERANK_BATCH_SIZE", 8))
    RERANK_MAX_LENGTH: int = int(os.getenv("RERANK_MAX_LENGTH", 512))
    RERANK_MAX_INPUT_CHARS: int = int(os.getenv("RERANK_MAX_INPUT_CHARS", 1400))
    RERANK_MIN_SCORE: float = float(os.getenv("RERANK_MIN_SCORE", 0.56))
    VLLM_API_URL: str = os.getenv("VLLM_API_URL", "http://127.0.0.1:8001/v1")

    @property
    def data_dir(self) -> Path:
        default_data_dir = Path(__file__).resolve().parents[2] / 'data'
        data_dir = Path(os.getenv('DATA_DIR', str(default_data_dir))).expanduser()
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir

    @property
    def SQLALCHEMY_DATABASE_URI(self) -> str:
        if self.POSTGRES_SERVER == 'sqlite' or not self.POSTGRES_SERVER:
            return f"sqlite+aiosqlite:///{self.data_dir / 'wiki.db'}"
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@"
            f"{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @property
    def chroma_persist_directory(self) -> Path:
        chroma_dir = Path(
            os.getenv('CHROMA_DB_PATH', str(self.data_dir / 'chroma_db'))
        ).expanduser()
        chroma_dir.mkdir(parents=True, exist_ok=True)
        return chroma_dir

    @property
    def rerank_cache_directory(self) -> Path:
        rerank_dir = Path(
            os.getenv('RERANK_CACHE_DIR', str(self.data_dir / 'models' / 'rerank'))
        ).expanduser()
        rerank_dir.mkdir(parents=True, exist_ok=True)
        return rerank_dir

    @property
    def chat_api_base(self) -> str:
        return self.OPENAI_API_BASE or self.LLM_BASE_URL

    @property
    def completion_api_base(self) -> str:
        if self.LLM_PROVIDER == 'vllm':
            return self.VLLM_API_URL
        return self.LLM_BASE_URL

    @property
    def embedding_api_base(self) -> str:
        return self.EMBEDDING_BASE_URL or self.chat_api_base

    class Config:
        env_file = '.env'
        extra = 'ignore'


settings = Settings()
