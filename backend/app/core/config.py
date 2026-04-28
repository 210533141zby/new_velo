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

from pathlib import Path
from typing import List, Optional

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:
    from pydantic import BaseSettings
    SettingsConfigDict = None


class Settings(BaseSettings):
    PROJECT_NAME: str = "Wiki AI"
    API_V1_STR: str = "/api/v1"

    POSTGRES_SERVER: str = "sqlite"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_DB: str = "wiki_db"
    POSTGRES_PORT: str = "5432"

    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379

    BACKEND_CORS_ORIGINS: List[str] = ["*"]

    OPENAI_API_KEY: str = "ollama"
    DEEPSEEK_API_KEY: Optional[str] = None
    OPENAI_API_BASE: Optional[str] = None
    DEEPSEEK_API_BASE: Optional[str] = "https://api.deepseek.com/v1"
    LLM_PROVIDER: str = "ollama"
    LLM_BASE_URL: str = "http://127.0.0.1:11434/v1"
    CHAT_MODEL: str = "qwen2.5:7b-instruct"
    COMPLETION_PROVIDER: Optional[str] = None
    COMPLETION_MODEL: str = "qwen2.5-coder:14b"
    COMPLETION_API_BASE: Optional[str] = None
    COMPLETION_API_KEY: Optional[str] = None
    COMPLETION_FALLBACK_ENABLED: bool = True
    COMPLETION_FALLBACK_BASE_URL: str = "http://127.0.0.1:11434/v1"
    COMPLETION_FALLBACK_MODEL: str = "qwen2.5-coder:14b"
    COMPLETION_FALLBACK_API_KEY: Optional[str] = "ollama"
    EMBEDDING_PROVIDER: str = "ollama"
    EMBEDDING_BASE_URL: str = "http://127.0.0.1:11434/v1"
    EMBEDDING_MODEL: str = "nomic-embed-text:latest"
    AI_WARMUP_ON_STARTUP: bool = True
    RAG_RESULT_LIMIT: int = 3
    RAG_VECTOR_SEARCH_LIMIT: int = 50
    RAG_BM25_SEARCH_LIMIT: int = 50
    RAG_HYBRID_CANDIDATE_LIMIT: int = 30
    RAG_JUDGE_ENABLED: bool = True
    RAG_JUDGE_MAX_TOKENS: int = 220
    RAG_JUDGE_CONTEXT_CHARS: int = 2200
    RERANK_ENABLED: bool = True
    RERANK_MODEL: str = "BAAI/bge-reranker-v2-m3"
    RERANK_CPU_MODEL: str = "BAAI/bge-reranker-base"
    RERANK_GPU_MODEL: str = "BAAI/bge-reranker-v2-m3"
    RERANK_DEVICE: str = "auto"
    RERANK_BATCH_SIZE: int = 8
    RERANK_MAX_LENGTH: int = 512
    RERANK_MAX_INPUT_CHARS: int = 1400
    RERANK_MIN_SCORE: float = 0.56
    VLLM_API_URL: str = "http://127.0.0.1:8001/v1"
    DATA_DIR: Optional[str] = None
    CHROMA_DB_PATH: Optional[str] = None
    RERANK_CACHE_DIR: Optional[str] = None

    @property
    def data_dir(self) -> Path:
        default_data_dir = Path(__file__).resolve().parents[2] / 'data'
        data_dir = Path(self.DATA_DIR or str(default_data_dir)).expanduser()
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
        chroma_dir = Path(self.CHROMA_DB_PATH or str(self.data_dir / 'chroma_db')).expanduser()
        chroma_dir.mkdir(parents=True, exist_ok=True)
        return chroma_dir

    @property
    def rerank_cache_directory(self) -> Path:
        rerank_dir = Path(self.RERANK_CACHE_DIR or str(self.data_dir / 'models' / 'rerank')).expanduser()
        rerank_dir.mkdir(parents=True, exist_ok=True)
        return rerank_dir

    @property
    def chat_api_base(self) -> str:
        if (self.LLM_PROVIDER or '').lower() == 'deepseek':
            return self.DEEPSEEK_API_BASE or self.OPENAI_API_BASE or self.LLM_BASE_URL
        return self.OPENAI_API_BASE or self.LLM_BASE_URL

    @property
    def completion_api_base(self) -> str:
        if self.COMPLETION_API_BASE:
            return self.COMPLETION_API_BASE
        if self.completion_provider == 'vllm':
            return self.VLLM_API_URL
        if self.completion_provider == 'deepseek':
            return self.DEEPSEEK_API_BASE or self.OPENAI_API_BASE or self.LLM_BASE_URL
        return self.LLM_BASE_URL

    @property
    def embedding_api_base(self) -> str:
        return self.EMBEDDING_BASE_URL or self.chat_api_base

    @property
    def llm_api_key(self) -> str:
        provider = (self.LLM_PROVIDER or '').lower()
        if provider == 'deepseek':
            if self.DEEPSEEK_API_KEY:
                return self.DEEPSEEK_API_KEY
            if self.OPENAI_API_KEY and self.OPENAI_API_KEY != 'ollama':
                return self.OPENAI_API_KEY
            return ''
        return self.DEEPSEEK_API_KEY or self.OPENAI_API_KEY

    @property
    def completion_provider(self) -> str:
        return (self.COMPLETION_PROVIDER or self.LLM_PROVIDER or 'ollama').strip().lower()

    @property
    def completion_api_key(self) -> str:
        if self.COMPLETION_API_KEY is not None:
            return self.COMPLETION_API_KEY
        return self.llm_api_key

    if SettingsConfigDict is not None:
        model_config = SettingsConfigDict(env_file='.env', extra='ignore')
    else:
        class Config:
            env_file = '.env'
            extra = 'ignore'


settings = Settings()
