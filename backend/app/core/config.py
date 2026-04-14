"""
=============================================================================
文件: config.py
描述: 全局配置管理

核心功能：
1. 环境变量加载：从 .env 文件或系统环境变量中读取配置。
2. 统一管理：集中管理数据库、Redis、AI 模型等所有关键参数。

依赖组件:
- pydantic-settings: 用于自动读取环境变量并进行类型检查。
=============================================================================
"""

import os
from pathlib import Path
from typing import List, Optional

try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings


class Settings(BaseSettings):
    """
    系统配置类

    Why:
    使用 Pydantic 的 BaseSettings 可以自动从环境变量中读取配置，并且自带类型检查。
    如果 .env 写错了（比如端口写成了字符串），程序启动时就会报错提醒，防止带着隐患运行。
    """

    PROJECT_NAME: str = "Wiki AI"
    API_V1_STR: str = "/api/v1"

    # =========================================================================
    # 数据库配置 (Database)
    # =========================================================================
    POSTGRES_SERVER: str = os.getenv("POSTGRES_SERVER", "sqlite")
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "postgres")
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "wiki_db")
    POSTGRES_PORT: str = os.getenv("POSTGRES_PORT", "5432")

    @property
    def SQLALCHEMY_DATABASE_URI(self) -> str:
        if self.POSTGRES_SERVER == "sqlite" or not self.POSTGRES_SERVER:
            default_data_dir = Path(__file__).resolve().parents[2] / "data"
            data_dir = Path(os.getenv("DATA_DIR", str(default_data_dir))).expanduser()
            data_dir.mkdir(parents=True, exist_ok=True)
            return f"sqlite+aiosqlite:///{data_dir / 'wiki.db'}"
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    # =========================================================================
    # Redis 缓存配置
    # =========================================================================
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", 6379))

    # =========================================================================
    # 安全与跨域配置 (CORS)
    # =========================================================================
    BACKEND_CORS_ORIGINS: List[str] = ["*"]

    # =========================================================================
    # AI 模型配置
    # =========================================================================
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "ollama")
    OPENAI_API_BASE: Optional[str] = os.getenv("OPENAI_API_BASE", None)
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama")
    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "http://127.0.0.1:11434/v1")
    CHAT_MODEL: str = os.getenv("CHAT_MODEL", "qwen2.5:7b-instruct")
    COMPLETION_MODEL: str = os.getenv("COMPLETION_MODEL", "qwen2.5-coder:14b")
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "ollama")
    EMBEDDING_BASE_URL: str = os.getenv("EMBEDDING_BASE_URL", "http://127.0.0.1:11434/v1")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest")
    VLLM_API_URL: str = os.getenv("VLLM_API_URL", "http://127.0.0.1:8001/v1")

    @property
    def chat_api_base(self) -> str:
        return self.OPENAI_API_BASE or self.LLM_BASE_URL

    @property
    def completion_api_base(self) -> str:
        if self.LLM_PROVIDER == "vllm":
            return self.VLLM_API_URL
        return self.LLM_BASE_URL

    @property
    def embedding_api_base(self) -> str:
        return self.EMBEDDING_BASE_URL or self.chat_api_base

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
