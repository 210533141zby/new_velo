"""
=============================================================================
文件: schemas.py
描述: Pydantic 数据模式定义

核心功能：
1. 数据校验：在请求进入业务逻辑前，自动检查数据格式是否正确（如必填项、类型等）。
2. 数据清洗：自动过滤掉用户多传的无用字段。
3. 序列化：将数据库模型（ORM 对象）转换为前端可读的 JSON 格式。

依赖组件:
- pydantic: Python 中最流行的数据验证库，FastAPI 的核心依赖。
=============================================================================
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel

# =============================================================================
# 文件夹模式 (Folder)
# =============================================================================


class FolderBase(BaseModel):
    """
    文件夹基础模式

    Logic Flow:
        定义了文件夹最基本的字段，用于被其他模式继承，避免重复代码。
    """

    title: str
    parent_id: Optional[int] = None


class FolderCreate(FolderBase):
    """
    创建文件夹请求模式

    Why:
        虽然目前看起来和 Base 一样，但未来如果要加“创建时必填，但更新时不可改”的字段，
        就可以在这里单独加，而不影响其他模式。
    """

    pass


class FolderUpdate(FolderBase):
    """
    更新文件夹请求模式
    """

    pass


class FolderResponse(FolderBase):
    """
    文件夹响应模式

    Logic Flow:
        这是返回给前端的数据结构。
        它比 Base 多了 id、created_at 等由数据库自动生成的字段。

    Why:
        为什么要有 Config 类？
        `from_attributes = True` 允许 Pydantic 直接从 SQLAlchemy 对象读取数据，
        而不用我们手动把 ORM 对象转成字典。
    """

    id: int
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# =============================================================================
# 文档模式 (Document)
# =============================================================================


class DocumentBase(BaseModel):
    """
    文档基础模式
    """

    title: str
    content: Optional[str] = None
    folder_id: Optional[int] = None


class DocumentCreate(DocumentBase):
    """
    创建文档请求模式
    """

    pass


class DocumentUpdate(BaseModel):
    """
    更新文档请求模式

    Logic Flow:
        所有字段都是 Optional 的，因为更新时用户可能只想改标题，或者只想改内容。
    """

    title: Optional[str] = None
    content: Optional[str] = None
    folder_id: Optional[int] = None
    summary: Optional[str] = None
    tags: Optional[str] = None


class DocumentSummary(BaseModel):
    """
    文档摘要响应模式

    Why:
        在文件列表页，我们不需要把成千上万字的文档内容 (content) 全部返回。
        只返回 id、标题和摘要，可以大大减少网络传输量，提高加载速度。
    """

    id: int
    title: str
    content: Optional[str] = None
    folder_id: Optional[int] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class DocumentResponse(DocumentBase):
    """
    文档详细响应模式

    Logic Flow:
        这是进入文档详情页时返回的数据，包含所有信息。
    """

    id: int
    summary: Optional[str] = None
    tags: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    folder_id: Optional[int] = None

    class Config:
        from_attributes = True


# =============================================================================
# 聊天模式 (Chat)
# =============================================================================


class ChatMessage(BaseModel):
    """
    聊天消息模式

    Logic Flow:
        代表一条聊天记录。
        role: 谁说的？(user, assistant, system)
        content: 说了什么？
    """

    role: str
    content: str


class ChatRequest(BaseModel):
    """
    聊天请求模式

    Logic Flow:
        前端发给后端的完整聊天上下文。
        当前只承载知识库问答所需的消息列表和可选文档定位信息。
    """

    messages: list[ChatMessage]
    doc_id: Optional[int] = None


class ChatSource(BaseModel):
    """
    RAG 引用来源模式

    Logic Flow:
        用于将最终引用文档、排序和置信度明确暴露给前端，避免返回松散字典。
    """

    title: str
    doc_id: Optional[int] = None
    rank: Optional[int] = None
    confidence: Optional[int] = None


class ChatResponse(BaseModel):
    """
    聊天响应模式

    Logic Flow:
        后端返回给前端的最终回答。
        除了回答内容 (response)，如果用了 RAG，还会带上参考来源 (sources)。
    """

    response: str
    sources: Optional[list[ChatSource]] = None
