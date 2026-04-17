"""API 路由汇总。"""

from fastapi import APIRouter

from app.api import completion, content, rag

api_router = APIRouter()
api_router.include_router(content.documents_router)
api_router.include_router(content.folders_router)
api_router.include_router(rag.rag_router)
api_router.include_router(completion.completion_router)
