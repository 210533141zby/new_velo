from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.schemas import ChatRequest, ChatResponse
from app.services.rag.rag_service import RagService

rag_router = APIRouter(prefix='/agent', tags=['agent'])


def get_rag_service(db: AsyncSession = Depends(get_db)) -> RagService:
    return RagService(db)


@rag_router.post('/chat', response_model=ChatResponse)
async def chat_with_rag(
    request: ChatRequest,
    rag_service: RagService = Depends(get_rag_service),
):
    user_query = request.messages[-1].content
    result = await rag_service.rag_qa(user_query)
    return ChatResponse(
        response=result.get('response', ''),
        sources=result.get('sources'),
    )
