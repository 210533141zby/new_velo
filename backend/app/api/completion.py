import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.completion.completion_service import complete_text

logger = logging.getLogger(__name__)
completion_router = APIRouter(tags=['completion'])


class CompletionRequest(BaseModel):
    prefix: str
    suffix: str
    language: Optional[str] = None
    trigger_mode: Optional[str] = None


@completion_router.post('/completion')
async def generate_completion(request: CompletionRequest):
    try:
        if not request.prefix:
            return {'completion': ''}

        logger.info(
            '收到补全请求: prefix_len=%s, suffix_len=%s, language=%s, trigger_mode=%s',
            len(request.prefix),
            len(request.suffix),
            request.language,
            request.trigger_mode,
        )

        result = await complete_text(
            request.prefix,
            request.suffix,
            language=request.language,
            trigger_mode=request.trigger_mode,
        )
        if result is None:
            logger.warning('LLM 服务返回 None')
            raise HTTPException(status_code=503, detail='AI Service Unavailable')
        return {'completion': result}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception('补全接口发生未捕获异常')
        raise HTTPException(status_code=500, detail=str(exc))
