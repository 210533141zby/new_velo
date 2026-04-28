import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.completion.completion_service import complete_text_detailed

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
            logger.info(
                '忽略空前缀补全请求',
                extra={
                    'extra_data': {
                        'event': 'completion_request_skipped',
                        'reason': 'empty_prefix',
                        'suffix_len': len(request.suffix),
                        'language': request.language or '',
                        'trigger_mode': request.trigger_mode or '',
                    }
                },
            )
            return {'completion': '', 'reason': 'empty_prefix'}

        logger.info(
            '收到补全请求',
            extra={
                'extra_data': {
                    'event': 'completion_api_request_received',
                    'prefix_len': len(request.prefix),
                    'suffix_len': len(request.suffix),
                    'language': request.language or '',
                    'trigger_mode': request.trigger_mode or '',
                    'prefix_tail': request.prefix[-80:].replace('\n', '\\n'),
                    'suffix_head': request.suffix[:80].replace('\n', '\\n'),
                }
            },
        )

        result = await complete_text_detailed(
            request.prefix,
            request.suffix,
            language=request.language,
            trigger_mode=request.trigger_mode,
        )
        if result['completion'] is None:
            logger.warning(
                'LLM 服务返回 None',
                extra={'extra_data': {'event': 'completion_api_invalid_result'}},
            )
            raise HTTPException(status_code=503, detail='AI Service Unavailable')
        logger.info(
            '补全请求处理完成',
            extra={
                'extra_data': {
                    'event': 'completion_api_response_sent',
                    'status': 'accepted' if result['completion'] else 'rejected',
                    'reason': result['reason'] or '',
                    'completion_len': len(result['completion'] or ''),
                    'completion_preview': (result['completion'] or '')[:80].replace('\n', '\\n'),
                }
            },
        )
        return {'completion': result['completion'], 'reason': result['reason']}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(
            '补全接口发生未捕获异常',
            extra={'extra_data': {'event': 'completion_api_unhandled_exception'}},
        )
        raise HTTPException(status_code=500, detail=str(exc))
