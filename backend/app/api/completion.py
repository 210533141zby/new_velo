"""
=============================================================================
文件: completion.py
描述: 代码补全接口

核心功能：
1. 代码智能提示：根据用户当前光标位置的前文 (prefix) 和后文 (suffix)，预测中间可能输入的代码。

依赖组件:
- llm_service: 提供底层的大模型调用能力。
=============================================================================
"""

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.llm_service import complete_text

router = APIRouter()
logger = logging.getLogger(__name__)

class CompletionRequest(BaseModel):
    """
    代码补全请求参数
    
    Logic Flow:
        FIM (Fill-In-the-Middle) 模式的标准参数。
        - prefix: 光标前面的代码
        - suffix: 光标后面的代码
        - language: 当前文件的语言 (python, markdown, js 等)，可选
    """
    prefix: str
    suffix: str
    language: Optional[str] = None
    trigger_mode: Optional[str] = None

@router.post("/completion")
async def generate_completion(request: CompletionRequest):
    """
    接收前端的光标前缀(prefix)和后缀(suffix)，返回补全结果
    """
    try:
        if not request.prefix:
            return {"completion": ""}

        # 2. 增加日志，确认请求到达
        logger.info(f"收到补全请求: prefix_len={len(request.prefix)}, suffix_len={len(request.suffix)}")

        result = await complete_text(request.prefix, request.suffix, trigger_mode=request.trigger_mode)
        
        if result is None:
            logger.warning("LLM 服务返回 None")
            raise HTTPException(status_code=503, detail="AI Service Unavailable")
            
        return {"completion": result}

    except Exception as e:
        # 3. 捕获所有未知错误并打印堆栈，防止前端只看到 500
        logger.exception("补全接口发生未捕获异常")
        raise HTTPException(status_code=500, detail=str(e))
