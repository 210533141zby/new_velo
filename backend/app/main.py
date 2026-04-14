"""
=============================================================================
文件: main.py
描述: 应用入口文件

核心功能：
1. 应用启动：配置 FastAPI 实例，加载中间件，注册路由。
2. 生命周期管理：处理应用启动时 (Startup) 和关闭时 (Shutdown) 的钩子事件。
3. 请求日志：通过中间件记录所有 HTTP 请求的详细信息。

依赖组件:
- FastAPI: Web 框架核心。
- Uvicorn: ASGI 服务器 (通常在启动脚本中调用)。
=============================================================================
"""

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api.api import api_router
from app.cache import redis_manager
from app.core.config import settings
from app.database import AsyncSessionLocal, engine
from app.db_init import init_db
from app.logger import InterceptHandler, ip_address_ctx, logger, request_id_ctx, user_id_ctx
from app.services.agent_service import AgentService
from app.services.llm_service import close_completion_http_client

# =============================================================================
# 生命周期管理 (Lifecycle)
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理器

    Logic Flow (执行流程):
        启动阶段 (Startup):
        1. 日志接管：让 Loguru 拦截并接管 uvicorn 的标准日志，统一日志格式。
        2. 资源初始化：建立 Redis 连接池。
        3. 数据库准备：执行数据库初始化检查 (如自动建表)。

        Yield:
        应用正常运行...

        关闭阶段 (Shutdown):
        1. 资源释放：优雅关闭 Redis 连接和数据库连接池，防止资源泄露。

    Why:
    FastAPI 推荐使用 lifespan 来替代旧版的 @app.on_event("startup")。
    它能更安全地管理那些需要在整个应用运行期间一直存在的资源（比如数据库连接）。
    """
    logging.getLogger().handlers = [InterceptHandler()]
    logging.getLogger("uvicorn.access").handlers = [InterceptHandler()]
    logging.getLogger("uvicorn.error").handlers = [InterceptHandler()]

    logger.info("应用正在启动...", extra={"extra_data": {"event": "startup"}})

    await redis_manager.init_redis()

    try:
        await init_db()
        print("数据库初始化完成，连接成功！", flush=True)
        logger.info("数据库连接成功", extra={"extra_data": {"event": "db_connected"}})
    except Exception:
        logger.exception("数据库初始化失败，后端停止启动")
        raise

    try:
        async with AsyncSessionLocal() as session:
            await AgentService(session).ensure_bootstrap_index()
    except Exception:
        logger.exception("RAG 索引预热失败")

    yield

    logger.info("应用正在关闭...", extra={"extra_data": {"event": "shutdown"}})
    await close_completion_http_client()
    await redis_manager.close()
    await engine.dispose()
    logger.info("数据库连接池已关闭", extra={"extra_data": {"event": "db_disconnected"}})

# =============================================================================
# 应用实例配置
# =============================================================================

app = FastAPI(
    title=settings.PROJECT_NAME,
    lifespan=lifespan,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# =============================================================================
# 中间件配置 (Middleware)
# =============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    请求日志中间件

    Logic Flow:
        1. 预处理：生成唯一的 Request ID，记录客户端 IP 和请求开始时间。
        2. 传递：调用 `call_next(request)` 将请求传给具体的路由处理函数。
        3. 后处理：拿到响应结果，计算总耗时，记录日志。
        4. 异常捕获：如果处理过程中崩了，记录错误日志并重新抛出异常。

    Why:
        在一个真实的生产系统中，知道“谁”在“什么时候”发了“什么请求”以及“花了多久”是非常重要的。
        这个中间件就像是一个监控探头，自动记录所有进出的流量。
    """
    start_time = time.time()

    request_id = str(uuid.uuid4())
    request_id_ctx.set(request_id)

    client_ip = request.client.host if request.client else "unknown"
    ip_address_ctx.set(client_ip)

    user_id = "Admin"
    user_id_ctx.set(user_id)

    url = request.url.path
    method = request.method

    if not url.startswith("/static") and url != "/health":
        logger.info(f"请求开始: {method} {url}", extra={
            "extra_data": {
                "event": "request_start",
                "method": method,
                "url": url,
                "user_agent": request.headers.get("user-agent"),
            }
        })

    try:
        response = await call_next(request)
        process_time = (time.time() - start_time) * 1000

        if not url.startswith("/static") and url != "/health":
            logger.info(f"请求结束: {response.status_code}", extra={
                "duration": process_time,
                "extra_data": {
                    "event": "request_end",
                    "status_code": response.status_code,
                    "method": method,
                    "url": url
                }
            })

        response.headers["X-Request-ID"] = request_id
        return response

    except Exception as e:
        process_time = (time.time() - start_time) * 1000
        logger.error(f"请求处理异常: {str(e)}", exc_info=True, extra={
            "duration": process_time,
            "extra_data": {
                "event": "request_error",
                "method": method,
                "url": url
            }
        })
        raise e

# =============================================================================
# 路由注册
# =============================================================================

app.include_router(api_router, prefix=settings.API_V1_STR)

# =============================================================================
# 根路由
# =============================================================================

@app.get("/")
async def read_root():
    """
    根路径检查
    """
    return {"message": f"Welcome to {settings.PROJECT_NAME} API"}


@app.get("/health")
async def health_check():
    return {"status": "ok"}
