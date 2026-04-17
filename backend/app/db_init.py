"""
=============================================================================
文件: db_init.py
描述: 数据库初始化脚本

核心功能：
1. 等待数据库就绪：在应用启动时，循环检查数据库是否连得上。
2. 自动建表：连接成功后，根据 ORM 模型自动创建数据表。

Why:
应用和数据库并不一定同时就绪。如果后端启动时立即连库，偶发启动顺序问题会直接让服务退出。
所以这里保留一个有限次数的重试等待机制，保证数据库真正可用后再继续初始化。
=============================================================================
"""

import asyncio
import logging

from sqlalchemy import text

from app.database import Base, engine
from app.models import Document, Folder, Log

logger = logging.getLogger(__name__)


async def init_db():
    """
    初始化数据库。

    Logic Flow:
        第一步：等待连接。
            尝试执行 `SELECT 1`，失败则每秒重试一次，最多 60 次。
        第二步：创建表。
            连接成功后，调用 `Base.metadata.create_all` 自动建表。
    """
    logger.info('正在初始化数据库...')

    retries = 0
    max_retries = 60

    while retries < max_retries:
        try:
            async with engine.begin() as conn:
                await conn.execute(text('SELECT 1'))
            logger.info('数据库连接建立成功！')
            break
        except Exception as exc:
            retries += 1
            logger.warning(f'等待数据库就绪 ({retries}/{max_retries})... 错误: {exc}')
            await asyncio.sleep(1)

    if retries >= max_retries:
        raise Exception('连接数据库超时 (60秒)，请检查数据库服务是否正常启动。')

    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            logger.info('数据库表结构已创建/更新。')
    except Exception as exc:
        logger.error(f'数据库初始化失败: {exc}')
        raise
