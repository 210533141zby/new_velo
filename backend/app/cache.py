"""
=============================================================================
文件: cache.py
描述: 缓存管理器

核心功能：
1. 混合缓存策略 (Hybrid Cache)：优先使用 Redis 缓存，如果 Redis 挂了，自动降级到进程内存缓存。
2. 容错性：即使 Redis 不可用，系统也能继续运行，不会因为缓存服务崩溃而导致整个应用停摆。
3. 性能优化：为 RAG 问答、文档列表等高频查询提供毫秒级的响应速度。

依赖组件:
- redis-py: 用于连接和操作 Redis 数据库。
=============================================================================
"""

import redis.asyncio as redis
from app.core.config import settings
import time
import logging
from typing import Optional, Any, Tuple

# 设置日志记录器 (专门记录缓存相关的操作)
logger = logging.getLogger(__name__)

class CacheManager:
    """
    混合策略缓存管理器
    
    Why:
    在一个健壮的系统中，不能假设所有依赖服务（如 Redis）永远 100% 可用。
    当 Redis 出现网络波动或未安装时，我们需要一个“备胎”方案，即内存缓存。
    """
    
    def __init__(self):
        self.redis: Optional[redis.Redis] = None
        self.use_redis = False
        # 内存缓存存储结构: {键: (值, 过期时间戳)}
        self._memory_cache: dict[str, Tuple[Any, float]] = {}

    async def init_redis(self):
        """
        初始化 Redis 连接
        
        Logic Flow:
            1. 连接配置：获取 Redis 地址和端口。
            2. 极速尝试：尝试建立连接，并设置 0.5 秒的极短超时时间。
            3. 状态标记：
               - 成功：设置 `use_redis = True`。
               - 失败：捕获异常，打印警告，设置 `use_redis = False` 并启动内存缓存。
               
        Why:
            为什么要设置 0.5 秒超时？
            因为如果 Redis 连不上，默认的超时时间可能长达几十秒，这会导致整个后端启动卡死。
            我们宁愿快速失败并切换到内存模式，也不要让应用一直卡着。
        """
        try:
            # 强制使用 IP 避免某些系统下 localhost 解析慢的问题
            host = "127.0.0.1" if settings.REDIS_HOST == "localhost" else settings.REDIS_HOST
            
            # 创建异步 Redis 客户端
            client = redis.from_url(
                f"redis://{host}:{settings.REDIS_PORT}",
                encoding="utf-8",
                decode_responses=True, # 自动把字节转成字符串，方便使用
                socket_connect_timeout=0.5, # 连接超时
                socket_timeout=0.5,         # 读取超时
                retry_on_timeout=False
            )
            
            # 发送 PING 命令测试是否真的通了
            await client.ping()
            
            self.redis = client
            self.use_redis = True
            logger.info("✅ Redis 缓存: 连接成功")
            
        except (redis.ConnectionError, redis.TimeoutError, Exception) as e:
            self.use_redis = False
            # 如果 client 已经创建但 ping 失败了，也要关掉它
            if 'client' in locals() and client:
                 await client.aclose()
            self.redis = None
            logger.warning(f"⚠️ Redis 不可用: {str(e)}. 切换到内存缓存模式")

    async def close(self):
        """
        关闭资源
        应用退出时调用，确保连接被优雅释放。
        """
        if self.redis:
            await self.redis.close()
        self._memory_cache.clear()

    async def get(self, key: str) -> Optional[Any]:
        """
        获取缓存值
        
        Logic Flow:
            1. 优先查 Redis：如果当前是 Redis 模式，先去 Redis 拿。
            2. 容错处理：如果 Redis 读取报错，不直接报错，而是记录日志并尝试查内存。
            3. 兜底查内存：如果 Redis 没命中或不可用，查 `_memory_cache`。
            4. 过期检查：拿到内存值后，判断是否已经过了过期时间。
        """
        # 1. 尝试从 Redis 读取
        if self.use_redis and self.redis:
            try:
                return await self.redis.get(key)
            except Exception as e:
                logger.error(f"Redis 读取错误: {e}. 尝试内存查询")
                pass

        # 2. 尝试从内存读取
        if key in self._memory_cache:
            value, expire_at = self._memory_cache[key]
            # 检查是否过期 (expire_at 为 0 或 None 表示永不过期)
            if expire_at and time.time() > expire_at:
                del self._memory_cache[key] # 发现过期，顺手清理掉
                return None
            return value
        
        return None

    async def set(self, key: str, value: Any, ex: int = None):
        """
        设置缓存值
        
        Input:
            key: 缓存键
            value: 缓存值
            ex: 有效期 (秒)
            
        Logic Flow:
            1. 写入 Redis：如果 Redis 可用，先存进 Redis。
            2. 写入内存：作为备份，同时也存一份到本地内存。
            
        Why:
            始终写入内存虽然多占了一点点 RAM，但能保证在 Redis 突然抖动时，本地还有一份热数据可用。
        """
        # 1. 写入 Redis
        if self.use_redis and self.redis:
            try:
                await self.redis.set(key, value, ex=ex)
                return
            except Exception as e:
                logger.error(f"Redis 写入错误: {e}. 存入内存备用")
        
        # 2. 写入本地内存
        expire_at = (time.time() + ex) if ex else float('inf')
        self._memory_cache[key] = (value, expire_at)

    async def delete(self, key: str):
        """
        删除缓存
        """
        if self.use_redis and self.redis:
            try:
                await self.redis.delete(key)
            except Exception:
                pass
        
        if key in self._memory_cache:
            del self._memory_cache[key]

# 全局唯一的缓存管理器实例
redis_manager = CacheManager()
