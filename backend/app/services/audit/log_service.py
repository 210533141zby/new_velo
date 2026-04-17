"""操作日志服务。"""

from __future__ import annotations

from app.database import AsyncSessionLocal
from app.logger import logger
from app.models import Log


class LogService:
    def __init__(self, db):
        self.db = db

    async def log_operation(
        self,
        action: str,
        resource_type: str,
        resource_id: str,
        user_id: str = 'system',
        details: dict | None = None,
    ):
        try:
            log = Log(
                user_id=user_id,
                action=action,
                resource_type=resource_type,
                resource_id=str(resource_id),
                details=details,
            )
            self.db.add(log)
            await self.db.commit()
        except Exception as exc:
            logger.error(
                f'日志记录失败: {exc}',
                exc_info=True,
                extra={'extra_data': {'event': 'log_operation_failed', 'resource_type': resource_type, 'resource_id': resource_id}},
            )


async def record_operation_log(
    action: str,
    resource_type: str,
    resource_id: str,
    user_id: str = 'system',
    details: dict | None = None,
):
    async with AsyncSessionLocal() as session:
        await LogService(session).log_operation(action, resource_type, resource_id, user_id=user_id, details=details)
