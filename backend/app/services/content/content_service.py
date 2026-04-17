"""内容域业务服务。"""

import json

from sqlalchemy import func, select

from app.cache import redis_manager
from app.logger import logger
from app.models import Document, Folder
from app.schemas import DocumentCreate, DocumentUpdate, FolderCreate


class DocumentService:
    def __init__(self, db):
        self.db = db

    async def create_document(self, doc_data: DocumentCreate) -> Document:
        new_doc = Document(title=doc_data.title, content=doc_data.content, folder_id=doc_data.folder_id)
        self.db.add(new_doc)
        await self.db.commit()
        await self.db.refresh(new_doc)

        await self._invalidate_cache()
        logger.info(
            f'创建文档: {new_doc.title}',
            extra={'extra_data': {'event': 'document_created', 'document_id': new_doc.id, 'folder_id': new_doc.folder_id, 'title': new_doc.title}},
        )
        return new_doc

    async def get_document(self, doc_id: int) -> Document | None:
        result = await self.db.execute(select(Document).where(Document.id == doc_id, Document.is_active == True))
        return result.scalar_one_or_none()

    async def update_document(self, doc_id: int, doc_update: DocumentUpdate) -> Document | None:
        db_doc = await self.get_document(doc_id)
        if not db_doc:
            return None

        update_data = doc_update.dict(exclude_unset=True)
        for key, value in update_data.items():
            setattr(db_doc, key, value)
        await self.db.commit()
        await self.db.refresh(db_doc)
        await self._invalidate_cache()
        return db_doc

    async def delete_document(self, doc_id: int) -> bool:
        db_doc = await self.get_document(doc_id)
        if not db_doc:
            return False

        db_doc.is_active = False
        await self.db.commit()
        await self._invalidate_cache()
        return True

    async def get_all_documents(self):
        try:
            cached_docs = await redis_manager.get('documents_list')
            if cached_docs:
                return json.loads(cached_docs)
        except Exception as exc:
            logger.error(f'缓存读取错误: {exc}', exc_info=True)

        result = await self.db.execute(
            select(
                Document.id,
                Document.title,
                func.substr(Document.content, 1, 200).label('content'),
                Document.created_at,
                Document.updated_at,
                Document.folder_id,
            )
            .where(Document.is_active == True)
            .order_by(Document.updated_at.desc())
        )
        rows = result.all()
        docs_data = [
            {
                'id': row.id,
                'title': row.title,
                'content': row.content,
                'created_at': row.created_at.isoformat() if row.created_at else None,
                'updated_at': row.updated_at.isoformat() if row.updated_at else None,
                'folder_id': row.folder_id,
            }
            for row in rows
        ]

        try:
            await redis_manager.set('documents_list', json.dumps(docs_data), ex=300)
        except Exception as exc:
            logger.error(f'缓存写入错误: {exc}', exc_info=True)
        return docs_data

    async def _invalidate_cache(self):
        try:
            await redis_manager.delete('documents_list')
        except Exception as exc:
            logger.error(f'缓存清理错误: {exc}', exc_info=True)


class FolderService:
    def __init__(self, db):
        self.db = db

    async def create_folder(self, folder: FolderCreate) -> Folder:
        db_folder = Folder(title=folder.title, parent_id=folder.parent_id)
        self.db.add(db_folder)
        await self.db.commit()
        await self.db.refresh(db_folder)
        return db_folder

    async def get_all_folders(self) -> list[Folder]:
        result = await self.db.execute(select(Folder).where(Folder.is_active == True))
        return result.scalars().all()

    async def get_folder(self, folder_id: int) -> Folder | None:
        result = await self.db.execute(select(Folder).where(Folder.id == folder_id))
        return result.scalar_one_or_none()

    async def get_folder_contents(self, folder_id: int | None = None) -> dict:
        folder_query = select(Folder).where(Folder.is_active == True)
        folder_query = folder_query.where(Folder.parent_id == (None if folder_id is None else folder_id))

        doc_query = select(Document).where(Document.is_active == True)
        doc_query = doc_query.where(Document.folder_id == (None if folder_id is None else folder_id))

        folders = (await self.db.execute(folder_query)).scalars().all()
        docs = (await self.db.execute(doc_query)).scalars().all()
        return {'folders': folders, 'documents': docs}
