"""内容域接口。"""

from typing import List

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.schemas import DocumentCreate, DocumentResponse, DocumentSummary, DocumentUpdate, FolderCreate, FolderResponse
from app.services.audit.log_service import record_operation_log
from app.services.content.content_service import DocumentService, FolderService
from app.services.rag.vector_index_service import delete_document_chunks, index_document_chunks

documents_router = APIRouter(prefix='/documents', tags=['documents'])
folders_router = APIRouter(prefix='/folders', tags=['folders'])


def get_document_service(db: AsyncSession = Depends(get_db)) -> DocumentService:
    return DocumentService(db)


def get_folder_service(db: AsyncSession = Depends(get_db)) -> FolderService:
    return FolderService(db)


@documents_router.post('/', response_model=DocumentResponse)
async def create_document(
    doc: DocumentCreate,
    background_tasks: BackgroundTasks,
    doc_service: DocumentService = Depends(get_document_service),
):
    new_doc = await doc_service.create_document(doc)
    if new_doc.content:
        background_tasks.add_task(index_document_chunks, new_doc.id, new_doc.title, new_doc.content)
    background_tasks.add_task(record_operation_log, 'CREATE_DOC', 'DOCUMENT', str(new_doc.id))
    return new_doc


@documents_router.get('/', response_model=List[DocumentSummary])
async def list_documents(doc_service: DocumentService = Depends(get_document_service)):
    return await doc_service.get_all_documents()


@documents_router.put('/{doc_id}', response_model=DocumentResponse)
async def update_document(
    doc_id: int,
    doc_update: DocumentUpdate,
    background_tasks: BackgroundTasks,
    doc_service: DocumentService = Depends(get_document_service),
):
    updated_doc = await doc_service.update_document(doc_id, doc_update)
    if not updated_doc:
        raise HTTPException(status_code=404, detail='未找到文档')
    if doc_update.title is not None or doc_update.content is not None:
        if updated_doc.content:
            background_tasks.add_task(index_document_chunks, updated_doc.id, updated_doc.title, updated_doc.content)
        else:
            background_tasks.add_task(delete_document_chunks, updated_doc.id)
    background_tasks.add_task(record_operation_log, 'UPDATE_DOC', 'DOCUMENT', str(doc_id))
    return updated_doc


@documents_router.get('/{doc_id}', response_model=DocumentResponse)
async def get_document(doc_id: int, doc_service: DocumentService = Depends(get_document_service)):
    doc = await doc_service.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail='未找到文档')
    return doc


@documents_router.delete('/{doc_id}')
async def delete_document(
    doc_id: int,
    background_tasks: BackgroundTasks,
    doc_service: DocumentService = Depends(get_document_service),
):
    success = await doc_service.delete_document(doc_id)
    if not success:
        raise HTTPException(status_code=404, detail='未找到文档')
    background_tasks.add_task(delete_document_chunks, doc_id)
    background_tasks.add_task(record_operation_log, 'DELETE_DOC', 'DOCUMENT', str(doc_id))
    return {'message': '文档已删除'}


@folders_router.post('/', response_model=FolderResponse)
async def create_folder(
    folder: FolderCreate,
    background_tasks: BackgroundTasks,
    folder_service: FolderService = Depends(get_folder_service),
):
    new_folder = await folder_service.create_folder(folder)
    background_tasks.add_task(record_operation_log, 'CREATE', 'FOLDER', str(new_folder.id))
    return new_folder


@folders_router.get('/all', response_model=List[FolderResponse])
async def read_all_folders(folder_service: FolderService = Depends(get_folder_service)):
    return await folder_service.get_all_folders()


@folders_router.get('/{folder_id}', response_model=FolderResponse)
async def read_folder(folder_id: int, folder_service: FolderService = Depends(get_folder_service)):
    folder = await folder_service.get_folder(folder_id)
    if folder is None:
        raise HTTPException(status_code=404, detail='未找到文件夹')
    return folder


@folders_router.get('/{folder_id}/contents')
async def read_folder_contents(folder_id: int, folder_service: FolderService = Depends(get_folder_service)):
    return await folder_service.get_folder_contents(None if folder_id == 0 else folder_id)
