import client from './client';
import type { DocumentRecord } from '@/types/document';

type DocumentDraft = Pick<DocumentRecord, 'title' | 'content'>;

export function listDocuments() {
  return client.get<never, DocumentRecord[]>('/documents/');
}

export function fetchDocument(documentId: number) {
  return client.get<never, DocumentRecord>(`/documents/${documentId}`);
}

export function createDocument(payload: DocumentDraft) {
  return client.post<never, DocumentRecord>('/documents/', payload);
}

export function updateDocument(documentId: number, payload: DocumentDraft) {
  return client.put<never, Partial<DocumentRecord>>(`/documents/${documentId}`, payload);
}

export function removeDocument(documentId: number) {
  return client.delete(`/documents/${documentId}`);
}
