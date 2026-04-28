import { defineStore } from 'pinia';
import { ref } from 'vue';
import { createDocument, fetchDocument, listDocuments, removeDocument, updateDocument } from '@/api/documents';
import type { DocumentRecord, SaveStatus } from '@/types/document';
import { normalizeDocumentTitle } from '@/utils/document';

type CursorPosition = {
  line: number;
  col: number;
};

type CompletionStatusTone = 'idle' | 'info' | 'success' | 'settled' | 'warning' | 'error';

const AUTO_SAVE_DELAY_MS = 2000;

function sanitizeDocument(document: DocumentRecord): DocumentRecord {
  return {
    ...document,
    title: normalizeDocumentTitle(document.title),
  };
}

export const useEditorStore = defineStore('editor', () => {
  const documents = ref<DocumentRecord[]>([]);
  const currentDocument = ref<DocumentRecord | null>(null);
  const saveStatus = ref<SaveStatus>('saved');
  const isSidebarOpen = ref(true);
  const isCopilotOpen = ref(false);
  const wordCount = ref(0);
  const cursorPosition = ref<CursorPosition>({ line: 1, col: 1 });
  const isAiThinking = ref(false);
  const completionStatusMessage = ref('补全就绪');
  const completionStatusTone = ref<CompletionStatusTone>('idle');

  let saveTimer: number | null = null;
  const loadedDocumentIds = new Set<number>();
  const loadingDocumentPromises = new Map<number, Promise<DocumentRecord | null>>();

  function clearScheduledSave() {
    if (saveTimer) {
      window.clearTimeout(saveTimer);
      saveTimer = null;
    }
  }

  function queueSave() {
    clearScheduledSave();
    saveTimer = window.setTimeout(() => {
      saveTimer = null;
      void saveCurrentDocument();
    }, AUTO_SAVE_DELAY_MS);
  }

  function syncDocumentInList(document: DocumentRecord) {
    const normalized = sanitizeDocument(document);
    const index = documents.value.findIndex((item) => item.id === normalized.id);

    if (index === -1) {
      documents.value.push(normalized);
    } else {
      documents.value[index] = normalized;
    }

    return normalized;
  }

  function markDocumentAsLoaded(documentId: number) {
    loadedDocumentIds.add(documentId);
  }

  function selectDocument(document: DocumentRecord | null) {
    clearScheduledSave();
    currentDocument.value = document ? sanitizeDocument(document) : null;
  }

  async function fetchDocumentDetail(documentId: number, options?: { select?: boolean }) {
    const shouldSelect = options?.select ?? false;
    const cachedDocument = documents.value.find((document) => document.id === documentId);

    if (cachedDocument && loadedDocumentIds.has(documentId)) {
      if (shouldSelect) {
        selectDocument(cachedDocument);
        saveStatus.value = 'saved';
      }
      return cachedDocument;
    }

    const existingRequest = loadingDocumentPromises.get(documentId);
    if (existingRequest) {
      const document = await existingRequest;
      if (shouldSelect && document) {
        selectDocument(document);
        saveStatus.value = 'saved';
      }
      return document;
    }

    const request = (async () => {
      try {
        const document = syncDocumentInList(await fetchDocument(documentId));
        markDocumentAsLoaded(document.id);
        return document;
      } catch (error) {
        console.error('Failed to load document', error);
        return null;
      } finally {
        loadingDocumentPromises.delete(documentId);
      }
    })();

    loadingDocumentPromises.set(documentId, request);
    const document = await request;

    if (shouldSelect && document) {
      selectDocument(document);
      saveStatus.value = 'saved';
    }

    return document;
  }

  async function prefetchDocumentDetails(documentIds: number[]) {
    const pendingIds = documentIds.filter(
      (documentId) => !loadedDocumentIds.has(documentId) && !loadingDocumentPromises.has(documentId),
    );

    if (!pendingIds.length) {
      return;
    }

    await Promise.all(pendingIds.map((documentId) => fetchDocumentDetail(documentId)));
  }

  async function fetchDocuments() {
    try {
      const existingById = new Map(documents.value.map((document) => [document.id, document]));
      const nextDocuments = (await listDocuments()).map((document) => {
        const normalized = sanitizeDocument(document);
        const cached = existingById.get(normalized.id);
        if (cached && loadedDocumentIds.has(normalized.id)) {
          return sanitizeDocument({
            ...normalized,
            content: cached.content,
          });
        }
        return normalized;
      });
      documents.value = nextDocuments;

      if (!nextDocuments.length) {
        selectDocument(null);
        return;
      }

      if (!currentDocument.value) {
        const initialDocument = await loadDocument(nextDocuments[0].id);
        const initialId = initialDocument?.id ?? nextDocuments[0].id;
        void prefetchDocumentDetails(nextDocuments.map((document) => document.id).filter((id) => id !== initialId));
        return;
      }

      const currentSummary = nextDocuments.find((document) => document.id === currentDocument.value?.id);
      if (!currentSummary) {
        const fallbackDocument = await loadDocument(nextDocuments[0].id);
        const fallbackId = fallbackDocument?.id ?? nextDocuments[0].id;
        void prefetchDocumentDetails(nextDocuments.map((document) => document.id).filter((id) => id !== fallbackId));
        return;
      }

      currentDocument.value = {
        ...currentDocument.value,
        ...currentSummary,
      };

      void prefetchDocumentDetails(
        nextDocuments.map((document) => document.id).filter((id) => id !== currentDocument.value?.id),
      );
    } catch (error) {
      console.error('Failed to fetch documents', error);
    }
  }

  async function createNewDocument() {
    try {
      const newDocument = sanitizeDocument(
        await createDocument({
          title: '',
          content: '',
        }),
      );

      documents.value.unshift(newDocument);
      markDocumentAsLoaded(newDocument.id);
      selectDocument(newDocument);
      saveStatus.value = 'saved';
      return newDocument;
    } catch (error) {
      console.error('Failed to create document', error);
      return null;
    }
  }

  async function loadDocument(documentId: number) {
    if (currentDocument.value?.id === documentId) {
      return currentDocument.value;
    }

    return fetchDocumentDetail(documentId, { select: true });
  }

  async function deleteDocument(documentId: number) {
    try {
      await removeDocument(documentId);
      loadedDocumentIds.delete(documentId);
      documents.value = documents.value.filter((document) => document.id !== documentId);

      if (currentDocument.value?.id !== documentId) {
        return;
      }

      if (!documents.value.length) {
        selectDocument(null);
        return;
      }

      await loadDocument(documents.value[0].id);
    } catch (error) {
      console.error('Failed to delete document', error);
    }
  }

  async function saveCurrentDocument() {
    if (!currentDocument.value) {
      return;
    }

    clearScheduledSave();

    const snapshot = {
      ...currentDocument.value,
    };

    saveStatus.value = 'saving';

    try {
      const response = await updateDocument(snapshot.id, {
        title: snapshot.title,
        content: snapshot.content,
      });

      const persistedDocument = sanitizeDocument({
        ...snapshot,
        ...response,
        title: response.title ?? snapshot.title,
        content: response.content ?? snapshot.content,
      });

      if (currentDocument.value?.id === snapshot.id) {
        const hasPendingLocalChanges =
          currentDocument.value.title !== snapshot.title || currentDocument.value.content !== snapshot.content;

        const nextCurrentDocument = hasPendingLocalChanges
          ? sanitizeDocument({
              ...currentDocument.value,
              updated_at: persistedDocument.updated_at,
              created_at: persistedDocument.created_at,
              folder_id: persistedDocument.folder_id,
            })
          : persistedDocument;

        currentDocument.value = nextCurrentDocument;
        syncDocumentInList(nextCurrentDocument);
        markDocumentAsLoaded(nextCurrentDocument.id);
        saveStatus.value = hasPendingLocalChanges ? 'unsaved' : 'saved';

        if (hasPendingLocalChanges) {
          queueSave();
        }

        return;
      }

      syncDocumentInList(persistedDocument);
      markDocumentAsLoaded(persistedDocument.id);
    } catch (error) {
      if (currentDocument.value?.id === snapshot.id) {
        saveStatus.value = 'error';
      }
      console.error('Failed to save document', error);
    }
  }

  function updateContent(content: string) {
    if (!currentDocument.value || currentDocument.value.content === content) {
      return;
    }

    currentDocument.value.content = content;
    saveStatus.value = 'unsaved';
    queueSave();
  }

  function updateTitle(title: string) {
    if (!currentDocument.value || currentDocument.value.title === title) {
      return;
    }

    currentDocument.value.title = title;
    saveStatus.value = 'unsaved';
    queueSave();
  }

  function updateStats(count: number, line: number, col: number) {
    wordCount.value = count;
    cursorPosition.value = { line, col };
  }

  function setAiThinking(nextValue: boolean) {
    isAiThinking.value = nextValue;
  }

  function setCompletionStatus(message: string, tone: CompletionStatusTone = 'info') {
    completionStatusMessage.value = message;
    completionStatusTone.value = tone;
  }

  function toggleSidebar() {
    isSidebarOpen.value = !isSidebarOpen.value;
  }

  function toggleCopilot() {
    isCopilotOpen.value = !isCopilotOpen.value;
  }

  return {
    documents,
    currentDocument,
    saveStatus,
    isSidebarOpen,
    isCopilotOpen,
    wordCount,
    cursorPosition,
    isAiThinking,
    completionStatusMessage,
    completionStatusTone,
    fetchDocuments,
    createDocument: createNewDocument,
    loadDocument,
    saveCurrentDocument,
    deleteDocument,
    updateContent,
    updateTitle,
    updateStats,
    setAiThinking,
    setCompletionStatus,
    toggleSidebar,
    toggleCopilot,
  };
});
