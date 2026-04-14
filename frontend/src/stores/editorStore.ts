/**
 * Editor Store (Pinia)
 * 
 * 描述:
 * 管理编辑器全局状态的核心 Store。负责文档的 CRUD、自动保存、
 * 以及 UI 状态 (侧边栏、Copilot) 的切换。
 * 
 * 核心设计:
 * 1. **Optimistic UI**: 大部分操作 (如更新内容) 先更新本地 State，再异步调用 API，
 *    提供流畅的用户体验。
 * 2. **Auto Save**: 使用 `useDebounceFn` 实现防抖自动保存 (2秒延迟)，
 *    避免频繁请求后端。
 * 3. **Single Source of Truth**: `currentDocument` 是当前编辑器渲染的唯一数据源。
 */

import { defineStore } from 'pinia';
import { ref } from 'vue';
import { useDebounceFn } from '@vueuse/core';
import request from '@/api/request';

export interface Document {
  id: number;
  title: string;
  content: string;
  created_at: string;
  updated_at: string;
  folder_id?: number | null;
}

export type SaveStatus = 'saved' | 'saving' | 'error' | 'unsaved';

export const useEditorStore = defineStore('editor', () => {
  // ==========================================================================
  // State
  // ==========================================================================
  
  // 所有文档列表 (侧边栏显示)
  const documents = ref<Document[]>([]);
  // 当前正在编辑的文档
  const currentDocument = ref<Document | null>(null);
  // 保存状态 (用于 UI 反馈: "已保存", "保存中"...)
  const saveStatus = ref<SaveStatus>('saved');
  // UI 开关状态
  const isSidebarOpen = ref(true);
  const isCopilotOpen = ref(false);
  // 统计信息
  const wordCount = ref(0);
  const cursorPosition = ref({ line: 1, col: 1 });
  // AI 思考状态
  const isAiThinking = ref(false);

  // ==========================================================================
  // Actions
  // ==========================================================================

  /**
   * 获取文档列表
   * 
   * Logic Flow:
   * 1. 调用 GET /documents/ 接口。
   * 2. 更新 `documents` 列表。
   * 3. 如果当前没有选中任何文档且列表不为空，自动加载第一个文档。
   */
  async function fetchDocuments() {
    try {
      const res = await request.get('/documents/');
      documents.value = res as Document[];
      if (!currentDocument.value && documents.value.length > 0) {
        await loadDocument(documents.value[0].id);
      }
    } catch (e) {
      console.error('Failed to fetch documents', e);
    }
  }

  /**
   * 创建新文档
   * 
   * Logic Flow:
   * 1. 调用 POST /documents/ 接口创建空文档。
   * 2. 将返回的新文档插入到 `documents` 列表头部。
   * 3. 立即将其设为 `currentDocument`。
   */
  async function createDocument() {
    try {
      const newDoc = await request.post('/documents/', {
        title: '',
        content: '',
      }) as Document;
      // Force title to be empty if it comes back as Untitled or null (defensive fix)
      // We check for various forms of 'Untitled' to be safe
      const titleLower = (newDoc.title || '').toLowerCase().trim();
      if (!newDoc.title || titleLower === 'untitled' || titleLower === 'untitled document') {
        newDoc.title = '';
      }
      documents.value.unshift(newDoc);
      currentDocument.value = newDoc;
      saveStatus.value = 'saved';
    } catch (e) {
      console.error('Failed to create document', e);
    }
  }

  /**
   * 加载指定文档
   */
  async function loadDocument(id: number) {
    try {
      const doc = await request.get(`/documents/${id}`) as Document;
      currentDocument.value = doc;
      saveStatus.value = 'saved';
    } catch (e) {
      console.error('Failed to load document', e);
    }
  }

  /**
   * 删除文档
   * 
   * Logic Flow:
   * 1. 调用 DELETE 接口。
   * 2. 从本地 `documents` 列表中移除。
   * 3. 如果删除的是当前正在编辑的文档:
   *    - 尝试切换到列表中的第一个文档。
   *    - 如果列表为空，置空 `currentDocument`。
   */
  async function deleteDocument(id: number) {
    try {
      await request.delete(`/documents/${id}`);
      const idx = documents.value.findIndex(d => d.id === id);
      if (idx !== -1) {
        documents.value.splice(idx, 1);
      }
      
      // If deleted current document, switch to another one
      if (currentDocument.value?.id === id) {
        if (documents.value.length > 0) {
          await loadDocument(documents.value[0].id);
        } else {
          currentDocument.value = null;
        }
      }
    } catch (e) {
      console.error('Failed to delete document', e);
    }
  }

  /**
   * 保存当前文档 (核心)
   * 
   * Logic Flow:
   * 1. 设置状态为 'saving'。
   * 2. 调用 PUT 接口发送 title 和 content。
   * 3. 更新本地 store 中的数据 (合并后端返回的可能更新的字段)。
   * 4. 设置状态为 'saved'。
   */
  async function saveCurrentDocument() {
    if (!currentDocument.value) return;
    saveStatus.value = 'saving';
    try {
      const updatedDoc = await request.put(`/documents/${currentDocument.value.id}`, {
        title: currentDocument.value.title,
        content: currentDocument.value.content,
      }) as Partial<Document>;
      currentDocument.value = { ...currentDocument.value, ...updatedDoc };
      
      const idx = documents.value.findIndex(d => d.id === currentDocument.value?.id);
      if (idx !== -1) documents.value[idx] = currentDocument.value;
      
      saveStatus.value = 'saved';
    } catch (e) {
      saveStatus.value = 'error';
      console.error('Failed to save', e);
    }
  }

  /**
   * 防抖自动保存
   * 延迟 2000ms 执行 saveCurrentDocument
   */
  const debouncedSave = useDebounceFn(() => {
    saveCurrentDocument();
  }, 2000);

  /**
   * 更新内容 (由编辑器调用)
   * 
   * Logic:
   * 1. 立即更新内存中的 content。
   * 2. 标记为 'unsaved'。
   * 3. 触发防抖保存。
   */
  function updateContent(content: string) {
    if (!currentDocument.value) return;
    currentDocument.value.content = content;
    saveStatus.value = 'unsaved'; // Set to unsaved immediately on change
    debouncedSave();
  }

  /**
   * 更新标题
   */
  function updateTitle(title: string) {
    if (!currentDocument.value) return;
    currentDocument.value.title = title;
    saveStatus.value = 'unsaved';
    debouncedSave();
  }

  /**
   * 更新统计信息
   */
  function updateStats(count: number, line: number, col: number) {
    wordCount.value = count;
    cursorPosition.value = { line, col };
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
    fetchDocuments,
    createDocument,
    loadDocument,
    saveCurrentDocument,
    deleteDocument,
    updateContent,
    updateTitle,
    updateStats,
    toggleSidebar,
    toggleCopilot
  };
});
