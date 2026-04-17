<script setup lang="ts">
import { onMounted, ref } from 'vue';
import { File, Plus, Settings, Trash2, User } from 'lucide-vue-next';
import ConfirmModal from '../modals/ConfirmModal.vue';
import { useEditorStore } from '@/stores/editorStore';
import { formatDocumentUpdatedAt, getDocumentDisplayTitle } from '@/utils/document';

const store = useEditorStore();
const isDeleteModalVisible = ref(false);
const fileToDeleteId = ref<number | null>(null);

function closeDeleteModal() {
  isDeleteModalVisible.value = false;
  fileToDeleteId.value = null;
}

async function createNewFile() {
  await store.createDocument();
}

function openDocument(documentId: number) {
  void store.loadDocument(documentId);
}

function confirmDelete(documentId: number, event: Event) {
  event.stopPropagation();
  fileToDeleteId.value = documentId;
  isDeleteModalVisible.value = true;
}

async function handleDelete() {
  if (fileToDeleteId.value === null) {
    return;
  }

  await store.deleteDocument(fileToDeleteId.value);
  closeDeleteModal();
}

onMounted(() => {
  void store.fetchDocuments();
});
</script>

<template>
  <div class="flex h-full flex-col border-r border-stone-200 bg-[#F2F3F5] font-['Nunito']">
    <div class="flex h-14 items-center justify-between border-b border-stone-200 bg-[#F2F3F5] px-5">
      <div class="flex items-center space-x-3">
        <div class="flex h-6 w-6 items-center justify-center rounded-lg bg-[#D06847] shadow-sm">
          <span class="text-xs font-bold italic text-white">V</span>
        </div>
        <span class="font-serif text-lg font-bold tracking-tight text-stone-900">Velo</span>
      </div>
      <button
        @click="createNewFile"
        class="rounded-lg p-1.5 text-stone-500 transition-all duration-200 hover:bg-stone-200 hover:text-[#D06847]"
        title="Create New File"
      >
        <Plus class="h-5 w-5" />
      </button>
    </div>

    <div class="flex-1 overflow-y-auto bg-[#F2F3F5] px-3 py-4">
      <div class="mb-3 flex items-center justify-between px-2 font-mono text-xs font-bold uppercase tracking-wider text-stone-900">
        <span>Workspace</span>
        <span class="rounded border border-stone-300 bg-stone-200 px-1.5 py-0.5 text-[10px] font-medium text-stone-900">Local</span>
      </div>
      <div class="space-y-1">
        <div
          v-for="document in store.documents"
          :key="document.id"
          class="group relative flex cursor-pointer items-center rounded-lg border border-transparent px-3 py-2.5 text-sm transition-all duration-200"
          :class="store.currentDocument?.id === document.id ? 'bg-white font-bold text-[#D06847] shadow-sm' : 'font-medium text-stone-900 hover:bg-stone-200 hover:text-black'"
          @click="openDocument(document.id)"
        >
          <div
            class="absolute bottom-2 left-0 top-2 w-1 rounded-r bg-[#D06847] transition-opacity"
            :class="store.currentDocument?.id === document.id ? 'opacity-100' : 'opacity-0'"
          ></div>

          <File
            class="mr-3 h-4 w-4 flex-shrink-0 transition-colors"
            :class="store.currentDocument?.id === document.id ? 'text-[#D06847]' : 'text-stone-500 group-hover:text-stone-700'"
          />

          <div class="flex min-w-0 flex-1 flex-col">
            <span class="truncate leading-tight">{{ getDocumentDisplayTitle(document.title) }}</span>
            <span class="mt-0.5 truncate text-[10px] text-stone-400 transition-colors group-hover:text-stone-500">
              {{ formatDocumentUpdatedAt(document.updated_at) }}
            </span>
          </div>

          <button
            @click="(event) => confirmDelete(document.id, event)"
            class="rounded-md p-1.5 text-stone-400 opacity-0 transition-all hover:scale-105 hover:bg-red-50 hover:text-red-500 group-hover:opacity-100"
            title="Delete File"
          >
            <Trash2 class="h-3.5 w-3.5" />
          </button>
        </div>
      </div>
    </div>

    <ConfirmModal
      v-model:visible="isDeleteModalVisible"
      message="确定要删除这个文档吗？此操作无法撤销。"
      @confirm="handleDelete"
    />

    <div class="border-t border-stone-200 bg-[#F5F5F5] p-4">
      <div class="flex items-center justify-between">
        <div class="flex items-center space-x-3">
          <div class="flex h-8 w-8 items-center justify-center overflow-hidden rounded-full border border-stone-300 bg-stone-200">
            <User class="h-5 w-5 text-stone-500" />
          </div>
          <div class="flex flex-col">
            <span class="text-xs font-bold text-stone-700">User</span>
            <span class="text-[10px] text-stone-400">Pro Plan</span>
          </div>
        </div>
        <button class="rounded-lg p-2 text-stone-400 transition-colors hover:bg-stone-200 hover:text-stone-600">
          <Settings class="h-5 w-5" />
        </button>
      </div>
    </div>
  </div>
</template>
