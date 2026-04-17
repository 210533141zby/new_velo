<script setup lang="ts">
import { computed, onMounted, onUnmounted } from 'vue';
import {
  Bot,
  Check,
  ChevronLeft,
  ChevronRight,
  Loader2,
  PanelLeft,
  Save,
  Share,
  Sparkles,
} from 'lucide-vue-next';
import ChatPanel from './components/Copilot/ChatPanel.vue';
import TiptapEditor from './components/Editor/TiptapEditor.vue';
import StatusBar from './components/Layout/StatusBar.vue';
import FileTree from './components/Sidebar/FileTree.vue';
import { useEditorStore } from './stores/editorStore';

const store = useEditorStore();

const hasCurrentDocument = computed(() => Boolean(store.currentDocument));
const currentTitle = computed({
  get: () => store.currentDocument?.title ?? '',
  set: (value: string) => {
    store.updateTitle(value);
  },
});

function handleKeydown(event: KeyboardEvent) {
  if ((event.ctrlKey || event.metaKey) && event.key === '\\') {
    event.preventDefault();
    store.toggleCopilot();
  }

  if (event.key === 'Escape' && store.isCopilotOpen) {
    store.toggleCopilot();
  }
}

onMounted(() => {
  window.addEventListener('keydown', handleKeydown);
});

onUnmounted(() => {
  window.removeEventListener('keydown', handleKeydown);
});
</script>

<template>
  <div class="flex h-screen w-full overflow-hidden bg-[#F5F5F5] font-sans text-stone-900">
    <aside
      class="flex-shrink-0 border-r border-stone-200 bg-[#F5F5F5] transition-all duration-300 ease-in-out"
      :class="store.isSidebarOpen ? 'w-[240px]' : 'w-0 overflow-hidden border-none opacity-0'"
    >
      <FileTree />
    </aside>

    <main class="relative z-0 flex min-w-0 flex-1 flex-col bg-[#F5F5F5]">
      <div class="z-20 flex h-14 items-center justify-between border-b border-stone-200 bg-[#F5F5F5] px-6">
        <div class="flex h-full items-center space-x-4">
          <button
            @click="store.toggleSidebar"
            class="flex h-9 w-9 items-center justify-center rounded-lg text-stone-500 transition-colors hover:bg-stone-50 hover:text-stone-900"
            title="Toggle Sidebar"
          >
            <PanelLeft class="h-5 w-5" />
          </button>

          <div class="flex items-center space-x-2 text-sm font-medium text-stone-500">
            <div class="hidden cursor-pointer items-center space-x-2 transition-colors hover:text-stone-900 sm:flex">
              <Sparkles class="h-4 w-4 text-[#D06847]" />
              <span class="font-serif text-stone-900">Velo</span>
            </div>
            <span class="hidden text-stone-300 sm:inline-block">/</span>
            <input
              v-if="hasCurrentDocument"
              v-model="currentTitle"
              class="w-[300px] border-b-2 border-transparent bg-transparent px-1 py-0.5 font-serif text-lg font-bold text-stone-900 outline-none transition-all placeholder:text-stone-300 hover:border-stone-200 focus:border-[#D06847]"
              placeholder="Enter document title..."
            />
            <span v-else class="px-1 font-serif text-lg text-stone-500">No Document Selected</span>
          </div>
        </div>

        <div class="flex items-center space-x-3">
          <button
            @click="store.saveCurrentDocument"
            class="flex h-8 w-8 items-center justify-center rounded-lg transition-all hover:bg-stone-100 disabled:cursor-not-allowed disabled:opacity-50"
            :class="{ 'cursor-wait': store.saveStatus === 'saving' }"
            :disabled="!hasCurrentDocument || store.saveStatus === 'saving'"
            :title="store.saveStatus === 'saving' ? 'Saving...' : (store.saveStatus === 'saved' ? 'Saved' : 'Save')"
          >
            <Loader2 v-if="store.saveStatus === 'saving'" class="h-4 w-4 animate-spin text-stone-400" />
            <Check v-else-if="store.saveStatus === 'saved'" class="h-4 w-4 text-green-600" />
            <Save v-else class="h-4 w-4 text-stone-400" />
          </button>

          <button
            class="rounded-lg border border-transparent p-2 text-stone-500 transition-colors hover:border-stone-200 hover:bg-stone-50 hover:text-stone-900 disabled:cursor-not-allowed disabled:opacity-50"
            title="Export PDF"
            :disabled="!hasCurrentDocument"
          >
            <Share class="h-5 w-5" />
          </button>

          <div class="mx-1 h-5 w-px bg-stone-200"></div>

          <button
            @click="store.toggleCopilot"
            class="flex items-center space-x-2 rounded-lg border px-3 py-2 text-sm font-medium transition-colors"
            :class="store.isCopilotOpen ? 'border-[#D06847] bg-[#D06847]/10 text-[#D06847]' : 'border-transparent text-stone-500 hover:bg-stone-50 hover:text-stone-900'"
          >
            <Bot class="h-4 w-4" />
            <span>AI</span>
          </button>
        </div>
      </div>

      <div class="relative flex-1 overflow-hidden bg-white">
        <TiptapEditor
          :model-value="store.currentDocument?.content || ''"
          @update:model-value="store.updateContent"
        />
      </div>

      <StatusBar />

      <button
        @click="store.toggleCopilot"
        class="absolute right-0 top-1/2 z-50 flex h-12 w-5 -translate-y-1/2 items-center justify-center rounded-l-lg border border-r-0 border-stone-200 bg-white text-stone-400 shadow-md transition-all duration-300 hover:text-[#D06847]"
        title="Toggle AI Assistant (Ctrl + \\)"
      >
        <ChevronRight v-if="store.isCopilotOpen" class="h-3 w-3" />
        <ChevronLeft v-else class="h-3 w-3" />
      </button>
    </main>

    <aside
      class="flex-shrink-0 border-l border-border-line bg-surface transition-all duration-300 ease-in-out"
      :class="store.isCopilotOpen ? 'w-[350px]' : 'w-0 overflow-hidden border-none opacity-0'"
    >
      <ChatPanel />
    </aside>
  </div>
</template>
