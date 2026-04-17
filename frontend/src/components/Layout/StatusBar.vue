<script setup lang="ts">
import { computed } from 'vue';
import { useEditorStore } from '@/stores/editorStore';

const store = useEditorStore();
const aiIndicatorClass = computed(() =>
  store.isAiThinking ? 'bg-[#D06847] ring-2 ring-[#D06847]/15' : 'bg-emerald-500 ring-2 ring-emerald-100',
);
const aiStatusText = computed(() => (store.isAiThinking ? 'Velo AI Thinking' : 'Velo AI Ready'));
</script>

<template>
  <div class="flex h-8 select-none items-center justify-between border-t border-border-line bg-surface px-4 font-sans text-[11px] text-text-sub">
    <div class="flex items-center space-x-4">
      <div class="cursor-pointer rounded-md border border-transparent px-2 py-1 font-medium transition-colors hover:border-neutral-200 hover:bg-neutral-50">
        Words: {{ store.wordCount }}
      </div>
      <div class="cursor-pointer rounded-md border border-transparent px-2 py-1 font-medium transition-colors hover:border-neutral-200 hover:bg-neutral-50">
        Ln {{ store.cursorPosition.line }}, Col {{ store.cursorPosition.col }}
      </div>
      <div class="cursor-pointer rounded-md border border-transparent px-2 py-1 font-medium transition-colors hover:border-neutral-200 hover:bg-neutral-50">
        UTF-8
      </div>
    </div>

    <div class="flex items-center space-x-4">
      <div class="hidden cursor-pointer rounded-md px-2 py-1 font-medium transition-colors hover:bg-neutral-50 sm:block">
        Markdown
      </div>
      <div class="flex items-center space-x-2 rounded-md border border-border-line bg-surface px-2 py-1 shadow-sm transition-colors hover:bg-neutral-50">
        <span class="h-1.5 w-1.5 rounded-full" :class="aiIndicatorClass"></span>
        <span class="font-bold text-text-main">{{ aiStatusText }}</span>
      </div>
    </div>
  </div>
</template>
