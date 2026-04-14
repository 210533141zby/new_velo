<script setup lang="ts">
import { nextTick, ref } from 'vue';
import request from '@/api/request';
import { useEditorStore } from '../../stores/editorStore';
import { X, Send, Bot, User as UserIcon, Database, Sparkles, ArrowUpLeft } from 'lucide-vue-next';

const store = useEditorStore();
const inputValue = ref('');
const isRAGMode = ref(false);
const isSending = ref(false);
const messagesContainer = ref<HTMLElement | null>(null);

type ChatSource = {
  title: string;
  doc_id?: number | null;
};

type ChatMessage = {
  id: number;
  role: 'user' | 'assistant';
  content: string;
  sources?: ChatSource[];
  isLoading?: boolean;
  mode?: 'rag' | 'chat';
};

const messages = ref<ChatMessage[]>([
  { id: 1, role: 'assistant', content: '你好！我是你的知识副驾。', mode: 'chat' }
]);

const scrollToBottom = (behavior: ScrollBehavior = 'smooth') => {
  nextTick(() => {
    const container = messagesContainer.value;
    if (container) {
      container.scrollTo({ top: container.scrollHeight, behavior });
    }
  });
};

const updateAssistantMessage = (messageId: number, payload: Partial<ChatMessage>) => {
  const index = messages.value.findIndex(msg => msg.id === messageId);
  if (index === -1) return;

  messages.value[index] = {
    ...messages.value[index],
    ...payload,
  };
};

const openSource = async (source: ChatSource) => {
  if (!source.doc_id) return;
  try {
    await store.loadDocument(source.doc_id);
  } catch (error) {
    console.error('Failed to open source document', error);
  }
};

const sendMessage = async () => {
  const trimmed = inputValue.value.trim();
  if (!trimmed || isSending.value) return;

  const mode: 'rag' | 'chat' = isRAGMode.value ? 'rag' : 'chat';
  const now = Date.now();
  const userMessage: ChatMessage = {
    id: now,
    role: 'user',
    content: trimmed,
    mode,
  };
  const loadingMessageId = now + 1;

  messages.value.push(userMessage, {
    id: loadingMessageId,
    role: 'assistant',
    content: '',
    isLoading: true,
    mode,
    sources: [],
  });

  inputValue.value = '';
  isSending.value = true;
  scrollToBottom('auto');

  try {
    const response = await request.post('/agent/chat', {
      messages: [{ role: 'user', content: trimmed }],
      use_rag: isRAGMode.value,
    }) as { response: string; sources?: ChatSource[] };

    updateAssistantMessage(loadingMessageId, {
      content: response.response,
      sources: response.sources ?? [],
      isLoading: false,
    });
  } catch (_error) {
    updateAssistantMessage(loadingMessageId, {
      content: '抱歉，我现在没法正常回复，请稍后再试。',
      sources: [],
      isLoading: false,
    });
  } finally {
    isSending.value = false;
    scrollToBottom();
  }
};
</script>

<template>
  <div class="h-full flex flex-col border-l border-stone-200 bg-gradient-to-b from-[#fbf8f2] via-[#f6f1e8] to-[#f2ebe0] shadow-xl">
    <div class="flex h-16 items-center justify-between border-b border-[#e5dccf] bg-[#fffaf2]/95 px-4 backdrop-blur">
      <div class="flex items-center gap-3">
        <div class="flex h-9 w-9 items-center justify-center rounded-full border border-[#e4d6c1] bg-white text-[#c56a46] shadow-sm">
          <Sparkles class="h-4 w-4" />
        </div>
        <div>
          <div class="text-sm font-semibold tracking-[0.12em] text-stone-700">Atlas</div>
        </div>
      </div>

      <button
        @click="store.toggleCopilot"
        class="rounded-md p-1 text-stone-400 transition-colors hover:bg-[#d06847]/10 hover:text-[#d06847]"
      >
        <X class="h-5 w-5" />
      </button>
    </div>

    <div ref="messagesContainer" class="flex-1 space-y-5 overflow-y-auto px-4 py-5">
      <div
        v-for="msg in messages"
        :key="msg.id"
        class="flex items-start gap-3"
        :class="msg.role === 'user' ? 'flex-row-reverse' : ''"
      >
        <div
          class="mt-1 flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-full border shadow-sm"
          :class="msg.role === 'assistant'
            ? 'border-[#dfd2bf] bg-white text-[#c56a46]'
            : 'border-stone-300 bg-stone-200 text-stone-700'"
        >
          <Database v-if="msg.role === 'assistant' && msg.mode === 'rag'" class="h-4 w-4" />
          <Bot v-else-if="msg.role === 'assistant'" class="h-4 w-4" />
          <UserIcon v-else class="h-4 w-4" />
        </div>

        <div class="max-w-[88%] space-y-2">
          <div v-if="msg.role === 'user'" class="rounded-2xl rounded-tr-sm border border-[#d06847] bg-[#d06847] px-4 py-3 text-sm leading-6 text-white shadow-sm">
            {{ msg.content }}
          </div>

          <template v-else>
            <div class="overflow-hidden rounded-2xl rounded-tl-sm border border-[#e1d7ca] bg-[#fffdf8] shadow-[0_12px_30px_rgba(125,92,52,0.06)]">
              <div class="border-b border-[#eee3d3] bg-[#f8f1e6] px-4 py-2 text-[11px] font-semibold uppercase tracking-[0.18em] text-stone-500">
                {{ msg.mode === 'rag' ? '知识库回答' : '模型回答' }}
              </div>

              <div v-if="msg.isLoading" class="flex items-center justify-between gap-3 px-4 py-4 text-sm text-stone-600">
                <div>{{ msg.mode === 'rag' ? '正在检索资料并组织答案' : '正在生成回答' }}</div>
                <div class="loading-dots" aria-label="正在加载">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>

              <div v-else class="px-4 py-4">
                <div class="assistant-body whitespace-pre-wrap text-[14px] leading-7 text-stone-700">
                  {{ msg.content }}
                </div>
              </div>
            </div>

            <div
              v-if="!msg.isLoading && msg.sources?.length"
              class="overflow-hidden rounded-2xl border border-[#ddd2c2] bg-white/80 shadow-[0_8px_24px_rgba(125,92,52,0.05)]"
            >
              <div class="flex items-center gap-2 border-b border-[#eee4d6] bg-[#f8f3ec] px-4 py-2 text-[11px] font-semibold uppercase tracking-[0.18em] text-stone-500">
                <Database class="h-3.5 w-3.5 text-[#c56a46]" />
                <span>参考引用</span>
              </div>

              <div class="space-y-2 px-4 py-4">
                <button
                  v-for="(source, index) in msg.sources"
                  :key="`${msg.id}-${index}`"
                  type="button"
                  class="flex w-full items-center justify-between rounded-xl border border-[#ece3d6] bg-[#fffaf3] px-3 py-3 text-left transition-colors"
                  :class="source.doc_id ? 'hover:border-[#d8b28f] hover:bg-[#fff4e8]' : ''"
                  :disabled="!source.doc_id"
                  @click="source.doc_id && openSource(source)"
                >
                  <span class="text-sm font-semibold text-stone-800">{{ source.title }}</span>
                  <ArrowUpLeft v-if="source.doc_id" class="h-4 w-4 text-[#b76543]" />
                </button>
              </div>
            </div>
          </template>
        </div>
      </div>
    </div>

    <div class="border-t border-[#e5dccf] bg-[#fffaf2] px-4 py-4">
      <div class="relative overflow-hidden rounded-2xl border border-[#e4d7c6] bg-white shadow-sm">
        <textarea
          v-model="inputValue"
          @keydown.enter.prevent="sendMessage"
          :placeholder="isRAGMode ? '向知识库提问...' : '输入指令或问题...'"
          class="w-full resize-none bg-transparent p-4 pr-12 text-sm leading-6 text-stone-800 outline-none placeholder:text-stone-400"
          rows="3"
          :disabled="isSending"
        ></textarea>
        <button
          @click="sendMessage"
          class="absolute bottom-3 right-3 rounded-full bg-[#d06847] p-2 text-white shadow-sm transition-colors hover:bg-[#b55739] disabled:cursor-not-allowed disabled:opacity-50"
          :disabled="!inputValue.trim() || isSending"
        >
          <Send class="h-3.5 w-3.5" />
        </button>
      </div>

      <div class="mt-3 flex items-center justify-between px-1">
        <div class="flex items-center gap-2">
          <button
            @click="isRAGMode = !isRAGMode"
            class="relative inline-flex h-5 w-9 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none"
            :class="isRAGMode ? 'bg-[#d06847]' : 'bg-stone-300'"
            title="点击切换知识库模式"
          >
            <span class="sr-only">使用知识库</span>
            <span
              aria-hidden="true"
              class="pointer-events-none inline-block h-4 w-4 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out"
              :class="isRAGMode ? 'translate-x-4' : 'translate-x-0'"
            ></span>
          </button>
          <span
            class="cursor-pointer select-none text-xs font-semibold tracking-[0.14em]"
            :class="isRAGMode ? 'text-[#d06847]' : 'text-stone-500'"
            @click="isRAGMode = !isRAGMode"
          >
            {{ isRAGMode ? '知识库模式' : '通用模式' }}
          </span>
        </div>

        <span class="text-xs text-stone-400">
          {{ isSending ? (isRAGMode ? '知识库检索中...' : '模型生成中...') : (isRAGMode ? '已启用本地检索' : '通用模型') }}
        </span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.assistant-body {
  font-family: 'Source Han Sans SC', 'PingFang SC', 'Noto Sans CJK SC', 'Microsoft YaHei', sans-serif;
  letter-spacing: 0.01em;
}

.loading-dots {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
}

.loading-dots span {
  width: 0.42rem;
  height: 0.42rem;
  border-radius: 9999px;
  background: #d06847;
  animation: loading-bounce 1s infinite ease-in-out;
}

.loading-dots span:nth-child(2) {
  animation-delay: 0.15s;
}

.loading-dots span:nth-child(3) {
  animation-delay: 0.3s;
}

@keyframes loading-bounce {
  0%, 80%, 100% {
    transform: translateY(0);
    opacity: 0.35;
  }
  40% {
    transform: translateY(-3px);
    opacity: 1;
  }
}
</style>
