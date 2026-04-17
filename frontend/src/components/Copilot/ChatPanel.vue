<script setup lang="ts">
import { nextTick, ref } from 'vue';
import { ChevronLeft, Database, Send, Sparkles, User as UserIcon } from 'lucide-vue-next';
import { sendChatMessage } from '@/api/chat';
import { useEditorStore } from '@/stores/editorStore';
import type { ChatMessage, ChatSource } from '@/types/chat';

const store = useEditorStore();
const inputValue = ref('');
const isSending = ref(false);
const messagesContainer = ref<HTMLElement | null>(null);
const messages = ref<ChatMessage[]>([
  { id: 1, role: 'assistant', content: '你好！我是你的知识副驾。', mode: 'rag' },
]);

function scrollToBottom(behavior: ScrollBehavior = 'smooth') {
  nextTick(() => {
    const container = messagesContainer.value;
    if (container) {
      container.scrollTo({ top: container.scrollHeight, behavior });
    }
  });
}

function updateAssistantMessage(messageId: number, payload: Partial<ChatMessage>) {
  const index = messages.value.findIndex((message) => message.id === messageId);
  if (index === -1) {
    return;
  }

  messages.value[index] = {
    ...messages.value[index],
    ...payload,
  };
}

async function openSource(source: ChatSource) {
  if (!source.doc_id) {
    return;
  }

  await store.loadDocument(source.doc_id);
}

function getSourceRank(source: ChatSource, index: number) {
  return source.rank ?? index + 1;
}

function getSourceConfidence(source: ChatSource) {
  return typeof source.confidence === 'number' ? `${source.confidence}%` : '未标注';
}

async function sendMessage() {
  const trimmed = inputValue.value.trim();
  if (!trimmed || isSending.value) {
    return;
  }

  const mode = 'rag';
  const now = Date.now();
  const loadingMessageId = now + 1;

  messages.value.push(
    {
      id: now,
      role: 'user',
      content: trimmed,
      mode,
    },
    {
      id: loadingMessageId,
      role: 'assistant',
      content: '',
      isLoading: true,
      mode,
      sources: [],
    },
  );

  inputValue.value = '';
  isSending.value = true;
  store.setAiThinking(true);
  scrollToBottom('auto');

  try {
    const response = await sendChatMessage(trimmed);
    updateAssistantMessage(loadingMessageId, {
      content: response.response,
      sources: response.sources ?? [],
      isLoading: false,
    });
  } catch {
    updateAssistantMessage(loadingMessageId, {
      content: '抱歉，我现在没法正常回复，请稍后再试。',
      sources: [],
      isLoading: false,
    });
  } finally {
    isSending.value = false;
    store.setAiThinking(false);
    scrollToBottom();
  }
}
</script>

<template>
  <div class="flex h-full flex-col border-l border-[#eadfce] bg-gradient-to-b from-[#fcfaf6] via-[#f7f3ec] to-[#f3ede4] shadow-[0_10px_34px_rgba(128,102,70,0.08)]">
    <div class="flex h-14 items-center gap-3 border-b border-[#ece2d3] bg-[#fffaf4]/92 px-4 backdrop-blur">
      <button
        @click="store.toggleCopilot"
        class="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full border border-[#eadfce] bg-white/88 text-stone-400 shadow-[0_4px_14px_rgba(126,96,61,0.08)] transition-colors hover:border-[#dcc4a6] hover:bg-[#fff7ee] hover:text-[#c06a49]"
      >
        <ChevronLeft class="h-3.5 w-3.5" />
      </button>

      <div class="h-5 w-px flex-shrink-0 bg-[#e9dfd2]"></div>

      <div class="flex min-w-0 items-center gap-2.5">
        <div class="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full border border-[#eadfce] bg-white/92 text-[#c47a58] shadow-[0_4px_14px_rgba(126,96,61,0.08)]">
          <Sparkles class="h-3.5 w-3.5" />
        </div>
        <div class="min-w-0">
          <div class="text-[13px] font-semibold tracking-[0.08em] text-stone-700">Atlas</div>
        </div>
      </div>
    </div>

    <div ref="messagesContainer" class="flex-1 space-y-4 overflow-y-auto px-3.5 py-4">
      <div
        v-for="message in messages"
        :key="message.id"
        class="flex items-start gap-2.5"
        :class="message.role === 'user' ? 'flex-row-reverse' : ''"
      >
        <div
          class="mt-0.5 flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full border shadow-[0_4px_14px_rgba(126,96,61,0.07)]"
          :class="message.role === 'assistant' ? 'border-[#e7ddcf] bg-white/92 text-[#c47a58]' : 'border-stone-300/80 bg-stone-100 text-stone-600'"
        >
          <Database v-if="message.role === 'assistant'" class="h-3.5 w-3.5" />
          <UserIcon v-else class="h-3.5 w-3.5" />
        </div>

        <div
          class="space-y-2"
          :class="message.role === 'assistant' ? 'min-w-0 flex-1 pr-2' : 'max-w-[84%]'"
        >
          <div
            v-if="message.role === 'user'"
            class="rounded-[20px] rounded-tr-md border border-[#d97a58] bg-[#d06847] px-4 py-3 text-[13px] leading-6 text-white shadow-[0_8px_20px_rgba(181,92,59,0.16)]"
          >
            {{ message.content }}
          </div>

          <template v-else>
            <div class="overflow-hidden rounded-[20px] rounded-tl-md border border-[#e7ddd0] bg-[#fffdf9] shadow-[0_8px_24px_rgba(125,92,52,0.05)]">
              <div class="border-b border-[#f0e7da] bg-[#faf5ee] px-4 py-2 text-[10px] font-semibold uppercase tracking-[0.14em] text-stone-500">
                知识库回答
              </div>

              <div v-if="message.isLoading" class="flex items-center justify-between gap-3 px-4 py-3.5 text-[13px] text-stone-600">
                <div>正在检索资料并组织答案</div>
                <div class="loading-dots" aria-label="正在加载">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>

              <div v-else class="px-4 py-3.5">
                <div class="assistant-body whitespace-pre-wrap text-[13px] leading-[1.9] text-stone-700">
                  {{ message.content }}
                </div>
              </div>
            </div>

            <div
              v-if="!message.isLoading && message.sources?.length"
              class="overflow-hidden rounded-[18px] border border-[#e6ddcf] bg-white/72 shadow-[0_6px_18px_rgba(125,92,52,0.04)]"
            >
              <div class="flex items-center gap-2 border-b border-[#efe6d8] bg-[#faf5ee] px-4 py-2 text-[10px] font-semibold uppercase tracking-[0.14em] text-stone-500">
                <Database class="h-3 w-3 text-[#c47a58]" />
                <span>参考引用</span>
              </div>

              <div class="space-y-2 px-4 py-3.5">
                <button
                  v-for="(source, index) in message.sources"
                  :key="`${message.id}-${index}`"
                  type="button"
                  class="w-full rounded-2xl border border-[#eee5d8] bg-[#fffaf4] px-3 py-2.5 text-left transition-colors"
                  :class="source.doc_id ? 'hover:border-[#dec8ae] hover:bg-[#fff6ea]' : ''"
                  :disabled="!source.doc_id"
                  @click="source.doc_id && openSource(source)"
                >
                  <div class="flex items-center gap-2">
                    <span
                      class="flex-shrink-0 rounded-full border border-[#ece1d2] bg-white px-1.5 py-0.5 text-[9px] font-semibold uppercase tracking-[0.04em] text-stone-500"
                    >
                      #{{ getSourceRank(source, index) }}
                    </span>
                    <span
                      class="min-w-0 flex-1 truncate text-[12px] font-medium leading-5 text-stone-700"
                      :title="source.title"
                    >
                      {{ source.title }}
                    </span>
                    <span
                      class="flex-shrink-0 rounded-full bg-[#f7efe3] px-1.5 py-0.5 text-[9px] font-semibold text-stone-500"
                    >
                      {{ getSourceConfidence(source) }}
                    </span>
                  </div>
                </button>
              </div>
            </div>
          </template>
        </div>
      </div>
    </div>

    <div class="border-t border-[#ece2d3] bg-[#fffaf4] px-3.5 py-3.5">
      <div class="relative overflow-hidden rounded-[22px] border border-[#e9dece] bg-white/92 shadow-[0_6px_18px_rgba(125,92,52,0.05)]">
        <textarea
          v-model="inputValue"
          @keydown.enter.prevent="sendMessage"
          placeholder="向知识库提问..."
          class="w-full resize-none bg-transparent p-4 pr-11 text-[13px] leading-6 text-stone-700 outline-none placeholder:text-stone-400"
          rows="3"
          :disabled="isSending"
        ></textarea>
        <button
          @click="sendMessage"
          class="absolute bottom-3 right-3 rounded-full bg-[#cf7a57] p-1.5 text-white shadow-[0_6px_16px_rgba(181,92,59,0.18)] transition-colors hover:bg-[#bb6848] disabled:cursor-not-allowed disabled:opacity-50"
          :disabled="!inputValue.trim() || isSending"
        >
          <Send class="h-3.5 w-3.5" />
        </button>
      </div>

      <div class="mt-3 flex items-center justify-between px-1">
        <div class="rounded-full border border-[#ebdcc8] bg-[#fff6ec] px-3 py-1 text-[10px] font-semibold tracking-[0.1em] text-[#be7352]">
          知识库问答
        </div>

        <span class="text-[11px] text-stone-400">
          {{ isSending ? '知识库检索中...' : '已启用本地检索' }}
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
