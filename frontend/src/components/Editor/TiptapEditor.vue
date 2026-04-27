<script setup lang="ts">
import { Editor, Extension } from '@tiptap/core';
import Image from '@tiptap/extension-image';
import TaskItem from '@tiptap/extension-task-item';
import TaskList from '@tiptap/extension-task-list';
import Typography from '@tiptap/extension-typography';
import { Plugin, PluginKey } from '@tiptap/pm/state';
import { Decoration, DecorationSet, EditorView } from '@tiptap/pm/view';
import StarterKit from '@tiptap/starter-kit';
import { EditorContent, useEditor } from '@tiptap/vue-3';
import MarkdownIt from 'markdown-it';
import taskLists from 'markdown-it-task-lists';
import { Markdown } from 'tiptap-markdown';
import { onBeforeUnmount, ref, watch } from 'vue';
import {
  Bold,
  CheckSquare,
  Code,
  Heading1,
  Heading2,
  Heading3,
  Italic,
  List,
  ListOrdered,
  Minus,
  Quote,
  Strikethrough,
} from 'lucide-vue-next';
import { requestCompletion } from '@/api/completion';
import { useEditorStore } from '@/stores/editorStore';

const props = defineProps<{ modelValue: string }>();
const emit = defineEmits<{
  (event: 'update:modelValue', value: string): void;
}>();

const store = useEditorStore();
const isComposing = ref(false);
const ghostText = ref('');
const ghostPos = ref<number | null>(null);
const markdownParser = new MarkdownIt({ html: true, breaks: true });

markdownParser.use(taskLists);

const AUTO_COMPLETION_DEBOUNCE_MS = 120;
const COMPLETION_REQUEST_TIMEOUT_MS = 9000;
const COMPLETION_CONTEXT_PROFILES = {
  manual: {
    prefixWindow: 896,
    suffixWindow: 160,
  },
  auto: {
    prefixWindow: 512,
    suffixWindow: 96,
  },
} as const;

type CompletionMode = keyof typeof COMPLETION_CONTEXT_PROFILES;
type MarkdownStorage = {
  markdown: {
    getMarkdown(): string;
  };
};

let completionTimer: ReturnType<typeof setTimeout> | null = null;
let completionRequestVersion = 0;
let activeCompletionAbortController: AbortController | null = null;
let isApplyingExternalContent = false;

function getMarkdown(editorInstance: Editor) {
  return (editorInstance.storage as unknown as MarkdownStorage).markdown.getMarkdown();
}

function dismissGhostText() {
  ghostText.value = '';
  ghostPos.value = null;
}

function invalidatePendingCompletion() {
  if (completionTimer) {
    clearTimeout(completionTimer);
    completionTimer = null;
  }

  if (activeCompletionAbortController) {
    activeCompletionAbortController.abort();
    activeCompletionAbortController = null;
  }

  completionRequestVersion += 1;
}

function syncGhostDecoration(editorInstance?: Editor | null) {
  const target = editorInstance ?? editorRef.value;
  if (target) {
    target.view.dispatch(target.state.tr);
  }
}

function clearGhostText(editorInstance?: Editor | null, shouldInvalidate = true) {
  if (shouldInvalidate) {
    invalidatePendingCompletion();
  } else if (completionTimer) {
    clearTimeout(completionTimer);
    completionTimer = null;
  }

  if (!ghostText.value && ghostPos.value === null) {
    return;
  }

  dismissGhostText();
  syncGhostDecoration(editorInstance);
}

async function fetchCompletion(prefix: string, suffix: string, triggerMode: CompletionMode) {
  const abortController = new AbortController();
  activeCompletionAbortController = abortController;
  const timeoutId = window.setTimeout(() => abortController.abort(), COMPLETION_REQUEST_TIMEOUT_MS);

  try {
    return await requestCompletion(
      {
        prefix,
        suffix,
        language: 'markdown',
        trigger_mode: triggerMode,
      },
      abortController.signal,
    );
  } catch (error) {
    if (!abortController.signal.aborted) {
      console.error('[GhostDebug] 请求失败:', error);
    }
    return '';
  } finally {
    window.clearTimeout(timeoutId);
    if (activeCompletionAbortController === abortController) {
      activeCompletionAbortController = null;
    }
  }
}

function isManualCompletionShortcut(event: KeyboardEvent) {
  const isMod = event.ctrlKey || event.metaKey;
  const isPlainMod = isMod && !event.altKey && !event.shiftKey;
  const isCtrlOrCmdEnter = isPlainMod && event.key === 'Enter';
  const isCtrlOrCmdSpace = isPlainMod && (event.code === 'Space' || event.key === ' ');
  const isAltSlash = !isMod && event.altKey && !event.shiftKey && (event.code === 'Slash' || event.key === '/');
  return isCtrlOrCmdEnter || isCtrlOrCmdSpace || isAltSlash;
}

async function triggerCoreCompletion(editorInstance: Editor, triggerMode: CompletionMode = 'manual') {
  if (isComposing.value) {
    return;
  }

  invalidatePendingCompletion();

  const { state } = editorInstance;
  const { from, empty } = state.selection;
  if (!empty) {
    clearGhostText(editorInstance, false);
    return;
  }

  if (ghostText.value || ghostPos.value !== null) {
    dismissGhostText();
    syncGhostDecoration(editorInstance);
  }

  const profile = COMPLETION_CONTEXT_PROFILES[triggerMode];
  const prefix = state.doc.textBetween(Math.max(0, from - profile.prefixWindow), from, '\n');
  const suffix = state.doc.textBetween(from, Math.min(state.doc.content.size, from + profile.suffixWindow), '\n');

  if (!prefix.trim()) {
    clearGhostText(editorInstance, false);
    return;
  }

  const requestVersion = completionRequestVersion;
  const completion = await fetchCompletion(prefix, suffix, triggerMode);
  if (requestVersion !== completionRequestVersion) {
    return;
  }

  if (!completion) {
    clearGhostText(editorInstance, false);
    return;
  }

  ghostText.value = completion;
  ghostPos.value = from;
  syncGhostDecoration(editorInstance);
}

function handleAutoCompletion(editorInstance: Editor) {
  if (isComposing.value || !editorInstance.isFocused) {
    return;
  }

  if (completionTimer) {
    clearTimeout(completionTimer);
  }

  const { state } = editorInstance;
  const { from, empty } = state.selection;
  if (!empty) {
    clearGhostText(editorInstance, false);
    return;
  }

  const suffixCheck = state.doc.textBetween(from, Math.min(state.doc.content.size, from + 50), '\n');
  if (suffixCheck.trim()) {
    clearGhostText(editorInstance, false);
    return;
  }

  completionTimer = setTimeout(() => {
    completionTimer = null;
    void triggerCoreCompletion(editorInstance, 'auto');
  }, AUTO_COMPLETION_DEBOUNCE_MS);
}

const ghostPluginKey = new PluginKey<DecorationSet>('ghostText');

const GhostTextExtension = Extension.create({
  name: 'ghostText',

  addProseMirrorPlugins() {
    const tiptapEditor = this.editor;

    return [
      new Plugin({
        key: ghostPluginKey,
        state: {
          init() {
            return DecorationSet.empty;
          },
          apply(transaction) {
            if (transaction.docChanged || transaction.selectionSet) {
              invalidatePendingCompletion();
              if (ghostText.value || ghostPos.value !== null) {
                dismissGhostText();
              }
              return DecorationSet.empty;
            }

            if (ghostText.value && ghostPos.value !== null) {
              const widget = document.createElement('span');
              widget.className = 'ghost-text';
              widget.textContent = ghostText.value;

              const decoration = Decoration.widget(ghostPos.value, widget, { side: 1 });
              return DecorationSet.create(transaction.doc, [decoration]);
            }

            return DecorationSet.empty;
          },
        },
        props: {
          decorations(state) {
            return ghostPluginKey.getState(state);
          },
          handleKeyDown(view: EditorView, event: KeyboardEvent) {
            if (isManualCompletionShortcut(event)) {
              event.preventDefault();
              void triggerCoreCompletion(tiptapEditor, 'manual');
              return true;
            }

            if (event.key === 'Tab' && ghostText.value && ghostPos.value !== null) {
              event.preventDefault();
              view.dispatch(view.state.tr.insertText(ghostText.value, ghostPos.value));
              clearGhostText(tiptapEditor, false);
              return true;
            }

            if (event.key === 'Escape' && ghostText.value) {
              event.preventDefault();
              clearGhostText(tiptapEditor, false);
              return true;
            }

            return false;
          },
        },
      }),
    ];
  },
});

const MarkdownPasteLogic = Extension.create({
  name: 'markdownPasteLogic',

  addProseMirrorPlugins() {
    const tiptapEditor = this.editor;

    return [
      new Plugin({
        key: new PluginKey('markdownPaste'),
        props: {
          handlePaste: (_view, event) => {
            const text = event.clipboardData?.getData('text/plain');
            const html = event.clipboardData?.getData('text/html');

            if (text) {
              const isMarkdownLike = /^(\s*#{1,6}\s|\s*>|\s*[-*]\s|\s*\d+\.\s|`{3})/.test(text);

              if (isMarkdownLike && (!html || html.length < text.length * 1.5)) {
                tiptapEditor.commands.insertContent(markdownParser.render(text));
                return true;
              }
            }

            return false;
          },
        },
      }),
    ];
  },
});

function updateStats(editorInstance: Editor) {
  const text = editorInstance.state.doc.textContent;
  const wordCount = text.trim() ? text.trim().split(/\s+/).length : 0;

  const selection = editorInstance.state.selection;
  const textBefore = editorInstance.state.doc.textBetween(0, selection.from, '\n');
  const lines = textBefore.split('\n');

  store.updateStats(wordCount, lines.length, lines[lines.length - 1].length + 1);
}

const editorRef = useEditor({
  content: props.modelValue,
  extensions: [
    StarterKit.configure({
      heading: { levels: [1, 2, 3, 4, 5, 6] },
    }),
    Markdown.configure({
      html: true,
      transformPastedText: true,
      transformCopiedText: true,
    }),
    Typography,
    Image,
    TaskList,
    TaskItem.configure({ nested: true }),
    GhostTextExtension,
    MarkdownPasteLogic,
  ],
  editorProps: {
    attributes: {
      class: 'prose prose-stone max-w-none focus:outline-none min-h-[calc(100vh-12rem)] px-12 py-10 bg-white shadow-sm mx-auto',
    },
    handleDOMEvents: {
      compositionend: () => {
        isComposing.value = false;
        setTimeout(() => {
          if (editorRef.value) {
            handleAutoCompletion(editorRef.value);
          }
        }, 0);
        return false;
      },
      compositionstart: () => {
        isComposing.value = true;
        clearGhostText(editorRef.value);
        return false;
      },
    },
  },
  onCreate: ({ editor }) => {
    updateStats(editor);
  },
  onUpdate: ({ editor, transaction }) => {
    const isExternalContentUpdate = isApplyingExternalContent;
    if (isExternalContentUpdate) {
      isApplyingExternalContent = false;
    }

    if (transaction.docChanged && !isComposing.value && !isExternalContentUpdate) {
      handleAutoCompletion(editor);
    }

    if (!isExternalContentUpdate) {
      emit('update:modelValue', getMarkdown(editor));
    }

    updateStats(editor);
  },
  onSelectionUpdate: ({ editor }) => {
    updateStats(editor);
  },
});

watch(
  () => props.modelValue,
  (newValue) => {
    if (!editorRef.value) {
      return;
    }

    const currentMarkdown = getMarkdown(editorRef.value);
    if (newValue === currentMarkdown) {
      return;
    }

    isApplyingExternalContent = true;
    clearGhostText(editorRef.value);
    editorRef.value.commands.setContent(newValue);
  },
);

const toggleBold = () => editorRef.value?.chain().focus().toggleBold().run();
const toggleItalic = () => editorRef.value?.chain().focus().toggleItalic().run();
const toggleStrike = () => editorRef.value?.chain().focus().toggleStrike().run();
const toggleCode = () => editorRef.value?.chain().focus().toggleCode().run();
const toggleH1 = () => editorRef.value?.chain().focus().toggleHeading({ level: 1 }).run();
const toggleH2 = () => editorRef.value?.chain().focus().toggleHeading({ level: 2 }).run();
const toggleH3 = () => editorRef.value?.chain().focus().toggleHeading({ level: 3 }).run();
const toggleBulletList = () => editorRef.value?.chain().focus().toggleBulletList().run();
const toggleOrderedList = () => editorRef.value?.chain().focus().toggleOrderedList().run();
const toggleTaskList = () => editorRef.value?.chain().focus().toggleTaskList().run();
const toggleBlockquote = () => editorRef.value?.chain().focus().toggleBlockquote().run();
const setHorizontalRule = () => editorRef.value?.chain().focus().setHorizontalRule().run();

onBeforeUnmount(() => {
  invalidatePendingCompletion();
  editorRef.value?.destroy();
});
</script>

<template>
  <div class="relative flex h-full w-full flex-col overflow-hidden bg-[#F9F9F9]">
    <div v-if="editorRef" class="z-10 flex flex-shrink-0 items-center justify-center border-b border-stone-200 bg-white p-2 shadow-sm">
      <div class="flex w-full max-w-5xl items-center gap-1 overflow-x-auto px-2">
        <button @click="toggleBold" :class="{ 'bg-stone-100 text-stone-900': editorRef.isActive('bold') }" class="rounded p-1.5 text-stone-500 hover:bg-stone-100" title="加粗">
          <Bold class="h-4 w-4" />
        </button>
        <button @click="toggleItalic" :class="{ 'bg-stone-100 text-stone-900': editorRef.isActive('italic') }" class="rounded p-1.5 text-stone-500 hover:bg-stone-100" title="斜体">
          <Italic class="h-4 w-4" />
        </button>
        <button @click="toggleStrike" :class="{ 'bg-stone-100 text-stone-900': editorRef.isActive('strike') }" class="rounded p-1.5 text-stone-500 hover:bg-stone-100" title="删除线">
          <Strikethrough class="h-4 w-4" />
        </button>
        <button @click="toggleCode" :class="{ 'bg-stone-100 text-stone-900': editorRef.isActive('code') }" class="rounded p-1.5 text-stone-500 hover:bg-stone-100" title="代码">
          <Code class="h-4 w-4" />
        </button>
        <div class="mx-1 h-4 w-px bg-stone-200"></div>
        <button @click="toggleH1" :class="{ 'bg-stone-100 text-stone-900': editorRef.isActive('heading', { level: 1 }) }" class="rounded p-1.5 text-stone-500 hover:bg-stone-100" title="标题 1">
          <Heading1 class="h-4 w-4" />
        </button>
        <button @click="toggleH2" :class="{ 'bg-stone-100 text-stone-900': editorRef.isActive('heading', { level: 2 }) }" class="rounded p-1.5 text-stone-500 hover:bg-stone-100" title="标题 2">
          <Heading2 class="h-4 w-4" />
        </button>
        <button @click="toggleH3" :class="{ 'bg-stone-100 text-stone-900': editorRef.isActive('heading', { level: 3 }) }" class="rounded p-1.5 text-stone-500 hover:bg-stone-100" title="标题 3">
          <Heading3 class="h-4 w-4" />
        </button>
        <div class="mx-1 h-4 w-px bg-stone-200"></div>
        <button @click="toggleBulletList" :class="{ 'bg-stone-100 text-stone-900': editorRef.isActive('bulletList') }" class="rounded p-1.5 text-stone-500 hover:bg-stone-100" title="无序列表">
          <List class="h-4 w-4" />
        </button>
        <button @click="toggleOrderedList" :class="{ 'bg-stone-100 text-stone-900': editorRef.isActive('orderedList') }" class="rounded p-1.5 text-stone-500 hover:bg-stone-100" title="有序列表">
          <ListOrdered class="h-4 w-4" />
        </button>
        <button @click="toggleTaskList" :class="{ 'bg-stone-100 text-stone-900': editorRef.isActive('taskList') }" class="rounded p-1.5 text-stone-500 hover:bg-stone-100" title="任务列表">
          <CheckSquare class="h-4 w-4" />
        </button>
        <button @click="toggleBlockquote" :class="{ 'bg-stone-100 text-stone-900': editorRef.isActive('blockquote') }" class="rounded p-1.5 text-stone-500 hover:bg-stone-100" title="引用">
          <Quote class="h-4 w-4" />
        </button>
        <div class="mx-1 h-4 w-px bg-stone-200"></div>
        <button @click="setHorizontalRule" class="rounded p-1.5 text-stone-500 hover:bg-stone-100" title="水平分割线">
          <Minus class="h-4 w-4" />
        </button>
      </div>
    </div>

    <div class="custom-scrollbar flex-1 overflow-y-auto p-8">
      <div class="mx-auto w-full max-w-4xl">
        <EditorContent :editor="editorRef" class="h-full" />
      </div>
    </div>
  </div>
</template>

<style>
.custom-scrollbar::-webkit-scrollbar { width: 6px; }
.custom-scrollbar::-webkit-scrollbar-thumb { background-color: #e5e5e5; border-radius: 3px; }

.ghost-text {
  color: #adb5bd;
  font-style: italic;
  pointer-events: none;
}

.ghost-text::after {
  content: 'Tab';
  margin-left: 4px;
  border-radius: 3px;
  background: #e9ecef;
  padding: 0 4px;
  color: #6c757d;
  font-size: 0.7em;
  vertical-align: middle;
}
</style>
