<script setup lang="ts">
/**
 * TiptapEditor.vue - 核心富文本编辑器组件 (修复版)
 */
import { useEditor, EditorContent } from '@tiptap/vue-3'
import { Extension, Editor } from '@tiptap/core'
import { Plugin, PluginKey } from '@tiptap/pm/state'
import { Decoration, DecorationSet } from '@tiptap/pm/view'
import { EditorView } from '@tiptap/pm/view'
// @ts-ignore
import StarterKit from '@tiptap/starter-kit'
import { Markdown } from 'tiptap-markdown'
import Typography from '@tiptap/extension-typography'
import Image from '@tiptap/extension-image'
import TaskList from '@tiptap/extension-task-list'
import TaskItem from '@tiptap/extension-task-item'
import { watch, onBeforeUnmount, ref } from 'vue'
import { useEditorStore } from '../../stores/editorStore'
import { useDebounceFn } from '@vueuse/core'
import MarkdownIt from 'markdown-it'
// @ts-ignore
import taskLists from 'markdown-it-task-lists'
import {
  Bold, Italic, Strikethrough, Code, Heading1, Heading2, Heading3,
  List, ListOrdered, Quote, Minus, CheckSquare
} from 'lucide-vue-next'

const props = defineProps<{ modelValue: string }>()
const emit = defineEmits(['update:modelValue'])
const store = useEditorStore()
const isComposing = ref(false)
const md = new MarkdownIt({ html: true, breaks: true })
md.use(taskLists)

// --- 1. Ghost Text (幽灵文本) 状态管理 ---
const ghostText = ref('')
const ghostPos = ref<number | null>(null)
let completionTimer: ReturnType<typeof setTimeout> | null = null
let completionRequestVersion = 0
let activeCompletionAbortController: AbortController | null = null
let isApplyingExternalContent = false

type CompletionMode = 'manual' | 'auto'

const AUTO_COMPLETION_DEBOUNCE_MS = 120
const COMPLETION_REQUEST_TIMEOUT_MS = 9000
const COMPLETION_CONTEXT_PROFILES: Record<CompletionMode, { prefixWindow: number; suffixWindow: number }> = {
  manual: {
    prefixWindow: 1280,
    suffixWindow: 260,
  },
  auto: {
    prefixWindow: 640,
    suffixWindow: 128,
  },
}

const dismissGhostText = () => {
  ghostText.value = ''
  ghostPos.value = null
}

const invalidatePendingCompletion = () => {
  if (completionTimer) {
    clearTimeout(completionTimer)
    completionTimer = null
  }

  if (activeCompletionAbortController) {
    activeCompletionAbortController.abort()
    activeCompletionAbortController = null
  }

  completionRequestVersion += 1
}

const syncGhostDecoration = (editorInstance?: Editor | null) => {
  const target = editorInstance ?? editor.value
  if (target) {
    target.view.dispatch(target.state.tr)
  }
}

const clearGhostText = (editorInstance?: Editor | null, invalidatePending = true) => {
  if (invalidatePending) {
    invalidatePendingCompletion()
  } else if (completionTimer) {
    clearTimeout(completionTimer)
    completionTimer = null
  }

  if (!ghostText.value && ghostPos.value === null) {
    return
  }

  dismissGhostText()
  syncGhostDecoration(editorInstance)
}

const fetchCompletion = async (prefix: string, suffix: string, triggerMode: CompletionMode) => {
  const abortController = new AbortController()
  activeCompletionAbortController = abortController
  const timeoutId = window.setTimeout(() => abortController.abort(), COMPLETION_REQUEST_TIMEOUT_MS)

  try {
    const response = await fetch('/api/v1/completion', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        prefix,
        suffix,
        language: 'markdown',
        trigger_mode: triggerMode,
      }),
      signal: abortController.signal,
    })

    if (!response.ok) {
      throw new Error(`Completion request failed: ${response.status}`)
    }

    const result = await response.json() as { completion?: string }
    return result.completion ?? ''
  } catch (error) {
    if (!abortController.signal.aborted) {
      console.error('[GhostDebug] 请求失败:', error)
    }
    return ''
  } finally {
    window.clearTimeout(timeoutId)
    if (activeCompletionAbortController === abortController) {
      activeCompletionAbortController = null
    }
  }
}

const triggerCoreCompletion = async (editorInstance: Editor, triggerMode: CompletionMode = 'manual') => {
  if (isComposing.value || !editorInstance) return

  invalidatePendingCompletion()

  const { state } = editorInstance
  const { from, empty } = state.selection
  if (!empty) {
    clearGhostText(editorInstance, false)
    return
  }

  if (ghostText.value || ghostPos.value !== null) {
    dismissGhostText()
    syncGhostDecoration(editorInstance)
  }

  const contextProfile = COMPLETION_CONTEXT_PROFILES[triggerMode]
  const prefix = state.doc.textBetween(Math.max(0, from - contextProfile.prefixWindow), from, '\n')
  const suffix = state.doc.textBetween(from, Math.min(state.doc.content.size, from + contextProfile.suffixWindow), '\n')
  if (!prefix.trim()) {
    clearGhostText(editorInstance, false)
    return
  }

  const requestVersion = completionRequestVersion
  const completion = await fetchCompletion(prefix, suffix, triggerMode)
  if (requestVersion !== completionRequestVersion) return

  if (!completion) {
    clearGhostText(editorInstance, false)
    return
  }

  ghostText.value = completion
  ghostPos.value = from
  syncGhostDecoration(editorInstance)
}

const handleAutoCompletion = (editorInstance: Editor) => {
  if (isComposing.value || !editorInstance) return
  if (!editorInstance.isFocused) return

  if (completionTimer) {
    clearTimeout(completionTimer)
  }

  const { state } = editorInstance
  const { from, empty } = state.selection
  if (!empty) {
    clearGhostText(editorInstance, false)
    return
  }

  const suffixCheck = state.doc.textBetween(from, Math.min(state.doc.content.size, from + 50), '\n')
  const isAtEnd = !suffixCheck.trim()
  if (!isAtEnd) {
    clearGhostText(editorInstance, false)
    return
  }

  completionTimer = setTimeout(() => {
    completionTimer = null
    void triggerCoreCompletion(editorInstance, 'auto')
  }, AUTO_COMPLETION_DEBOUNCE_MS)
}

// --- 2. Ghost Text Tiptap Extension ---
const ghostPluginKey = new PluginKey<DecorationSet>('ghostText')

const GhostTextExtension = Extension.create({
  name: 'ghostText',

  addProseMirrorPlugins() {
    const editorInstance = this.editor

    return [
      new Plugin({
        key: ghostPluginKey,
        state: {
          init() {
            return DecorationSet.empty
          },
          apply(tr) {
            if (tr.docChanged || tr.selectionSet) {
              invalidatePendingCompletion()
              if (ghostText.value || ghostPos.value !== null) {
                dismissGhostText()
              }
              return DecorationSet.empty
            }

            if (ghostText.value && ghostPos.value !== null) {
              const widget = document.createElement('span')
              widget.className = 'ghost-text'
              widget.textContent = ghostText.value

              const deco = Decoration.widget(ghostPos.value, widget, {
                side: 1
              })
              return DecorationSet.create(tr.doc, [deco])
            }

            return DecorationSet.empty
          }
        },
        props: {
          decorations(state) {
            return ghostPluginKey.getState(state)
          },
          handleKeyDown(view: EditorView, event: KeyboardEvent) {
            const isMod = event.ctrlKey || event.metaKey
            const isAlt = event.altKey
            const isManualShortcut =
              (isMod && isAlt) ||
              (isMod && event.code === 'Space') ||
              (isMod && event.key === 'Enter')

            if (isManualShortcut) {
              event.preventDefault()
              void triggerCoreCompletion(editorInstance, 'manual')
              return true
            }

            if (event.key === 'Tab' && ghostText.value && ghostPos.value !== null) {
              event.preventDefault()
              const tr = view.state.tr.insertText(ghostText.value, ghostPos.value)
              view.dispatch(tr)
              clearGhostText(editorInstance, false)
              return true
            }

            if (event.key === 'Escape' && ghostText.value) {
              event.preventDefault()
              clearGhostText(editorInstance, false)
              return true
            }

            return false
          }
        }
      })
    ]
  }
})

// --- 3. Markdown Paste Logic Extension ---
const MarkdownPasteLogic = Extension.create({
  name: 'markdownPasteLogic',

  addProseMirrorPlugins() {
    const editor = (this as any).editor as Editor

    return [
      new Plugin({
        key: new PluginKey('markdownPaste'),
        props: {
          handlePaste: (_view, event) => {
            const text = event.clipboardData?.getData('text/plain')
            const html = event.clipboardData?.getData('text/html')

            if (text) {
              const isMarkdownLike = /^(\s*#{1,6}\s|\s*>|\s*[-*]\s|\s*\d+\.\s|`{3})/.test(text)

              if (isMarkdownLike && (!html || html.length < text.length * 1.5)) {
                const parsedHtml = md.render(text)
                editor.commands.insertContent(parsedHtml)
                return true
              }
            }
            return false
          }
        }
      })
    ]
  }
})

const debouncedSave = useDebounceFn(() => {
  // Save logic
}, 1000)

const updateStats = (editorInstance: Editor) => {
  const text = editorInstance.state.doc.textContent
  const wordCount = text.trim() ? text.trim().split(/\s+/).length : 0

  const selection = editorInstance.state.selection
  const { from } = selection
  const textBefore = editorInstance.state.doc.textBetween(0, from, '\n')
  const lines = textBefore.split('\n')
  const line = lines.length
  const col = lines[lines.length - 1].length + 1

  store.updateStats(wordCount, line, col)
}

// --- 初始化编辑器 ---
const editor = useEditor({
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
        isComposing.value = false
        setTimeout(() => {
          if (editor.value) {
            handleAutoCompletion(editor.value)
          }
        }, 0)
        return false
      },
      compositionstart: () => {
        isComposing.value = true
        clearGhostText(editor.value)
        return false
      }
    },
  },
  onUpdate: ({ editor, transaction }) => {
    const isExternalContentUpdate = isApplyingExternalContent

    if (isExternalContentUpdate) {
      isApplyingExternalContent = false
    }

    if (transaction.docChanged && !isComposing.value && !isExternalContentUpdate) {
      handleAutoCompletion(editor)
    }

    const markdown = (editor.storage as any).markdown.getMarkdown()
    emit('update:modelValue', markdown)
    store.updateContent(markdown)
    debouncedSave()
    updateStats(editor)
  },
  onSelectionUpdate: ({ editor }) => {
    updateStats(editor)
  },
})

watch(() => props.modelValue, (newValue) => {
  if (editor.value) {
    const currentMarkdown = (editor.value.storage as any).markdown.getMarkdown()
    if (newValue !== currentMarkdown) {
      isApplyingExternalContent = true
      clearGhostText(editor.value)
      editor.value.commands.setContent(newValue)
      queueMicrotask(() => {
        isApplyingExternalContent = false
      })
    }
  }
})

// Toolbar Actions
const toggleBold = () => editor.value?.chain().focus().toggleBold().run()
const toggleItalic = () => editor.value?.chain().focus().toggleItalic().run()
const toggleStrike = () => editor.value?.chain().focus().toggleStrike().run()
const toggleCode = () => editor.value?.chain().focus().toggleCode().run()
const toggleH1 = () => editor.value?.chain().focus().toggleHeading({ level: 1 }).run()
const toggleH2 = () => editor.value?.chain().focus().toggleHeading({ level: 2 }).run()
const toggleH3 = () => editor.value?.chain().focus().toggleHeading({ level: 3 }).run()
const toggleBulletList = () => editor.value?.chain().focus().toggleBulletList().run()
const toggleOrderedList = () => editor.value?.chain().focus().toggleOrderedList().run()
const toggleTaskList = () => editor.value?.chain().focus().toggleTaskList().run()
const toggleBlockquote = () => editor.value?.chain().focus().toggleBlockquote().run()
const setHorizontalRule = () => editor.value?.chain().focus().setHorizontalRule().run()

onBeforeUnmount(() => {
  invalidatePendingCompletion()
  editor.value?.destroy()
})
</script>

<template>
  <div class="flex flex-col h-full w-full bg-[#F9F9F9] relative overflow-hidden">
    <div v-if="editor" class="flex items-center justify-center p-2 border-b border-stone-200 bg-white z-10 shadow-sm flex-shrink-0">
      <div class="flex items-center gap-1 overflow-x-auto max-w-5xl w-full px-2">
        <button @click="toggleBold" :class="{ 'bg-stone-100 text-stone-900': editor.isActive('bold') }" class="p-1.5 rounded hover:bg-stone-100 text-stone-500" title="加粗">
          <Bold class="w-4 h-4" />
        </button>
        <button @click="toggleItalic" :class="{ 'bg-stone-100 text-stone-900': editor.isActive('italic') }" class="p-1.5 rounded hover:bg-stone-100 text-stone-500" title="斜体">
          <Italic class="w-4 h-4" />
        </button>
        <button @click="toggleStrike" :class="{ 'bg-stone-100 text-stone-900': editor.isActive('strike') }" class="p-1.5 rounded hover:bg-stone-100 text-stone-500" title="删除线">
          <Strikethrough class="w-4 h-4" />
        </button>
        <button @click="toggleCode" :class="{ 'bg-stone-100 text-stone-900': editor.isActive('code') }" class="p-1.5 rounded hover:bg-stone-100 text-stone-500" title="代码">
          <Code class="w-4 h-4" />
        </button>
        <div class="w-px h-4 bg-stone-200 mx-1"></div>
        <button @click="toggleH1" :class="{ 'bg-stone-100 text-stone-900': editor.isActive('heading', { level: 1 }) }" class="p-1.5 rounded hover:bg-stone-100 text-stone-500" title="标题 1">
          <Heading1 class="w-4 h-4" />
        </button>
        <button @click="toggleH2" :class="{ 'bg-stone-100 text-stone-900': editor.isActive('heading', { level: 2 }) }" class="p-1.5 rounded hover:bg-stone-100 text-stone-500" title="标题 2">
          <Heading2 class="w-4 h-4" />
        </button>
        <button @click="toggleH3" :class="{ 'bg-stone-100 text-stone-900': editor.isActive('heading', { level: 3 }) }" class="p-1.5 rounded hover:bg-stone-100 text-stone-500" title="标题 3">
          <Heading3 class="w-4 h-4" />
        </button>
        <div class="w-px h-4 bg-stone-200 mx-1"></div>
        <button @click="toggleBulletList" :class="{ 'bg-stone-100 text-stone-900': editor.isActive('bulletList') }" class="p-1.5 rounded hover:bg-stone-100 text-stone-500" title="无序列表">
          <List class="w-4 h-4" />
        </button>
        <button @click="toggleOrderedList" :class="{ 'bg-stone-100 text-stone-900': editor.isActive('orderedList') }" class="p-1.5 rounded hover:bg-stone-100 text-stone-500" title="有序列表">
          <ListOrdered class="w-4 h-4" />
        </button>
        <button @click="toggleTaskList" :class="{ 'bg-stone-100 text-stone-900': editor.isActive('taskList') }" class="p-1.5 rounded hover:bg-stone-100 text-stone-500" title="任务列表">
          <CheckSquare class="w-4 h-4" />
        </button>
        <button @click="toggleBlockquote" :class="{ 'bg-stone-100 text-stone-900': editor.isActive('blockquote') }" class="p-1.5 rounded hover:bg-stone-100 text-stone-500" title="引用">
          <Quote class="w-4 h-4" />
        </button>
        <div class="w-px h-4 bg-stone-200 mx-1"></div>
        <button @click="setHorizontalRule" class="p-1.5 rounded hover:bg-stone-100 text-stone-500" title="水平分割线">
          <Minus class="w-4 h-4" />
        </button>
      </div>
    </div>

    <div class="flex-1 overflow-y-auto custom-scrollbar p-8">
      <div class="max-w-4xl mx-auto w-full">
        <editor-content
          :editor="editor"
          class="h-full"
        />
      </div>
    </div>
  </div>
</template>

<style>
.custom-scrollbar::-webkit-scrollbar { width: 6px; }
.custom-scrollbar::-webkit-scrollbar-thumb { background-color: #e5e5e5; border-radius: 3px; }
.rotate-90 { transform: rotate(90deg); }

.ghost-text {
  color: #adb5bd;
  font-style: italic;
  pointer-events: none;
}

.ghost-text::after {
  content: 'Tab';
  font-size: 0.7em;
  background: #e9ecef;
  border-radius: 3px;
  padding: 0 4px;
  margin-left: 4px;
  color: #6c757d;
  vertical-align: middle;
}
</style>
