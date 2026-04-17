declare module 'markdown-it-task-lists' {
  import type MarkdownIt from 'markdown-it';

  type MarkdownItTaskListsPlugin = (md: MarkdownIt, options?: Record<string, unknown>) => void;

  const plugin: MarkdownItTaskListsPlugin;
  export default plugin;
}
