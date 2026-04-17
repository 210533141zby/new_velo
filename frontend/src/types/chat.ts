export type ChatMode = 'rag' | 'chat';

export interface ChatSource {
  title: string;
  doc_id?: number | null;
  rank?: number;
  confidence?: number;
}

export interface ChatMessage {
  id: number;
  role: 'user' | 'assistant';
  content: string;
  sources?: ChatSource[];
  isLoading?: boolean;
  mode?: ChatMode;
}

export interface ChatResponsePayload {
  response: string;
  sources?: ChatSource[];
}
