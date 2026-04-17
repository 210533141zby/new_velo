import client from './client';
import type { ChatResponsePayload } from '@/types/chat';

const CHAT_REQUEST_TIMEOUT_MS = 120000;

export function sendChatMessage(content: string) {
  return client.post<never, ChatResponsePayload>(
    '/agent/chat',
    {
      messages: [{ role: 'user', content }],
    },
    {
      timeout: CHAT_REQUEST_TIMEOUT_MS,
    },
  );
}
