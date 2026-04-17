export interface CompletionRequestPayload {
  prefix: string;
  suffix: string;
  language: string;
  trigger_mode: 'manual' | 'auto';
}

export async function requestCompletion(payload: CompletionRequestPayload, signal: AbortSignal) {
  const response = await fetch('/api/v1/completion', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
    signal,
  });

  if (!response.ok) {
    throw new Error(`Completion request failed: ${response.status}`);
  }

  const result = (await response.json()) as { completion?: string };
  return result.completion ?? '';
}
