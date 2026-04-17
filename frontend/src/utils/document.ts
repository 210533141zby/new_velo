const SHANGHAI_FORMATTER = new Intl.DateTimeFormat('zh-CN', {
  timeZone: 'Asia/Shanghai',
  hour12: false,
  year: 'numeric',
  month: '2-digit',
  day: '2-digit',
  hour: '2-digit',
  minute: '2-digit',
  second: '2-digit',
});

function normalizeBackendDate(dateString: string) {
  if (dateString.includes('Z') || /[+-]\d{2}:\d{2}$/.test(dateString)) {
    return dateString;
  }

  return `${dateString.replace(' ', 'T')}Z`;
}

export function normalizeDocumentTitle(title?: string | null) {
  const normalized = (title ?? '').trim();
  const titleLower = normalized.toLowerCase();

  if (!normalized || titleLower === 'untitled' || titleLower === 'untitled document') {
    return '';
  }

  return normalized;
}

export function getDocumentDisplayTitle(title?: string | null) {
  return normalizeDocumentTitle(title) || 'Untitled';
}

export function formatDocumentUpdatedAt(dateString?: string | null) {
  if (!dateString) {
    return '';
  }

  const date = new Date(normalizeBackendDate(dateString));
  if (Number.isNaN(date.getTime())) {
    return '';
  }

  return SHANGHAI_FORMATTER.format(date);
}
