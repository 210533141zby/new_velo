export interface DocumentRecord {
  id: number;
  title: string;
  content: string;
  created_at: string;
  updated_at: string;
  folder_id?: number | null;
}

export type SaveStatus = 'saved' | 'saving' | 'error' | 'unsaved';
