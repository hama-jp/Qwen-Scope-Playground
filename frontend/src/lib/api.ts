import type {
  CorpusStatus,
  EncodeRequest,
  EncodeResponse,
  FeatureSamplesResponse,
  FeatureSearchRequest,
  FeatureSearchResponse,
  GenerateRequest,
  MetaResponse,
  Note,
  NoteCreate,
  SseEvent
} from './types';

export async function fetchMeta(): Promise<MetaResponse> {
  const r = await fetch('/api/meta');
  if (!r.ok) throw new Error(`meta: ${r.status}`);
  return r.json();
}

export async function loadLayer(layer: number): Promise<{ layer: number; loaded: boolean }> {
  const r = await fetch('/api/load_layer', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ layer })
  });
  if (!r.ok) throw new Error(`load_layer: ${r.status} ${await r.text()}`);
  return r.json();
}

/**
 * POST /api/generate and consume the SSE response via fetch streaming.
 *
 * The standard EventSource API only supports GET, so we parse SSE frames
 * ourselves. Each frame is delimited by a blank line ("\n\n"); within a frame
 * we read `event:` and `data:` lines.
 */
export async function streamGenerate(
  req: GenerateRequest,
  onEvent: (e: SseEvent) => void,
  signal: AbortSignal
): Promise<void> {
  const r = await fetch('/api/generate', {
    method: 'POST',
    headers: { 'content-type': 'application/json', accept: 'text/event-stream' },
    body: JSON.stringify(req),
    signal
  });
  if (!r.ok || !r.body) {
    throw new Error(`generate: ${r.status} ${r.statusText} ${await r.text().catch(() => '')}`);
  }

  const reader = r.body.getReader();
  const decoder = new TextDecoder('utf-8');
  let buf = '';

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      // Normalise CRLF -> LF so the rest of the parser only deals with LF.
      // sse-starlette emits CRLF terminators per the SSE spec; the spec also
      // accepts plain LF, so any server is covered by this normalisation.
      buf += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');

      // Process complete frames (blank line as delimiter).
      while (true) {
        const idx = buf.indexOf('\n\n');
        if (idx < 0) break;
        const frame = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        const ev = parseFrame(frame);
        if (ev) onEvent(ev);
      }
    }
  } finally {
    try {
      reader.releaseLock();
    } catch {
      /* noop */
    }
  }
}

// ─── v0.2 ─────────────────────────────────────────────────────────────────

export async function fetchCorpusStatus(): Promise<CorpusStatus> {
  const r = await fetch('/api/corpus_status');
  if (!r.ok) throw new Error(`corpus_status: ${r.status}`);
  return r.json();
}

export async function fetchFeatureSamples(
  layer: number,
  featureIdx: number,
  n = 10
): Promise<FeatureSamplesResponse> {
  const r = await fetch(
    `/api/feature_samples?layer=${layer}&feature_idx=${featureIdx}&n=${n}`
  );
  if (r.status === 404) {
    throw new Error('corpus_unavailable');
  }
  if (!r.ok) throw new Error(`feature_samples: ${r.status} ${await r.text()}`);
  return r.json();
}

export async function postEncode(req: EncodeRequest): Promise<EncodeResponse> {
  const r = await fetch('/api/encode', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(req)
  });
  if (!r.ok) throw new Error(`encode: ${r.status} ${await r.text()}`);
  return r.json();
}

export async function postFeatureSearch(
  req: FeatureSearchRequest
): Promise<FeatureSearchResponse> {
  const r = await fetch('/api/feature_search', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(req)
  });
  if (!r.ok) throw new Error(`feature_search: ${r.status} ${await r.text()}`);
  return r.json();
}

export async function listNotes(layer?: number): Promise<{ notes: Note[] }> {
  const url = layer == null ? '/api/notes' : `/api/notes?layer=${layer}`;
  const r = await fetch(url);
  if (!r.ok) throw new Error(`notes: ${r.status}`);
  return r.json();
}

export async function createNote(payload: NoteCreate): Promise<Note> {
  const r = await fetch('/api/notes', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(payload)
  });
  if (!r.ok) throw new Error(`create_note: ${r.status} ${await r.text()}`);
  return r.json();
}

export async function deleteNote(id: number): Promise<void> {
  const r = await fetch(`/api/notes/${id}`, { method: 'DELETE' });
  if (!r.ok) throw new Error(`delete_note: ${r.status}`);
}

// ──────────────────────────────────────────────────────────────────────────

function parseFrame(frame: string): SseEvent | null {
  let event = 'message';
  const dataLines: string[] = [];
  for (const line of frame.split('\n')) {
    if (!line || line.startsWith(':')) continue;
    if (line.startsWith('event:')) {
      event = line.slice(6).trim();
    } else if (line.startsWith('data:')) {
      dataLines.push(line.slice(5).trimStart());
    }
  }
  if (dataLines.length === 0) return null;
  const dataStr = dataLines.join('\n');
  let data: unknown = {};
  try {
    data = JSON.parse(dataStr);
  } catch {
    return null;
  }
  return { event, data } as SseEvent;
}
