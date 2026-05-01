import { writable } from 'svelte/store';
import type {
  CorpusStatus,
  EncodeResponse,
  FeatureRef,
  InspectorTarget,
  MetaResponse,
  Note,
  VariantState
} from './types';

export const meta = writable<MetaResponse | null>(null);

export const layer = writable<number>(10);
export const prompt = writable<string>('Once upon a time in a small village,');
export const featureStack = writable<FeatureRef[]>([
  { idx: 16503, strength_multiplier: 1.0 }
]);
export const strengths = writable<number[]>([0, 5, 20, 50, 100]);
export const maxNewTokens = writable<number>(80);
export const repetitionPenalty = writable<number>(1.1);

export const variants = writable<VariantState[]>([]);
export const generating = writable<boolean>(false);
export const lastError = writable<string | null>(null);

// v0.3: when true, ResultsGrid sends `with_topk_logits: true` and ResultCard
// renders the TokenHeatmap instead of plain text. Toggle is disabled while
// generating to avoid mid-stream payload mismatches.
export const heatmapMode = writable<boolean>(false);

// ─── v0.2 ───────────────────────────────────────────────────────────────────

export const corpusStatus = writable<CorpusStatus | null>(null);
export const inspectorTarget = writable<InspectorTarget | null>(null);
export const promptEncode = writable<EncodeResponse | null>(null);
export const promptEncodeLoading = writable<boolean>(false);
export const notes = writable<Note[]>([]);
