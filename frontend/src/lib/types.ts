export interface FeatureRef {
  idx: number;
  strength_multiplier: number;
}

export interface MetaResponse {
  model_id: string;
  sae_id: string;
  num_layers: number;
  loaded_layers: number[];
  sae_width: number;
  d_model: number;
  top_k: number;
}

export interface GenerateRequest {
  prompt: string;
  max_new_tokens: number;
  do_sample: boolean;
  repetition_penalty: number;
  layer: number;
  feature_stack: FeatureRef[];
  strengths: number[];
  temperature?: number;
  top_p?: number;
  seed?: number | null;
  // v0.3: opt-in heatmap data on each token event.
  with_topk_logits?: boolean;
  topk_logits_k?: number;
}

// v0.3: a piece emitted by the streamer plus its top-k confidence data.
// `chosen_prob`, `token_id`, and `topk` are present only when the request was
// made with `with_topk_logits: true`. Plain v0.2 streams populate only `text`.
export interface TokenWithLogits {
  text: string;
  token_id?: number;
  chosen_prob?: number;
  topk?: [string, number][];
}

export interface VariantState {
  variant_id: number;
  strength: number;
  label: string;
  text: string;
  tokens: TokenWithLogits[];
  // v0.3: true once at least one token event with `topk` has arrived. ResultCard
  // uses this to decide whether heatmap mode can render or has to fall back.
  has_logits: boolean;
  done: boolean;
}

export type SseTokenV02 = { variant_id: number; token: string };
export type SseTokenV03 = SseTokenV02 & {
  token_id: number;
  chosen_prob: number;
  topk: [string, number][];
};

export type SseEvent =
  | { event: 'variant_start'; data: { variant_id: number; strength: number; label: string } }
  | { event: 'token'; data: SseTokenV02 | SseTokenV03 }
  | { event: 'variant_end'; data: { variant_id: number; full_text: string } }
  | { event: 'done'; data: Record<string, never> };

// ─── v0.2 ───────────────────────────────────────────────────────────────────

export interface EncodeRequest {
  prompt: string;
  layer: number;
  top_k_per_token?: number;
  skip_first?: boolean;
}

export interface EncodeTokenEntry {
  feature_acts: [number, number][];
}

export interface EncodeResponse {
  tokens: string[];
  per_token: EncodeTokenEntry[];
}

export interface CorpusStatus {
  available: boolean;
  layers?: number[];
  num_samples?: number;
  top_n?: number;
  built_at?: string | null;
  dataset?: string | null;
  chunk_tokens?: number | null;
}

export interface FeatureSample {
  rank: number;
  activation: number;
  sample_id: number;
  text: string;
  tokens: string[];
  highlight_pos: number;
  context_start: number;
  context_end: number;
}

export interface FeatureSamplesResponse {
  layer: number;
  feature_idx: number;
  samples: FeatureSample[];
}

export interface FeatureSearchRequest {
  layer: number;
  seed_positive: string[];
  seed_negative: string[];
  top_k: number;
}

export interface FeatureCandidate {
  feature_idx: number;
  score: number;
  pos_mean: number;
  neg_mean: number;
}

export interface FeatureSearchResponse {
  candidates: FeatureCandidate[];
}

export interface NoteCreate {
  layer: number;
  feature_idx: number;
  label: string;
  memo?: string;
}

export interface Note {
  id: number;
  layer: number;
  feature_idx: number;
  label: string;
  memo: string;
  created_at: string;
}

export interface InspectorTarget {
  layer: number;
  feature_idx: number;
}
