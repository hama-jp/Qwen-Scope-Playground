"""Pydantic schemas shared across endpoints."""
from __future__ import annotations

from typing import List, Optional, Tuple

from pydantic import BaseModel, Field


class FeatureRef(BaseModel):
    idx: int = Field(..., description="SAE feature index (0..sae_width-1)")
    strength_multiplier: float = Field(
        1.0, description="Per-feature scale applied before the global strength."
    )


class MetaResponse(BaseModel):
    model_id: str
    sae_id: str
    num_layers: int
    loaded_layers: List[int]
    sae_width: int
    d_model: int
    top_k: int


class LoadLayerRequest(BaseModel):
    layer: int


class LoadLayerResponse(BaseModel):
    layer: int
    loaded: bool


class EncodeRequest(BaseModel):
    prompt: str
    layer: int
    top_k_per_token: int = 10
    # When True, drop tokens[0] / per_token[0] from the response. The first
    # position acts as a positional sink in causal LMs and contaminates
    # top-K lists with positional rather than semantic features (SPEC §8).
    skip_first: bool = False


class EncodeTokenEntry(BaseModel):
    # [[feature_idx, activation], ...] — idx is int, act is float
    feature_acts: List[Tuple[int, float]]


class EncodeResponse(BaseModel):
    tokens: List[str]
    per_token: List[EncodeTokenEntry]


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 80
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    repetition_penalty: float = 1.1
    layer: int
    feature_stack: List[FeatureRef] = Field(default_factory=list)
    strengths: List[float] = Field(default_factory=lambda: [0.0, 5.0, 20.0, 50.0, 100.0])
    seed: Optional[int] = None
    # v0.3: when True, emit per-token top-k logits on each `token` SSE event.
    with_topk_logits: bool = False
    topk_logits_k: int = Field(5, ge=1, le=20)


# ─── v0.2: corpus / inspector ────────────────────────────────────────────────


class CorpusStatus(BaseModel):
    available: bool
    layers: List[int] = Field(default_factory=list)
    num_samples: int = 0
    top_n: int = 0
    built_at: Optional[str] = None
    dataset: Optional[str] = None
    chunk_tokens: Optional[int] = None


class FeatureSample(BaseModel):
    rank: int
    activation: float
    sample_id: int
    text: str
    tokens: List[str]
    highlight_pos: int
    context_start: int
    context_end: int


class FeatureSamplesResponse(BaseModel):
    layer: int
    feature_idx: int
    samples: List[FeatureSample]


# ─── v0.2: feature search ────────────────────────────────────────────────────


class FeatureSearchRequest(BaseModel):
    layer: int
    seed_positive: List[str]
    seed_negative: List[str] = Field(default_factory=list)
    top_k: int = 20


class FeatureCandidate(BaseModel):
    feature_idx: int
    score: float
    pos_mean: float
    neg_mean: float


class FeatureSearchResponse(BaseModel):
    candidates: List[FeatureCandidate]


# ─── v0.2: notes CRUD ────────────────────────────────────────────────────────


class NoteCreate(BaseModel):
    layer: int
    feature_idx: int
    label: str
    memo: str = ""


class Note(BaseModel):
    id: int
    layer: int
    feature_idx: int
    label: str
    memo: str
    created_at: str


class NotesResponse(BaseModel):
    notes: List[Note]
