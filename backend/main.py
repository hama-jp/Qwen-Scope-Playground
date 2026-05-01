"""FastAPI application entrypoint.

Run with:
    uvicorn backend.main:app --reload --port 8000
or  uvicorn main:app --reload --port 8000   (when CWD is `backend/`)
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from . import corpus_db, notes_db
from .core import MODEL_ID, SAE_REPO_ID, TOP_K, ModelManager, get_manager
from .encode import encode_prompt
from .feature_search import search_features
from .generate import stream_generate
from .schemas import (
    CorpusStatus,
    EncodeRequest,
    EncodeResponse,
    FeatureSamplesResponse,
    FeatureSearchRequest,
    FeatureSearchResponse,
    GenerateRequest,
    LoadLayerRequest,
    LoadLayerResponse,
    MetaResponse,
    Note,
    NoteCreate,
    NotesResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    mgr = get_manager()  # triggers model + default-layer SAE load
    app.state.mgr = mgr
    yield


app = FastAPI(title="Qwen-Scope Playground", lifespan=lifespan)

# Allow the SvelteKit dev server (5173) and any localhost origin during dev.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _mgr(request: Request) -> ModelManager:
    return request.app.state.mgr  # type: ignore[no-any-return]


@app.get("/api/meta", response_model=MetaResponse)
def get_meta(request: Request) -> MetaResponse:
    mgr = _mgr(request)
    return MetaResponse(
        model_id=MODEL_ID,
        sae_id=SAE_REPO_ID,
        num_layers=mgr.num_layers,
        loaded_layers=mgr.loaded_layers,
        sae_width=mgr.sae_width,
        d_model=mgr.d_model,
        top_k=TOP_K,
    )


@app.post("/api/load_layer", response_model=LoadLayerResponse)
def load_layer(req: LoadLayerRequest, request: Request) -> LoadLayerResponse:
    mgr = _mgr(request)
    try:
        mgr.load_layer(req.layer)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"failed to load: {e!r}")
    return LoadLayerResponse(layer=req.layer, loaded=True)


@app.post("/api/encode", response_model=EncodeResponse)
def encode(req: EncodeRequest, request: Request) -> EncodeResponse:
    mgr = _mgr(request)
    if not (0 <= req.layer < mgr.num_layers):
        raise HTTPException(
            status_code=400, detail=f"layer out of range [0, {mgr.num_layers})"
        )
    return encode_prompt(
        mgr, req.prompt, req.layer, req.top_k_per_token, req.skip_first,
    )


@app.post("/api/generate")
async def generate(req: GenerateRequest, request: Request) -> EventSourceResponse:
    mgr = _mgr(request)
    if not (0 <= req.layer < mgr.num_layers):
        raise HTTPException(
            status_code=400, detail=f"layer out of range [0, {mgr.num_layers})"
        )
    if not req.strengths:
        raise HTTPException(status_code=400, detail="strengths must be non-empty")

    async def is_disconnected() -> bool:
        return await request.is_disconnected()

    return EventSourceResponse(stream_generate(mgr, req, is_disconnected))


@app.get("/api/corpus_status", response_model=CorpusStatus)
def corpus_status() -> CorpusStatus:
    return corpus_db.get_status()


@app.get("/api/feature_samples", response_model=FeatureSamplesResponse)
def feature_samples(
    request: Request, layer: int, feature_idx: int, n: int = 10,
) -> FeatureSamplesResponse:
    mgr = _mgr(request)
    if not (0 <= layer < mgr.num_layers):
        raise HTTPException(
            status_code=400, detail=f"layer out of range [0, {mgr.num_layers})"
        )
    if n <= 0 or n > 100:
        raise HTTPException(status_code=400, detail="n must be in (0, 100]")

    tokenizer = mgr.tokenizer

    def _decode(tid: int) -> str:
        return tokenizer.decode([tid])

    try:
        return corpus_db.get_feature_samples(layer, feature_idx, n, _decode)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/api/feature_search", response_model=FeatureSearchResponse)
def feature_search(
    req: FeatureSearchRequest, request: Request,
) -> FeatureSearchResponse:
    mgr = _mgr(request)
    if not (0 <= req.layer < mgr.num_layers):
        raise HTTPException(
            status_code=400, detail=f"layer out of range [0, {mgr.num_layers})"
        )
    if not req.seed_positive or all(not s.strip() for s in req.seed_positive):
        raise HTTPException(
            status_code=400, detail="seed_positive must contain at least one non-empty string"
        )
    return search_features(
        mgr, req.layer, req.seed_positive, req.seed_negative, req.top_k,
    )


@app.get("/api/notes", response_model=NotesResponse)
def list_notes(layer: int | None = None) -> NotesResponse:
    return NotesResponse(notes=notes_db.list_notes(layer))


@app.post("/api/notes", response_model=Note)
def create_note(payload: NoteCreate) -> Note:
    if not payload.label.strip():
        raise HTTPException(status_code=400, detail="label is required")
    return notes_db.create_note(payload)


@app.delete("/api/notes/{note_id}")
def delete_note(note_id: int) -> dict:
    deleted = notes_db.delete_note(note_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"note {note_id} not found")
    return {"deleted": True, "id": note_id}


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}
