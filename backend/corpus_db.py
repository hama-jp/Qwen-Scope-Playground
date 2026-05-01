"""Read-only access layer for `backend/data/corpus.db`.

Mining is offline (`backend.corpus_mine`); the FastAPI app only reads.
Connections are opened per call — SQLite open is cheap, and this avoids
threading concerns for FastAPI's sync handlers running in a thread pool.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import List, Optional

import numpy as np

from .schemas import CorpusStatus, FeatureSample, FeatureSamplesResponse

DB_PATH = Path(__file__).resolve().parent / "data" / "corpus.db"
CONTEXT_WINDOW = 16  # tokens shown on each side of highlight_pos


def _open_ro() -> Optional[sqlite3.Connection]:
    if not DB_PATH.exists():
        return None
    # URI mode lets us mark the connection read-only.
    uri = f"file:{DB_PATH.as_posix()}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def get_status() -> CorpusStatus:
    con = _open_ro()
    if con is None:
        return CorpusStatus(available=False)
    try:
        cur = con.cursor()
        cur.execute("SELECT key, value FROM meta")
        rows = dict(cur.fetchall())
        layers = json.loads(rows.get("layers", "[]"))
        # Prefer multi-source `sources` JSON if present; fall back to the
        # single-source `dataset` string from older mines.
        sources_json = rows.get("sources")
        if sources_json:
            sources = json.loads(sources_json)
            short_names = [s["dataset"].split("/")[-1] for s in sources]
            dataset_str = " + ".join(short_names)
        else:
            dataset_str = rows.get("dataset")
        return CorpusStatus(
            available=True,
            layers=layers,
            num_samples=int(rows.get("num_samples", "0")),
            top_n=int(rows.get("top_n", "0")),
            built_at=rows.get("built_at"),
            dataset=dataset_str,
            chunk_tokens=int(rows.get("chunk_tokens", "0")) or None,
        )
    finally:
        con.close()


def get_feature_samples(
    layer: int, feature_idx: int, n: int, decode_token,
) -> FeatureSamplesResponse:
    """Return up to `n` top-activating snippets for the given feature.

    `decode_token` is a callable `int -> str` (e.g. tokenizer.decode([tid])).
    Snippets are clipped to ±CONTEXT_WINDOW tokens around the activation peak.
    Raises FileNotFoundError if the corpus DB does not exist.
    """
    con = _open_ro()
    if con is None:
        raise FileNotFoundError(f"corpus db not found at {DB_PATH}")
    try:
        cur = con.cursor()
        cur.execute(
            "SELECT rank, sample_id, token_pos, activation "
            "FROM feature_top "
            "WHERE layer = ? AND feature_idx = ? "
            "ORDER BY rank ASC LIMIT ?",
            (layer, feature_idx, n),
        )
        rows = cur.fetchall()
        if not rows:
            return FeatureSamplesResponse(
                layer=layer, feature_idx=feature_idx, samples=[],
            )

        sample_ids = sorted({r[1] for r in rows})
        placeholders = ",".join("?" * len(sample_ids))
        cur.execute(
            f"SELECT id, text, token_ids FROM samples WHERE id IN ({placeholders})",
            sample_ids,
        )
        sample_rows = {
            sid: (text, np.frombuffer(blob, dtype=np.int32).tolist())
            for sid, text, blob in cur.fetchall()
        }

        out: List[FeatureSample] = []
        for rank, sample_id, token_pos, activation in rows:
            text, token_ids = sample_rows[sample_id]
            tokens = [decode_token(int(tid)) for tid in token_ids]
            cs = max(0, token_pos - CONTEXT_WINDOW)
            ce = min(len(tokens), token_pos + CONTEXT_WINDOW + 1)
            out.append(FeatureSample(
                rank=int(rank),
                activation=float(activation),
                sample_id=int(sample_id),
                text=text,
                tokens=tokens,
                highlight_pos=int(token_pos),
                context_start=cs,
                context_end=ce,
            ))
        return FeatureSamplesResponse(
            layer=layer, feature_idx=feature_idx, samples=out,
        )
    finally:
        con.close()
