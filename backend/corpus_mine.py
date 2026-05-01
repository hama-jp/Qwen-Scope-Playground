"""Offline corpus mining: build top-activating snippet index for SAE features.

For each target layer, stream a small corpus, run the model, encode the
residual through the SAE, and keep — per feature — the top-N (activation,
sample_id, token_pos) tuples. The result is a SQLite database used by the
Inspector pane to ground each feature in concrete text.

Run:
    python -m backend.corpus_mine --layer 10 --num-samples 1000 \\
        --output backend/data/corpus.db

Token position 0 of every chunk is skipped: in causal LMs, the very first
position tends to act as a "register" / BOS sink and contaminates top-k
lists with positional rather than semantic outliers (see SPEC §8).
"""
from __future__ import annotations

import argparse
import datetime
import heapq
import json
import sqlite3
import time
from pathlib import Path
from typing import Iterable, Iterator, List, Tuple

# IMPORTANT: import `datasets` (and therefore pyarrow) BEFORE anything that
# initialises CUDA. Importing pyarrow after a CUDA context is alive triggers
# a hard process segfault on Windows (pyarrow's libstdc++ vs torch's mismatch).
# Hoisting these imports to module top is the simplest reliable workaround.
from datasets import load_dataset

import numpy as np
import torch

from .core import TOP_K, ModelManager, SAEBundle, get_manager


# Default mining mix: prose + code + Japanese. Concept search needs each
# concept domain to actually appear in the corpus, otherwise it returns the
# closest available proxy instead of the requested concept. The 3-source
# default splits `num_samples` evenly across these so a single mining run
# covers English prose, Python code, and Japanese in one corpus.db.
DEFAULT_SOURCES: list[tuple[str, str | None, str]] = [
    ("HuggingFaceFW/fineweb-edu", "sample-10BT", "text"),
    ("codeparrot/codeparrot-clean-valid", None, "content"),
    ("wikimedia/wikipedia", "20231101.ja", "text"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--layer", type=int, nargs="+", default=[10],
                   help="One or more layer indices to mine (default: 10).")
    p.add_argument("--num-samples", type=int, default=1000)
    p.add_argument("--chunk-tokens", type=int, default=256)
    p.add_argument("--top-n", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--output", type=str, default="backend/data/corpus.db")
    p.add_argument("--dataset", type=str, default=None,
                   help="Override the default mix with a single HF dataset id. "
                        "If omitted, mines from prose + code sources.")
    p.add_argument("--dataset-config", type=str, default=None,
                   help="Config name for --dataset (e.g. sample-10BT).")
    p.add_argument("--text-field", type=str, default="text",
                   help="Field name for --dataset (only used when --dataset is set).")
    p.add_argument("--max-text-chars", type=int, default=20000,
                   help="Cap each source document length before tokenisation "
                        "to keep memory bounded.")
    return p.parse_args()


def resolve_sources(args: argparse.Namespace) -> list[tuple[str, str | None, str]]:
    """Return the list of (dataset, config, text_field) triples to mine."""
    if args.dataset is not None:
        return [(args.dataset, args.dataset_config, args.text_field)]
    return list(DEFAULT_SOURCES)


def stream_chunks(
    mgr: ModelManager, dataset: str, config: str, text_field: str,
    chunk_tokens: int, max_text_chars: int,
) -> Iterator[List[int]]:
    ds = load_dataset(dataset, config, split="train", streaming=True)
    tokenizer = mgr.tokenizer

    buf: List[int] = []
    for example in ds:
        text = example.get(text_field) or ""
        if not text:
            continue
        if len(text) > max_text_chars:
            text = text[:max_text_chars]
        ids = tokenizer.encode(text, add_special_tokens=False)
        buf.extend(ids)
        while len(buf) >= chunk_tokens:
            chunk = buf[:chunk_tokens]
            buf = buf[chunk_tokens:]
            yield chunk


def collect_chunks(
    mgr: ModelManager,
    args: argparse.Namespace,
    sources: list[tuple[str, str | None, str]],
) -> Tuple[List[List[int]], List[str]]:
    """Pull `num_samples` chunks total, split evenly across `sources`.

    Sources are processed sequentially so the resulting `samples` table is
    grouped by source — sample IDs `[0, k)` come from sources[0],
    `[k, 2k)` from sources[1], etc. This makes it easy to spot which source
    contributed a top-activating snippet during inspection.
    """
    n_sources = len(sources)
    base_per_source = args.num_samples // n_sources
    chunks_ids: List[List[int]] = []
    chunks_text: List[str] = []
    t_total = time.time()

    for i, (dataset, config, text_field) in enumerate(sources):
        # Last source absorbs the remainder so totals match exactly.
        target = (
            args.num_samples - len(chunks_ids)
            if i == n_sources - 1
            else base_per_source
        )
        if target <= 0:
            continue
        print(f"[mine] [{i + 1}/{n_sources}] streaming {dataset} "
              f"({config}) field={text_field!r} target={target} chunks")
        gen = stream_chunks(
            mgr, dataset, config, text_field,
            args.chunk_tokens, args.max_text_chars,
        )
        t0 = time.time()
        before = len(chunks_ids)
        for chunk in gen:
            chunks_ids.append(chunk)
            chunks_text.append(
                mgr.tokenizer.decode(chunk, skip_special_tokens=True)
            )
            if len(chunks_ids) - before >= target:
                break
            if (len(chunks_ids) - before) % 200 == 0:
                print(f"[mine]   tokenised {len(chunks_ids) - before}/{target} "
                      f"({time.time() - t0:.1f}s)")
        print(f"[mine]   source {i + 1} done: "
              f"{len(chunks_ids) - before} chunks in {time.time() - t0:.1f}s")

    print(f"[mine] collected {len(chunks_ids)} chunks total "
          f"in {time.time() - t_total:.1f}s")
    return chunks_ids, chunks_text


@torch.no_grad()
def mine_layer(
    mgr: ModelManager, layer: int, sae: SAEBundle,
    chunks_ids: List[List[int]], top_n: int, batch_size: int,
) -> dict[int, List[Tuple[float, int, int]]]:
    """Return {feature_idx: list of top (activation, sample_id, token_pos)} for one layer."""
    device = mgr.device
    model = mgr.model

    captured: dict = {}

    def hook(module, inp, out):  # noqa: ARG001
        h = out[0] if isinstance(out, tuple) else out
        captured["h"] = h.detach().to(torch.float32)

    handle = model.model.layers[layer].register_forward_hook(hook)

    top_per_feature: dict[int, List[Tuple[float, int, int]]] = {}
    n = len(chunks_ids)
    t0 = time.time()
    try:
        for batch_start in range(0, n, batch_size):
            batch = chunks_ids[batch_start: batch_start + batch_size]
            ids = torch.tensor(batch, dtype=torch.long, device=device)
            model(input_ids=ids)
            h = captured["h"]  # [B, seq, d_model]
            pre = torch.relu(h @ sae.W_enc_T + sae.b_enc)  # [B, seq, sae_width]
            vals, idxs = pre.topk(TOP_K, dim=-1)
            vals_np = vals.cpu().numpy()
            idx_np = idxs.cpu().numpy()
            B, S, K = idx_np.shape

            # Skip token_pos 0 to avoid the first-position register / BOS-sink
            # outlier described in SPEC §8.
            for bi in range(B):
                sid = batch_start + bi
                for t in range(1, S):
                    row_idx = idx_np[bi, t]
                    row_val = vals_np[bi, t]
                    for k in range(K):
                        f = int(row_idx[k])
                        a = float(row_val[k])
                        heap = top_per_feature.get(f)
                        if heap is None:
                            heap = []
                            top_per_feature[f] = heap
                        if len(heap) < top_n:
                            heapq.heappush(heap, (a, sid, t))
                        elif a > heap[0][0]:
                            heapq.heapreplace(heap, (a, sid, t))

            done = min(batch_start + batch_size, n)
            elapsed = time.time() - t0
            rate = done / max(1e-3, elapsed)
            eta = (n - done) / max(1e-3, rate)
            print(f"[mine] layer {layer}: {done}/{n} "
                  f"({rate:.1f} chunk/s, eta {eta:.0f}s, "
                  f"features touched={len(top_per_feature)})")
    finally:
        handle.remove()

    return top_per_feature


def write_db(
    out_path: Path, layers: List[int], top_n: int,
    chunks_ids: List[List[int]], chunks_text: List[str],
    per_layer_top: dict[int, dict[int, List[Tuple[float, int, int]]]],
    args: argparse.Namespace,
    sources: list[tuple[str, str | None, str]],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    con = sqlite3.connect(out_path)
    cur = con.cursor()
    cur.executescript(
        """
        CREATE TABLE samples (
            id INTEGER PRIMARY KEY,
            text TEXT NOT NULL,
            token_ids BLOB NOT NULL
        );
        CREATE TABLE feature_top (
            layer INTEGER NOT NULL,
            feature_idx INTEGER NOT NULL,
            rank INTEGER NOT NULL,
            sample_id INTEGER NOT NULL,
            token_pos INTEGER NOT NULL,
            activation REAL NOT NULL,
            PRIMARY KEY (layer, feature_idx, rank),
            FOREIGN KEY (sample_id) REFERENCES samples(id)
        );
        CREATE INDEX idx_feature_top ON feature_top(layer, feature_idx);
        CREATE TABLE meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """
    )

    sample_rows = [
        (i, txt, np.asarray(ids, dtype=np.int32).tobytes())
        for i, (ids, txt) in enumerate(zip(chunks_ids, chunks_text))
    ]
    cur.executemany(
        "INSERT INTO samples (id, text, token_ids) VALUES (?, ?, ?)",
        sample_rows,
    )

    feat_rows: List[Tuple[int, int, int, int, int, float]] = []
    for layer, top_per_feature in per_layer_top.items():
        for fidx, heap in top_per_feature.items():
            ordered = sorted(heap, key=lambda x: -x[0])
            for rank, (act, sid, tpos) in enumerate(ordered):
                feat_rows.append((layer, fidx, rank, sid, tpos, act))
    cur.executemany(
        "INSERT INTO feature_top "
        "(layer, feature_idx, rank, sample_id, token_pos, activation) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        feat_rows,
    )

    built_at = datetime.datetime.now(datetime.timezone.utc).isoformat(
        timespec="seconds"
    ).replace("+00:00", "Z")
    sources_for_meta = [
        {"dataset": d, "config": c, "text_field": f} for (d, c, f) in sources
    ]
    meta_kv = [
        ("layers", json.dumps(layers)),
        ("num_samples", str(len(chunks_ids))),
        ("top_n", str(top_n)),
        ("chunk_tokens", str(args.chunk_tokens)),
        ("sources", json.dumps(sources_for_meta)),
        # Back-compat single-source fields (first source) for older readers.
        ("dataset", sources[0][0] if sources else ""),
        ("dataset_config", sources[0][1] or "" if sources else ""),
        ("built_at", built_at),
    ]
    cur.executemany(
        "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", meta_kv,
    )
    con.commit()
    con.close()
    size_mb = out_path.stat().st_size / 1e6
    print(f"[mine] wrote {out_path} ({size_mb:.1f} MB), "
          f"{len(sample_rows)} samples, {len(feat_rows)} feature_top rows")


def main() -> None:
    args = parse_args()
    layers: List[int] = sorted(set(args.layer))
    out_path = Path(args.output)

    print(f"[mine] target db: {out_path.absolute()}")
    print(f"[mine] layers={layers} num_samples={args.num_samples} "
          f"chunk_tokens={args.chunk_tokens} top_n={args.top_n}")

    mgr = get_manager()
    sources = resolve_sources(args)
    print(f"[mine] sources: {sources}")
    chunks_ids, chunks_text = collect_chunks(mgr, args, sources)
    if not chunks_ids:
        raise SystemExit("[mine] no chunks collected — check dataset config")

    per_layer_top: dict[int, dict[int, List[Tuple[float, int, int]]]] = {}
    for layer in layers:
        sae = mgr.get_sae(layer)
        per_layer_top[layer] = mine_layer(
            mgr, layer, sae, chunks_ids, args.top_n, args.batch_size,
        )

    write_db(out_path, layers, args.top_n, chunks_ids, chunks_text,
             per_layer_top, args, sources)


if __name__ == "__main__":
    main()
