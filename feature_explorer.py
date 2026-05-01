"""Surface interesting SAE features from backend/data/corpus.db.

The 999-sample corpus has 3 source bands by sample_id:
  - [  0, 333) fineweb-edu (English prose)
  - [333, 666) codeparrot   (Python code)
  - [666, 999) wikipedia.ja (Japanese)

Filters that strip out the not-interesting noise:
  - token_pos <= 2 in every top snippet -> register/BOS feature, skipped
  - mean activation > 200 -> register-magnitude, skipped
  - n hits < 8 -> rare/dead feature, skipped

Categories:
  A. cross-source: hits in all 3 bands (≥ 2 each), strict "concept that
     transcends domain". Often these turn out to be very general — punctuation,
     numbers, capitalisation — but the strong ones are real cross-cutting
     concepts
  B. source-specific: top-N entirely in one band, ≥ 12 hits, mean act ≥ 8
  C. sharp detector: low CV (std/mean) with mean act ≥ 8 — one consistent
     thing fires it
  D. cross-lingual EN ↔ JA (no code): fires in both EN and JA, code <= 1.
     Best place to find genuine semantic features (concept independent of
     language)
"""
import io
import sqlite3
import sys
from pathlib import Path
from statistics import mean, stdev

import numpy as np
from transformers import AutoTokenizer

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

DB = Path(__file__).parent / "backend" / "data" / "corpus.db"
TOKENIZER_ID = "Qwen/Qwen3-1.7B-Base"
EN_HI, CODE_HI, SAMPLE_TOTAL = 333, 666, 999

REGISTER_POS_THR = 2          # mean token_pos <= this -> register feature
REGISTER_ACT_THR = 200.0      # mean activation > this -> register magnitude
MIN_HITS = 8


def src_band(sid: int) -> str:
    return "EN" if sid < EN_HI else ("CODE" if sid < CODE_HI else "JA")


print(f"loading tokenizer {TOKENIZER_ID}...")
tok = AutoTokenizer.from_pretrained(TOKENIZER_ID)
print(f"opening {DB}...")
con = sqlite3.connect(f"file:{DB.as_posix()}?mode=ro", uri=True)
cur = con.cursor()


# Pre-fetch every (feature_idx -> rows) for layer 10.
features: dict[int, list[tuple[float, int, int]]] = {}
cur.execute(
    "SELECT feature_idx, activation, sample_id, token_pos "
    "FROM feature_top WHERE layer=10 ORDER BY feature_idx, rank"
)
for fidx, act, sid, tpos in cur.fetchall():
    features.setdefault(fidx, []).append((act, sid, tpos))


_token_cache: dict[int, list[int]] = {}


def get_tokens(sid: int) -> list[int]:
    if sid not in _token_cache:
        cur.execute("SELECT token_ids FROM samples WHERE id=?", (sid,))
        row = cur.fetchone()
        _token_cache[sid] = (
            np.frombuffer(row[0], dtype=np.int32).tolist() if row else []
        )
    return _token_cache[sid]


def render_window(sid: int, tpos: int, w: int = 10) -> str:
    """Render ±w tokens around tpos with peak in【】brackets."""
    ids = get_tokens(sid)
    lo, hi = max(0, tpos - w), min(len(ids), tpos + w + 1)
    pre = tok.decode(ids[lo:tpos], skip_special_tokens=False) if tpos > lo else ""
    peak = tok.decode([ids[tpos]], skip_special_tokens=False) if 0 <= tpos < len(ids) else "?"
    post = tok.decode(ids[tpos + 1:hi], skip_special_tokens=False) if tpos + 1 < hi else ""
    out = f"{pre}【{peak}】{post}"
    return out.replace("\n", "\\n")


def feature_summary(fidx: int) -> dict | None:
    rows = features[fidx]
    if len(rows) < MIN_HITS:
        return None
    acts = [r[0] for r in rows]
    poses = [r[2] for r in rows]
    bands = [src_band(r[1]) for r in rows]
    mean_pos = mean(poses)
    mean_act = mean(acts)
    if mean_pos <= REGISTER_POS_THR:
        return None
    if mean_act > REGISTER_ACT_THR:
        return None
    s = {
        "idx": fidx, "n": len(rows),
        "max_act": max(acts), "mean_act": mean_act,
        "std_act": stdev(acts) if len(acts) > 1 else 0.0,
        "mean_pos": mean_pos,
        "en": bands.count("EN"),
        "code": bands.count("CODE"),
        "ja": bands.count("JA"),
        "rows": rows,
    }
    s["cv"] = s["std_act"] / s["mean_act"] if s["mean_act"] else 9.0
    return s


# Pre-compute summaries (filtered).
print("computing summaries...")
summaries: dict[int, dict] = {}
for fidx in features:
    s = feature_summary(fidx)
    if s is not None:
        summaries[fidx] = s
print(f"  {len(summaries)} features after filter (out of {len(features)})")


def cat_A():
    return sorted(
        (s for s in summaries.values() if s["en"] >= 2 and s["code"] >= 2 and s["ja"] >= 2),
        key=lambda s: -s["mean_act"],
    )


def cat_B(target: str, min_hits: int = 12, min_mean: float = 8.0):
    out = []
    for s in summaries.values():
        if s["n"] < min_hits or s["mean_act"] < min_mean:
            continue
        if target == "EN" and s["en"] == s["n"]:
            out.append(s)
        elif target == "CODE" and s["code"] == s["n"]:
            out.append(s)
        elif target == "JA" and s["ja"] == s["n"]:
            out.append(s)
    out.sort(key=lambda s: -s["mean_act"])
    return out


def cat_C(min_hits: int = 10, min_mean: float = 8.0, max_cv: float = 0.10):
    out = [s for s in summaries.values() if s["n"] >= min_hits
           and s["mean_act"] >= min_mean and s["cv"] <= max_cv]
    out.sort(key=lambda s: s["cv"])
    return out


def cat_D():
    out = [s for s in summaries.values()
           if s["en"] >= 3 and s["ja"] >= 3 and s["code"] <= 1]
    out.sort(key=lambda s: -s["mean_act"])
    return out


def show(s: dict, n: int = 3):
    rows = sorted(s["rows"], key=lambda r: -r[0])[:n]
    for act, sid, tpos in rows:
        print(f"    [{src_band(sid)} sid={sid:3d} tpos={tpos:3d} act={act:5.1f}]  "
              f"{render_window(sid, tpos)}")


def report(label: str, items, top: int = 10):
    print(f"\n{'=' * 78}\n{label}  ({len(items)} candidates, showing top {top})"
          f"\n{'=' * 78}")
    for s in items[:top]:
        print(f"\nfeat #{s['idx']:5d}  "
              f"max={s['max_act']:5.1f}  mean={s['mean_act']:5.1f}  "
              f"cv={s['cv']:.2f}  pos≈{s['mean_pos']:5.1f}  "
              f"src EN/CODE/JA={s['en']}/{s['code']}/{s['ja']}")
        show(s)


report("[A] cross-source (all 3 bands ≥ 2 hits each)", cat_A(), top=15)
report("[D] cross-lingual EN ↔ JA (code ≤ 1)", cat_D(), top=15)
report("[C] sharp detector (cv ≤ 0.10, mean ≥ 8)", cat_C(), top=12)
report("[B-CODE] code-only (top-N entirely CODE)", cat_B("CODE"), top=15)
report("[B-JA] Japanese-only", cat_B("JA"), top=15)
report("[B-EN] English-prose-only", cat_B("EN"), top=15)
