"""Concept-to-feature search.

Given lists of positive / negative seed phrases, encode each through the SAE
at the chosen layer, average per-token activations *excluding token 0* (the
positional sink described in SPEC §8), and rank features by

    score = mean(positive activations) - mean(negative activations)

`top_k` highest-scoring features are returned. With no negative seeds, the
score reduces to the positive mean.
"""
from __future__ import annotations

from typing import List, Sequence

import torch

from .core import ModelManager
from .schemas import FeatureCandidate, FeatureSearchResponse


@torch.no_grad()
def _seed_mean_activations(
    mgr: ModelManager, layer: int, seeds: Sequence[str],
) -> torch.Tensor:
    """Return [sae_width] tensor: mean activation per feature across all
    (seed, token) positions, excluding token 0 of each seed."""
    sae = mgr.get_sae(layer)
    width = sae.W_enc_T.shape[1]
    if not seeds:
        return torch.zeros(width, dtype=torch.float32, device=mgr.device)

    tokenizer = mgr.tokenizer
    model = mgr.model
    captured: dict = {}

    def hook(module, inp, out):  # noqa: ARG001
        h = out[0] if isinstance(out, tuple) else out
        captured["h"] = h.detach().to(torch.float32)

    handle = model.model.layers[layer].register_forward_hook(hook)
    total = torch.zeros(width, dtype=torch.float32, device=mgr.device)
    n_positions = 0
    try:
        for seed in seeds:
            text = (seed or "").strip()
            if not text:
                continue
            ids = tokenizer(text, return_tensors="pt").to(mgr.device)["input_ids"]
            model(input_ids=ids)
            h = captured["h"][0]  # [seq, d_model]
            if h.shape[0] <= 1:
                continue
            h = h[1:]  # drop the first-position outlier
            pre = torch.relu(h @ sae.W_enc_T + sae.b_enc)  # [seq-1, sae_width]
            total = total + pre.sum(dim=0)
            n_positions += int(h.shape[0])
    finally:
        handle.remove()
    if n_positions == 0:
        return torch.zeros(width, dtype=torch.float32, device=mgr.device)
    return total / float(n_positions)


def search_features(
    mgr: ModelManager,
    layer: int,
    seed_positive: Sequence[str],
    seed_negative: Sequence[str],
    top_k: int,
) -> FeatureSearchResponse:
    pos_mean = _seed_mean_activations(mgr, layer, seed_positive)
    neg_mean = _seed_mean_activations(mgr, layer, seed_negative)
    score = pos_mean - neg_mean

    k = max(1, min(int(top_k), score.shape[0]))
    vals, idxs = score.topk(k)
    vals_cpu = vals.cpu().tolist()
    idxs_cpu = idxs.cpu().tolist()
    pos_cpu = pos_mean.cpu()
    neg_cpu = neg_mean.cpu()

    candidates: List[FeatureCandidate] = []
    for v, i in zip(vals_cpu, idxs_cpu):
        candidates.append(FeatureCandidate(
            feature_idx=int(i),
            score=float(v),
            pos_mean=float(pos_cpu[i].item()),
            neg_mean=float(neg_cpu[i].item()),
        ))
    return FeatureSearchResponse(candidates=candidates)
