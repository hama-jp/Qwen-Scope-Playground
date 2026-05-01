"""Per-token TopK feature extraction for /api/encode."""
from __future__ import annotations

from typing import List

import torch

from .core import ModelManager, TOP_K
from .schemas import EncodeResponse, EncodeTokenEntry


@torch.no_grad()
def encode_prompt(
    mgr: ModelManager, prompt: str, layer: int, top_k_per_token: int,
    skip_first: bool = False,
) -> EncodeResponse:
    sae = mgr.get_sae(layer)
    tokenizer = mgr.tokenizer
    model = mgr.model

    enc = tokenizer(prompt, return_tensors="pt").to(mgr.device)
    input_ids = enc["input_ids"]
    tok_strs = [tokenizer.decode([tid]) for tid in input_ids[0].tolist()]

    captured: dict = {}

    def _hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captured["h"] = h.detach().to(torch.float32)

    handle = model.model.layers[layer].register_forward_hook(_hook)
    try:
        model(input_ids=input_ids)
    finally:
        handle.remove()

    h = captured["h"][0]  # [seq, d_model]
    pre = torch.relu(h @ sae.W_enc_T + sae.b_enc)  # [seq, sae_width]
    # TopK SAE: only top-K of the ReLU activations are non-zero.
    vals_topk, idx_topk = pre.topk(TOP_K, dim=-1)  # both [seq, K]

    # Limit per-token output to top_k_per_token (already sorted desc by topk)
    n_out = min(top_k_per_token, TOP_K)
    vals_out = vals_topk[:, :n_out].cpu().tolist()
    idx_out = idx_topk[:, :n_out].cpu().tolist()

    per_token: List[EncodeTokenEntry] = []
    for row_idx, row_vals in zip(idx_out, vals_out):
        feature_acts = [(int(i), float(v)) for i, v in zip(row_idx, row_vals)]
        per_token.append(EncodeTokenEntry(feature_acts=feature_acts))

    if skip_first and len(tok_strs) > 1:
        tok_strs = tok_strs[1:]
        per_token = per_token[1:]

    return EncodeResponse(tokens=tok_strs, per_token=per_token)
