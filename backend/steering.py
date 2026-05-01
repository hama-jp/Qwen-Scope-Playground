"""Build the steering vector from a feature stack and register the hook.

steering vector = sum_i (multiplier_i * W_dec[:, idx_i])

The hook adds `global_strength * steer_vec` to the residual stream output of
`model.model.layers[LAYER]`. Both prefill (full prompt) and decode-step
(KV-cache, seq=1) call paths are handled. The hook also tolerates the case
where the layer returns a tuple (Qwen3 dense) vs a plain tensor (Qwen3MoE).
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable, Sequence

import torch

from .core import ModelManager
from .schemas import FeatureRef


def build_steer_vec(
    mgr: ModelManager, layer: int, feature_stack: Sequence[FeatureRef]
) -> torch.Tensor:
    """Return a [d_model] float32 tensor on the model device."""
    sae = mgr.get_sae(layer)
    W_dec = sae.W_dec  # [d_model, sae_width]
    d_model = W_dec.shape[0]
    device = mgr.device
    if not feature_stack:
        return torch.zeros(d_model, device=device, dtype=torch.float32)
    sv = torch.zeros(d_model, device=device, dtype=torch.float32)
    for f in feature_stack:
        if not (0 <= f.idx < W_dec.shape[1]):
            raise ValueError(f"feature idx {f.idx} out of range")
        sv = sv + float(f.strength_multiplier) * W_dec[:, f.idx]
    return sv


@contextmanager
def steering_hook(
    mgr: ModelManager,
    layer: int,
    steer_vec: torch.Tensor,
    global_strength: float,
):
    """Context manager that registers a forward hook on the chosen layer.

    `global_strength = 0` short-circuits to a no-op (no hook registered).
    """
    if global_strength == 0.0 or torch.linalg.vector_norm(steer_vec).item() == 0.0:
        yield
        return

    sv_const = (global_strength * steer_vec).detach()

    def _hook(module, inp, out):
        is_tuple = isinstance(out, tuple)
        h = (out[0] if is_tuple else out)
        # in-place is fine: the layer already produced its own buffer this step.
        sv = sv_const.to(device=h.device, dtype=h.dtype)
        h = h + sv
        return (h, *out[1:]) if is_tuple else h

    handle = mgr.model.model.layers[layer].register_forward_hook(_hook)
    try:
        yield
    finally:
        handle.remove()
