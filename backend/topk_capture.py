"""LogitsProcessor that records top-k probabilities per generation step.

Inserted LAST in the LogitsProcessorList so the captured distribution reflects
all warpers (temperature, top_p, repetition_penalty) — i.e. the same
distribution `model.generate` actually samples from.
"""
from __future__ import annotations

from queue import Queue

import torch
from transformers import LogitsProcessor


class TopKLogitsCapture(LogitsProcessor):
    def __init__(self, k: int, queue: "Queue[dict]") -> None:
        self.k = k
        self.queue = queue

    def __call__(self, input_ids, scores):  # type: ignore[override]
        # scores: [batch, vocab]. We assume batch=1 (the v0.3 generate path
        # streams one variant at a time).
        probs = torch.softmax(scores, dim=-1)
        topk_p, topk_i = probs.topk(self.k, dim=-1)
        self.queue.put(
            {
                "topk_idx": topk_i[0].cpu().tolist(),
                "topk_probs": topk_p[0].cpu().tolist(),
            }
        )
        return scores
