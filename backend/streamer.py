"""Per-token streamer for v0.3 heatmap mode.

Unlike `TextIteratorStreamer`, this emits one queue entry per generation step,
preserving 1:1 alignment with `LogitsProcessor` calls. That alignment is what
lets the v0.3 generate path zip token IDs with captured top-k distributions.
"""
from __future__ import annotations

from queue import Queue
from typing import Optional, Tuple

import torch
# transformers 5.x exposes BaseStreamer under transformers.generation; the
# top-level package only re-exports TextIteratorStreamer / TextStreamer.
from transformers.generation import BaseStreamer


class PerTokenStreamer(BaseStreamer):
    """Yield (token_id, decoded_str) per generation step via a Queue.

    `model.generate` calls `put` once per new token. The first call carries
    the prompt input_ids — we drop that when `skip_prompt=True` so the queue
    only holds generated tokens. `end()` writes a single `None` sentinel.
    """

    SENTINEL: Optional[Tuple[int, str]] = None

    def __init__(self, tokenizer, skip_prompt: bool = True) -> None:
        self.tokenizer = tokenizer
        self.queue: "Queue[Optional[Tuple[int, str]]]" = Queue()
        self.skip_prompt = skip_prompt
        self._first = True

    def put(self, value) -> None:  # type: ignore[override]
        # Tensor shape varies: prompt is [batch, seq_len]; subsequent calls are
        # [batch] with one new token. Either way we want the last token of the
        # batch=0 row, which is what makes the streamer prompt-aware.
        if isinstance(value, torch.Tensor):
            if self.skip_prompt and self._first:
                self._first = False
                return
            t = value
            if t.dim() > 1:
                t = t[0, -1]
            elif t.dim() == 1:
                t = t[0]
            token_id = int(t.item())
        else:
            token_id = int(value)
        token_str = self.tokenizer.decode([token_id], skip_special_tokens=True)
        self.queue.put((token_id, token_str))

    def end(self) -> None:  # type: ignore[override]
        self.queue.put(self.SENTINEL)
