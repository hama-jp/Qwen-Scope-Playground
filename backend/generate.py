"""SSE streaming generator for /api/generate.

Iterates `strengths` in order, generating one variant at a time:
  variant_start -> token* -> variant_end  (per variant)
  done                                    (after all variants)

`strength == 0` is the baseline (no hook registered).

v0.3: when `req.with_topk_logits` is True, take the per-token + logits-capture
path. Otherwise behave exactly like v0.2 (TextIteratorStreamer).
"""
from __future__ import annotations

import asyncio
import json
import threading
from queue import Queue
from typing import AsyncIterator, Optional

import torch
from transformers import (
    LogitsProcessorList,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)

from .core import ModelManager
from .schemas import GenerateRequest
from .steering import build_steer_vec, steering_hook
from .streamer import PerTokenStreamer
from .topk_capture import TopKLogitsCapture


class _AbortStopping(StoppingCriteria):
    """Stop generation as soon as `event` is set (used for client-disconnect)."""

    def __init__(self, event: threading.Event) -> None:
        super().__init__()
        self.event = event

    def __call__(self, input_ids, scores, **kwargs) -> bool:  # type: ignore[override]
        return self.event.is_set()


def _sse(event: str, data: dict) -> dict:
    return {"event": event, "data": json.dumps(data, ensure_ascii=False)}


async def stream_generate(
    mgr: ModelManager, req: GenerateRequest, is_disconnected
) -> AsyncIterator[dict]:
    """Async generator yielding sse-starlette dicts.

    `is_disconnected` is a zero-arg awaitable that returns True if the client
    has gone away — Starlette's `request.is_disconnected`.
    """
    tokenizer = mgr.tokenizer
    model = mgr.model

    enc = tokenizer(req.prompt, return_tensors="pt").to(mgr.device)
    input_ids = enc["input_ids"]

    steer_vec = build_steer_vec(mgr, req.layer, req.feature_stack)

    base_kwargs = dict(
        max_new_tokens=req.max_new_tokens,
        do_sample=req.do_sample,
        repetition_penalty=req.repetition_penalty,
        pad_token_id=tokenizer.eos_token_id,
    )
    if req.do_sample:
        base_kwargs["temperature"] = req.temperature
        base_kwargs["top_p"] = req.top_p

    loop = asyncio.get_running_loop()

    for variant_id, strength in enumerate(req.strengths):
        if await is_disconnected():
            return

        label = "baseline" if strength == 0.0 else f"strength={strength:g}"
        yield _sse(
            "variant_start",
            {"variant_id": variant_id, "strength": float(strength), "label": label},
        )

        if req.seed is not None and req.do_sample:
            torch.manual_seed(int(req.seed))

        if req.with_topk_logits:
            async for ev in _run_variant_with_logits(
                mgr, req, variant_id, float(strength), steer_vec, input_ids,
                base_kwargs, loop, is_disconnected,
            ):
                yield ev
        else:
            async for ev in _run_variant_plain(
                mgr, req, variant_id, float(strength), steer_vec, input_ids,
                base_kwargs, loop, is_disconnected,
            ):
                yield ev

        if await is_disconnected():
            return

    yield _sse("done", {})


async def _run_variant_plain(
    mgr, req, variant_id, strength, steer_vec, input_ids,
    base_kwargs, loop, is_disconnected,
) -> AsyncIterator[dict]:
    """v0.2 path — TextIteratorStreamer. SSE bytes match v0.2 exactly."""
    tokenizer = mgr.tokenizer
    model = mgr.model

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    abort_event = threading.Event()
    stopping = StoppingCriteriaList([_AbortStopping(abort_event)])

    gen_kwargs = {
        **base_kwargs,
        "input_ids": input_ids,
        "streamer": streamer,
        "stopping_criteria": stopping,
    }

    mgr.gen_lock.acquire()
    full_text_parts: list[str] = []
    gen_thread: Optional[threading.Thread] = None
    try:
        with steering_hook(mgr, req.layer, steer_vec, strength):
            def _run():
                try:
                    with torch.no_grad():
                        model.generate(**gen_kwargs)
                except Exception as e:  # noqa: BLE001
                    streamer.text_queue.put(streamer.stop_signal)
                    print(f"[generate] variant {variant_id} error: {e!r}")

            gen_thread = threading.Thread(target=_run, daemon=True)
            gen_thread.start()

            while True:
                try:
                    piece = await loop.run_in_executor(
                        None, streamer.text_queue.get
                    )
                except Exception:
                    break
                if piece is streamer.stop_signal:
                    break
                full_text_parts.append(piece)
                yield _sse(
                    "token",
                    {"variant_id": variant_id, "token": piece},
                )
                if await is_disconnected():
                    abort_event.set()
                    break

            if gen_thread is not None:
                gen_thread.join(timeout=30)
    finally:
        mgr.gen_lock.release()

    full_text = "".join(full_text_parts)
    yield _sse(
        "variant_end",
        {"variant_id": variant_id, "full_text": full_text},
    )


async def _run_variant_with_logits(
    mgr, req, variant_id, strength, steer_vec, input_ids,
    base_kwargs, loop, is_disconnected,
) -> AsyncIterator[dict]:
    """v0.3 path — PerTokenStreamer + TopKLogitsCapture, 1:1 alignment.

    Order of operations per generation step inside model.generate:
      1. LogitsProcessorList(scores) is called — TopKLogitsCapture pushes one
         dict onto logits_q
      2. sampling/greedy picks a token from the post-warped scores
      3. Streamer.put is called with the new token id — PerTokenStreamer
         pushes one (token_id, str) onto tok_streamer.queue

    So per step we get exactly one entry in each queue, and zipping them is
    safe. If transformers ever changes that ordering this assertion-style
    debug log catches it on the first 5 tokens.
    """
    tokenizer = mgr.tokenizer
    model = mgr.model

    tok_streamer = PerTokenStreamer(tokenizer, skip_prompt=True)
    logits_q: "Queue[dict]" = Queue()
    proc = TopKLogitsCapture(k=req.topk_logits_k, queue=logits_q)

    abort_event = threading.Event()
    stopping = StoppingCriteriaList([_AbortStopping(abort_event)])

    gen_kwargs = {
        **base_kwargs,
        "input_ids": input_ids,
        "streamer": tok_streamer,
        "stopping_criteria": stopping,
        "logits_processor": LogitsProcessorList([proc]),
    }

    full_text_parts: list[str] = []
    gen_thread: Optional[threading.Thread] = None
    mgr.gen_lock.acquire()
    try:
        with steering_hook(mgr, req.layer, steer_vec, strength):
            def _run():
                try:
                    with torch.no_grad():
                        model.generate(**gen_kwargs)
                except Exception as e:  # noqa: BLE001
                    print(f"[generate] variant {variant_id} error: {e!r}")
                finally:
                    # Make sure both consumers wake up in error / abort cases.
                    tok_streamer.queue.put(PerTokenStreamer.SENTINEL)
                    logits_q.put({"_done": True})

            gen_thread = threading.Thread(target=_run, daemon=True)
            gen_thread.start()

            step = 0
            while True:
                item = await loop.run_in_executor(
                    None, tok_streamer.queue.get
                )
                if item is None:
                    break
                token_id, token_str = item
                # Pull the matching logits record. If `_run` errored before
                # producing one, we'll get the `_done` sentinel and break.
                rec = await loop.run_in_executor(None, logits_q.get)
                if "_done" in rec:
                    break

                topk_idx = rec["topk_idx"]
                topk_probs = rec["topk_probs"]
                # chosen_prob: prob of the actually-emitted token. If
                # repetition_penalty knocked it out of top-k, fall back to 0.0.
                chosen_prob = 0.0
                for i, p in zip(topk_idx, topk_probs):
                    if i == token_id:
                        chosen_prob = float(p)
                        break

                if step < 5:
                    in_topk = token_id in topk_idx
                    if not in_topk:
                        print(
                            f"[generate.heatmap] variant={variant_id} step={step} "
                            f"token_id={token_id} NOT in topk_idx={topk_idx[:5]} — "
                            "rep_penalty likely knocked it out; chosen_prob falls back to 0"
                        )
                step += 1

                topk_strs = [
                    tokenizer.decode([i], skip_special_tokens=True) for i in topk_idx
                ]
                topk_pairs = [[s, float(p)] for s, p in zip(topk_strs, topk_probs)]

                full_text_parts.append(token_str)
                yield _sse(
                    "token",
                    {
                        "variant_id": variant_id,
                        "token": token_str,
                        "token_id": token_id,
                        "chosen_prob": chosen_prob,
                        "topk": topk_pairs,
                    },
                )
                if await is_disconnected():
                    abort_event.set()
                    # Drain the generator thread cleanly.
                    break

            if gen_thread is not None:
                gen_thread.join(timeout=30)
    finally:
        mgr.gen_lock.release()

    full_text = "".join(full_text_parts)
    yield _sse(
        "variant_end",
        {"variant_id": variant_id, "full_text": full_text},
    )
