"""ModelManager: holds the Qwen3-1.7B-Base model + per-layer SAE tensors.

Loaded once at FastAPI lifespan startup and reused across requests.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Dict, Optional

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-1.7B-Base"
SAE_REPO_ID = "Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50"
DEFAULT_LAYER = 10
TOP_K = 50  # TopK SAE k


@dataclass
class SAEBundle:
    """Pre-converted SAE weights ready for hot-path use.

    `W_enc_T` is float32 [d_model, sae_width] so we can do `hidden @ W_enc_T`
    without re-transposing on every encode call.
    `W_dec` is kept as float32 [d_model, sae_width] for steering vector lookup.
    """

    layer: int
    W_enc_T: torch.Tensor  # [d_model, sae_width] f32
    b_enc: torch.Tensor    # [sae_width] f32
    W_dec: torch.Tensor    # [d_model, sae_width] f32
    b_dec: torch.Tensor    # [d_model] f32


class ModelManager:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model = None
        self.tokenizer = None
        self.saes: Dict[int, SAEBundle] = {}
        self._sae_lock = threading.Lock()
        # generation lock: model.generate is not thread-safe across concurrent calls
        self.gen_lock = threading.Lock()

    # ─── lifecycle ────────────────────────────────────────────────────────────

    def load(self) -> None:
        print(f"[core] loading {MODEL_ID} on {self.device} ({self.dtype})")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, dtype=self.dtype, device_map=self.device
        )
        self.model.eval()
        self.load_layer(DEFAULT_LAYER)
        print(f"[core] ready. d_model={self.d_model} num_layers={self.num_layers}")

    # ─── derived metadata ─────────────────────────────────────────────────────

    @property
    def d_model(self) -> int:
        return int(self.model.config.hidden_size)

    @property
    def num_layers(self) -> int:
        return int(self.model.config.num_hidden_layers)

    @property
    def sae_width(self) -> int:
        if not self.saes:
            return 32768
        return int(next(iter(self.saes.values())).b_enc.shape[0])

    @property
    def loaded_layers(self) -> list[int]:
        return sorted(self.saes.keys())

    # ─── SAE management ───────────────────────────────────────────────────────

    def get_sae(self, layer: int) -> SAEBundle:
        with self._sae_lock:
            if layer in self.saes:
                return self.saes[layer]
        return self.load_layer(layer)

    def load_layer(self, layer: int) -> SAEBundle:
        with self._sae_lock:
            if layer in self.saes:
                return self.saes[layer]
            if not (0 <= layer < self.num_layers):
                raise ValueError(
                    f"layer {layer} out of range [0, {self.num_layers})"
                )
            filename = f"layer{layer}.sae.pt"
            print(f"[core] downloading/locating SAE: {filename}")
            path = hf_hub_download(repo_id=SAE_REPO_ID, filename=filename)
            print(f"[core] torch.load {path}")
            sae = torch.load(path, map_location=self.device, weights_only=True)
            bundle = SAEBundle(
                layer=layer,
                W_enc_T=sae["W_enc"].T.to(dtype=torch.float32).contiguous(),
                b_enc=sae["b_enc"].to(dtype=torch.float32).contiguous(),
                W_dec=sae["W_dec"].to(dtype=torch.float32).contiguous(),
                b_dec=sae["b_dec"].to(dtype=torch.float32).contiguous(),
            )
            self.saes[layer] = bundle
            print(f"[core] SAE layer {layer} ready: {tuple(bundle.W_enc_T.shape)}")
            return bundle


_manager: Optional[ModelManager] = None


def get_manager() -> ModelManager:
    global _manager
    if _manager is None:
        _manager = ModelManager()
        _manager.load()
    return _manager
