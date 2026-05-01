"""Qwen-Scope ローカル動作デモ。

(1) Qwen3-1.7B-Base に SAE (layer 10) を被せて、プロンプトに対する
    発火特徴 (top-50 のうち上位) を取り出す。
(2) 取り出した特徴ベクトル W_dec[:, idx] を residual stream に足し込み、
    生成出力がどう変わるかを比較する。
"""

from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "Qwen/Qwen3-1.7B-Base"
SAE_PATH = Path(
    r"C:\Users\hama\.cache\huggingface\hub"
    r"\models--Qwen--SAE-Res-Qwen3-1.7B-Base-W32K-L0_50"
    r"\snapshots\f370f8d66e5afa376987679dbdf277552ee6b78a\layer10.sae.pt"
)
LAYER = 10
TOP_K = 50

device = "cuda"
dtype = torch.bfloat16

print("=== Loading Qwen3-1.7B-Base ===")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=dtype, device_map=device)
model.eval()

print(f"=== Loading SAE (layer {LAYER}) ===")
sae = torch.load(SAE_PATH, map_location=device, weights_only=True)
W_enc = sae["W_enc"].float()   # (32768, 2048)
b_enc = sae["b_enc"].float()   # (32768,)
W_dec = sae["W_dec"].float()   # (2048, 32768)
print(f"  W_enc {tuple(W_enc.shape)} / W_dec {tuple(W_dec.shape)}")


def topk_features(residual):
    """residual (..., 2048) -> sparse acts (..., 32768) with TOP_K non-zero."""
    pre = residual.float() @ W_enc.T + b_enc
    vals, idx = pre.topk(TOP_K, dim=-1)
    out = torch.zeros_like(pre)
    out.scatter_(-1, idx, vals)
    return out


def collect_residual(text):
    captured = {}

    def hook(_mod, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captured["h"] = h.detach()

    handle = model.model.layers[LAYER].register_forward_hook(hook)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        model(**inputs)
    handle.remove()
    return captured["h"], inputs


# ───────────────── Part 1: feature activations ─────────────────
print("\n=== Part 1: top features at layer 10 ===")
prompts = [
    ("English", "The capital of France is"),
    ("Chinese", "中国的首都是"),
    ("Code",    "def fibonacci(n):\n    if n <"),
]

prompt_top_feats = {}
for label, text in prompts:
    residual, _ = collect_residual(text)
    feats = topk_features(residual)
    last = feats[0, -1]                          # (32768,)
    nz = last.nonzero(as_tuple=True)[0]
    order = last[nz].argsort(descending=True)
    top10 = nz[order[:10]].tolist()
    print(f"\n[{label}] prompt: {text!r}")
    print("  top-10 active feature indices (last token):")
    for k in top10:
        print(f"    feat {k:5d}  act={last[k].item():7.3f}")
    prompt_top_feats[label] = top10


# ───────────────── Part 2: steering ─────────────────
# Pick a feature that fires on Chinese but not English -> inject into English prompt.
zh_set = set(prompt_top_feats["Chinese"])
en_set = set(prompt_top_feats["English"])
zh_only = [f for f in prompt_top_feats["Chinese"] if f not in en_set]
steer_feat = zh_only[0] if zh_only else prompt_top_feats["Chinese"][0]
print(f"\n=== Part 2: steering with feature {steer_feat} "
      f"(top-on-Chinese, not in English top-10) ===")

steer_text = "Once upon a time in a small village,"
inputs = tokenizer(steer_text, return_tensors="pt").to(device)
gen_kwargs = dict(max_new_tokens=60, do_sample=False, repetition_penalty=1.1)

# baseline (no steering)
with torch.no_grad():
    out_base = model.generate(**inputs, **gen_kwargs)
base_text = tokenizer.decode(
    out_base[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
)
print("\n[baseline, no steering]")
print(steer_text + base_text)

# steered
steer_vec = W_dec[:, steer_feat]   # (2048,)


def make_steer_hook(strength):
    def _hook(_mod, _inp, out):
        is_tuple = isinstance(out, tuple)
        h = (out[0] if is_tuple else out).clone()
        sv = steer_vec.to(device=h.device, dtype=h.dtype)
        h = h + strength * sv
        return (h, *out[1:]) if is_tuple else h
    return _hook


for strength in (5.0, 20.0, 100.0):
    handle = model.model.layers[LAYER].register_forward_hook(make_steer_hook(strength))
    with torch.no_grad():
        out_s = model.generate(**inputs, **gen_kwargs)
    handle.remove()
    text = tokenizer.decode(
        out_s[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    )
    label = {5.0: "Light", 20.0: "Medium", 100.0: "Strong"}[strength]
    print(f"\n[steered: feat={steer_feat}, layer={LAYER}, strength={strength} ({label})]")
    print(steer_text + text)
