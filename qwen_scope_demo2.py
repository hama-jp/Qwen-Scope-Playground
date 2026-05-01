"""デモ2: コードプロンプトで強く発火する特徴を、英語の物語プロンプトに注入。"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

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

print("Loading model + SAE...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map=device)
model.eval()
sae = torch.load(SAE_PATH, map_location=device, weights_only=True)
W_enc = sae["W_enc"].float()
b_enc = sae["b_enc"].float()
W_dec = sae["W_dec"].float()


def topk_features(residual):
    pre = residual.float() @ W_enc.T + b_enc
    vals, idx = pre.topk(TOP_K, dim=-1)
    out = torch.zeros_like(pre)
    out.scatter_(-1, idx, vals)
    return out


# Find a feature that fires on code but not on the story prompt.
def collect_top(text):
    captured = {}

    def hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        captured["h"] = h.detach()

    handle = model.model.layers[LAYER].register_forward_hook(hook)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        model(**inputs)
    handle.remove()
    feats = topk_features(captured["h"])[0, -1]
    nz = feats.nonzero(as_tuple=True)[0]
    order = feats[nz].argsort(descending=True)
    return [(int(i), float(feats[i])) for i in nz[order].tolist()]


code_top = collect_top("def fibonacci(n):\n    if n <")
story_top = collect_top("Once upon a time in a small village,")
story_set = {f for f, _ in story_top[:20]}
code_only = [f for f, v in code_top if f not in story_set]
print(f"\ntop story features: {[f for f, _ in story_top[:10]]}")
print(f"top code  features: {[f for f, _ in code_top[:10]]}")
print(f"code-only (top of code, not in story top-20): {code_only[:5]}")
steer_feat = code_only[0]
print(f"=> use feature {steer_feat} for steering\n")

prompt = "Once upon a time in a small village,"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
gen_kwargs = dict(max_new_tokens=60, do_sample=False, repetition_penalty=1.1)

with torch.no_grad():
    out_base = model.generate(**inputs, **gen_kwargs)
print("[baseline]")
print(prompt + tokenizer.decode(out_base[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))

steer_vec = W_dec[:, steer_feat]
for strength in (5.0, 20.0, 50.0):
    def make_hook(s):
        def _hook(_m, _i, out):
            tup = isinstance(out, tuple)
            h = (out[0] if tup else out).clone()
            h = h + s * steer_vec.to(device=h.device, dtype=h.dtype)
            return (h, *out[1:]) if tup else h
        return _hook

    handle = model.model.layers[LAYER].register_forward_hook(make_hook(strength))
    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    handle.remove()
    print(f"\n[steered feat={steer_feat} strength={strength}]")
    print(prompt + tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
