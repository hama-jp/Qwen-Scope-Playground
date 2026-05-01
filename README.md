# Qwen-Scope Playground

A local web playground for **exploring SAE features in Qwen3-1.7B-Base**.
Built on top of [Qwen-Scope](https://qwen.ai/blog?id=qwen-scope)'s public Sparse
Autoencoder weights, with an Inspector / Concept search / Steering / Heatmap UI
for hands-on feature exploration.

> 32,768 個の SAE feature を、自分の目で覗き、概念から逆引きで釣り、
> 名前を付けて永続化し、生成に注入して効きどころを観測する — そのループを
> ローカル PC 1 台で回せるツール。

![3-pane overview of Qwen-Scope Playground](docs/screenshots/01-overview.png)

## What it does

| Pane | Function |
| --- | --- |
| **Discover** (left) | Concept search — positive/negative seed の差分活性で feature を釣る。Notes — 命名した feature を SQLite に永続保存 |
| **Compose** (center) | プロンプト + Feature stack + Strength sweep で steering 実験。Show top features でプロンプトに反応する feature を chip 表示 |
| **Inspector** (right) | 任意の `(layer, feature_idx)` の top-activating snippet を corpus から表示。意味の検算用 |
| **Heatmap mode** | 生成された各トークンの `chosen_prob` を背景色で可視化。steering の効きどころが定量で見える |

## Quick Start

### Hardware

- NVIDIA GPU **8 GB+ VRAM** (RTX 3090 / 4090 / A100 等で動作確認)
- Disk **~20 GB** (Qwen3-1.7B-Base ~3.4 GB + SAE weights 537 MB/layer + corpus.db ~80 MB)

### Setup

```bash
# Python venv (Python 3.12)
uv venv --python 3.12
source .venv/Scripts/activate         # Windows bash. PowerShell は .venv\Scripts\Activate.ps1
uv pip install -e ./backend           # FastAPI / transformers / datasets 等

# Frontend
cd frontend && pnpm install && cd ..
```

### Run

```bash
# Terminal 1: backend
uvicorn backend.main:app --reload --port 8000

# Terminal 2: frontend
cd frontend && pnpm dev               # http://localhost:5173
```

初回起動で Qwen3-1.7B-Base (~3.4 GB) を Hugging Face cache に取得します。

### Corpus mining (Inspector / Concept search を使うために 1 度だけ)

```bash
# uvicorn を Ctrl+C で止めてから (GPU 競合)
uv run --python .venv/Scripts/python.exe -m backend.corpus_mine \
    --layer 10 --num-samples 24000
```

既定で **fineweb-edu (英語) + codeparrot-clean-valid (Python) + wikipedia.ja (日本語)** を
1:1:1 でミックスして mine。RTX 3090 で 25〜35 分。

## Hands-On Tour

機能を一巡する 30〜40 分のハンズオン原稿を `docs/tour.md` に置いてあります。
Inspector で feature の意味を覗く → プロンプトから feature を釣る → 概念から逆引き →
Notes に名前を付ける → Steering で物語を変える → Heatmap で確信度を観測する、まで触ります。

→ **[docs/tour.md](docs/tour.md)**

## Architecture

```
backend/        FastAPI + SSE。Model + SAE weights を lifespan で常駐
  main.py       endpoints
  core.py       ModelManager (model + per-layer SAE)
  steering.py   feature_stack -> steer_vec hook
  generate.py   /api/generate (SSE, with_topk_logits opt-in)
  encode.py     /api/encode (per-token TopK)
  corpus_mine.py CLI: HF dataset → SAE TopK → SQLite
  corpus_db.py  /api/feature_samples (read-only sqlite)
  feature_search.py /api/feature_search
  notes_db.py   /api/notes (CRUD)

frontend/       SvelteKit + Tailwind + TypeScript
  src/lib/components/
    ComposePane / FeatureStack / ResultsGrid / ResultCard
    Inspector / NotesPane / ConceptSearch / FeatureChip
    TokenHeatmap (v0.3 token confidence visualization)
  src/lib/api.ts  fetch + ReadableStream で SSE 手動パース
```

## Acknowledgments

- **Qwen Team** — [Qwen-Scope](https://qwen.ai/blog?id=qwen-scope) の SAE weights、
  および本ツールの参考にした [元 Gradio app](https://huggingface.co/spaces/Qwen/QwenScope)
- **Hugging Face** — model / dataset hosting
- **Anthropic** — Sparse Autoencoder + steering 系の研究の流れを作ったコミュニティ全体に

## License

This repository is **MIT** licensed (see [LICENSE](LICENSE)).

ただし以下は **それぞれの上流ライセンスに従います**:

- Qwen3-1.7B-Base — Qwen License (上流の HF model card 参照)
- SAE-Res-Qwen3-* weights — Qwen License (Qwen-Scope コレクション参照)
- Hugging Face datasets (fineweb-edu / codeparrot / wikipedia) — 各 dataset card 参照

本リポジトリのコード自体は MIT で、SAE weights や model はダウンロード時に
それぞれの利用条件に同意する形になります。
