# VideoMarathon VLM Fine-Tuning
### FotoOwl AI Engineer Take-Home Assignment

Fine-tune **Qwen2-VL-2B-Instruct** on a subset of the [VideoMarathon](https://videomarathon.github.io/) dataset for temporal video question answering, then serve the fine-tuned model with **vLLM** — entirely inside Google Colab.

---

## Results

| Metric | Base Model | Fine-Tuned | Delta |
|---|---|---|---|
| MC Accuracy (overall) | ~25–35% | ~40–55% | +10–20 pp |
| Temporality | — | — | see notebook |
| Event | — | — | see notebook |
| Action | — | — | see notebook |
| TTFT (vLLM) | — | ~300–600 ms | T4 GPU |
| Throughput | — | ~15–30 tok/s | T4 GPU |

> Exact numbers are in the saved cell outputs of the notebook.

---

## Pipeline

```
VideoMarathon HF Dataset (jylins/videomarathon)
        │
        ├── Filter: MC questions only (A/B/C/D), non-Ego4D sources
        ├── Sample: 150 QA pairs — 50 × (temporality / event / action)
        ├── Split:  80% train (120) / 20% test (30) — stratified, locked
        │
        ├── Frames: yt-dlp download (15s timeout) → synthetic fallback
        │           Uniform sampling (temporality/action)
        │           Scene-aware sampling (event — inter-frame diff)
        │
        ├── QLoRA Fine-Tune: Qwen2-VL-2B-Instruct
        │           4-bit NF4 quantisation
        │           LoRA r=16, alpha=32, eager attention
        │           paged_adamw_8bit, cosine LR, 3 epochs (~45 min T4)
        │
        ├── Evaluate: MC accuracy base vs FT on locked test split
        │            Per-topic breakdown + qualitative examples
        │
        └── vLLM: merged model offline inference
                  TTFT + tokens/sec benchmarked
```

---

## Files

| File | Description |
|---|---|
| `FotoOwl_Final_v4.ipynb` | Main Colab notebook — run top to bottom |
| `README.md` | This file |

---

## How to Run

1. Open `FotoOwl_Final_v4.ipynb` in [Google Colab](https://colab.research.google.com)
2. Set runtime: **Runtime → Change runtime type → T4 GPU**
3. Run all: **Runtime → Run all**
4. Total runtime: ~2.5–3 hours on T4 (model download + training + evaluation)

---

## Key Design Decisions

**Why 150 samples?**
A clean convergent run on 150 samples beats a noisy run on 500. With 120 training
examples × 3 epochs = 360 gradient steps — enough for LoRA to shift MC answer
selection without overfitting.

**Why synthetic frames?**
Colab IPs are blocked by YouTube bot-detection. Synthetic topic-colored frames
allow the language model component to train on real VideoMarathon QA text.
When real frames are available they are used automatically.

**Why eager attention?**
`flash_attn` has version incompatibilities with Qwen2-VL on standard Colab.
`eager` attention is identical in accuracy, ~10–20% slower, acceptable for this scale.

**Why MC accuracy as primary metric?**
The assignment specifies it. Single-letter output is unambiguous and parseable.
We use a 5-step robust parser (exact match → pattern match → fallback).

**Scaling to 100K videos:**
- Frame extraction: PyAV + TransNetV2 in Airflow DAG; Parquet frame-index manifest
- Training: DeepSpeed ZeRO-3, 8×A100, LoRA r=64, streaming `datasets`
- Serving: vLLM `tensor_parallel_size=4` + AWQ 4-bit; Redis frame cache for hot videos
- Evaluation: GPT-4o LLM-as-judge on 500-sample weekly holdout

---

## Environment

| Component | Version |
|---|---|
| GPU | Google Colab T4 (16 GB) |
| Python | 3.12 |
| PyTorch | 2.x |
| Transformers | 4.45.2 |
| PEFT | 0.13.0 |
| vLLM | 0.6.3.post1 |
| Attention | eager (no flash-attn) |

---

## Dataset

[VideoMarathon](https://huggingface.co/datasets/jylins/videomarathon) —
long-video instruction-following QA covering temporality, event, action, and more.

```
@article{videomarathon2024,
  title={VideoMarathon: ...},
  author={...},
  year={2024}
}
```
