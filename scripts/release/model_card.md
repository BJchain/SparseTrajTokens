---
license: apache-2.0
base_model: Qwen/Qwen3-VL-8B-Instruct
library_name: peft
tags:
  - video
  - 4d
  - trajectory
  - spatial-reasoning
  - lora
---

# Sparse Trajectory Tokens — released checkpoints

Weights for *Sparse Trajectory Tokens for Metric Motion Queries in Multimodal Language Models* (ICLR 2027
submission). Code and instructions: see the accompanying repository (`README.md` there); the dataset is in the
companion dataset repo `TrajQA-B`.

Every directory is a PEFT / transformers checkpoint directory as written by `src/training/g1/train.py`
(tokenizer and processor files included) and is loaded by `src/training/g1/eval.py --ckpt <dir>` on top of the
base model [`Qwen/Qwen3-VL-8B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) (not included).

| Directory | Paper | Files | Steps |
|---|---|---|---|
| `checkpoints/stage1_projector/` | Stage 1: trajectory-token projector warmed up on rule-generated motion captions | `traj_encoder.pt` (fp32, 85.93M params) | 26,000 |
| `checkpoints/stage2_token_only/` | Stage 2: token-only QA (LoRA r128 on the LLM + projector) | `adapter_model.safetensors` (349.18M params), `traj_encoder.pt` | 8,400 |
| `checkpoints/stage3_final/` | **Stage 3: final model** (frames + trajectory tokens) — all reported numbers | same as above | 3,507 |
| `checkpoints/rgb_only_arm/` | Frames-only arm fine-tuned on the same questions (ablation) | `adapter_model.safetensors` | 9,002 |

`checkpoints/SHA256SUMS` lists every file; verify with `sha256sum -c SHA256SUMS` inside `checkpoints/`.

## Results (TrajQA-B, numeric families, accuracy %)

| Model | Pooled | Held-out | Test |
|---|---|---|---|
| `stage3_final` — frames + trajectory tokens | **82.2** | 81.7 | 82.5 |
| `stage3_final` — trajectory tokens only (no frames) | 82.2 | 81.5 | 82.8 |
| `stage3_final` — frames + donor tokens (control) | 29.6 | 30.6 | 28.8 |
| `rgb_only_arm` — frames only | 48.3 | 47.8 | 48.8 |
| Qwen3-VL-8B zero-shot | 28.1 | 28.5 | 27.7 |

## Usage

```bash
# from the code repository root, with the dataset placed under data/
QWEN3VL_PATH=<path to Qwen3-VL-8B-Instruct> PY=python bash scripts/eval/eval_final.sh heldout checkpoints/stage3_final
```

Training recipes (three stages, global batch 16, AdamW, seed 0) are in `scripts/train/` of the code repository;
hyper-parameters of every run are in the dataset repo under `results/training_logs/<run>/meta.json`.

## License

Apache License 2.0 (see `LICENSE` in the code repository). The weights are LoRA adapters and a projector on top of
[Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct), which keeps its own license.
