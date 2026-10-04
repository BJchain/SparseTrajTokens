# Released checkpoints

Distributed separately from the git repository. Every directory is a PEFT/transformers checkpoint directory as
written by `src/training/g1/train.py` (tokenizer / processor files included) and is loaded by
`src/training/g1/eval.py --ckpt <dir>` on top of the base model `Qwen/Qwen3-VL-8B-Instruct`
(`QWEN3VL_PATH`, default `checkpoints/Qwen3-VL-8B-Instruct`, not included).

| Directory | Paper | Files | Training log |
|---|---|---|---|
| `stage1_projector/` | Stage 1: trajectory projector warmed up on motion captions (26,000 steps) | `traj_encoder.pt` (fp32, 85.93M params) | `results/training_logs/1a_fit_const_s0`, `1a_fit_tail24k_s0` |
| `stage2_token_only/` | Stage 2: token-only QA (8,400 steps) | `adapter_model.safetensors` (LoRA r128, 349.18M params), `traj_encoder.pt` | `results/training_logs/pB_wsd_tok_s0`, `pB_wsd_tok_s0_tail7k_d20` |
| `stage3_final/` | **Stage 3: final model** (frames + tokens, 3,507 steps) — all reported numbers | same as above | `results/training_logs/pB_s2_cos1ep_s0` |
| `rgb_only_arm/` | Frames-only arm (same data and targets, no trajectory tokens; 9,002 steps) | `adapter_model.safetensors` | `results/training_logs/pB_rgbsw_9002_s0` |

In the released `adapter_config.json`, `base_model_name_or_path` was changed from the training machine's local
path to `Qwen/Qwen3-VL-8B-Instruct`; the evaluation code loads the base model from `--base` / `QWEN3VL_PATH` anyway.
