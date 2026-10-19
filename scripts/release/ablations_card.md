---
license: apache-2.0
base_model: Qwen/Qwen3-VL-8B-Instruct
library_name: peft
tags:
  - ablation
  - trajectory
---

# SparseTrajTokens — ablation and search checkpoints

Final checkpoints of runs that are **not** the released model (see `SparseTrajTokens` for the four released ones).
Kept for reproducibility of the ablations and of the schedule search; every run has its `meta.json`
(all hyper-parameters), `train_log.jsonl` and, where applicable, `val_curve.jsonl` / `PREREG.md`. Optimizer states
and intermediate steps are not included.

| Group | Runs | What |
|---|---|---|
| `runs_sft110k/` | `pB_rgbsw_3000_s0`, `pB_rgbsw_6000_s0` | frames-only budget sweep (the 9,002-step arm is the released `rgb_only_arm`) |
| | `pB_f_tok_6k_s0`, `pB_f_s2_1500_s0`, `pB_f_s2_3000_s0` | the previous (cosine) stage-2 / stage-3 recipe |
| | `pB_d2_no1a_s{0,1}`, `pB_d2_with1a_s{0,1}` | value of the stage-1 warm-up, two seeds each |
| | `pB_rgb_S_e8_s0`, `pB_traj_only_S_e8_s0`, `1a_concat_tsec_v4td` | earlier structured-target arms and the earlier stage-1 projector |
| `runs_t2/` | 35 stage-1 projector runs | tokenization-structure ablations (fusion / time axis / camera), see `PAPER_TOKENIZATION.md` in the archive |
| `runs_ablation_0824/` | 14 runs | August tokenization ablations |

Evaluation outputs of all these runs are in `TrajQA-B/results/` and in the workspace archive (`sparsetraj-archive`).
