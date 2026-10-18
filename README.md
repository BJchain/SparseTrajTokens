# Sparse Trajectory Tokens for Metric Motion Queries in Multimodal Language Models

Code, data and checkpoints for the paper (ICLR 2027 submission).
We convert videos into **sparse, object-centric, metric 4D keypoint tracks**, compress them with an
**event-aware B-spline trajectory tokenizer**, and inject the resulting records into **Qwen3-VL-8B** through a
small trajectory-token projector. On **TrajQA-B** (65,360 source-video-disjoint questions about metric
displacement and speed of objects and the camera) the model reaches **82.2%** numeric accuracy
(held-out 81.7 / test 82.5), while every zero-shot baseline we evaluated — including GPT-6 Astra (39.2) and
MLLM-4D-SFT (25.0) — stays close to chance (25.0), and a frames-only model fine-tuned on the same data reaches 48.3.

```
video ──► Object-Centric Track Constructor ──► Event-Aware Trajectory Tokenizer ──► Trajectory-Language Adapter ──► answer
           (Qwen3-VL-Plus inventory, SAM 3,      (scene normalisation, event split,    (85.9M projector + LoRA on
            released Stereo4D tracks / Track4World, cubic B-spline nodes, rigid &        Qwen3-VL-8B; 3 training stages)
            Fast-SAM3D, PartSAM, keypoints)         camera pseudo-objects)
```

## Repository layout

| Path | Content | In git |
|---|---|---|
| `src/stage0 … stage4/`, `src/run_stereo4d_batch_v3.py`, `src/run_davis_batch_v3.py` | Object-Centric Track Constructor (data pipeline) | ✓ |
| `src/tools/stereo4d_{ingest,prod,queue}.*`, `src/tools/qwen_prefetch.py` | Stereo4D ingestion and the multi-GPU production queue | ✓ |
| `src/tokenization/` | Event-Aware Trajectory Tokenizer (`tokenizer.py`, `curve_fit_v3.py`, `build_tokens.py`) | ✓ |
| `src/training/g1/` | Trajectory-Language Adapter: projector (`traj_encoder.py`, `modeling_traj.py`), serialisation, data loading, training (`train.py`), evaluation (`eval.py`, `eval_permute.py`), caption and TrajQA-B generators | ✓ |
| `src/tools/trajqa_b_*.py`, `src/tools/evidence_metrics.py` | TrajQA-B Gate 0 / balancing / validation / shortcut audit; evaluation metrics | ✓ |
| `src/bakeoff2/` | Tracker bake-off (App. Table "Tracker bake-off") | ✓ |
| `scripts/` | One-command entry points: data pipeline, tokens, TrajQA-B, three training stages, final evaluation | ✓ |
| `baselines/` | Zero-shot evaluation of the 15 baseline models, scoring and table/figure generation (`tools/`), their per-question predictions (`preds/`) | tools ✓ / preds ✗ |
| `paper/latex/` | Final LaTeX source of the paper; `paper/figure_scripts/` renders every figure from the released data | ✓ (figure assets ✗) |
| `paper/demo/` | Demo web page (trajectories rendered into the videos) | page ✓ / videos ✗ |
| `results/` | Final evaluation outputs (per-question), checkpoint-selection evaluations, training logs, tracker bake-off scoreboard | ✗ (data release) |
| `environment/` | Exact `pip freeze` of every environment used | ✓ |
| `third_party/` | Upstream repositories (not vendored) + our patches | README/patches ✓ |
| `data/` | Stereo4D raw videos, frames, 42,151 processed scenes, trajectory tokens, TrajQA-B, captions | ✗ (data release) |
| `checkpoints/` | Released weights of the three training stages and the frames-only arm | ✗ (model release) |

Everything marked ✗ is listed in `.gitignore` and distributed through Hugging Face (see `data/README.md` and
`checkpoints/README.md` for the exact layout):

* dataset (TrajQA-B, processed scenes, tokens, captions, results, baseline predictions): https://huggingface.co/datasets/Chenfy0407/TrajQA-B
* checkpoints: https://huggingface.co/Chenfy0407/SparseTrajTokens (private until the paper is public) Paths in the code are relative to the repository root; run all
commands from there.

### Internal names ↔ paper names

The code keeps the names used during development, because they are also stored inside result files:

| Paper | Code / run name |
|---|---|
| Stage 1 (projector warm-up on captions) | `--stage 1a`, run `1a_fit_tail24k_s0` → `checkpoints/stage1_projector` |
| Stage 2 (token-only QA) | `--stage 1b --arm traj_only`, run `pB_wsd_tok_s0_tail7k_d20` → `checkpoints/stage2_token_only` |
| Stage 3 (frames + tokens, final model) | `--stage 1b --arm traj --rgb-dropout 0.2`, run `pB_s2_cos1ep_s0` → `checkpoints/stage3_final` |
| Frames-only arm | `--arm rgb`, run `pB_rgbsw_9002_s0` → `checkpoints/rgb_only_arm` |
| Stage-1 motion captions | "T8" (`g1/t8_gen.py`, `data/captions/t8_train.jsonl`) |
| Trajectory tokens | tokenizer version `v4td` (`data/tokens_v4td`) |
| Training / evaluation package | `g1` (`src/training/g1`) |
| held-out / test split | `heldout_test.jsonl` ("hld") / `test.jsonl` ("dev") |

## Setup

1. **Environments.** Each pipeline stage runs in its own environment (they have incompatible torch versions);
   exact package lists are in `environment/` (see `environment/README.md`). Training and evaluation only need `g1train`.
2. **Third-party code and weights.** Clone the upstream repositories into `third_party/` and apply our patches as
   described in `third_party/README.md` (only needed to re-run the data pipeline).
3. **Base model.** Download [`Qwen/Qwen3-VL-8B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) to
   `checkpoints/Qwen3-VL-8B-Instruct` or point `QWEN3VL_PATH` to it.
4. **Data and checkpoints.** Place the released data under `data/` and the weights under `checkpoints/`
   (`data/README.md`, `checkpoints/README.md`).

## Reproducing the paper

All commands are run from the repository root. We used 48 GB GPUs (NVIDIA L20); the global batch is always 16,
whatever the number of GPUs.

**1. Evaluate the released model** (Tables 2–3; one GPU per input condition)
```bash
PY=<g1train>/bin/python bash scripts/eval/eval_final.sh heldout          # frames+tokens / tokens only / donor tokens / question only
PY=<g1train>/bin/python bash scripts/eval/eval_final.sh dev
MODES="rgb blind" TAG=cmp PY=... bash scripts/eval/eval_final.sh heldout checkpoints/rgb_only_arm   # frames-only arm
```
Our outputs are in `results/eval/final/` (`final_{hld,dev}_*.json.items.jsonl` = per-question predictions,
`final_{heldout,dev}.jsonl` = pooled metrics).

**2. Main table and ablation from per-question predictions** (no GPU)
```bash
cd baselines && python tools/make_table.py --out figure/trajqa_b_baseline.png \
  && python tools/export_table_csv.py --out results_table.csv --md results_table.md \
  && python tools/make_ablation.py --md figure/Ablation.md --csv figure/Ablation.csv
```

**3. Train the three stages** (wall-clock times of our runs are in `results/training_logs/*/train_log.jsonl`)
```bash
PY=<g1train>/bin/python bash scripts/train/stage1_projector.sh
PY=<g1train>/bin/python bash scripts/train/stage2_token_only.sh       # INIT=checkpoints/stage1_projector/traj_encoder.pt to start from ours
PY=<g1train>/bin/python bash scripts/train/stage3_joint.sh            # SRC=checkpoints/stage2_token_only to start from ours
PY=<g1train>/bin/python bash scripts/train/rgb_only_arm.sh            # frames-only arm
```
Checkpoint-selection evidence (all decisions were taken on the validation split only) is in
`results/training_logs/` and `results/eval/selection_val/`; `src/tools/{val_curve,wsd1b,wsd_s2}_report.py`
reprint the selection tables from these files.

**4. Rebuild TrajQA-B from the trajectory tokens** (CPU)
```bash
bash scripts/data/build_trajqa_b.sh outputs/trajqa_b      # candidates → Gate 0 → balancing → splits → validation + shortcut audit
```

**5. Rebuild tokens and captions from the processed scenes** (CPU)
```bash
bash scripts/data/build_tokens_and_captions.sh outputs
```

**6. Re-run the data pipeline on Stereo4D** (GPU; needs `third_party/` and a DashScope key for Qwen3-VL-Plus)
```bash
DASHSCOPE_API_KEY=... GPUS="0 1 2 3" bash scripts/data/run_stereo4d_pipeline.sh <clip_list> outputs/scenes
```
The curation outcome of every Stereo4D clip (108,015 → 42,152) is recorded in `data/curation/`.

**7. Figures.** `paper/figure_scripts/render_figs.py` renders the per-frame panels from the released scenes
(into `paper/figure_assets/`); `render_fig{1,2,3,4,4b}.py` compose Figures 1–4 and re-check every number printed
in them against the data (Figure 2 needs `tar xzf data/curation/skipped_static_records.tar.gz -C data/curation`).
Set `G1_PY` to the `g1train` python.
`paper/demo/render_demo_videos.py` renders the demo videos.

## Results (TrajQA-B, numeric families, %)

| Model | Pooled | Held-out | Test | Cam. disp. | Obj. disp. | Speed |
|---|---|---|---|---|---|---|
| GPT-6 Astra (zero-shot, 1,000-question subset) | 39.2 | 37.9 | 40.5 | 46.0 | 32.7 | 33.9 |
| Qwen3-VL-8B (zero-shot) | 28.1 | 28.5 | 27.7 | 28.9 | 27.9 | 26.9 |
| MLLM-4D-SFT | 25.0 | 24.7 | 25.3 | 25.3 | 24.9 | 24.8 |
| Frames only (fine-tuned on TrajQA-B) | 48.3 | 47.8 | 48.8 | 53.7 | 44.5 | 43.9 |
| Frames + donor tokens (control) | 29.6 | 30.6 | 28.8 | 30.7 | 28.8 | 28.6 |
| **Ours: frames + trajectory tokens** | **82.2** | **81.7** | **82.5** | **89.4** | **77.6** | **75.2** |

All 15 baselines are in `baselines/results_table.md`. All trained-model numbers are from a single seed.

## Reproduction notes

* **Determinism.** Tokenization, TrajQA-B generation (incl. Gate 0), caption generation, metric computation and
  greedy evaluation are deterministic; re-running them on samples of the released data reproduced the released
  content (evaluation: identical generated text for the same question set). The donor control draws donors from the
  evaluated question pool, so it only matches when the full split is evaluated. In the data pipeline, SAM 3 and the
  released-track attribution are deterministic, whereas the mesh branch (SAM 3D / Fast-SAM3D, PartSAM) is not, so
  re-running it changes the keypoints of non-rigid objects and of rigid objects that were not routed past the mesh.
* **`gt_stereo4d.npz` in the release is camera-only.** To save disk space, the official per-clip 3D tracks
  (`track3d_world`) were dropped after processing and only the fields used downstream were kept
  (`src/tools/compress_gt_npz.py`). The attributed tracks are kept in `data/scenes/*/dense_tracks_*.npz`.
  To re-run the track attribution, re-download the official npz (`INGEST=1` in `scripts/data/run_stereo4d_pipeline.sh`);
  frames re-extracted this way are byte-identical to the released ones.
* **Stage-1 captions and QA evaluation scenes.** The Stage-1 caption corpus was not built with the QA splits:
  the 64,523 training captions cover 2,891 of the 3,103 QA evaluation scenes. The captions contain no questions
  or options, but they do state displacements and speeds computed from the same tracks.

## Data and licences

* Code in this repository and the released checkpoints: **Apache License 2.0** (`LICENSE`).
* TrajQA-B, the processed scenes, tokens, captions and results (`data/`, `results/`, `baselines/preds`): **CC BY 4.0**,
  the same terms as the [Stereo4D](https://stereo4d.github.io/) annotations they are derived from; cite Stereo4D
  together with this paper. The raw videos are not redistributed and remain subject to their original terms.
* Third-party code under `third_party/` and the base model keep their own licences.
