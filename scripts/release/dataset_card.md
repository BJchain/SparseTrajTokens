---
license: cc-by-4.0
task_categories:
  - visual-question-answering
  - video-classification
language:
  - en
tags:
  - video
  - 4d
  - trajectory
  - metric
  - spatial-reasoning
  - stereo4d
size_categories:
  - 10K<n<100K
pretty_name: TrajQA-B and Sparse 4D Trajectory Scenes
---

# TrajQA-B — metric motion questions on sparse 4D trajectories

Data release of *Sparse Trajectory Tokens for Metric Motion Queries in Multimodal Language Models* (ICLR 2027
submission). The code (data pipeline, tokenizer, training, evaluation, figure scripts) is in the accompanying
repository; the model weights are in the companion model repo `SparseTrajTokens`.

Everything here is derived from [Stereo4D](https://stereo4d.github.io/) (YouTube VR180 videos and their released
3D tracks). **Video frames and raw videos are not redistributed**; the code repository extracts them from the
official release deterministically (frames re-extracted this way are byte-identical to the ones we trained on).

## Contents

| Path | Size | Content |
|---|---|---|
| `trajqa_b/final/` | 68 MB | **TrajQA-B**: `train.jsonl` 56,120 / `val.jsonl` 1,318 / `test.jsonl` 4,492 / `heldout_test.jsonl` 3,430 questions; `_audit.json` (text-shortcut audit), `_heldout_test_report.json` |
| `trajqa_b/candidates_m6/`, `trajqa_b/candidates_m1/` | 360 MB | Generator candidates (`m6`: 333,661) and their Gate-0 token-side recomputation (`m6` → train / held-out, `m1` → val / test) |
| `trajqa_b/build_m1/` | 23 MB | Intermediate default build (val/test identical to `final/`; its train split is the family-proportion reference) |
| `captions/t8_train.jsonl` | 27 MB | 65,556 rule-generated motion captions used for Stage 1 (42,030 scenes) |
| `tokens_v4td.tar` (+ `tokens_v4td_meta.json`) | 1.3 GB | Trajectory tokens of the 42,151 scenes (`tokens_v4td/scene_<clip>.pkl`, pickled `SceneTokens`; needs `src/tokenization/schema.py` on `sys.path`) |
| `scenes/scenes-NN.tar` (64 shards) + `scenes/INDEX.tsv` | 108 GB | Processed scenes `scene_<clip>/`: `final_repr.pkl` (sparse metric keypoint tracks per object), `dense_tracks_<iid>.npz` (released tracks attributed to each instance), `stage0_pre.json` (VLM object inventory), `rigid_route.json`, `gt_dense_stats.json`, `scene_manifest.json`, `stereo4d_meta.json`, `gt_stereo4d.npz` (camera poses / timestamps only), … `INDEX.tsv` maps each scene to its shard |
| `figure_scenes.tar` | 2.7 GB | SAM 3 re-run (all frames) for the 22 clips shown in the paper figures and the demo |
| `curation/` | 21 MB | Outcome of every Stereo4D clip (108,015 → 42,152): delivered list, rejection reason per clip, rejected-clip evidence (`skipped_static_records.tar.gz`) |
| `results/` | 30 MB | Final per-question evaluation outputs, checkpoint-selection evaluations on the validation split, training logs (`meta.json`, `train_log.jsonl`, `val_curve.jsonl`, `PREREG.md`), tracker bake-off scoreboard |
| `baselines/` | 47 MB | Per-question predictions of the 15 baseline models (`preds/`), the evaluation questions in baseline format (`qa/`), result tables and figures |
| `paper_assets/` | 250 MB | Rendered figure panels and demo videos |
| `MANIFEST.tsv` | | SHA-256 of every tar |

## Question format

One JSON object per line: `id`, `video_id` (= scene), `event_subtype` (family: `cam_disp_grid`, `disp_grid`,
`speed_grid`, `interval_disp`, `interval_speed`, `motion_source`, `motion_duration_binary`), `question`,
`options`, `answer`, `evidence` (structured target), `frame_range`, `gt_payload` (reference value, grid window and
cell, Gate-0 recomputation, …).

Splits are formed on a hash of the source YouTube video id: test `u < 0.20`, val `[0.20, 0.25)`, held-out
`[0.25, 0.30)`, train otherwise — the four files share no source video.

## Unpacking

```bash
mkdir -p data && cd data
tar xf tokens_v4td.tar && tar xf figure_scenes.tar
mkdir scenes && for t in scenes/scenes-*.tar; do tar xf "$t" -C scenes; done      # 108 GB
sha256sum -c <(awk -F'\t' '{print $3"  "$1}' MANIFEST.tsv)                        # optional
```
The resulting `data/` directory matches the layout expected by the code repository (`data/README.md` there).

## Notes

* The Stage-1 caption corpus was not built with the QA splits: its 64,523 training captions cover 2,891 of the
  3,103 QA evaluation scenes (question-free descriptions of displacement and speed from the same tracks).
* Reference answers inherit the errors of the Stereo4D reconstruction (tracks, camera poses, masks); see the
  paper's limitations section.
* **License: CC BY 4.0**, the same terms as the Stereo4D annotations this release is derived from
  (https://creativecommons.org/licenses/by/4.0/). Please cite Stereo4D (Jin et al., 2025) together with our paper.
  The underlying videos belong to their YouTube creators and are not part of this release.
