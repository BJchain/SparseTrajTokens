# Data release layout

Everything below `data/` is distributed separately from the git repository (see `.gitignore`), as the Hugging Face
dataset https://huggingface.co/datasets/Chenfy0407/TrajQA-B (scenes and tokens packed as tar shards; unpacking
instructions are in its dataset card; `MANIFEST.tsv` there lists the SHA-256 of every tar). Frames, raw videos and
third-party data (`stereo4d/`, `external/`) are not on the Hub.
All paths are relative to the repository root, which is also the working directory of every script.

| Path | Size | Content | Produced by |
|---|---|---|---|
| `stereo4d/raw_mp4/{train,test}_mp4s.tar` | 105 GB | Official Stereo4D rectified left/right mp4s (Hugging Face mirror of the official release). **Not redistributed** — download from the Stereo4D release. | official |
| `stereo4d/tar_index.pkl`, `stereo4d/no_npz_clips.txt` | 4.6 MB | Index of the two tars; clips for which no official 3D-track npz exists (18,561) | `src/tools/stereo4d_ingest.py` |
| `stereo4d/frames/<clip>/` | 102 GB | For the 42,151 delivered clips: left-eye perspective frames `%05d.jpg` (512×512, 60° hfov, adaptive stride → 20–38 frames), `gt_stereo4d.npz` (official camera poses / timestamps, compressed to the fields used downstream), `stereo4d_meta.json` (stride, effective fps, intrinsics) | `src/stage0/stereo4d_adapter.py` |
| `scenes/scene_<clip>/` | 102 GB | Output of the Object-Centric Track Constructor for the 42,151 delivered clips: `final_repr.pkl` (per-object sparse metric keypoint tracks, visibility, parts, rigid SE(3), labels), `dense_tracks_<iid>.npz` (released tracks attributed to each instance), `stage0_pre.json` (Qwen3-VL-Plus inventory), `rigid_route.json`, `stage1_candidates.jsonl`, `stage2b_early.json`, `pass2_meta.json`, `gt_dense_stats.json`, `scene_manifest.json`, … | `scripts/data/run_stereo4d_pipeline.sh` |
| `tokens_v4td/scene_<clip>.pkl` | 1.3 GB | Trajectory tokens (`SceneTokens`) used for training and TrajQA-B; `_meta.json` has the build statistics (median 165 tokens / scene, 3,088 of 1,729,834 pieces miss the fit bound) | `src/tokenization/build_tokens.py --camera` |
| `captions/t8_train.jsonl` | 27 MB | 65,556 rule-generated motion captions for Stage 1 (42,030 scenes). Training holds out 2% of source videos (64,523 train / 1,033 val captions). **These captions are not QA-split-aware:** the 64,523 training captions cover 2,891 of the 3,103 QA evaluation scenes (5,322 captions; question-free descriptions of displacement / speed computed from the same tracks) | `src/training/g1/t8_gen.py` |
| `trajqa_b/final/` | 68 MB | **TrajQA-B**: `train.jsonl` 56,120 / `val.jsonl` 1,318 / `test.jsonl` 4,492 / `heldout_test.jsonl` 3,430 questions; build reports and the shortcut audit | `scripts/data/build_trajqa_b.sh` |
| `trajqa_b/candidates_m6/`, `trajqa_b/candidates_m1/` | 254 MB + 107 MB | Generator candidates (333,661 for `m6`) and their Gate-0 token-side recomputation; `m6` → train / held-out, `m1` → val / test | `g1.trajqa_b_gen`, `src/tools/trajqa_b_gate0.py` |
| `trajqa_b/build_m1/` | 23 MB | Intermediate default build on `candidates_m1` (its val/test are byte-identical to `final/`; its train split is the family-proportion reference for `trajqa_b_mixmatch.py`) | `g1.trajqa_b_build` |
| `figure_scenes/scene_<clip>/` | 2.7 GB | SAM 3 re-run (production parameters, all frames) for the 22 clips shown in the paper figures and the demo | `src/stage0/run_stage0_batch.py` |
| `external/MLLM-4D-Datasets/` | 2.3 GB | Third-party annotations of MLLM-4D (`flow666/MLLM-4D-Datasets` on Hugging Face; **not ours to redistribute** — fetch with `baselines/tools/dl_mllm4d_ann.sh`). Used only for the MLLM-4D overlap / answer-position analysis (`baselines/tools/overlap_mllm4d.py`, `mllm4d_answer_position.py`) | official |
| `curation/` | 21 MB | Outcome of every Stereo4D clip: `delivered_scenes.txt` (42,151), `skipped_static_clips.json` (reason per rejected clip), `skipped_static_records.tar.gz` (the `stage0_pre.json` / `gt_dense_stats.json` evidence of the 47,291 rejected clips; extract to `data/curation/` for `paper/figure_scripts/render_fig2.py`), `qwen_rejected.txt`, `gave_up.txt`, `_rej.txt`, `clips_ready.txt`, `ingested.txt`, `no_npz_clips.txt` | pipeline queue |

**Question format** (`trajqa_b/final/*.jsonl`, one question per line): `id`, `video_id` (= scene), `event_subtype`
(family), `question`, `options` (A–D or 2-way), `answer`, `evidence` (structured target used for training),
`frame_range`, `gt_payload` (reference value, window, grid cell, Gate-0 recomputation, …).

**Splits** are defined on a hash of the source YouTube video id (`src/training/g1/split.py`):
test `u < 0.20`, val `[0.20, 0.25)`, held-out `[0.25, 0.30)`, train otherwise; the four files share no video.

**Curation funnel** (paper Fig. 2): 108,015 official clips → 89,454 with released 3D tracks → removed: 29,915
all-static by the VLM, 9,207 no dynamic mask, 5,725 too few attributed tracks, 2,444 demoted by the motion gate,
11 failures → **42,152** delivered (42,151 used for QA; one clip has a corrupt reconstruction file).
