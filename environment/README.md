# Environments

`*.pip.txt` are the exact `pip freeze` outputs of the environments used for the paper (local editable installs
are rewritten to `./third_party/...`). Every stage of the data pipeline runs as a subprocess in its own
environment; `src/run_davis_batch_v3.py` finds them as `$CONDA_ENVS_DIR/<name>/bin/python`
(default `~/miniconda3/envs`), or through the per-environment variables below.

| Environment | Python | torch / CUDA | Used for | Override |
|---|---|---|---|---|
| `g1train` | 3.11 | 2.5.1 / cu121 (transformers 4.57.3, peft 0.19) | training, evaluation, tokenization, TrajQA-B, captions, figures | `PY=` in `scripts/` |
| `sam3` | 3.12 | 2.10.0 / cu128 | Stage 0 SAM 3, rigid routing, Stage 2b/3, finalize | `SAM3_PY` |
| `sam3d-objects` | 3.11 | 2.5.1 / cu121 | Stage 1 (SAM 3D / Fast-SAM3D), track attribution, keypoint linking, Qwen prefetch | `SAM3D_PY`, `QWEN_PY` |
| `PartSAM` | 3.11 | 2.4.1 / cu124 | Stage 2a PartSAM | `PARTSAM_PY` |
| `track4world` | 3.11 | 2.5.1 / cu121 | optional Track4World branch, bake-off | `T4W_PY` |
| `stereo4d` | 3.11 | — (CPU, JAX) | Stereo4D ingestion and frame extraction (`src/tools/stereo4d_ingest.py`), demo rendering | `PY=` in `scripts/data/run_stereo4d_pipeline.sh` |

Example: `conda create -n g1train python=3.11 && conda run -n g1train pip install -r environment/g1train.pip.txt`
(install torch from the matching CUDA wheel index first).

Baseline models were evaluated on a different machine with their own environments; their installation scripts
are `baselines/tools/install_*.sh`.
