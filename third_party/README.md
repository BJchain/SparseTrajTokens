# Third-party code

Upstream repositories are **not vendored**. Clone them into this directory (the code resolves them as
`third_party/<name>` relative to the repository root) and install them into the environments listed in
`environment/`. Only the data pipeline (`src/stage0 … stage4`) and the tracker bake-off need them; training,
evaluation, tokenization and TrajQA-B generation do not.

| Directory | Upstream | Revision used | Used by | Weights (place inside the clone) |
|---|---|---|---|---|
| `sam3/` | https://github.com/facebookresearch/sam3 | snapshot of 2026-05-23 | Stage 0 masks (`src/stage0/sam3_masklets.py`), env `sam3` | `checkpoints/sam3.1/sam3.1_multiplex.pt` from HF `facebook/sam3.1` |
| `sam-3d-objects/` | https://github.com/facebookresearch/sam-3d-objects | snapshot of 2026-06-03 + **`patches/sam-3d-objects.patch`** | Stage 1 meshes (weights and shared code), env `sam3d-objects` | `checkpoints/hf/` from HF `facebook/sam-3d-objects` (incl. `moge-vitl`) |
| `Fast-SAM3D/` | https://github.com/wlfeng0509/Fast-SAM3D | snapshot of 2026-07-25 | Stage 1 engine used for all 42k scenes (`--recon fastsam3d`, `src/stage1/sam3d_engine.py`) | none (reuses `sam-3d-objects/checkpoints/hf`) |
| `PartSAM/` | PartSAM (Zhu et al., 2026), official release | snapshot of 2026-02-21, unmodified | Stage 2a part segmentation (`src/stage2/partsam_runner.py`), env `PartSAM` | `pretrained/model.safetensors`; its CUDA extension `_build/SAMPart3D/libs/pointops` is built locally |
| `Track4World/` | Track4World (Lu et al., 2026), official release | snapshot of 2026-04-27 (submodules 2026-06-26), unmodified | optional monocular tracking branch (`src/stage4/run_dense_frontend_t4w.py`) and bake-off, env `track4world` | `checkpoints/track4world_da3.pth`, `checkpoints/sam2.1_hiera_large.pt` |
| `stereo4d-code/` | https://github.com/Stereo4d/stereo4d-code | `3f40d4139d6ed4126b50ae13cea80e77bd2affd9` | reading the official Stereo4D npz / rectification (`src/stage0/stereo4d_adapter.py`) | — |
| `pytorch3d/`, `gsplat/` | facebookresearch/pytorch3d, nerfstudio-project/gsplat | `75ebeeaea0908c5527e7b1e305fbc7681382db47`, `2323de5905d5e90e035f792fe65bad0fedd413e7` | dependencies of sam-3d-objects (built from source) | — |
| `cc_torch/` | https://github.com/ronghanghu/cc_torch | `446e9360ed10d3c8d03925ccb5f220ccb62009de` | dependency of SAM 3 | — |
| `SpaTrackerV2/`, `TAPIP3D/`, `4RC/`, `D4rt/` | henry123-boy/SpaTrackerV2 (`7e12274c52077860cebfe007a6290777db43b63c` + **`patches/SpaTrackerV2.patch`**), TAPIP3D, 4RC, D4RT official releases | — | tracker bake-off only (`src/bakeoff2/adapters/`) | per upstream |

## Patches

* `patches/sam-3d-objects.patch` — `notebook/inference.py`: expose `with_layout_postprocess` as an argument of
  `Inference.__call__` (upstream hard-codes `False`); our Stage 1 calls it with `True`.
  Apply with `cd third_party/sam-3d-objects && patch -p1 < ../patches/sam-3d-objects.patch`.
* `patches/SpaTrackerV2.patch` — local changes used only for the bake-off adapter (`git apply` on the revision above).
* Offline use (optional): we pointed `pretrained_model_name_or_path` of the MoGe block in
  `sam-3d-objects/checkpoints/hf/pipeline.yaml` to the local file `checkpoints/hf/moge-vitl/model.pt` instead of
  `Ruicheng/moge-vitl`, and ran every stage with `HF_HUB_OFFLINE=1`.
