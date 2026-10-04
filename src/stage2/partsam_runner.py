"""Thin PartSAM runner for Stage 2 (Milestone 1 — single-object feasibility probe).

This module REUSES the official PartSAM model-loading, preprocessing
(`utils.ValDataset`), `predict_masks`, and `nms`, but bypasses the official
eval script's "vote to faces → export colored .ply" tail. Instead it keeps the
NMS-selected mask LOGITS (not binarized), so the downstream can build a
per-vertex soft (simplex) part assignment as required by REFERENCE §3.4.

Nothing under third_party/PartSAM/ is modified; we only import from it.

Integration facts established by static reading + probing (2026-06):
- The released checkpoint is hardwired to d_in = coords(3) + cat[color(3),
  normal(3)]; no d_in=6/3 mode. Textureless meshes get uniform gray 192 inside
  ValDataset, which is exactly our case (the adapter emits no texture).
- ValDataset loads the mesh with trimesh default (process=True), which MERGES
  duplicate vertices (9912: 136020 -> 62499) but PRESERVES face order exactly
  (per-face centroid diff = 0 vs a process=False load). Therefore PartSAM's
  `point_to_face` is a valid index into our process=False `ref_mesh_F`
  (same triangle at each face index), and we can aggregate point->face->vertex
  directly without any coordinate-space transfer.
- The encoder runs once per object (model caches `pc_embeddings`); only the
  decoder repeats per prompt batch. fps_point_number=512 is divisible by
  batch_size=8, so every prompt batch is exactly size 8 and the cached
  embedding batch dim stays consistent.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

# --- make third_party/PartSAM importable (PartSAM / partfield / utils / configs) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_PARTSAM_ROOT = os.path.join(_REPO_ROOT, "third_party", "PartSAM")
if _PARTSAM_ROOT not in sys.path:
    sys.path.insert(0, _PARTSAM_ROOT)

import hydra  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from safetensors.torch import load_model  # noqa: E402
import pointops  # noqa: E402

from PartSAM.utils.torch_utils import replace_with_fused_layernorm  # noqa: E402
from utils.ValDataset import ValDataset, collate_fn_eval  # noqa: E402
from utils.point import sample_surface  # noqa: E402
from utils.infer_utils import nms  # noqa: E402
import trimesh  # noqa: E402


PARTSAM_CONFIGS_DIR = os.path.join(_PARTSAM_ROOT, "configs")
PARTSAM_CKPT = os.path.join(_PARTSAM_ROOT, "pretrained", "model.safetensors")

# Official defaults (调研报告里拍板沿用，不调超参).
FPS_POINT_NUMBER = 512
IOU_THRESHOLD = 0.65
NMS_THRESHOLD = 0.3
NUM_POINTS = 100000
SAMPLE_SEED = 666  # ValDataset default seed


def assert_face_order_preserved(trimesh_mesh, ref_mesh_F, ref_mesh_V,
                                n_samples: int = 20, tol: float = 1e-5,
                                seed: int = 0) -> float:
    """Defensive input check for every mesh PartSAM processes.

    ValDataset loads meshes with trimesh's default process=True, which MERGES
    duplicate vertices but must PRESERVE FACE ORDER; otherwise PartSAM's
    `point_to_face` (indexing the merged mesh's faces) would not align with the
    process=False `ref_mesh_F` used downstream to aggregate logits ->
    faces -> vertices. This helper verifies (a) face count matches and (b)
    `n_samples` random faces have matching centroids in original coords.

    Raises AssertionError on violation — see MILESTONE1_ADAPTER_NOTES.md §4.4,
    which mandates reusing one such helper across Milestones. Returns the max
    centroid distance observed (for logging).
    """
    faces_mesh = np.asarray(trimesh_mesh.faces)
    nf_mesh = faces_mesh.shape[0]
    nf_ref = int(np.asarray(ref_mesh_F).shape[0])
    assert nf_mesh == nf_ref, (
        f"face count mismatch: ValDataset mesh has {nf_mesh} faces, "
        f"ref_mesh_F has {nf_ref} -> point_to_face cannot index ref_mesh_F")
    rng = np.random.default_rng(seed)
    k = min(n_samples, nf_ref)
    idx = rng.choice(nf_ref, size=k, replace=False)
    mv = np.asarray(trimesh_mesh.vertices)
    c_mesh = mv[faces_mesh[idx]].mean(axis=1)
    c_ref = np.asarray(ref_mesh_V)[np.asarray(ref_mesh_F)[idx]].mean(axis=1)
    max_d = float(np.linalg.norm(c_mesh - c_ref, axis=1).max())
    assert max_d < tol, (
        f"face order NOT preserved: max face-centroid dist {max_d:.3e} >= {tol}; "
        f"PartSAM point_to_face would misalign with ref_mesh_F")
    return max_d


def load_partsam_model(config_dir: str = PARTSAM_CONFIGS_DIR,
                       ckpt_path: str = PARTSAM_CKPT):
    """Reuse the official load path (hydra instantiate + fused LN + safetensors)."""
    with hydra.initialize_config_dir(config_dir=os.path.abspath(config_dir), version_base=None):
        cfg = hydra.compose(config_name="partsam")
        OmegaConf.resolve(cfg)
    model = hydra.utils.instantiate(cfg.model)
    model.apply(replace_with_fused_layernorm)
    load_model(model, ckpt_path)
    model.eval()
    model.cuda()
    return model


def _preprocess_single_mesh(mesh_path: str):
    """Reuse ValDataset preprocessing for exactly one mesh.

    ValDataset walks a directory for meshes; we point it at the file's dir and
    pick the matching index, then run the official collate (surface sample 100K,
    gray-192 color, face-normal per point, internal bbox + center normalization).
    """
    mesh_dir = os.path.dirname(os.path.abspath(mesh_path))
    ds = ValDataset(root_dir=mesh_dir, num_points=NUM_POINTS, seed=SAMPLE_SEED)
    target = os.path.abspath(mesh_path)
    matches = [i for i, f in enumerate(ds.files) if os.path.abspath(f) == target]
    if not matches:
        raise FileNotFoundError(f"{mesh_path} not picked up by ValDataset under {mesh_dir}")
    item = ds[matches[0]]
    batch = collate_fn_eval([item])
    return batch


def _original_space_samples(mesh_path: str):
    """Re-sample the ORIGINAL-coordinate mesh with the same seed to recover the
    sample points in pre-normalization world coords. Face areas scale uniformly
    under ValDataset's normalization, so face picks (hence point_to_face) are
    identical; only the absolute coords differ. Returns (coords[N,3], ptf[N])."""
    ext = os.path.splitext(mesh_path)[1].lower().lstrip(".")
    mesh = trimesh.load(mesh_path, force="mesh", file_type=ext)  # default: merged, original coords
    samples, face_index = sample_surface(mesh, count=NUM_POINTS, sample_color=False, seed=SAMPLE_SEED)
    return np.asarray(samples, dtype=np.float64), np.asarray(face_index, dtype=np.int64)


@torch.no_grad()
def run_partsam_on_mesh(model, mesh_path: str, use_graph_cut: bool = False,
                        batch_size: int = 8, iou_threshold: float | None = None,
                        iou_fallbacks: "tuple | list" = ()) -> dict:
    """Run the 'segment every part' auto mode and KEEP NMS-selected mask logits.

    `iou_fallbacks` (2026-08-11): descending thresholds to retry ONLY when `iou_threshold` lets
    no mask through (K=0 hard failure). The threshold is pure post-processing on scores the
    forward pass already produced, so this costs nothing and — unlike calling this function again
    with a lower threshold — reuses the SAME scores (PartSAM is non-deterministic; a re-run would
    score differently). When the primary threshold already yields a mask the fallback code does
    not execute at all, so currently-succeeding instances take a bit-identical path.

    `iou_threshold` overrides the module-level IOU_THRESHOLD (PartSAM's official 0.65) for THIS
    mesh; None keeps 0.65. Callers lower it for reconstructions PartSAM is less confident on —
    on TripoSR fast-path meshes 0.65 leaves K=0 (hard failure) on 17% of instances and yields
    only ~4 parts where SAM 3D yields ~9; 0.50 measured 10 parts and 8% failure (2026-08-08,
    see STAGE1_FASTPATH_QUALITY_FINDINGS.md §6). The value actually used is returned as
    `iou_threshold` so products stay auditable.

    `use_graph_cut` is accepted for interface parity with the official pipeline;
    this runner does not run the .ply/graph-cut post-processing at all (it
    returns per-point logits instead), so the flag is effectively a no-op here
    and kept False per the F-b decision.
    """
    batch = _preprocess_single_mesh(mesh_path)

    # Defensive input check (every mesh): ValDataset's default-process load must
    # preserve face order vs the process=False ref mesh, else point_to_face is
    # invalid. Raise immediately on violation. (MILESTONE1_ADAPTER_NOTES §4.4)
    _ext = os.path.splitext(mesh_path)[1].lower().lstrip(".")
    _mesh_default = trimesh.load(mesh_path, force="mesh", file_type=_ext)            # as ValDataset loads
    _ref_mesh = trimesh.load(mesh_path, force="mesh", file_type=_ext, process=False)
    assert_face_order_preserved(_mesh_default,
                                np.asarray(_ref_mesh.faces, dtype=np.int64),
                                np.asarray(_ref_mesh.vertices, dtype=np.float64),
                                n_samples=20)

    coords = batch["coords"].cuda(non_blocking=True)      # [1, N, 3], normalized to ~[-1,1]
    color = batch["color"].cuda(non_blocking=True)        # [1, N, 3]
    normal = batch["normal"].cuda(non_blocking=True)      # [1, N, 3]
    point_to_face = batch["point_to_face"]                # [1, N] (cpu long)
    vertices0 = batch["vertices"][0].cuda(non_blocking=True)  # [1, Nv, 3] (unused by model)
    faces0 = batch["faces"][0].cuda(non_blocking=True)        # [1, Nf, 3] (unused by model)

    coord = coords[0].contiguous()                        # [N, 3]
    N = coord.shape[0]
    coord_offset = torch.tensor([N], device="cuda")
    new_offset = torch.tensor([FPS_POINT_NUMBER], device="cuda")
    fps_idx = pointops.farthest_point_sampling(coord, coord_offset, new_offset)  # [512]
    prompt_labels = torch.tensor([1], dtype=torch.long).unsqueeze(0).cuda()      # [1,1]

    ptf_cuda = point_to_face.cuda(non_blocking=True)

    masks_logits, scores = [], []
    for s in range(0, fps_idx.size(0), batch_size):
        e = min(s + batch_size, fps_idx.size(0))
        B = e - s
        sel = fps_idx[s:e]
        prompt_pts = coord[sel].unsqueeze(1)              # [B, 1, 3]
        data_in = {
            "coords": coords.repeat(B, 1, 1),
            "color": color.repeat(B, 1, 1),
            "normal": normal.repeat(B, 1, 1),
            "point_to_face": ptf_cuda.repeat(B, 1),
            "vertices": vertices0.repeat(B, 1, 1),
            "faces": faces0.repeat(B, 1, 1),
            "prompt_coords": prompt_pts,
            "selected_indices": sel,
            "prompt_labels": prompt_labels.expand(B, -1),
        }
        bm, bs_ = model.predict_masks(**data_in)          # [B, num_outputs, N], [B, num_outputs]
        masks_logits.append(bm.float().cpu())
        scores.append(bs_.float().cpu())
        del bm, bs_
        # 这里【不要】加 torch.cuda.empty_cache()+gc.collect()(2026-08-12 实测删除)。
        # 原来每个 batch 后各调一次(bs=8 -> 64 次/实例),占 decoder 循环墙钟的 52%:
        # 26.56s 里有 13.81s 是这两句。而它们【没有任何显存收益】—— 删除前后峰值显存
        # 一字不变(典型 mesh 10.6GB / 827k 顶点的最大 mesh 10.9GB),保留显存也不变
        # (18.2GB),6 个实例全部不 OOM。原因:caching allocator 本就复用已释放的块,
        # empty_cache 只是把缓存还给驱动、下个 batch 再 cudaMalloc;而 max_memory_allocated
        # 测的是分配量不是缓存量,所以它根本不受影响。gc.collect() 则是每次遍历整个 Python 堆。
        # 产物影响:逐顶点标签 ARI 0.9997~0.9999,落在 PartSAM 自身噪声地板(0.9995~1.0)以内。

    # reset cached embeddings (object done)
    model.pc_embeddings = None
    model.patches = None
    model.pf_feat = None
    model.part_planes = None
    model.labels = None

    masks_logits = torch.cat(masks_logits, dim=0)         # [512, num_outputs, N]
    scores = torch.cat(scores, dim=0)                     # [512, num_outputs]
    masks_logits = masks_logits.reshape(-1, masks_logits.size(2))  # [512*num_outputs, N]
    scores = scores.reshape(-1, 1)                        # [512*num_outputs, 1]

    masks_bin = masks_logits > 0
    thr_req = IOU_THRESHOLD if iou_threshold is None else float(iou_threshold)
    thr = thr_req
    top = (scores > thr).squeeze(1)
    # ---- K=0 兜底阶梯(2026-08-11) ------------------------------------------------------------
    # 阈值只是【后处理】:上面 512 个 prompt batch 的前向已经跑完,scores/masks_logits 都在手上。
    # 所以"降阈值重试"是零额外算力的纯筛选,而且用的是【同一批 scores】—— 比重跑一次
    # run_partsam_on_mesh 更可比(PartSAM 非确定,重跑会得到另一批 scores)。
    # 只在一个 mask 都没过线时才降,且只降不升;primary 阈值本来就有 mask 时下面这段完全不执行,
    # 现有能成功的实例走的是与改动前【逐位相同】的代码路径。
    # 阶梯取值依据:实测失败 mesh 的 score_max 落在 0.48~0.64(成功的 0.84~0.91),
    # 见 STAGE1_FASTPATH_QUALITY_FINDINGS.md §6。
    for _fb in (iou_fallbacks or ()):
        if top.any():
            break
        _fb = float(_fb)
        if _fb >= thr:                       # 只降不升;阶梯里 >= 当前阈值的项直接跳过
            continue
        thr = _fb
        top = (scores > thr).squeeze(1)
    if not top.any():
        # 原来会一路走到 argmax 空序列,抛 "ValueError: attempt to get argmax of an empty
        # sequence" —— 上层只能把这句原样写进 FAILED.json,看不出是阈值问题还是别的。
        _sm = float(scores.max().item()) if scores.numel() else float("nan")
        raise RuntimeError(
            f"PartSAM: no mask above threshold (score_max={_sm:.3f}; tried "
            f"{[thr_req] + [float(f) for f in (iou_fallbacks or ())]})")
    ml_top = masks_logits[top]
    mb_top = masks_bin[top]
    sc_top = scores[top]

    # 恰好 1 个 mask 过线时绕开上游 nms(2026-08-11):`utils.infer_utils.nms` 里
    # `torch.argsort(scores.squeeze(), ...)` —— scores 形状是 [M,1],M==1 时 squeeze() 会把
    # 两个维度都压掉,得到 0 维张量,随后 `sorted_indices[i]` 抛
    # "IndexError: invalid index of a 0-dim tensor"。单个 mask 的 NMS 本来就是恒等
    # (没有别的框能压制它,可用 nms 逻辑逐行验证:keep=[0] 后 subsequent 为空即 break),
    # 所以这里直接给 [0],与"上游 nms 正常工作时的返回值"完全一致;M>=2 走原路径,逐位不变。
    # 触发条件很窄(K=0 兜底阶梯把阈值降下来后只捞到 1 个 mask),但不挡住就等于兜底白做。
    keep = [0] if mb_top.shape[0] == 1 else nms(mb_top, sc_top, threshold=NMS_THRESHOLD)
    final_logits = ml_top[keep]                           # [K, N]
    K = int(final_logits.shape[0])

    logits_per_point = final_logits.t().contiguous().numpy().astype(np.float32)  # [N, K]
    hard_label_per_point = logits_per_point.argmax(axis=1).astype(np.int64)      # [N]
    ptf_np = point_to_face[0].cpu().numpy().astype(np.int64)                      # [N]

    # original-space sample coords (for completeness; algorithm uses face centroids)
    sample_coords, ptf_resample = _original_space_samples(mesh_path)
    ptf_match = bool(np.array_equal(ptf_resample, ptf_np))

    return {
        "K": K,
        "logits_per_point": logits_per_point,          # [N_sample, K]
        "hard_label_per_point": hard_label_per_point,  # [N_sample]
        "point_to_face": ptf_np,                       # [N_sample], indexes ref_mesh_F
        "sample_coords": sample_coords,                # [N_sample, 3], original world coords
        "ptf_resample_matches": ptf_match,             # sanity: re-sampled ptf == PartSAM's
        "n_masks_after_threshold": int(top.sum().item()),
        "iou_threshold": float(thr),                   # the value actually used (audit)
        "iou_threshold_requested": float(thr_req),     # != iou_threshold => 兜底阶梯救回来的
        "score_max": float(scores.max().item()) if scores.numel() else float("nan"),
    }


def partsam_output_to_soft_assign(partsam_out: dict,
                                  ref_mesh_V: np.ndarray,
                                  ref_mesh_F: np.ndarray) -> dict:
    """Point logits -> face logits -> vertex logits -> per-vertex soft simplex.

    Uses `point_to_face` (valid against ref_mesh_F, face order preserved). Faces
    with no sample point fall back to the nearest non-empty face's logits (KNN on
    face centroids). Vertex logits = mean over incident faces. Row softmax -> simplex.
    """
    from scipy.spatial import cKDTree

    logits = partsam_out["logits_per_point"].astype(np.float64)  # [Ns, K]
    ptf = partsam_out["point_to_face"]                           # [Ns]
    K = logits.shape[1]
    N_face = ref_mesh_F.shape[0]
    N_vertex = ref_mesh_V.shape[0]

    # ---- 1) point logits -> face logits ----
    face_sum = np.zeros((N_face, K), dtype=np.float64)
    face_cnt = np.zeros((N_face,), dtype=np.float64)
    np.add.at(face_sum, ptf, logits)
    np.add.at(face_cnt, ptf, 1.0)
    nonempty = face_cnt > 0
    face_logits = np.zeros((N_face, K), dtype=np.float64)
    face_logits[nonempty] = face_sum[nonempty] / face_cnt[nonempty, None]

    n_empty = int((~nonempty).sum())
    centroids = ref_mesh_V[ref_mesh_F].mean(axis=1)  # [N_face, 3]
    if n_empty > 0:
        tree = cKDTree(centroids[nonempty])
        _, nn = tree.query(centroids[~nonempty])
        face_logits[~nonempty] = face_logits[nonempty][nn]

    # ---- 2) face logits -> vertex logits (mean over incident faces) ----
    vsum = np.zeros((N_vertex, K), dtype=np.float64)
    vcnt = np.zeros((N_vertex,), dtype=np.float64)
    flat = ref_mesh_F.reshape(-1)                         # [3*N_face]
    np.add.at(vsum, flat, np.repeat(face_logits, 3, axis=0))
    np.add.at(vcnt, flat, 1.0)
    unref = vcnt == 0
    vcnt_safe = np.maximum(vcnt, 1.0)
    vertex_logits = vsum / vcnt_safe[:, None]

    # ---- 3) softmax -> simplex ----
    m = vertex_logits.max(axis=1, keepdims=True)
    ex = np.exp(vertex_logits - m)
    soft = ex / ex.sum(axis=1, keepdims=True)
    hard = soft.argmax(axis=1).astype(np.int64)
    max_dev = float(np.abs(soft.sum(axis=1) - 1.0).max())

    return {
        "soft_assign": soft.astype(np.float32),          # [N_vertex, K] simplex
        "hard_label_per_vertex": hard,                   # [N_vertex]
        "K": int(K),
        "simplex_max_row_dev": max_dev,
        "n_empty_faces": n_empty,
        "n_unreferenced_vertices": int(unref.sum()),
    }
