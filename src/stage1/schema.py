"""Stage 1 schema + logic — split out of the former monolithic SCHEMA.py.
Reference-frame selection, SAM 3D single-frame reconstruction, scene composition (Path 1/2),
and the stage0_to_stage1 bridge. InstanceMesh/SceneReconstruction/QualityThresholds/RefSelectConfig
etc. are re-exported by the SCHEMA.py shim."""
from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Optional
import numpy as np

from src.stage0.schema import VideoInput, Masklet
from src.geometry import (
    umeyama_similarity, _apply_transform, _centroid, _extent, _scale_about_centroid,
    _scale_about_point, _principal_axes, _obb_axes_from_corners, _rotation_angle_deg,
    _axis_alignment, _rotation_check, _to_np, _quat_wxyz_to_matrix, _similarity_from_quat,
    _extract_mesh, _layout_to_transform,
)
from src.stage1.gt_coverage import eligible_frames


# =============================================================================
# Sub-stage A: reference frame selection  (one t* per instance)
# =============================================================================

# =============================================================================
# Sub-stage A: reference frame selection  (one t* per instance)
# =============================================================================
# DESIGN (compromise): the PRIMARY selection method is a *learnable* scorer
# (LinearFrameScorer), trained by learning-to-rank against Path 2 reconstruction quality.
# The hand-set weights in RefSelectConfig are NOT the default scoring path; they serve two
# narrower roles only: (1) warm-start the scorer's training, (2) fall back when no trained
# checkpoint is loaded, so the pipeline is runnable on day one. As soon as a trained scorer
# is passed in, it takes over. Pipeline of this sub-stage:
#     mask_features (CONCRETE)  ->  LinearFrameScorer (learned weights | prior fallback)
#     -> select_reference_frames -> one ReferenceChoice (t*) per instance.

@dataclass
class RefSelectConfig:
    # PRIOR weights (hand-set). Used only to warm-start training and as a no-checkpoint
    # fallback — NOT the default once a trained scorer exists. Keep in FEATURE_NAMES order.
    w_area: float = 0.3
    w_completeness: float = 1.0
    w_occlusion: float = 0.8
    w_compactness: float = 0.4
    min_area_px: int = 256        # reject frames where the instance is too small
    border_penalty: float = 3.0   # punishes edge contact inside the completeness feature

    def prior_weights(self) -> np.ndarray:
        """Hand-prior weight vector, MUST stay in FEATURE_NAMES order."""
        return np.array([self.w_area, self.w_completeness, self.w_occlusion, self.w_compactness], float)


# Canonical feature order. mask_features(), RefSelectConfig.prior_weights(), and every
# scorer's `weights` MUST all agree on this ordering.
FEATURE_NAMES = ("area", "completeness", "occlusion", "compactness")


@dataclass
class ReferenceChoice:
    instance_id: int
    frame_index: int              # chosen reference frame t*
    mask: np.ndarray              # (H, W) bool at t*, full-frame aligned
    score: float


# -----------------------------------------------------------------------------
# Features (CONCRETE) — mask geometry, each normalized to [0,1]. This is the scorer's
# INPUT and is identical whether you use the hand prior or a trained model.
# -----------------------------------------------------------------------------

def mask_features(mask_t: np.ndarray, frame_t: np.ndarray, cfg: RefSelectConfig) -> np.ndarray:
    """Return the (4,) feature vector in FEATURE_NAMES order for one candidate frame.
    `frame_t` is currently unused (kept so appearance features like blur/contrast can be
    added later without changing call sites).
      area         : sqrt(visible-area / image-area).        Bigger silhouette -> finer recon.
      completeness : exp(-border_penalty * edge_coverage).   1 = not clipped by frame border.
      occlusion    : solidity = area / convex_hull_area.      1 = no concave occluder notch.
      compactness  : largest_component / total_area.          1 = single blob, not split.
    """
    m = np.asarray(mask_t, bool)
    H, W = m.shape
    area = int(m.sum())
    if area == 0:
        return np.zeros(len(FEATURE_NAMES), float)

    area_term = float(np.sqrt(area / float(H * W)))
    edge_coverage = float(m[0, :].mean() + m[-1, :].mean() + m[:, 0].mean() + m[:, -1].mean())
    completeness_term = float(np.exp(-cfg.border_penalty * edge_coverage))
    hull_area = _convex_hull_area(m)
    occlusion_term = float(np.clip(area / hull_area, 0.0, 1.0)) if hull_area > 0 else 0.0
    largest = _largest_component_area(m)
    compactness_term = float(largest / area)
    return np.array([area_term, completeness_term, occlusion_term, compactness_term], float)


def _convex_hull_area(m: np.ndarray) -> float:
    """Area of the convex hull of the mask's foreground pixels.
    Uses scipy if present; falls back to the bounding-box area (a looser upper bound,
    so solidity is under-estimated but still monotone enough to rank frames)."""
    ys, xs = np.nonzero(m)
    if xs.size < 3:
        return float(xs.size)
    try:
        from scipy.spatial import ConvexHull
        return float(ConvexHull(np.column_stack([xs, ys])).volume)  # 2D 'volume' == area
    except Exception:
        return float((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1))


def _largest_component_area(m: np.ndarray) -> int:
    """Pixel count of the largest 8-connected component.
    Needs scipy.ndimage.label; if unavailable, returns total area (i.e. compactness term
    degrades to 1.0 and stops discriminating — install scipy to keep this signal alive)."""
    try:
        from scipy.ndimage import label
        structure = np.ones((3, 3), dtype=int)  # 8-connectivity
        lab, n = label(m, structure=structure)
        if n <= 1:
            return int(m.sum())
        sizes = np.bincount(lab.ravel())[1:]
        return int(sizes.max())
    except Exception:
        return int(m.sum())


# -----------------------------------------------------------------------------
# Scorer (the LEARNABLE part, primary method). Lightweight tier = linear in the 4 features:
#     score = weights · features + bias
# `weights` come from learning-to-rank training; with no checkpoint, they default to the
# hand prior so selection still works.
# -----------------------------------------------------------------------------

@dataclass
class LinearFrameScorer:
    weights: np.ndarray          # (4,) in FEATURE_NAMES order
    bias: float = 0.0
    trained: bool = False        # True ONLY if weights came from a trained checkpoint

    @classmethod
    def from_prior(cls, cfg: RefSelectConfig) -> "LinearFrameScorer":
        """Hand-prior scorer — used both as the training warm-start and the day-one fallback.
        trained=False marks that this is the prior, not a learned model."""
        return cls(weights=cfg.prior_weights(), bias=0.0, trained=False)

    @classmethod
    def load(cls, path: str) -> "LinearFrameScorer":
        """Load a trained scorer (weights + bias) and set trained=True.
        TODO: implement (e.g. np.load a .npz with 'weights','bias'). Validate weights.shape == (4,)."""
        raise NotImplementedError

    def save(self, path: str) -> None:
        """Persist weights + bias. TODO: implement (e.g. np.savez(path, weights=..., bias=...))."""
        raise NotImplementedError

    def score(self, feats: np.ndarray) -> float:
        return float(self.weights @ feats + self.bias)


def score_frame(mask_t: np.ndarray, frame_t: np.ndarray, cfg: RefSelectConfig,
                scorer: "Optional[LinearFrameScorer]" = None) -> float:
    """Score one candidate frame. PRIMARY path uses a (trained) LinearFrameScorer; if
    `scorer` is None it falls back to the hand prior, so this is always runnable.
    Empty mask -> -inf (can never be chosen)."""
    if np.asarray(mask_t, bool).sum() == 0:
        return float("-inf")
    if scorer is None:
        scorer = LinearFrameScorer.from_prior(cfg)
    return scorer.score(mask_features(mask_t, frame_t, cfg))


def select_reference_frames(video: VideoInput, cfg: RefSelectConfig,
                            scorer: "Optional[LinearFrameScorer]" = None,
                            gt_cov: "Optional[dict]" = None) -> list[ReferenceChoice]:
    """Pick one t* per instance. Pass a trained `scorer` to use the learned model; omit it
    to fall back to the hand prior (day-one runnable).

    `gt_cov` (2026-08-10): per-instance GT dense-track coverage per frame, from
    `src/stage1/gt_coverage.load_gt_track_coverage`. Frames where this instance has too few
    visible GT tracks are DROPPED from the candidate set — stage ⑦ pairs strictly at t*, so
    such a t* is structurally unlinkable no matter how good the frame looks to the scorer.
    Ranking WITHIN the surviving frames is unchanged. None / no eligible frame => no constraint
    (see gt_coverage.eligible_frames), so datasets without GT tracks behave exactly as before."""
    if scorer is None:
        scorer = LinearFrameScorer.from_prior(cfg)
    choices: list[ReferenceChoice] = []
    T = video.frames.shape[0]
    for ml in video.masklets:
        elig = eligible_frames(gt_cov, ml.instance_id, T)
        best: Optional[ReferenceChoice] = None
        best_any: Optional[ReferenceChoice] = None      # 不加 GT 约束时会选的那一帧(仅用于日志)
        for t in range(T):
            m = ml.masks[t]
            if m.sum() < cfg.min_area_px:
                continue
            s = score_frame(m, video.frames[t], cfg, scorer)
            if best_any is None or s > best_any.score:
                best_any = ReferenceChoice(ml.instance_id, t, m, s)
            if elig is not None and not elig[t]:
                continue
            if best is None or s > best.score:
                best = ReferenceChoice(ml.instance_id, t, m, s)
        if best is None:
            best = best_any        # 合格帧全都没过面积门槛 -> 诚实退回原选择
        elif best_any is not None and best.frame_index != best_any.frame_index:
            cov_i = gt_cov[int(ml.instance_id)]
            print(f"[stage1] inst {ml.instance_id}: t* {best_any.frame_index} -> "
                  f"{best.frame_index} (GT 覆盖 {int(cov_i[best_any.frame_index])} -> "
                  f"{int(cov_i[best.frame_index])} 条)", flush=True)
        if best is None:
            # Instance never well-observed (typically an over-enumerated tiny/background concept).
            # SKIP it instead of raising — a missing reference frame just means this ONE instance
            # gets no mesh; killing the whole scene over a background pole/plant loses the real
            # subjects too (per-instance robustness, mirrors Stage 2a's per-instance SKIP).
            print(f"[stage1] instance {ml.instance_id} ({ml.concept!r}): no valid reference frame "
                  f"-> skipped", flush=True)
            continue
        choices.append(best)
    return choices


def select_reference_frames_topk(video: VideoInput, cfg: RefSelectConfig,
                                 scorer: "Optional[LinearFrameScorer]" = None,
                                 k: int = 4, min_gap: int = 2,
                                 gt_cov: "Optional[dict]" = None) -> "dict[int, list[ReferenceChoice]]":
    """Per instance, return up to `k` candidate reference frames (best hand-prior/learned score
    first), kept temporally APART by >= `min_gap` frames so the candidates show genuinely
    different views (a different t* changes self-occlusion AND the SAM 3D reconstruction). The
    multi-candidate driver reconstructs each and keeps the one whose placed mesh best matches the
    mask — the GT-free way to escape a bad single-frame reconstruction. The first entry equals
    select_reference_frames()'s pick, so best-of-1 == the single-frame behaviour.

    `gt_cov`: same GT dense-track coverage filter as select_reference_frames — applied to the
    candidate POOL, so every candidate (not just the first) is one stage ⑦ can actually link."""
    if scorer is None:
        scorer = LinearFrameScorer.from_prior(cfg)
    out: dict[int, list[ReferenceChoice]] = {}
    T = video.frames.shape[0]
    for ml in video.masklets:
        elig = eligible_frames(gt_cov, ml.instance_id, T)
        scored, scored_any = [], []
        for t in range(T):
            m = ml.masks[t]
            if m.sum() < cfg.min_area_px:
                continue
            e = (score_frame(m, video.frames[t], cfg, scorer), t, m)
            scored_any.append(e)
            if elig is None or elig[t]:
                scored.append(e)
        if not scored:
            scored = scored_any        # 合格帧全都没过面积门槛 -> 诚实退回原候选集
        scored.sort(key=lambda x: -x[0])
        scored_any.sort(key=lambda x: -x[0])
        if scored and scored_any and scored[0][1] != scored_any[0][1]:
            cov_i = gt_cov[int(ml.instance_id)]
            print(f"[stage1] inst {ml.instance_id}: t* {scored_any[0][1]} -> {scored[0][1]} "
                  f"(GT 覆盖 {int(cov_i[scored_any[0][1]])} -> {int(cov_i[scored[0][1]])} 条)",
                  flush=True)
        chosen: list[ReferenceChoice] = []
        for s, t, m in scored:
            if all(abs(t - c.frame_index) >= min_gap for c in chosen):
                chosen.append(ReferenceChoice(ml.instance_id, t, m, s))
            if len(chosen) >= k:
                break
        if not chosen:
            print(f"[stage1] instance {ml.instance_id} ({ml.concept!r}): no valid reference frame "
                  f"-> skipped", flush=True)
            continue
        out[ml.instance_id] = chosen
    return out


# -----------------------------------------------------------------------------
# Training the scorer (LEARNING-TO-RANK). OFFLINE and Kubric-only: labels come from
# actually reconstructing each candidate frame (Sub-stage B) and grading it with the
# Path 2 quality check (_grade_against_gt -> ReconQuality). Only the ARGMAX frame matters
# at inference, so we rank frames WITHIN an instance rather than regress absolute scores.
# -----------------------------------------------------------------------------

@dataclass
class FrameSample:
    """One labeled candidate frame for training. Pairs are formed only WITHIN a group_id."""
    features: np.ndarray   # (4,) from mask_features, FEATURE_NAMES order
    quality: float         # supervision target, higher = better recon (see build_quality_label)
    group_id: str          # e.g. f"{video_id}:{instance_id}" — one ranking group per instance


def build_quality_label(rq: "ReconQuality") -> float:
    """Map a Path 2 ReconQuality into a scalar 'higher = better' training target.
    NOTE (GT-ray Path 2): scale_ratio is now ~1 (uninformative — drop it), and position_error
    measures reconstruction/placement residual, NOT SAM 3D pose — which is exactly the signal
    frame selection wants (pick the frame that reconstructs best). rotation_error_deg is a new
    quality signal but only for non-degenerate (non-symmetric) objects.
    Suggested form (TODO tune the coefficients):
        if rq.flagged: return a large negative (e.g. -1e3) so bad recons sink in their group
        label = -rq.position_error
        if rq.rotation_error_deg is not None: label -= beta * rq.rotation_error_deg
    Keep it MONOTONE in reconstruction quality. Must be comparable within a group_id."""
    raise NotImplementedError


def make_training_samples(videos, inference, cfg: RefSelectConfig,
                          qt: "QualityThresholds") -> list[FrameSample]:
    """Build FrameSamples by running Sub-stage B + Path 2 grading on candidate frames.
    OFFLINE and EXPENSIVE (one SAM 3D reconstruction per candidate frame).
    TODO per (video, instance), for each candidate frame t (passing cfg.min_area_px):
        feats = mask_features(mask_t, frame_t, cfg)
        out   = run_sam3d_single(frame_t, mask_t, seed, inference)         # Sub-stage B
        place into the GT world frame, then rq = _grade_against_gt(...)     # Path 2
        sample = FrameSample(feats, build_quality_label(rq), f"{video_id}:{instance_id}")
    Return the flat list across all instances. To bound cost, subsample candidate frames."""
    raise NotImplementedError


def train_linear_scorer(samples: list[FrameSample], cfg: RefSelectConfig,
                        lr: float = 1e-2, epochs: int = 200, l2: float = 1e-4) -> LinearFrameScorer:
    """Learning-to-rank on the 4 mask features. PAIRWISE LOGISTIC loss WITHIN each group:
        for every pair (i, j) in the same group_id with quality_i > quality_j,
            d = score_i - score_j ;  loss += softplus(-d)
        plus l2 * ||weights||^2 .
    Warm-start `weights` from cfg.prior_weights() (the hand prior) and `bias`=0.
    Plain numpy gradient descent is enough (4 parameters). Set trained=True on the result.
    TODO: implement the optimizer loop; return the trained LinearFrameScorer."""
    raise NotImplementedError


# =============================================================================
# Sub-stage B: per-instance SAM 3D Objects call
# =============================================================================

@dataclass
class SAM3DRawOutput:
    instance_id: int
    mesh: object                       # exported mesh (verts/faces) or path — STATIC, canonical
    gaussians: object = None           # optional gs (output["gs"])
    object_pose: Optional[np.ndarray] = None   # (4,4) object->camera at reference frame
    layout: object = None              # SAM 3D layout info, used for scene composition
    reference_frame_index: int = -1    # carried through for Stage 4 seeding
    sam3d_intrinsics: Optional[np.ndarray] = None  # (3,3) pixel K of SAM 3D's own (pointmap)
    #   camera; with object_pose this projects keypoints back onto the t* image (Stage 4 query),
    #   GT-independent. Derived from out["pointmap"] (no 'intrinsics' key in the output dict).
    # (a `pointmap` passthrough was removed 2026-06-30 along with the Path-1 re-anchor that used it.)


def run_sam3d_single(image_rgb: np.ndarray, mask: np.ndarray, seed: int, inference) -> SAM3DRawOutput:
    """Wrap ONE SAM 3D Objects call.
    image_rgb : full frame (H,W,3) uint8 — not cropped
    mask      : full-frame-aligned (H,W) bool for this single instance
    inference : a `notebook.inference.Inference` instance (loaded from checkpoints/hf/pipeline.yaml).

    `inference(image, mask, seed=...)` runs the full image->3D pipeline and returns a dict
    (sam3d_objects InferencePipelinePointMap.run; structure verified on this build). We keep:
      - 'mesh' : list[MeshExtractResult] (.vertices/.faces) in the CANONICAL frame
                 (verts ~[-0.5, 0.5]); the pose below maps it into the camera frame.
      - 'gs'   : Gaussian splat (optional downstream).
      - pose   : 'rotation' (1,4) wxyz quaternion (pytorch3d convention), 'translation' (1,3),
                 'scale' (1,3 per-axis) — places the canonical mesh into the camera/pointmap
                 frame:  v_cam = R(quat) @ (scale ⊙ v_canonical) + t   (camera looks +z).
    object_pose (object->camera) is composed here as a (4,4) similarity; compose_scene then
    chains it with the (GT or SAM3D-layout) camera->scene transform. `glb` (a posed Trimesh)
    is export-only and intentionally NOT used (we keep canonical verts + an explicit pose).
    """
    # with_layout_postprocess=True: SAM 3D optimizes the object pose against its own pointmap
    # (orientation-aware) -> correct placement. We previously disabled it and rolled our own
    # re-anchor + silhouette-IoU optimize, which (after the R->R.T rotation fix) was shown to
    # DEGRADE orientation (silhouette IoU is orientation-blind); both removed 2026-06-30.
    out = inference(image_rgb, mask, seed=seed, with_layout_postprocess=True)   # __call__ -> dict
    mesh_list = out["mesh"]
    mesh = mesh_list[0] if isinstance(mesh_list, (list, tuple)) else mesh_list   # canonical MeshExtractResult
    quat = _to_np(out["rotation"]).reshape(-1)[:4]         # (4,) wxyz
    trans = _to_np(out["translation"]).reshape(-1)[:3]     # (3,)
    scale = _to_np(out["scale"]).reshape(-1)[:3]           # (3,) per-axis (usually uniform)
    object_pose = _similarity_from_quat(quat, trans, scale)   # (4,4) object(canonical)->camera
    return SAM3DRawOutput(
        instance_id=-1,                                    # identity filled by reconstruct_all
        mesh=mesh,
        gaussians=out.get("gs"),
        object_pose=object_pose,
        layout={"rotation_wxyz": quat, "translation": trans, "scale": scale, "matrix": object_pose},
        sam3d_intrinsics=_intrinsics_from_pointmap(out.get("pointmap")),
    )


def _intrinsics_from_pointmap(pointmap) -> Optional[np.ndarray]:
    """Fit SAM 3D's pixel intrinsics (3,3) from its pointmap (H,W,3 camera-frame points,
    at the INPUT frame resolution). For each valid pixel (u,v): u = fx*X/Z+cx, v = fy*Y/Z+cy
    -> least squares. object_pose shares this camera's rays, so K @ (object_pose @ canonical)
    projects keypoints back onto the t* image (scale-invariant; only X/Z ray matters). fx may
    come out NEGATIVE (SAM 3D's axis convention) — that is consistent and correct here."""
    if pointmap is None:
        return None
    pm = _to_np(pointmap)
    if pm.ndim != 3 or pm.shape[-1] != 3:
        return None
    H, W = pm.shape[:2]
    Z = pm[..., 2]
    valid = Z > 0.05
    if int(valid.sum()) < 100:
        return None
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    xz = (pm[..., 0] / Z)[valid]
    yz = (pm[..., 1] / Z)[valid]
    fx, cx = np.linalg.lstsq(np.stack([xz, np.ones_like(xz)], 1), us[valid].astype(float), rcond=None)[0]
    fy, cy = np.linalg.lstsq(np.stack([yz, np.ones_like(yz)], 1), vs[valid].astype(float), rcond=None)[0]
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], float)


def reconstruct_all(video: VideoInput, choices: list[ReferenceChoice],
                    inference, seed: int = 42) -> list[SAM3DRawOutput]:
    outs: list[SAM3DRawOutput] = []
    for ch in choices:
        img = video.frames[ch.frame_index]          # FULL frame
        out = run_sam3d_single(img, ch.mask, seed, inference)
        out.reference_frame_index = ch.frame_index  # keep the t* -> Stage 4 needs it
        out.instance_id = ch.instance_id            # identity from the masklet/ReferenceChoice
        outs.append(out)
    return outs


# =============================================================================
# Sub-stage C: compose into one shared scene coordinate frame
# =============================================================================

@dataclass
class InstanceMesh:
    """Stage 1 deliverable for one instance -> consumed by Stage 2."""
    instance_id: int
    verts: np.ndarray                 # (V,3) in the SHARED scene frame
    faces: np.ndarray                 # (F,3)
    reference_frame_index: int        # t* this mesh was built from
    object_to_scene: np.ndarray       # (4,4) transform used to place it in the scene
    # --- carried for Stage 4 keypoint seeding (project scene-frame pts -> ref image) ---
    camera_to_scene: Optional[np.ndarray] = None  # (4,4) GT camera->scene (Path 2); None on Path 1
    intrinsics: Optional[np.ndarray] = None        # (3,3) GT K at t*
    # SAM 3D's OWN camera (always available, Path 1 & 2). This is the RELIABLE projection for
    # Stage 4 queries: object_pose_cam @ inv(object_to_scene) @ p -> SAM 3D camera frame, then
    # sam3d_intrinsics -> the t* pixel where the keypoint actually appears (GT camera mis-projects
    # due to the SAM3D-vs-scene FOV/orientation mismatch — verified ~70% on-mask vs ~0% via GT).
    object_pose_cam: Optional[np.ndarray] = None   # (4,4) canonical -> SAM 3D camera (raw pose)
    sam3d_intrinsics: Optional[np.ndarray] = None   # (3,3) pixel K of SAM 3D's pointmap camera
    # --- Path 2 only: applied to put SAM 3D's arbitrary scale into metric, + quality grade ---
    scale_correction: float = 1.0     # 1.0 if not applied (Path 1)
    quality: Optional["ReconQuality"] = None   # None under Path 1 (no GT to grade against)


@dataclass
class ReconQuality:
    """Path 2 sanity check of SAM 3D's reconstruction against Kubric GT.
    Each field is None when not computable, so 'not checked' never masquerades as 'passed'.
    """
    position_error: Optional[float] = None     # world-frame centroid distance to GT object (metric)
    scale_ratio: Optional[float] = None         # placed-extent / GT-extent  (1.0 = ideal)
    rotation_error_deg: Optional[float] = None  # needs GT mesh / correspondences; else None
    flagged: bool = False                       # True => bad Stage 1 sample, exclude downstream
    notes: str = ""


@dataclass
class QualityThresholds:
    max_position_error: float = 0.15   # metric units of your GT world (tune to scene scale)
    scale_ratio_lo: float = 0.7        # flag if placed/GT extent outside [lo, hi]
    scale_ratio_hi: float = 1.4
    max_rotation_error_deg: float = 30.0


@dataclass
class SceneReconstruction:
    instances: list[InstanceMesh]
    scene_frame: str = "sam3d_layout"  # "sam3d_layout" (estimated) | "gt_camera" (known)
    flagged_ids: list = None           # instance_ids that failed the Path 2 check


def compose_scene(raw_outputs: list[SAM3DRawOutput],
                  video: VideoInput,
                  qt: QualityThresholds = QualityThresholds()) -> SceneReconstruction:
    """Place every per-instance mesh into ONE shared coordinate frame.

    Dispatch:
      - GT camera available (video.camera_extrinsics is not None)  -> Path 2
      - otherwise                                                  -> Path 1 (always works)

    A and B (reference-frame selection, SAM 3D meshing) have ALREADY run regardless of
    path; only the placement differs here. Path 1 is the unconditional fallback, so every
    dataset has a route through C.
    """
    if video.camera_extrinsics is not None:
        return _compose_path2_gt_camera(raw_outputs, video, qt)
    return _compose_path1_sam3d_layout(raw_outputs, video)


# ---- Path 1: SAM 3D native multi-object layout (no GT) ----------------------

def _project_cam(verts_cam: np.ndarray, K: np.ndarray):
    """Project (V,3) camera-frame points with pixel intrinsics K -> ((V,2) uv, valid mask).
    Matches viz_stage1 / build_tracker_query (fx may be negative — SAM 3D axis convention)."""
    z = verts_cam[:, 2]
    safe = (np.abs(z) > 1e-9) & (z > 0)
    zc = np.where(np.abs(z) > 1e-9, z, 1.0)
    u = K[0, 0] * verts_cam[:, 0] / zc + K[0, 2]
    v = K[1, 1] * verts_cam[:, 1] / zc + K[1, 2]
    return np.stack([u, v], 1), safe


def _k_grid_scale(K: np.ndarray, W: int, H: int) -> float:
    """Upscale factor from K's OWN pixel grid to the (W,H) frame. SAM3D's pipeline returns a
    pointmap that may be an ISOTROPIC power-of-two DOWNSAMPLE of the input frame
    (_down_sample_img: x0.5 for max_side>1200, x0.25 for >1900 — e.g. 1920x1280 -> 480x320),
    and `sam3d_intrinsics` is least-squares-fit IN THAT GRID (principal point ~ grid center).
    Comparing K-projections against the full-res mask then shrinks them by the factor ->
    on-mask collapses to ~0 (TAPVid-3D drivetrack; same root cause as the stage-⑦ fix, see
    src/stage4/run_link_dense.py::_k_grid_wh). Recover the factor SELF-ADAPTIVELY from K
    (2*cx ~ grid width), snapped to the pipeline's power-of-two set; frames that were not
    downsampled (DAVIS 854, PO 960) snap to 1.0 -> bit-identical behavior there."""
    fac = np.array([1.0, 2.0, 4.0, 8.0])
    gw, gh = max(2.0 * float(K[0, 2]), 1e-9), max(2.0 * float(K[1, 2]), 1e-9)
    err = np.abs(np.log(gw * fac / W)) + np.abs(np.log(gh * fac / H))
    return float(fac[int(np.argmin(err))])


def _on_mask_frac(verts_cam: np.ndarray, K: np.ndarray, mask: np.ndarray) -> float:
    """Fraction of projected (front-facing) vertices that land inside `mask`.
    Projection happens in K's own pixel grid, then upscaled to the mask's frame (no-op factor
    1.0 unless SAM3D downsampled its pointmap — see _k_grid_scale)."""
    uv, valid = _project_cam(verts_cam, K)
    H, W = mask.shape
    uv = uv[valid] * _k_grid_scale(K, W, H)
    inb = (uv[:, 1] >= 0) & (uv[:, 1] < H) & (uv[:, 0] >= 0) & (uv[:, 0] < W)
    if not inb.any():
        return 0.0
    xi = uv[inb].astype(int)
    return float(mask[xi[:, 1], xi[:, 0]].mean())


def _place_one_path1(r: SAM3DRawOutput, mask_t=None) -> InstanceMesh:
    """Path 1 placement for ONE raw SAM 3D output: use SAM 3D's OWN pose, which is now produced WITH
    its layout_post_optimization (optimizes the pose against the pointmap = orientation-aware) and
    with the corrected R->R.T rotation. We previously re-anchored to the pointmap centroid and then
    maximized silhouette-IoU here; an orientation metric (posed-mesh vs pointmap 3D) showed that
    raised on-mask but DEGRADED orientation (silhouette IoU is orientation-blind), so both were
    removed 2026-06-30. `mask_t` kept for call-compat (unused). Scene frame == SAM 3D camera frame."""
    verts, faces = _extract_mesh(r.mesh)
    obj_to_scene = _layout_to_transform(r.layout)     # SAM 3D's layout-post-optimized, rotation-fixed pose
    verts_scene = _apply_transform(verts, obj_to_scene)
    return InstanceMesh(
        instance_id=r.instance_id, verts=verts_scene, faces=faces,
        reference_frame_index=r.reference_frame_index,
        object_to_scene=obj_to_scene,
        object_pose_cam=obj_to_scene,                 # Path 1: scene frame == SAM 3D camera frame
        sam3d_intrinsics=r.sam3d_intrinsics, quality=None)


def _instance_on_mask(inst: InstanceMesh, mask_t) -> float:
    """Fraction of the placed mesh's vertices that project inside `mask_t` (the selection metric
    used to pick among candidate reference frames). 0.0 when un-projectable."""
    if mask_t is None or inst.sam3d_intrinsics is None or inst.object_pose_cam is None:
        return 0.0
    M = np.asarray(inst.object_pose_cam) @ np.linalg.inv(np.asarray(inst.object_to_scene))
    v_cam = _apply_transform(inst.verts, M)
    return _on_mask_frac(v_cam, inst.sam3d_intrinsics, np.asarray(mask_t, bool))


def _compose_path1_sam3d_layout(raw_outputs: list[SAM3DRawOutput],
                                video: "Optional[VideoInput]" = None) -> SceneReconstruction:
    """Use SAM 3D's own estimated layout/pose (produced WITH its layout_post_optimization, which
    optimizes the pose against the pointmap) to compose the scene. No GT reference => no quality
    grade (quality stays None; do not fake a pass). (The former re-anchor + silhouette-IoU pose
    correction was removed 2026-06-30 — it degraded orientation; see _place_one_path1.)
    """
    mask_by_id = {ml.instance_id: ml.masks for ml in video.masklets} if video is not None else {}
    instances = []
    for r in raw_outputs:
        masks_i = mask_by_id.get(r.instance_id)
        mask_t = (masks_i[r.reference_frame_index]
                  if masks_i is not None and 0 <= r.reference_frame_index < len(masks_i) else None)
        instances.append(_place_one_path1(r, mask_t))
    return SceneReconstruction(instances=instances, scene_frame="sam3d_layout", flagged_ids=[])


# ---- Path 2: GT camera placement + reconstruction sanity check --------------

# pytorch3d camera (SAM 3D's internal convention) -> OpenCV camera (our GT extrinsics/intrinsics).
# 180 deg roll about the viewing axis. See the long note at its use site in Path 2.
_CAM_CONV_FIX = np.diag([-1.0, -1.0, 1.0, 1.0])


def _compose_path2_gt_camera(raw_outputs: list[SAM3DRawOutput],
                             video: VideoInput,
                             qt: QualityThresholds) -> SceneReconstruction:
    """Place each mesh into the GT world frame via the GT camera at its t*,
    then grade SAM 3D's reconstruction against GT (position, scale; rotation if possible).

    PLACEMENT (GT-ray): SAM 3D's object pose lives in *its own* canonical camera (assumed
    FOV/framing), so trusting its lateral translation under the real GT camera mis-maps the
    object's direction — the error grows with field angle (validated: tangential error ∝
    image eccentricity, r≈0.79). SAM 3D's *range* (depth), however, is fine once de-scaled.
    So we split the two:
      - DIRECTION comes from the GT camera ray through the object's MASK CENTROID — exactly
        known, no dependence on SAM 3D's camera model.
      - RANGE (distance along that ray) = f * W_perp / px, also from GT geometry: the metric
        mesh's extent W_perp perpendicular to the ray must project to the mask's pixel extent
        px under the GT focal length f. (SAM 3D's own range is FOV-contaminated too, so it is
        used only as a fallback when the mask is degenerate.)
      - SHAPE/metric size come from SAM 3D (scaled by s = gt_extent/placed_extent).
      - ORIENTATION: SAM 3D's, kept as-is, but CHECKED against the GT OBB principal axes
        (from gt_object_bbox) -> rotation_error_deg, recorded as a DIAGNOSTIC only. (Re-
        orienting to GT was net-negative for position, so it is not applied; rotation does
        not gate `flagged`.)
    The object centroid is thus pinned on the true viewing ray at the GT-implied metric range;
    any residual is the depth/silhouette error + GT-pivot-vs-centroid offset, NOT the FOV
    mismatch. Placement (not pose diagnosis) is the goal: since direction AND range come from
    GT, position_error no longer measures SAM 3D's pose. Falls back to a camera-origin rescale
    when intrinsics/mask/extent are missing.
    """
    mask_by_id = {ml.instance_id: ml.masks for ml in video.masklets}
    instances, flagged = [], []
    for r in raw_outputs:
        verts, faces = _extract_mesh(r.mesh)
        t_star = r.reference_frame_index

        # world<-camera at t*  (GT). Accept (T,4,4) or a single (4,4).
        E = video.camera_extrinsics
        world_from_cam = E[t_star] if E.ndim == 3 else E
        K = video.camera_intrinsics
        intr = (K[t_star] if (K is not None and K.ndim == 3) else K)

        # CAMERA-CONVENTION FIX (2026-08-09). SAM 3D works in the pytorch3d camera convention
        # (+X left, +Y up, +Z forward — see the `camera_to_pytorch3d_camera` transforms its own
        # pipeline applies to the MoGe pointmap), while GT extrinsics/intrinsics here are OpenCV
        # (+X right, +Y down, +Z forward). The two differ by a 180 deg roll about the viewing
        # axis: diag(-1,-1,1). Composing `world_from_cam @ M_fix @ cam_from_obj` without it left
        # every placed mesh rolled 180 deg in the world frame.
        # EVIDENCE (Stereo4D, 127 instances): stage-7's fitted W_track->W_mesh rotation — which
        # must be ~0 once both frames are W_gt — had median 174.7 deg before and 15.6 deg after
        # (the fast path, whose pose is ICP-fitted to GT points, sits at 16.8 deg). Position and
        # scale are UNCHANGED by this (the centroid still lands at `rng * d_cam`, and the roll
        # preserves the perpendicular bbox extent that sets `rng`), so it cannot regress the
        # Path-2 position_error/scale_ratio grades on MOVi/Kubric — it only fixes orientation.
        # Stage-4's own projection is unaffected too: `object_pose_cam` below still carries SAM
        # 3D's RAW pose in SAM 3D's own camera, which is what `sam3d_intrinsics` pairs with.
        cam_from_obj = _CAM_CONV_FIX @ r.object_pose   # (4,4), canonical -> GT-convention camera
        v_sam = _apply_transform(verts, cam_from_obj)  # mesh in the GT camera frame
        c_sam = _centroid(v_sam)
        gt_ext = (video.gt_object_extent or {}).get(r.instance_id)
        # FRONTAL-size input (2026-08-10). See VideoInput.gt_object_frontal_extent: when the
        # dataset can only measure the object's silhouette size, feeding it as `gt_object_extent`
        # is a UNIT MISMATCH — the divisor `placed_ext` is the mesh's FULL bbox diagonal (includes
        # the along-ray thickness) while the silhouette has none, so every mesh comes out too
        # small by perp/full (measured median 0.906, p10 0.681 — shape-dependent, not a constant).
        # With this field the divisor becomes the mesh's own PERPENDICULAR extent, so both sides
        # are frontal and the range formula collapses to rng = f*frontal/w_pix = the GT depth.
        frontal = (getattr(video, "gt_object_frontal_extent", None) or {}).get(r.instance_id)

        masks_i = mask_by_id.get(r.instance_id)
        mask_t = masks_i[t_star] if masks_i is not None else None
        placed_ext = _extent(v_sam)
        use_ray = (intr is not None and (gt_ext is not None or frontal is not None)
                   and placed_ext > 1e-9
                   and mask_t is not None and bool(np.asarray(mask_t).any())
                   and float(np.linalg.norm(c_sam)) > 1e-9)

        rot_err, rot_degenerate = None, False
        if use_ray:
            # ---- GT-ray placement: DIRECTION + RANGE both from GT geometry ----
            mask_arr = np.asarray(mask_t)
            ys, xs = np.nonzero(mask_arr)
            u, v = float(xs.mean()), float(ys.mean())          # mask centroid pixel
            fx, fy, cx, cy = intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]
            d_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
            d_cam /= np.linalg.norm(d_cam)                      # true viewing ray (camera frame)
            shape0 = v_sam - c_sam                              # unscaled, centred at origin
            # perpendicular extent of the UNSCALED mesh — the divisor that matches a frontal size
            perp0 = shape0 - np.outer(shape0 @ d_cam, d_cam)
            perp0_ext = float(np.linalg.norm(perp0.max(0) - perp0.min(0)))
            if frontal is not None and perp0_ext > 1e-9:
                scale, size_src = frontal / perp0_ext, "frontal/perp"
            else:
                scale, size_src = gt_ext / placed_ext, "full/full"
            metric_shape = scale * shape0                      # metric-sized, centred at origin

            # ---- rotation DIAGNOSTIC (no re-orient), in CAMERA frame ----
            # Compare the SAM mesh's PCA axes to the GT OBB principal axes -> SAM's raw
            # orientation error. RECORDED ONLY: re-orienting the mesh to GT was net-negative
            # for position and destructive at large/uncertain angles (3-scene batch study), so
            # SAM 3D's orientation is kept. Symmetric objects (ambiguous axis order) -> ungraded.
            corners_gt = None
            if video.gt_object_bbox is not None:
                cb = video.gt_object_bbox.get(r.instance_id)
                if cb is not None:
                    cb = np.asarray(cb, float)
                    corners_gt = cb[t_star] if cb.ndim == 3 else cb
            if corners_gt is not None:
                R_c2w, t_c2w = world_from_cam[:3, :3], world_from_cam[:3, 3]
                corners_cam = (corners_gt.reshape(-1, 3) - t_c2w) @ R_c2w   # world -> camera
                A_sam, _ = _principal_axes(metric_shape)
                A_gt, ext_gt = _obb_axes_from_corners(corners_cam)
                e = np.sort(ext_gt)[::-1]
                rot_degenerate = bool((e[1] > 1e-9 and e[0] / e[1] < 1.15)
                                      or (e[2] > 1e-9 and e[1] / e[2] < 1.15) or e[2] < 1e-9)
                _, rot_err = _axis_alignment(A_sam, A_gt)

            # RANGE from GT geometry, NOT SAM 3D's (FOV-contaminated) depth: the metric mesh's
            # extent perpendicular to the ray must project to the mask's pixel extent.
            #   proj_px = f * W_perp / R   ->   R = f * W_perp / px
            # (width_metric == scale * perp0_ext; with a frontal size that is EXACTLY `frontal`,
            #  so rng = f*frontal/w_pix — the object lands at the depth the silhouette implies.)
            w_pix = float(np.hypot(xs.max() - xs.min(), ys.max() - ys.min()))
            width_metric = scale * perp0_ext
            f_px = 0.5 * (float(fx) + float(fy))
            if w_pix > 1.0 and width_metric > 1e-9:
                rng = f_px * width_metric / w_pix
                method = f"gt_ray+gt_range({size_src})"
            else:
                rng = scale * float(np.linalg.norm(c_sam))
                method = "gt_ray"
            target_c = rng * d_cam                             # object centroid on the GT ray

            M_fix = np.eye(4)                                  # v_cam -> scaled, recentred on ray
            M_fix[:3, :3] = scale * np.eye(3)
            M_fix[:3, 3] = target_c - scale * c_sam
            obj_to_scene = world_from_cam @ M_fix @ cam_from_obj
        else:
            # ---- fallback: camera-origin rescale (no ray; keeps SAM 3D direction) ----
            world_from_obj = world_from_cam @ cam_from_obj
            scale = 1.0
            obj_to_scene = world_from_obj
            if gt_ext is not None and placed_ext > 1e-9:
                scale = gt_ext / placed_ext
                cam_center = world_from_cam[:3, 3]
                obj_to_scene = _scale_about_point(cam_center, scale) @ world_from_obj
            method = "cam_origin_rescale"

        verts_world = _apply_transform(verts, obj_to_scene)

        # ---- sanity check against GT ----
        quality = _grade_against_gt(verts_world, r.instance_id, t_star, video, qt)
        quality.notes = f"placement={method}; " + quality.notes
        # rotation: recorded as a DIAGNOSTIC only (trustworthy for non-degenerate objects);
        # does NOT gate `flagged` — orientation is uniformly poor and position drives usability.
        if rot_err is not None and not rot_degenerate:
            quality.rotation_error_deg = rot_err
        elif rot_degenerate:
            quality.notes = "rot=degenerate(symmetric,ungraded); " + quality.notes
        if quality.flagged:
            flagged.append(r.instance_id)

        instances.append(InstanceMesh(
            instance_id=r.instance_id,
            verts=verts_world, faces=faces,
            reference_frame_index=t_star,
            object_to_scene=obj_to_scene,
            camera_to_scene=world_from_cam,
            intrinsics=intr,
            object_pose_cam=r.object_pose,            # SAM 3D camera (for Stage 4 query)
            sam3d_intrinsics=r.sam3d_intrinsics,
            scale_correction=scale,
            quality=quality,
        ))
    return SceneReconstruction(instances=instances, scene_frame="gt_camera", flagged_ids=flagged)


def _grade_against_gt(verts_world: np.ndarray, instance_id: int, t_star: int,
                      video: VideoInput, qt: QualityThresholds) -> ReconQuality:
    """Grade a placed mesh against GT. Robust checks (position, scale) are computed here;
    rotation is left None unless GT correspondences are supplied (see umeyama_similarity).
    """
    q = ReconQuality()
    poses = video.gt_object_poses or {}
    if instance_id not in poses:
        q.notes = "no GT object pose; placement done but ungraded"
        return q

    P = poses[instance_id]
    world_from_obj_gt = P[t_star] if P.ndim == 3 else P
    gt_center = world_from_obj_gt[:3, 3]

    # Position: centroid distance is convention-light (origin offset cancels at centroid).
    q.position_error = float(np.linalg.norm(_centroid(verts_world) - gt_center))

    # Scale: placed extent vs GT extent (1.0 ideal). Only meaningful if GT extent known.
    gt_ext = (video.gt_object_extent or {}).get(instance_id)
    if gt_ext is not None and gt_ext > 1e-9:
        q.scale_ratio = float(_extent(verts_world) / gt_ext)

    # Rotation: left None here; _compose_path2_gt_camera fills rotation_error_deg from the
    # GT OBB principal axes (canonical-frame-free) when gt_object_bbox is available.
    q.notes = "position+scale graded; rotation set by compose if GT OBB available"

    q.flagged = (
        (q.position_error is not None and q.position_error > qt.max_position_error)
        or (q.scale_ratio is not None and not (qt.scale_ratio_lo <= q.scale_ratio <= qt.scale_ratio_hi))
    )
    return q


def _reconstruct_best_of_k(video: VideoInput, inference, ref_cfg: RefSelectConfig,
                           scorer, seed: int, n_candidates: int,
                           accept_onmask: float, gt_cov=None,
                           cand_log=None, group_prefix: str = "",
                           abandon_frac: float = 0.0, abandon_cap: int = 2,
                           engine: str = "sam3d", onmask_gate=None
                           ) -> list[SAM3DRawOutput]:
    """Multi-candidate reconstruction: per instance, reconstruct from up to `n_candidates`
    reference frames (best score first) and KEEP the raw output whose placed mesh best matches
    the mask (on-mask). Early-stops once a candidate reaches `accept_onmask`, so a good first
    frame costs exactly one SAM 3D call — only the hard instances pay for extra frames. Lifts
    the worst single-frame reconstructions (a different t* changes self-occlusion + shape).

    Returns RAW outputs, not placed instances (2026-08-09 refactor): selection and placement
    used to be fused here, which made this Path-1-only and silently disabled multi-candidate
    the moment a GT camera appeared (Path 2). They are independent — the selection metric only
    needs SAM 3D's OWN camera (`object_pose_cam` + `sam3d_intrinsics`), which both paths have —
    so the caller now feeds these through the normal `compose_scene` dispatch. Path 1 output is
    bit-identical to before (`_place_one_path1` is deterministic and re-runs on the same input).

    `cand_log` (P0 插桩,2026-08-12):每试一个候选就落一行 JSONL(rank/frame/4 特征/on_mask/
    是否被接受)。这个量本来**算完就丢** —— 它是早停判据本身,却从没被持久化,导致所有关于
    "候选数该怎么调"的分析都只能靠事后反算。零 GPU 成本,是自适应停止规则唯一的数据来源。

    `abandon_frac` > 0:用 top-1 候选的 mask 特征预判"这个实例注定达不到 accept_onmask",
    命中就把候选数砍到 `abandon_cap`(见 src/stage1/abandon.py 的实测收益表)。

    `engine` / `onmask_gate`(2026-08-19):提前放弃的两个模型是**按引擎分开**的 —— 同一批
    mask 特征在官方 SAM3D 上 CV AUC 0.716,在 Fast-SAM3D 上只有 0.546(失败成因多了一层加速
    管线的网格退化,看 mask 看不出来)。`engine` 决定取哪个 npz,取不到就不放弃。
    `onmask_gate` = 第 1 个候选**实测** on_mask 低于此值就不再试后面的候选;None = 用模型里
    带的值(官方 0 = 关,Fast-SAM3D 0.65),显式传 0 可关掉。Fast-SAM3D 上这一刀省 15.7%
    调用且**一个真达标实例都不丢**,官方同阈值要丢 2.8% —— 详见 abandon.py 的表。

    **`abandon_frac=0` 且 gate=0 时本函数与改动前逐位一致(模型都不加载)。**"""
    topk = select_reference_frames_topk(video, ref_cfg, scorer, k=n_candidates, gt_cov=gt_cov)
    picked: list[SAM3DRawOutput] = []
    amodel = None
    athr = float("inf")
    gate = float(onmask_gate or 0.0)
    # gate=None 表示"用模型里带的值",所以那种情况也要加载模型;两个门都关时一行都不跑。
    if (abandon_frac and abandon_frac > 0) or onmask_gate is None:
        from src.stage1.abandon import get_model
        amodel = get_model(engine)
        if amodel is not None:
            if onmask_gate is None:
                gate = amodel.onmask_gate
            if abandon_frac and abandon_frac > 0:
                athr = amodel.threshold_for(abandon_frac)
                print(f"[stage1] 提前放弃(预判门)启用:frac={abandon_frac} -> 分数阈值 "
                      f"{athr:.3f}, cap={abandon_cap}", flush=True)
                if abandon_frac > amodel.rec_frac:
                    print(f"[stage1] ⚠️ frac={abandon_frac} 超过该引擎模型的建议值 "
                          f"{amodel.rec_frac:g}(依据见 abandon.py) —— 照用户给的值执行",
                          flush=True)
        else:
            amodel = None
    if gate > 0:
        print(f"[stage1] 提前放弃(on-mask 门)启用:rank-1 on_mask < {gate:g} 就不再试后续候选",
              flush=True)
    logf = open(cand_log, "a") if cand_log else None
    try:
        for iid, cands in topk.items():
            # 提前放弃:只看 top-1 候选的特征(决策时唯一拿得到的东西),不额外跑任何 SAM 3D
            a_score, cap = None, len(cands)
            if amodel is not None and cands:
                f0 = mask_features(cands[0].mask, video.frames[cands[0].frame_index], ref_cfg)
                a_score = amodel.score(f0)
                if a_score >= athr:
                    cap = min(abandon_cap, len(cands))
            best_r, best_on = None, -1.0
            for rank, ch in enumerate(cands[:cap], start=1):
                r = run_sam3d_single(video.frames[ch.frame_index], ch.mask, seed, inference)
                r.reference_frame_index = ch.frame_index
                r.instance_id = iid
                on = _instance_on_mask(_place_one_path1(r, ch.mask), ch.mask)
                if on > best_on:
                    best_on, best_r = on, r
                # on-mask 门:第 1 次调用实测就够差 -> 后面的候选救不回来(引擎特异,见 abandon.py)
                gate_stop = bool(gate > 0 and rank == 1 and on < gate and on < accept_onmask)
                if logf is not None:
                    feats = mask_features(ch.mask, video.frames[ch.frame_index], ref_cfg)
                    logf.write(json.dumps({
                        "group": f"{group_prefix}{iid}", "iid": int(iid), "rank": rank,
                        "frame": int(ch.frame_index), "sel_score": float(ch.score),
                        "feats": [round(float(x), 6) for x in feats],
                        "on_mask": round(float(on), 6),
                        "accepted": bool(on >= accept_onmask),
                        "n_cand": len(cands), "cap": cap,
                        "abandon_score": None if a_score is None else round(a_score, 4),
                        "gate": gate or None, "gate_stop": gate_stop,
                    }) + "\n")
                if on >= accept_onmask or gate_stop:
                    break
            if best_r is not None:
                picked.append(best_r)
    finally:
        if logf is not None:
            logf.close()
    return picked


def stage0_to_stage1(video: VideoInput,
                     inference,
                     ref_cfg: RefSelectConfig = RefSelectConfig(),
                     qt: QualityThresholds = QualityThresholds(),
                     scorer: "Optional[LinearFrameScorer]" = None,
                     seed: int = 42,
                     n_candidates: int = 1,
                     accept_onmask: float = 0.9,
                     gt_cov: "Optional[dict]" = None,
                     cand_log=None, group_prefix: str = "",
                     abandon_frac: float = 0.0, abandon_cap: int = 2,
                     engine: str = "sam3d", onmask_gate=None) -> SceneReconstruction:
    """Full bridge: SAM 3 masklets -> shared-frame per-instance static meshes.
    `scorer`: a trained LinearFrameScorer for reference-frame selection; if None, the hand
    prior is used as fallback. Path is auto-selected in compose_scene (Path 2 if GT camera).
    `n_candidates` > 1: try several reference frames per instance and keep the best-on-mask
    reconstruction (`accept_onmask` early-stop) — works on BOTH paths since 2026-08-09 (the
    selection metric uses SAM 3D's own camera, not the scene frame).
    `gt_cov`: GT dense-track coverage per instance/frame — drops frames stage ⑦ could never link
    from the t* candidate set (see select_reference_frames)."""
    if n_candidates > 1:                                            # A+B: best-of-k per instance
        raw = _reconstruct_best_of_k(video, inference, ref_cfg, scorer, seed,
                                     n_candidates, accept_onmask, gt_cov=gt_cov,
                                     cand_log=cand_log, group_prefix=group_prefix,
                                     abandon_frac=abandon_frac, abandon_cap=abandon_cap,
                                     engine=engine, onmask_gate=onmask_gate)
    else:
        choices = select_reference_frames(video, ref_cfg, scorer, gt_cov=gt_cov)  # A: one t*/inst
        raw     = reconstruct_all(video, choices, inference, seed=seed)  # B: per-instance SAM 3D
    scene   = compose_scene(raw, video, qt)                         # C: unify coordinate frame (+grade)
    return scene                                                    # -> Stage 2
