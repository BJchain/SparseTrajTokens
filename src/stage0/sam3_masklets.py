"""
Stage 0 (v2): SAM 3.1 promptable concept segmentation  ->  per-instance masklets.

    video frames (T,H,W,3) uint8  +  noun-phrase prompt(s)
      -> Sam3MultiplexVideoPredictor  (start_session / add_prompt / propagate_in_video)
      -> per-instance masklets: stable obj_id + (T,H,W) bool mask over the WHOLE video
      -> SCHEMA.VideoInput.masklets   (the input consumed by stage0_to_stage1)

This is the front door of the v2 pipeline. Everything downstream (the SAM 3D
single-frame mesh, part decomposition, the tracker seed) keys off the stable
instance IDs produced here.

Design invariants (mirror SCHEMA.py)
------------------------------------
  * Masks are FULL-FRAME aligned, (T,H,W) bool, never tight crops. SAM 3D needs the
    surrounding scene context, so we keep masks at the native frame resolution.
  * One stable instance_id per tracked object, consistent across all frames. SAM 3.1
    assigns an obj_id per tracked instance within a session; we remap those (per
    concept) onto a single global instance_id space.
  * Time only lives in the mask sequence here — no per-frame meshing, no geometry.

SAM 3.1 video predictor API (verified against examples/sam3.1_video_predictor_example.ipynb)
  start_session(resource_path=<dir of NNNNN.jpg | .mp4>) -> {"session_id"}
  reset_session(session_id)                              # REQUIRED before re-prompting a used session
  add_prompt(session_id, frame_index, text=<concept>)    -> {"outputs"}
  handle_stream_request(propagate_in_video, session_id)  yields {"frame_index", "outputs"}
      outputs: {out_boxes_xywh, out_probs, out_obj_ids, out_binary_masks}
      out_binary_masks[i] may be LOWER-RES than the frame -> resize (nearest) to (H,W).
  close_session(session_id)

Run notes (this server)
  CUDA_VISIBLE_DEVICES=1          # only free GPU
  TMPDIR=/tmp    # root /tmp is full (other users); frame dump goes here
  HF_HUB_OFFLINE=1
  weights: third_party/sam3/checkpoints/sam3.1/sam3.1_multiplex.pt
  env:     conda env `sam3`   (py3.12 + torch2.10+cu128; see environment/)
"""
from __future__ import annotations

import os
import sys
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

# --- locate the repo + import the concrete SCHEMA dataclasses --------------------
_STAGE04_ROOT = Path(__file__).resolve().parents[2]          # .../3D_VLM_Fusion/stage0_4
_SAM3_REPO = _STAGE04_ROOT / "third_party" / "sam3"
_DEFAULT_CKPT = _SAM3_REPO / "checkpoints" / "sam3.1" / "sam3.1_multiplex.pt"

if str(_STAGE04_ROOT) not in sys.path:
    sys.path.insert(0, str(_STAGE04_ROOT))
from src.stage0.schema import Masklet, VideoInput  # noqa: E402  (concrete module, not the SCHEMA shim: avoids shim->stage0/__init__->sam3_masklets import cycle)


# =============================================================================
# Concept prompt spec (Stage 0-pre output -> SAM 3 prompt)
# =============================================================================
@dataclass
class ConceptPrompt:
    """One thing to prompt SAM 3 with. From Stage 0-pre (qwen) we get a noun phrase,
    a seed frame, an optional box, and the authoritative rigid/non-rigid type."""
    concept: str
    seed_frame: int = 0
    bbox_xywh: Optional[Sequence[float]] = None  # normalised [x, y, w, h] in [0,1]
    object_type: Optional[str] = None            # "rigid" | "non-rigid"
    motion: Optional[str] = None                 # "dynamic" | "static" (None -> treated dynamic)


def _as_concept_prompts(concepts) -> list[ConceptPrompt]:
    """Accept a str, a list of str, or a list of dict/ConceptPrompt -> [ConceptPrompt].
    Keeps backward-compat with the old text-only callers (list[str])."""
    if isinstance(concepts, (str, ConceptPrompt, dict)):
        concepts = [concepts]
    out: list[ConceptPrompt] = []
    for c in concepts:
        if isinstance(c, ConceptPrompt):
            out.append(c)
        elif isinstance(c, str):
            out.append(ConceptPrompt(concept=c))
        elif isinstance(c, dict):
            out.append(ConceptPrompt(
                concept=c["concept"],
                seed_frame=int(c.get("seed_frame", 0)),
                bbox_xywh=c.get("bbox_xywh"),
                object_type=c.get("object_type"),
                motion=c.get("motion"),
            ))
        else:
            raise TypeError(f"unsupported concept spec: {type(c)}")
    return out


def concept_prompts_from_stage0_pre(stage0_pre) -> list[ConceptPrompt]:
    """Build ConceptPrompts from a stage0_pre.json path / loaded dict (run_stage0_pre output)."""
    import json
    if isinstance(stage0_pre, (str, Path)):
        stage0_pre = json.loads(Path(stage0_pre).read_text())
    return [ConceptPrompt(
        concept=i["concept"], seed_frame=int(i.get("seed_frame", 0)),
        bbox_xywh=i.get("bbox_xywh"), object_type=i.get("object_type"),
        motion=i.get("motion"),
    ) for i in stage0_pre["instances"]]


# =============================================================================
# Predictor construction
# =============================================================================
def build_predictor(
    checkpoint_path: Optional[Union[str, Path]] = None,
    version: str = "sam3.1",
    use_fa3: bool = False,
    **kwargs,
):
    """Build the SAM 3.1 multiplex video predictor.

    Thin wrapper over sam3.model_builder.build_sam3_predictor that defaults to our
    local sam3.1 checkpoint (so nothing reaches out to HuggingFace at load time).
    The model lands on the default CUDA device — pin it with CUDA_VISIBLE_DEVICES=1.

    use_fa3 defaults to False: Flash Attention 3's Hopper kernel only supports
    fp16/bf16 on Ampere/Ada GPUs (this server's cards), and SAM 3 feeds it fp32 in
    places -> it raises. Standard attention works fine. Flip to True only on Hopper.
    """
    if str(_SAM3_REPO) not in sys.path:
        sys.path.insert(0, str(_SAM3_REPO))
    from sam3.model_builder import build_sam3_predictor

    ckpt = Path(checkpoint_path) if checkpoint_path is not None else _DEFAULT_CKPT
    if not ckpt.is_file():
        raise FileNotFoundError(
            f"SAM 3.1 checkpoint not found at {ckpt}. Expected the modelscope download "
            f"facebook/sam3.1 -> sam3.1_multiplex.pt (see project_sam3_setup memory)."
        )
    predictor = build_sam3_predictor(
        checkpoint_path=str(ckpt), version=version, use_fa3=use_fa3, **kwargs
    )
    _patch_init_state_kwargs(predictor)
    return predictor


def _patch_init_state_kwargs(predictor) -> None:
    """Work around a repo-snapshot API mismatch.

    base start_session unconditionally forwards offload_state_to_cpu (and possibly
    video_loader_type) into self.model.init_state, but the multiplex model's
    init_state — Sam3MultiplexTrackingWithInteractivity.init_state — does not accept
    them. Both are no-op defaults (offload_state_to_cpu=False) for our small clips,
    so we wrap init_state to drop any kwargs outside its real signature.
    """
    import functools
    import inspect

    model = getattr(predictor, "model", None)
    init = getattr(model, "init_state", None)
    if init is None:
        return
    try:
        params = inspect.signature(init).parameters
    except (TypeError, ValueError):
        return
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return  # already accepts **kwargs; nothing to filter
    accepted = set(params)

    @functools.wraps(init)
    def _filtered(*a, **k):
        return init(*a, **{kk: vv for kk, vv in k.items() if kk in accepted})

    model.init_state = _filtered


# =============================================================================
# Internal helpers
# =============================================================================
def _to_numpy(x) -> np.ndarray:
    """torch.Tensor | np.ndarray | list -> np.ndarray, without importing torch here."""
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):          # torch tensor
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _mask_to_bool_hw(mask, H: int, W: int) -> np.ndarray:
    """A single SAM 3 binary mask -> (H,W) bool at frame resolution.

    SAM 3.1 may emit masks at a lower internal resolution; resize with nearest
    interpolation (same convention as the repo's render_masklet_frame).
    """
    m = _to_numpy(mask)
    m = np.squeeze(m)
    if m.ndim != 2:
        raise ValueError(f"expected a 2-D mask, got shape {m.shape}")
    if m.shape != (H, W):
        import cv2
        m = cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
    return m > 0.5


def _is_oom(e: BaseException) -> bool:
    """True for a CUDA out-of-memory, however it surfaces (torch.cuda.OutOfMemoryError,
    a plain RuntimeError carrying the message, or one wrapped by SAM 3's request layer)."""
    try:
        import torch
        if isinstance(e, torch.cuda.OutOfMemoryError):
            return True
    except Exception:
        pass
    return "out of memory" in str(e).lower()


def _empty_cuda_cache() -> None:
    """Return the caching allocator's reserved-but-unused blocks to the driver. Matters when
    the model stays resident across clips (batch runner) — the per-clip-process design used to
    get this for free at process exit."""
    try:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _write_frames_jpg(frames: np.ndarray, out_dir: Path) -> None:
    """Dump (T,H,W,3) uint8 RGB frames as NNNNN.jpg (zero-padded -> int-sortable)."""
    from PIL import Image
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, fr in enumerate(frames):
        Image.fromarray(np.ascontiguousarray(fr)).save(out_dir / f"{i:05d}.jpg", quality=95)


# =============================================================================
# Core: video + concepts -> masklets
# =============================================================================
def _prompt_and_collect(predictor, session_id, seed, concept, bbox, T, H, W) -> dict:
    """Add ONE prompt (text, optionally +box) on `seed` and accumulate per-obj_id (T,H,W) bool
    masks over the full bidirectional propagation. Returns {obj_id: masks}. Assumes the session was
    just reset. Raises whatever SAM 3 raises (caller handles the box->text fallback)."""
    req = dict(type="add_prompt", session_id=session_id, frame_index=int(seed), text=concept)
    if bbox is not None:
        # SAM 3 box prompt: normalised [xmin, ymin, w, h], label 1 = positive.
        # text+box together: text grounds all instances, box seeds/locates one.
        req["bounding_boxes"] = [[float(v) for v in bbox]]
        req["bounding_box_labels"] = [1]
    predictor.handle_request(request=req)

    obj_masks: dict[int, np.ndarray] = {}
    for response in predictor.handle_stream_request(
            request=dict(type="propagate_in_video", session_id=session_id)):
        f = int(response["frame_index"])
        out = response["outputs"]
        obj_ids = _to_numpy(out["out_obj_ids"]).reshape(-1).tolist()
        bin_masks = out["out_binary_masks"]
        for idx, oid in enumerate(obj_ids):
            bm = _mask_to_bool_hw(bin_masks[idx], H, W)
            if not bm.any():
                continue
            buf = obj_masks.get(oid)
            if buf is None:
                buf = np.zeros((T, H, W), dtype=bool)
                obj_masks[oid] = buf
            buf[f] = bm
    return obj_masks


def segment_video(
    frames: np.ndarray,
    concepts: Union[str, Sequence],
    predictor,
    *,
    prompt_frame_index: int = 0,
    min_area_frac: float = 0.0,
    dedup_iou: float = 0.5,
    frames_dir: Optional[Union[str, Path]] = None,
    keep_frames_dir: bool = False,
    materialize: bool = True,
) -> list[Masklet]:
    """Run SAM 3.1 over a video for one or more concept prompts.

    Parameters
    ----------
    frames : (T,H,W,3) uint8 RGB.
    concepts : a noun phrase / list of noun phrases (text-only, legacy), OR a list of
        ConceptPrompt / dict specs (Stage 0-pre). A spec carries the noun phrase, a
        per-concept SEED FRAME, an optional normalised BOX (text+box are BOTH sent to
        SAM 3 — the box seeds/locates, the text grounds all instances), and the
        authoritative object_type which is stamped onto every resulting Masklet.
        Each concept is prompted independently; every instance SAM 3.1 returns becomes
        its own Masklet with a fresh global instance_id.
    prompt_frame_index : fallback seed frame for specs that don't carry their own
        (text propagates BIDIRECTIONALLY, so any frame yields full-video coverage).
    min_area_frac : drop a masklet whose largest single-frame mask covers less than
        this fraction of the image (noise). 0 disables.
    dedup_iou : merge masklets from DIFFERENT concepts whose temporal IoU exceeds this
        (the same physical object grounded under two names, e.g. swan / black swan).
        On a type conflict the merged masklet is kept NON-RIGID (conservative). 0 disables.
    frames_dir : where to dump the JPG frames SAM 3 reads. Defaults to a temp dir
        under TMPDIR; removed afterwards unless keep_frames_dir.
    materialize : write `frames` into frames_dir as NNNNN.jpg (default). Pass False ONLY
        when frames_dir ALREADY holds exactly these T frames under that naming (e.g. the
        dataset's own frame dir, used verbatim with stride=1 / no max-frames) — it skips a
        full re-encode per clip. ⚠️ materialize=False + a wrong/partial dir silently
        segments the wrong frames; and materialize=True on a SOURCE dir would overwrite the
        dataset with a lossy re-encode, so the two must always be set together with care.

    Returns
    -------
    list[Masklet], each with instance_id (globally unique), masks (T,H,W) bool,
    concept, object_type. Empty masklets (never positive) are dropped.
    """
    prompts = _as_concept_prompts(concepts)
    if frames.dtype != np.uint8:
        frames = frames.astype(np.uint8)
    T, H, W = frames.shape[:3]

    # 1. Materialise frames where SAM 3 can read them (a dir of NNNNN.jpg).
    owns_dir = frames_dir is None
    if owns_dir:
        base_tmp = os.environ.get("TMPDIR") or tempfile.gettempdir()
        frames_dir = Path(tempfile.mkdtemp(prefix="sam3_frames_", dir=base_tmp))
        materialize = True                       # a fresh temp dir is always empty
    else:
        frames_dir = Path(frames_dir)
    if materialize:
        _write_frames_jpg(frames, frames_dir)
    else:
        # Reusing a pre-existing dir: fail loudly on a count mismatch rather than
        # silently segmenting a different set of frames than `frames` describes.
        have = sum(1 for p in frames_dir.iterdir() if p.suffix.lower() == ".jpg")
        if have != T:
            raise ValueError(f"materialize=False but {frames_dir} holds {have} .jpg, "
                             f"expected {T} — refusing to segment mismatched frames")

    masklets: list[Masklet] = []
    next_iid = 0
    session_id = None
    try:
        resp = predictor.handle_request(
            request=dict(type="start_session", resource_path=str(frames_dir))
        )
        session_id = resp["session_id"]

        for cp in prompts:
            seed = cp.seed_frame if 0 <= cp.seed_frame < T else prompt_frame_index
            # text+box first; if SAM 3's geometric-prompt path errors (e.g. a tiny/empty box that
            # seeds the tracker with no points -> "No points are provided"), retry the SAME concept
            # TEXT-ONLY (text grounding is the primary signal; the box is only a seed). A concept
            # that still fails or grounds nothing is skipped — never kills the whole sequence.
            attempts = [cp.bbox_xywh, None] if cp.bbox_xywh is not None else [None]
            obj_masks: dict[int, np.ndarray] = {}
            for box in attempts:
                try:
                    # A used session must be reset before a new prompt (harmless on a fresh one).
                    predictor.handle_request(
                        request=dict(type="reset_session", session_id=session_id))
                    obj_masks = _prompt_and_collect(predictor, session_id, seed, cp.concept,
                                                    box, T, H, W)
                    break
                except Exception as e:
                    how = "text+box" if box is not None else "text-only"
                    # A CUDA OOM is NOT "this prompt is bad" — it is an environment problem, and
                    # swallowing it here silently drops a real instance from the output. (Measured
                    # 2026-08-12: a resident-model batch run sharing a GPU lost 10 of 49 masklets
                    # this way, and the pickle looked perfectly normal.) Drop the caches and retry
                    # once; if it still OOMs, fail the clip loudly so the caller can requeue it.
                    if _is_oom(e):
                        print(f"[segment_video] concept {cp.concept!r} {how}: CUDA OOM — "
                              f"emptying cache and retrying once", flush=True)
                        _empty_cuda_cache()
                        try:
                            predictor.handle_request(
                                request=dict(type="reset_session", session_id=session_id))
                            obj_masks = _prompt_and_collect(predictor, session_id, seed,
                                                            cp.concept, box, T, H, W)
                            break
                        except Exception as e2:
                            if _is_oom(e2):
                                raise MemoryError(
                                    f"SAM 3 OOM on concept {cp.concept!r} even after "
                                    f"empty_cache — refusing to write a silently incomplete "
                                    f"VideoInput. Free VRAM or lower concurrency.") from e2
                            e = e2
                    print(f"[segment_video] concept {cp.concept!r} {how} prompt failed: "
                          f"{type(e).__name__}: {e}", flush=True)
                    obj_masks = {}

            # Promote each tracked object to a global Masklet.
            min_px = min_area_frac * H * W
            for oid in sorted(obj_masks):
                seq = obj_masks[oid]
                if min_px > 0 and int(seq.reshape(T, -1).sum(axis=1).max()) < min_px:
                    continue
                masklets.append(
                    Masklet(instance_id=next_iid, masks=seq,
                            concept=cp.concept, object_type=cp.object_type,
                            motion=cp.motion)
                )
                next_iid += 1
    finally:
        if session_id is not None:
            try:
                predictor.handle_request(
                    request=dict(type="close_session", session_id=session_id)
                )
            except Exception:
                pass
        if owns_dir and not keep_frames_dir:
            shutil.rmtree(frames_dir, ignore_errors=True)

    if dedup_iou > 0:
        masklets = _dedup_masklets(masklets, dedup_iou)
    return masklets


# =============================================================================
# Cross-concept masklet de-duplication
# =============================================================================
def _temporal_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Volume IoU of two (T,H,W) bool masklets over the whole clip."""
    inter = int(np.logical_and(a, b).sum())
    if inter == 0:
        return 0.0
    union = int(np.logical_or(a, b).sum())
    return inter / union if union else 0.0


def _dedup_masklets(masklets: list[Masklet], iou_thresh: float) -> list[Masklet]:
    """Merge masklets (from different concepts) that overlap heavily in space-time — the
    same physical object grounded under two names. Keep the larger-area one as the
    survivor; on object_type conflict keep NON-RIGID (conservative: preserves trajectories).
    instance_ids are re-assigned contiguously."""
    n = len(masklets)
    if n < 2:
        return masklets
    areas = [int(m.masks.sum()) for m in masklets]
    # union-find over masklets that exceed the IoU threshold
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if _temporal_iou(masklets[i].masks, masklets[j].masks) >= iou_thresh:
                parent[find(i)] = find(j)

    groups: dict = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    merged: list[Masklet] = []
    for members in groups.values():
        # survivor = largest area; OR-union the masks so the kept mask is the most complete
        keep = max(members, key=lambda i: areas[i])
        masks = masklets[keep].masks
        if len(members) > 1:
            masks = np.logical_or.reduce([masklets[i].masks for i in members])
        types = {masklets[i].object_type for i in members}
        otype = "non-rigid" if "non-rigid" in types else masklets[keep].object_type
        # motion conflict -> DYNAMIC (conservative: runs the full mesh pipeline, loses nothing)
        motions = {getattr(masklets[i], "motion", None) for i in members}
        motion = "dynamic" if "dynamic" in motions else getattr(masklets[keep], "motion", None)
        concepts = "/".join(sorted({masklets[i].concept for i in members if masklets[i].concept}))
        merged.append(Masklet(instance_id=0, masks=masks,
                              concept=concepts or None, object_type=otype, motion=motion))

    merged.sort(key=lambda m: int(m.masks.sum()), reverse=True)
    for new_id, m in enumerate(merged):
        m.instance_id = new_id
    return merged


def video_input_from_frames(
    frames: np.ndarray,
    concepts: Union[str, Sequence[str]],
    predictor,
    *,
    camera_intrinsics: Optional[np.ndarray] = None,
    camera_extrinsics: Optional[np.ndarray] = None,
    gt_object_poses: Optional[dict] = None,
    gt_object_extent: Optional[dict] = None,
    **seg_kwargs,
) -> VideoInput:
    """Convenience: frames + concepts -> a fully-formed SCHEMA.VideoInput.

    Optional camera / GT fields are passed straight through (e.g. Kubric MOVi-E GT),
    so the Path-2 placement and its sanity check downstream can use them.
    """
    masklets = segment_video(frames, concepts, predictor, **seg_kwargs)
    return VideoInput(
        frames=frames.astype(np.uint8) if frames.dtype != np.uint8 else frames,
        masklets=masklets,
        camera_intrinsics=camera_intrinsics,
        camera_extrinsics=camera_extrinsics,
        gt_object_poses=gt_object_poses,
        gt_object_extent=gt_object_extent,
    )
