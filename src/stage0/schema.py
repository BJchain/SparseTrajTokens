"""Stage 0 (SAM 3) output dataclasses — split out of the former monolithic SCHEMA.py.
`Masklet`/`VideoInput` are PICKLED into video_input.pkl; the SCHEMA.py shim re-exports them so
old pickles referencing SCHEMA.VideoInput by module path still unpickle."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class Masklet:
    """One instance's segmentation across the whole video (a SAM 3 masklet)."""
    instance_id: int
    masks: np.ndarray                 # (T, H, W) bool — full-frame aligned, NOT cropped
    concept: Optional[str] = None     # noun-phrase / exemplar prompt that produced it
    object_type: Optional[str] = None  # "rigid" | "non-rigid", from Stage 0-pre (VLM). The
                                       # AUTHORITATIVE type carried downstream (consumed at §4);
                                       # replaces Stage 2b's tracker-dependent residual gate.
    motion: Optional[str] = None       # "dynamic" | "static", from Stage 0-pre (VLM). Static
                                       # instances skip SAM3D/2a/pass1/⑦/2b/pass2 and finalize to
                                       # 4 representative points. None (old pickles / no VLM) is
                                       # treated as DYNAMIC downstream — consumers must use
                                       # getattr(m, 'motion', None) so pre-field pickles still load.


@dataclass
class VideoInput:
    frames: np.ndarray                # (T, H, W, 3) uint8 RGB
    masklets: list[Masklet]
    # Optional known camera (e.g. Kubric MOVi-E GT). If provided, the shared frame
    # is built on the GT world frame instead of SAM 3D's estimated layout (more reliable).
    camera_intrinsics: Optional[np.ndarray] = None   # (3,3) or (T,3,3)
    camera_extrinsics: Optional[np.ndarray] = None   # (T,4,4)  world<-camera per frame
    # Optional GT per-object state (Kubric provides this). Used ONLY by Path 2's
    # sanity check — NOT needed to place the mesh, only to grade the placement.
    gt_object_poses: Optional[dict] = None    # instance_id -> (T,4,4) world<-object, or (4,4)
    gt_object_extent: Optional[dict] = None   # instance_id -> float, metric bbox diagonal
    # ALTERNATIVE size input for Path 2 (2026-08-10). `gt_object_extent` above is the object's
    # FULL 3D AABB diagonal sqrt(w^2+h^2+d^2) — that is what Kubric/MOVi hands us. Datasets that
    # can only measure the object's FRONTAL (silhouette) size sqrt(w^2+h^2) — e.g. Stereo4D, where
    # it is mask_pixel_diag * gt_depth / f — must use THIS field instead: Path 2 then divides by
    # the mesh's perpendicular extent rather than its full diagonal, so the two sides share one
    # convention. Feeding a frontal size as `gt_object_extent` shrinks every mesh (and pulls it
    # closer) by perp/full — measured median 0.906, p10 0.681, i.e. shape-dependent, NOT a
    # constant you can divide out. See _compose_path2_gt_camera.
    gt_object_frontal_extent: Optional[dict] = None   # instance_id -> float, metric FRONTAL size
    # Optional GT oriented bbox corners (Kubric inst_bboxes_3d). Used ONLY by Path 2's rotation
    # check / re-orient: gives the object's principal axes in world without needing its mesh.
    gt_object_bbox: Optional[dict] = None     # instance_id -> (T,8,3) world OBB corners, or (8,3)
