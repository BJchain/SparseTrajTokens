"""Stage 3->4 tracker-query handoff — split out of the former monolithic SCHEMA.py.
TrackerQuery, project_scene_point_to_image, build_tracker_query (SAM 3D-camera projection +
on-mask gate), and pass2 gap detection/re-query. Re-exported by the SCHEMA.py shim."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np

from src.geometry import _centroid
from src.stage3.schema import Keypoint
from src.stage1.schema import SceneReconstruction, InstanceMesh, VideoInput


@dataclass
class TrackerQuery:
    """Everything a sequence-input point tracker needs to begin tracking.
    `frames` is the sequence; `queries_2d` says WHERE (pixels) and WHEN (frame) each
    point starts. `index` maps every query row back to a keypoint identity so the
    returned trajectories can be attributed to instance / part / keypoint.
    """
    frames: np.ndarray                # (T,H,W,3) uint8 — the sequence the tracker consumes
    queries_2d: np.ndarray            # (N,3) rows of (t_query, x, y) in pixels
    index: list                       # len N: (keypoint_id, instance_id, part_id)
    valid: np.ndarray                 # (N,) bool — False if projection invalid/behind camera/off-object
    queries_3d: Optional[np.ndarray] = None   # (N,3) scene-frame XYZ, for 3D trackers
    on_mask: Optional[np.ndarray] = None      # (N,) bool — projected pixel landed on the SAM 3 mask (diag)


def project_scene_point_to_image(p_scene: np.ndarray,
                                 world_from_cam: np.ndarray,
                                 K: np.ndarray):
    """Project a scene/world 3D point into pixels for the camera `world_from_cam` (world<-camera).
    Returns (x, y, z_cam). CONVENTION: assumes camera looks down +z and pixel = K @ (X/z).
    If your GT uses OpenGL-style -z forward (Kubric can), negate z / flip before projecting —
    this is the one place a convention mismatch silently corrupts queries, so verify it once.
    """
    cam_from_world = np.linalg.inv(world_from_cam)
    p_cam = (cam_from_world @ np.append(p_scene, 1.0))[:3]
    z = p_cam[2]
    if abs(z) < 1e-9:
        return float("nan"), float("nan"), z
    uv = K @ (p_cam / z)
    return float(uv[0]), float(uv[1]), float(z)


def build_tracker_query(keypoints: list[Keypoint],
                        scene: SceneReconstruction,
                        video: VideoInput,
                        drop_flagged: bool = True,
                        on_mask_filter: bool = True,
                        min_onmask: int = 3) -> TrackerQuery:
    """Project Stage 3 keypoints back to their instance's t* image space and assemble
    the tracker query over video.frames.

    KNOWN ISSUE (2026-07-23, V2-only path, deliberately NOT fixed): `sam3d_intrinsics` lives in
    the pixel grid of SAM3D's pointmap, which is a power-of-two DOWNSAMPLE of the frame when
    max_side>1200 (e.g. 1920x1280 -> 480x320). The projection below treats it as full-frame
    pixels, so queries/on-mask are wrong on such videos. V3 (run_link_dense/_k_grid_wh) and
    stage1 (_k_grid_scale) are fixed; port the same grid-scale logic here if V2 is ever revived.

    PROJECTION via SAM 3D's OWN camera (GT-independent, works on Path 1 & 2). SAM 3D
    reconstructs each instance FROM its t* frame, so its mesh projects onto the object under
    SAM 3D's pointmap camera. The GT camera does NOT (the SAM3D-vs-scene FOV/orientation
    mismatch sends queries off the object — empirically ~0% on-mask vs ~70% via SAM 3D's
    camera). So per keypoint:
        v_sam = object_pose_cam @ inv(object_to_scene) @ p     # back to SAM 3D camera frame
        (x, y) = sam3d_intrinsics @ v_sam / v_sam.z            # the t* pixel
    The 3D query keeps the metric scene-frame position. If SAM 3D's camera is missing, fall
    back to the GT camera (with the Path-2 scale-about-centroid undo); flag if neither exists.

    Per-point query frame: instances generally have DIFFERENT t*, so queries_2d[:,0] carries
    a per-point query frame. The active backends D4RT and SpaTrackerV2 accept arbitrary per-point
    query timesteps ([t,x,y]), so this is consumed directly (also required by the re-anchor pass,
    which re-queries each lost point at its own last-on-mask frame). NOTE: 4RC and Track4World do
    NOT support per-point query frames (frame-level dense-grid query only), so they are not drop-in
    backends here — see project_tracker_bakeoff.

    ON-MASK GATE (on_mask_filter, default on): SAM 3D's recon/pose on real video is imperfect, so
    a fraction of keypoints project OFF the object. Tracking those wastes the tracker on background
    (which often tracks stably -> high visibility -> the vis-aware pass2 would then PREFER background
    points; measured ~53% of final DAVIS keypoints were off-object). So a query whose projected pixel
    falls outside its instance's SAM 3 mask (at its query frame) is marked invalid -> never tracked.
    Per-instance FALLBACK: if an instance keeps fewer than `min_onmask` on-mask points, revert it to
    its on-frame points (don't nuke an object whose projection is wholesale misaligned).
    """
    by_inst = {im.instance_id: im for im in scene.instances}
    flagged = set(scene.flagged_ids or [])
    H, W = video.frames.shape[1], video.frames.shape[2]
    mask_by_inst = {m.instance_id: m.masks for m in (video.masklets or [])}

    q2d, q3d, index, valid, on_mask = [], [], [], [], []
    for kp in keypoints:
        im = by_inst.get(kp.instance_id)
        if im is None or (drop_flagged and kp.instance_id in flagged):
            continue

        x = y = float("nan"); z = 0.0
        if im.object_pose_cam is not None and im.sam3d_intrinsics is not None:
            # SAM 3D camera (reliable, GT-independent)
            canon = np.linalg.inv(im.object_to_scene) @ np.append(kp.position, 1.0)
            v_sam = (im.object_pose_cam @ canon)[:3]
            z = float(v_sam[2])
            if abs(z) > 1e-9:
                K = im.sam3d_intrinsics
                x = float(K[0, 0] * v_sam[0] / z + K[0, 2])
                y = float(K[1, 1] * v_sam[1] / z + K[1, 2])
        elif im.camera_to_scene is not None and im.intrinsics is not None:
            # fallback: GT camera (undo Path-2 metric rescale first)
            p = kp.position
            if im.scale_correction != 1.0:
                c = _centroid(im.verts)
                p = (p - c) / im.scale_correction + c
            x, y, z = project_scene_point_to_image(p, im.camera_to_scene, im.intrinsics)

        ok = (z > 0) and np.isfinite(x) and np.isfinite(y) and (0 <= x < W) and (0 <= y < H)
        # on-mask: projected pixel inside the instance's SAM 3 mask at its query frame
        onm = False
        if ok:
            mk = mask_by_inst.get(kp.instance_id)
            t = int(im.reference_frame_index)
            if mk is not None and 0 <= t < mk.shape[0]:
                onm = bool(mk[t, int(y), int(x)])
        q2d.append([im.reference_frame_index, x, y])
        q3d.append(kp.position)                 # 3D query stays metric scene-frame
        index.append((kp.keypoint_id, kp.instance_id, kp.part_id))
        valid.append(bool(ok))
        on_mask.append(onm)

    q2d = np.asarray(q2d, float)
    valid = np.asarray(valid, bool)
    on_mask = np.asarray(on_mask, bool)

    final_valid = valid
    if on_mask_filter and len(valid):
        final_valid = valid.copy()
        inst_arr = np.array([ix[1] for ix in index])
        for iid in np.unique(inst_arr):
            sel = inst_arr == iid
            if int((on_mask & sel).sum()) >= min_onmask:
                final_valid[sel] = on_mask[sel]          # keep only on-object points
            # else: too few on-mask -> leave this instance's on-frame `valid` untouched (fallback)

    return TrackerQuery(
        frames=video.frames,
        queries_2d=q2d,
        queries_3d=np.asarray(q3d, float),
        index=index,
        valid=final_valid,
        on_mask=on_mask,
    )


# ---- C3: pass2 subset enforcement + gap detection + gap re-query (v2) --------
# pass2 selects a SUBSET of pass1 candidate keypoints. By construction (B4) every
# selected id should already have a trajectory from the single pass1 tracker run.
# A "gap" = a pass2-desired id NOT present in pass1 (the rare escape hatch). Gaps are
# never silently dropped: the missing points get a small dedicated re-query and a
# GapWarning is recorded so the gap RATE can be monitored (probe, not silent failure).

@dataclass
class GapWarning:
    """One pass2 gap event — pass2 wanted keypoints absent from pass1 (so no trajectory).
    Recorded in output metadata; high gap rate => pass1 coverage (A2/A3) needs tuning."""
    instance_id: int
    missing_keypoint_ids: list      # ids pass2 wanted but pass1 never tracked
    note: str = "pass2 keypoints not in pass1; gap re-query issued"


def find_pass2_gaps(pass2_keypoint_ids: list, pass1_index: list) -> list:
    """Return the pass2 ids that have NO trajectory in pass1 (i.e. not in pass1_index).
    pass1_index = the TrackerQuery.index from the pass1 run [(kp_id, inst_id, part_id), ...]."""
    have = {row[0] for row in pass1_index}
    return [kid for kid in pass2_keypoint_ids if kid not in have]


def build_gap_query(gap_keypoints: list[Keypoint],
                    scene: SceneReconstruction,
                    video: VideoInput) -> tuple[TrackerQuery, list[GapWarning]]:
    """Thin wrapper: build a tracker query for ONLY the gap keypoints (a small batch),
    to be run through the tracker separately and merged back by keypoint_id. Also emit a
    GapWarning per instance. This is the ONLY place a second (tiny) tracker pass happens;
    the main pass1 tracker still runs once over all candidates.
    Returns (gap_query, warnings). Empty gap_keypoints -> (empty query, [])."""
    warnings: list[GapWarning] = []
    by_inst: dict = {}
    for kp in gap_keypoints:
        by_inst.setdefault(kp.instance_id, []).append(kp.keypoint_id)
    for inst_id, ids in by_inst.items():
        warnings.append(GapWarning(instance_id=inst_id, missing_keypoint_ids=ids))
    gap_query = build_tracker_query(gap_keypoints, scene, video)  # reuse the same builder
    return gap_query, warnings
