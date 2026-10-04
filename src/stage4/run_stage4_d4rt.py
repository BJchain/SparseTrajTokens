"""Stage 4 (v2): D4RT sparse keypoint tracking.

pass1 keypoints + Stage1 InstanceMeshes + video
  -> SCHEMA.build_tracker_query (project keypoints to t* image via SAM 3D's own camera)
  -> D4RT _infer_tracks (per-point 3D trajectory + visibility + confidence)
  -> per-keypoint trajectory [K,T,3] keyed by keypoint_id / instance_id / part_id.

Run in env d4rt (build_tracker_query is pure numpy+SCHEMA; D4RT needs the d4rt env):
  TMPDIR=/tmp CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  <d4rt-env-python> src/stage4/run_stage4_d4rt.py \
      --inst-dir outputs/m0_runs/movi_scene0 \
      --keypoints outputs/m1_runs/movi_scene0/pass1_keypoints.pkl \
      --video-input outputs/m0_runs/movi_scene0/video_input.pkl \
      --out outputs/m1_runs/movi_scene0/tracks_d4rt.npz
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_D4RT = os.path.join(_ROOT, "third_party", "D4rt")
for p in (_ROOT, _D4RT):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402
from SCHEMA import InstanceMesh, SceneReconstruction, build_tracker_query  # noqa: E402
# NOTE: `src` collides — the D4RT repo also has a top-level `src` package (src.core/src.model),
# which wins on sys.path. So we do NOT import our own src.* here; VideoInput is just a pickle
# (SCHEMA is importable for unpickling), and we use D4RT's src.core/src.model.
from src.core import load_yaml_config, load_checkpoint  # noqa: E402  (D4RT repo)
from src.model import build_model  # noqa: E402  (D4RT repo)
from infer_track_3d import _infer_tracks, _unwrap_state_dict  # noqa: E402

_CKPT_DIR = os.path.join(_D4RT, "checkpoints", "OpenD4RT_48CLIP_9Mix_NoCropAUG")


def load_instances(inst_dir: str) -> list[InstanceMesh]:
    insts = []
    for p in sorted(glob.glob(os.path.join(inst_dir, "inst_*.npz"))):
        d = np.load(p)
        iid = int(os.path.splitext(os.path.basename(p))[0].replace("inst_", ""))
        g = lambda k: (d[k] if k in d.files else None)  # noqa: E731
        insts.append(InstanceMesh(
            instance_id=iid, verts=d["verts"], faces=d["faces"],
            reference_frame_index=int(d["reference_frame_index"]),
            object_to_scene=d["object_to_scene"],
            camera_to_scene=g("camera_to_scene"), intrinsics=g("intrinsics"),
            object_pose_cam=g("object_pose_cam"), sam3d_intrinsics=g("sam3d_intrinsics"),
            scale_correction=float(d["scale_correction"]),
        ))
    return insts


def load_d4rt(ckpt_dir: str = _CKPT_DIR):
    cfg = load_yaml_config(os.path.join(ckpt_dir, "model.yaml"))
    model = build_model(cfg["model"]).eval()
    sd = _unwrap_state_dict(load_checkpoint(os.path.join(ckpt_dir, "opend4rt.ckpt"), map_location="cpu"))
    res = model.load_state_dict(sd, strict=False)
    print(f"[stage4] D4RT loaded (missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)})")
    return model.cuda().eval()


def mask_consistency_vis(uv_track, vis_d4rt, inst_id, masklets, tol=4):
    """Cross-frame mask-consistency gate (REFERENCE §6.1 ①). D4RT's own per-frame visibility is
    unreliable on real video (measured: ~64% of D4RT-"visible" points have drifted OFF the object
    mask). We have the SAM 3 mask for ALL frames, so re-derive a TRUSTWORTHY visibility: a tracked
    point is visible at frame t ONLY IF D4RT says visible AND its 2D pixel lands inside that
    instance's mask at frame t (with a few-pixel tolerance for mask/edge slack).

    Per-frame, per-point — keeps the good frames of a point and drops only the drifted frames (a
    point tracked correctly for frames 0..9 then lost stays valid for 0..9). Returns the corrected
    (N,T) bool visibility; the raw D4RT one is kept separately for diagnostics."""
    import cv2
    mask_by_inst = {m.instance_id: m.masks for m in (masklets or [])}
    N, Tt, _ = uv_track.shape
    out = vis_d4rt.copy()
    # pre-dilate each instance's per-frame mask by `tol` px (edge/recon slack)
    k = np.ones((2 * tol + 1, 2 * tol + 1), np.uint8)
    dil = {iid: np.stack([cv2.dilate(m[t].astype(np.uint8), k) for t in range(m.shape[0])])
           for iid, m in mask_by_inst.items()}
    for n in range(N):
        md = dil.get(int(inst_id[n]))
        if md is None:
            continue
        Tm, H, W = md.shape
        for t in range(Tt):
            if not out[n, t]:
                continue
            if t >= Tm:
                out[n, t] = False; continue
            x = int(np.clip(uv_track[n, t, 0] * W, 0, W - 1))
            y = int(np.clip(uv_track[n, t, 1] * H, 0, H - 1))
            if not md[t, y, x]:
                out[n, t] = False           # D4RT said visible but it drifted off the object
    return out


def reanchor_tracks(model, frames, xyz, xyzl, uv, conf, vis, query_src, inst_id, masklets,
                    chunk, max_rounds=4):
    """② chained re-anchoring (REFERENCE §6.3, depends on ① mask-consistency vis). A point that
    drifts off the mask before the last frame is RE-QUERIED at its LAST on-mask frame L, using the
    tracker's own (mask-verified) 2D position there — giving D4RT a fresh NEARBY anchor so it can
    hold more frames. Forward-tracked frames t>L that pass ① are spliced in. Repeat until no point
    extends or max_rounds. `xyz`(=xyz_ref0) is anchor-INDEPENDENT (verified ~0.002*bbox), so spliced
    segments are consistent. Honest bound: a point that drifts IMMEDIATELY after re-anchor (camel-
    type) makes no progress and is dropped from further rounds — ② only helps 'tracks-a-while-then-
    drifts' (blackswan-type). All arrays (N,T,...) are edited in place; returns n_extended_frames."""
    N, Tt = vis.shape
    last_anchor = query_src.astype(int).copy()
    gained = 0
    for _ in range(max_rounds):
        # last on-mask (mask-corrected visible) frame per point
        L = np.where(vis.any(1), (vis * np.arange(Tt)).argmax(1), -1)
        todo = [n for n in range(N) if 0 <= L[n] < Tt - 1 and L[n] > last_anchor[n]]
        if not todo:
            break
        q_uv = np.array([uv[n, L[n]] for n in todo], np.float32)
        q_src = np.array([L[n] for n in todo], np.int64)
        o = _infer_tracks(model=model, video_model_rgb=frames, query_uv_norm=q_uv,
                          query_chunk_size=chunk, query_src_indices_global=q_src)
        nuv = o["tracks_uv_norm"].astype(np.float32)
        nvis = mask_consistency_vis(nuv, o["tracks_visibility"], inst_id[todo], masklets)
        for i, n in enumerate(todo):
            Ln = L[n]
            ext = (np.arange(Tt) > Ln) & nvis[i]      # new on-mask frames beyond L
            if ext.any():
                xyz[n, ext] = o["tracks_xyz_ref0"][i, ext]
                xyzl[n, ext] = o["tracks_xyz_local"][i, ext]
                uv[n, ext] = nuv[i, ext]
                conf[n, ext] = o["tracks_confidence"][i, ext]
                vis[n, ext] = True
                gained += int(ext.sum())
            last_anchor[n] = Ln                       # advance so a no-progress point is excluded
    return gained


def process_scene(inst_dir, keypoints_pkl, video_input_pkl, out_npz, model, chunk,
                  reanchor=True):
    """Build tracker query (SAM 3D camera projection) + run D4RT for one scene (model reused)."""
    with open(video_input_pkl, "rb") as f:
        video = pickle.load(f)            # SCHEMA.VideoInput (SCHEMA importable for unpickle)
    insts = load_instances(inst_dir)
    scene = SceneReconstruction(instances=insts, scene_frame="gt_camera", flagged_ids=[])
    with open(keypoints_pkl, "rb") as f:
        keypoints = pickle.load(f)
    T, H, W = video.frames.shape[:3]

    tq = build_tracker_query(keypoints, scene, video, drop_flagged=False)  # SAM 3D camera proj
    valid = tq.valid
    qv = tq.queries_2d[valid]
    query_uv_norm = np.stack([qv[:, 1] / W, qv[:, 2] / H], axis=1).astype(np.float32)
    query_src = qv[:, 0].astype(np.int64)
    index_valid = [tq.index[i] for i in np.nonzero(valid)[0]]

    out = _infer_tracks(model=model, video_model_rgb=video.frames.astype(np.uint8),
                        query_uv_norm=query_uv_norm, query_chunk_size=chunk,
                        query_src_indices_global=query_src)
    kp_id = np.array([ix[0] for ix in index_valid], np.int64)
    inst_id = np.array([ix[1] for ix in index_valid], np.int64)
    part_id = np.array([(-1 if ix[2] is None else ix[2]) for ix in index_valid], np.int64)
    # ① cross-frame mask-consistency gate -> trustworthy per-frame visibility (downstream consumes this)
    uv_tr = out["tracks_uv_norm"].astype(np.float32)
    vis_d4rt = out["tracks_visibility"]
    vis_mask = mask_consistency_vis(uv_tr, vis_d4rt, inst_id, video.masklets)
    xyz = out["tracks_xyz_ref0"].astype(np.float32)
    xyzl = out["tracks_xyz_local"].astype(np.float32)
    conf = out["tracks_confidence"].astype(np.float32)
    gained = 0
    if reanchor:
        # ② chained re-anchor: extend tracks that drift mid-video (edits xyz/xyzl/uv_tr/conf/vis_mask)
        gained = reanchor_tracks(model, video.frames.astype(np.uint8), xyz, xyzl, uv_tr, conf,
                                 vis_mask, query_src, inst_id, video.masklets, chunk)
    os.makedirs(os.path.dirname(out_npz) or ".", exist_ok=True)
    np.savez_compressed(
        out_npz,
        xyz_ref0=xyz, xyz_local=xyzl,
        visibility=vis_mask,                                  # mask-corrected + re-anchor (downstream)
        visibility_d4rt=vis_d4rt,                             # raw D4RT single-anchor (diagnostics)
        confidence=conf,
        query_t=query_src, query_uv_norm=query_uv_norm,
        uv_track=uv_tr,                                       # (N,T,2) per-frame 2D pixel (norm)
        keypoint_id=kp_id, instance_id=inst_id, part_id=part_id,
    )
    print(f"[stage4] {os.path.basename(os.path.dirname(out_npz)) or out_npz}: "
          f"{valid.sum()}/{len(valid)} on-frame -> tracked {xyz.shape} "
          f"vis {vis_d4rt.mean():.2f}->{vis_mask.mean():.2f} (d4rt->①②, +{gained}f reanchor)", flush=True)


def main():
    import glob
    ap = argparse.ArgumentParser()
    ap.add_argument("--inst-dir", help="single-scene: dir with inst_*.npz")
    ap.add_argument("--keypoints", help="single-scene: pass1_keypoints.pkl")
    ap.add_argument("--video-input", help="single-scene: video_input.pkl")
    ap.add_argument("--out", help="single-scene: output tracks npz")
    ap.add_argument("--batch-root", help="dir of scene_*/ each with inst_*.npz + pass1_keypoints.pkl "
                                         "+ video_input.pkl (batch: load D4RT once, tracks saved in-place)")
    ap.add_argument("--query-chunk-size", type=int, default=512)
    args = ap.parse_args()

    model = load_d4rt()
    if args.batch_root:
        scene_dirs = sorted(d for d in glob.glob(os.path.join(args.batch_root, "scene_*"))
                            if os.path.exists(os.path.join(d, "pass1_keypoints.pkl")))
        print(f"[stage4] BATCH: {len(scene_dirs)} scenes under {args.batch_root}", flush=True)
        for sd in scene_dirs:
            try:
                process_scene(sd, os.path.join(sd, "pass1_keypoints.pkl"),
                              os.path.join(sd, "video_input.pkl"), os.path.join(sd, "tracks_d4rt.npz"),
                              model, args.query_chunk_size)
            except Exception as e:
                print(f"[stage4] {os.path.basename(sd)}: FAILED {type(e).__name__}: {e}", flush=True)
    else:
        if not all([args.inst_dir, args.keypoints, args.video_input, args.out]):
            ap.error("single-scene mode needs --inst-dir --keypoints --video-input --out (or --batch-root)")
        process_scene(args.inst_dir, args.keypoints, args.video_input, args.out, model, args.query_chunk_size)


if __name__ == "__main__":
    main()
