"""Stage 4 (v2) — SpaTrackerV2 backend (drop-in replacement for run_stage4_d4rt.py).

Same contract as the D4RT stage 4: pass1 keypoints + Stage1 InstanceMeshes + video
  -> SCHEMA.build_tracker_query (project keypoints to t* via SAM 3D's own camera)
  -> _infer_tracks (SpaTrackerV2)  [per-point (t,x,y) query incl. per-point query frame]
  -> mask-consistency gate (①) + chained re-anchor (②)
  -> tracks_d4rt.npz   (SAME filename + fields, so stage2b/pass2/finalize are UNCHANGED).

Why STV2: the bake-off (project_tracker_bakeoff) showed STV2 ≳ all candidates on DAVIS
(on-mask 0.71 vs D4RT 0.10), AND it is the only candidate that natively satisfies the full V2
tracker contract — arbitrary per-point pixel queries WITH per-point query frame ([t,x,y], used
by re-anchor), native per-frame 2D + per-point 3D + visibility, self-contained VGGT depth front-end
(monocular RGB, no GT camera). `_infer_tracks` mirrors D4RT's signature/return so the
backend-agnostic mask_consistency_vis + reanchor_tracks below are reused verbatim.

Run in env SpaTrack2:
  TMPDIR=/tmp CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  <SpaTrack2-env-python> src/stage4/run_stage4_stv2.py --batch-root <dir of scene_*/>
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys
from pathlib import Path

import numpy as np

# --- env bring-up (VGGT/predictor load from local HF cache; no network) ---------
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_STV2 = os.path.join(_ROOT, "third_party", "SpaTrackerV2")
for p in (_ROOT, _STV2):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402
from SCHEMA import InstanceMesh, SceneReconstruction, build_tracker_query  # noqa: E402
# STV2 model (its top-level package is `models`, no collision with our `src`/`SCHEMA`)
from models.SpaTrackV2.models.predictor import Predictor  # noqa: E402
from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track  # noqa: E402
from models.SpaTrackV2.models.vggt4track.utils.load_fn import preprocess_image  # noqa: E402


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


def load_stv2():
    """Return (vggt_front, predictor) — both from the local HF cache."""
    vggt = VGGT4Track.from_pretrained("Yuxihenry/SpatialTrackerV2_Front").eval().cuda()
    predictor = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Offline")
    predictor.spatrack.track_num = 756
    predictor.eval().cuda()
    print("[stage4-stv2] STV2 (VGGT front + Offline predictor) loaded", flush=True)
    return (vggt, predictor)


# SPEED: fixed_cam=True makes STV2's predictor SKIP the per-window pycolmap/pyceres bundle
# adjustment (TrackRefiner: fixed_cam branch bypasses ba_pycolmap) and trust the VGGT cameras
# directly — the dominant stage-4 cost. Tracking (2D/vis) is the tracker's job, not BA's, so this
# is mostly a 3D-camera-refinement trade. Toggle via --no-fixed-cam.
FIXED_CAM = True

# VGGT depth/camera is query-independent -> compute once per scene, reuse across the main call
# AND each re-anchor round (keyed by the frames array identity; cleared per scene).
_GEOM_CACHE: dict = {}


def _vggt_geom(vggt, frames: np.ndarray):
    key = id(frames)
    if key in _GEOM_CACHE:
        return _GEOM_CACHE[key]
    vt = preprocess_image(torch.from_numpy(frames).permute(0, 3, 1, 2).float())   # (T,3,Hn,Wn)
    Hn, Wn = vt.shape[2:]
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        pred = vggt(vt[None].cuda() / 255)
    depth = pred["points_map"][..., 2].squeeze().float().cpu().numpy()            # (T,Hn,Wn)
    intrs = pred["intrs"].squeeze().float().cpu().numpy()                          # (T,3,3)
    extrs = pred["poses_pred"].squeeze().float().cpu().numpy()                     # (T,4,4)
    unc = (pred["unc_metric"].squeeze().cpu().numpy() > 0.5)
    # KNOWN EDGE CASE (5/80 DAVIS scenes: rallye/sheep/scooter-board/gold-fish/dance-jump):
    # STV2's forward (SpaTrack.py:618) filters query pixels by VGGT depth-conf and can drop ALL of a
    # scene's queries -> torch.randint(0,0) crash. Passing all-confident unc / single-window did NOT
    # bypass it (internal grid_sample of the conf at query locs returns 0 for these) — left as a
    # documented gap; main() per-scene try/except keeps the batch resilient (those scenes get no tracks).
    geom = (vt, depth, intrs, extrs, unc, int(Hn), int(Wn))
    _GEOM_CACHE[key] = geom
    return geom


def _infer_tracks(model, video_model_rgb, query_uv_norm, query_chunk_size,
                  query_src_indices_global):
    """SpaTrackerV2 backend mirroring D4RT's _infer_tracks(...) signature + return dict.
    query_uv_norm (N,2) in [0,1] (orig-res normalized); query_src_indices_global (N,) per-point
    query frame. Returns (N,T,*) arrays: tracks_xyz_ref0 (world), tracks_xyz_local (camera),
    tracks_uv_norm (orig-res normalized), tracks_visibility (bool), tracks_confidence."""
    vggt, predictor = model
    vt, depth, intrs, extrs, unc, Hn, Wn = _vggt_geom(vggt, video_model_rgb)
    T = vt.shape[0]
    N = int(query_uv_norm.shape[0])
    qt = query_src_indices_global.astype(np.int64)
    qx = query_uv_norm[:, 0] * Wn
    qy = query_uv_norm[:, 1] * Hn
    if N == 0:
        z = lambda c: np.zeros((0, T, c), np.float32)  # noqa: E731
        return dict(tracks_xyz_ref0=z(3), tracks_xyz_local=z(3), tracks_uv_norm=z(2),
                    tracks_visibility=np.zeros((0, T), bool), tracks_confidence=np.zeros((0, T), np.float32))

    # BALANCED CHUNKING: split into ceil(N/chunk) near-equal groups so NO chunk has <2 queries.
    # STV2's forward mis-shapes a SINGLE-query chunk into a 5-D tensor -> TrackRefiner crash
    # ("too many values to unpack (expected 4)"); a 512-size chunk over N=513 left a 1-query tail
    # (snowboard). array_split keeps sizes within 1 of each other (513 -> [257,256]), never a
    # singleton tail. N==1 (a lone valid query) can't be chunked away -> duplicate then slice back.
    pad1 = (N == 1)
    if pad1:
        qt = np.concatenate([qt, qt]); qx = np.concatenate([qx, qx]); qy = np.concatenate([qy, qy]); N = 2
    cs = max(2, int(query_chunk_size))
    n_chunks = max(1, (N + cs - 1) // cs)

    worlds, cams, uvs, viss, confs = [], [], [], [], []
    c2w0 = None
    for idx in np.array_split(np.arange(N), n_chunks):
        s, e = int(idx[0]), int(idx[-1]) + 1                                          # contiguous range
        queries = np.stack([qt[s:e], qx[s:e], qy[s:e]], axis=1).astype(np.float32)   # (Nc,3) [t,x,y]
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            (c2w, _io, _pm, _cd, tr3d, tr2d, vis, conf, _v) = predictor.forward(
                vt, depth=depth, intrs=intrs, extrs=extrs, queries=queries, fps=1,
                full_point=False, iters_track=4, query_no_BA=True, fixed_cam=FIXED_CAM, stage=1,
                unc_metric=unc, support_frame=T - 1, replace_ratio=0.2)
        c2w = c2w.float().cpu().numpy()[:T]
        if c2w0 is None:
            c2w0 = c2w                                  # fix world frame across chunks (query-indep)
        tr3dc = tr3d.float().cpu().numpy()[:T, :, :3]   # (T,Nc,3) camera-frame xyz
        tr2dc = tr2d.float().cpu().numpy()[:T, :, :2]   # (T,Nc,2) px in (Wn,Hn)
        visc = (vis.float().cpu().numpy()[:T, :, 0] > 0.5)
        confc = conf.float().cpu().numpy()[:T]
        if confc.ndim == 3:
            confc = confc[..., 0]
        world = np.einsum("tij,tnj->tni", c2w0[:, :3, :3], tr3dc) + c2w0[:, :3, 3][:, None]
        worlds.append(world); cams.append(tr3dc); uvs.append(tr2dc); viss.append(visc); confs.append(confc)

    world = np.concatenate(worlds, axis=1)              # (T,N,3)
    cam = np.concatenate(cams, axis=1)
    uv = np.concatenate(uvs, axis=1) / np.array([Wn, Hn], float)   # (T,N,2) orig-res normalized
    vis = np.concatenate(viss, axis=1)
    conf = np.concatenate(confs, axis=1)
    if pad1:                                            # undo the N==1 duplication
        world, cam, uv, vis, conf = world[:, :1], cam[:, :1], uv[:, :1], vis[:, :1], conf[:, :1]
    return dict(
        tracks_xyz_ref0=world.transpose(1, 0, 2).astype(np.float32),
        tracks_xyz_local=cam.transpose(1, 0, 2).astype(np.float32),
        tracks_uv_norm=uv.transpose(1, 0, 2).astype(np.float32),
        tracks_visibility=vis.transpose(1, 0),
        tracks_confidence=conf.transpose(1, 0).astype(np.float32),
    )


def mask_consistency_vis(uv_track, vis_d4rt, inst_id, masklets, tol=4):
    """① Cross-frame mask-consistency gate (REFERENCE §6.1). A tracked point is visible at frame t
    ONLY IF the tracker says visible AND its 2D pixel lands inside that instance's mask at t
    (few-px tolerance). Per-frame per-point. (Backend-agnostic — copied from run_stage4_d4rt.)"""
    import cv2
    mask_by_inst = {m.instance_id: m.masks for m in (masklets or [])}
    N, Tt, _ = uv_track.shape
    out = vis_d4rt.copy()
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
                out[n, t] = False
    return out


def reanchor_tracks(model, frames, xyz, xyzl, uv, conf, vis, query_src, inst_id, masklets,
                    chunk, max_rounds=4):
    """② chained re-anchoring (REFERENCE §6.3, depends on ①). A point that drifts off the mask
    before the last frame is RE-QUERIED at its LAST on-mask frame L (per-point query frame!) and
    forward-tracked frames t>L that pass ① are spliced in. xyz(=world) is query-anchor-independent
    (VGGT geometry is deterministic), so spliced segments are consistent. (Copied verbatim from
    run_stage4_d4rt — backend-agnostic, drives the swapped _infer_tracks.)"""
    N, Tt = vis.shape
    last_anchor = query_src.astype(int).copy()
    gained = 0
    for _ in range(max_rounds):
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
            ext = (np.arange(Tt) > Ln) & nvis[i]
            if ext.any():
                xyz[n, ext] = o["tracks_xyz_ref0"][i, ext]
                xyzl[n, ext] = o["tracks_xyz_local"][i, ext]
                uv[n, ext] = nuv[i, ext]
                conf[n, ext] = o["tracks_confidence"][i, ext]
                vis[n, ext] = True
                gained += int(ext.sum())
            last_anchor[n] = Ln
    return gained


def _longest_run_mid(v: np.ndarray) -> int:
    """Midpoint frame of the longest contiguous True run in (T,) bool v; -1 if none.
    Anchoring at the densest-visible center maximizes bidirectional reach of a good anchor."""
    T = len(v); best_len = 0; best_mid = -1; i = 0
    while i < T:
        if v[i]:
            j = i
            while j < T and v[j]:
                j += 1
            if j - i > best_len:
                best_len = j - i; best_mid = (i + j - 1) // 2
            i = j
        else:
            i += 1
    return best_mid


def active_anchor_tracks(model, frames, xyz, xyzl, uv, conf, vis, inst_id, masklets, chunk,
                         cov_thresh=0.9):
    """PROACTIVE multi-anchor (REFERENCE §6.3, complements reactive reanchor). After the initial
    t* pass, RE-ANCHOR each under-covered keypoint at its BEST on-mask frame (midpoint of its
    longest contiguous on-mask run — gives a reliable anchor with bidirectional reach), using that
    frame's tracked 2D position. Adopt the re-anchored trajectory WHOLE iff it covers MORE on-mask
    frames (best-of stitch; guarded -> never reduces coverage, keeps each keypoint's trajectory
    internally consistent for SE(3)). Needs the tracker's per-point query frame ([t,x,y]); seeding
    at the best frame is the V2 lever that took breakdance dense@best 66->91% in the bake-off.
    Edits arrays in place; returns #keypoints improved."""
    N, T = vis.shape
    cur_cov = vis.mean(1)
    t_best = np.array([_longest_run_mid(vis[n]) for n in range(N)])
    todo = [n for n in range(N) if t_best[n] >= 0 and cur_cov[n] < cov_thresh
            and t_best[n] != 0]                            # 0 == nothing-visible fallback guard
    if not todo:
        return 0
    q_uv = np.array([uv[n, t_best[n]] for n in todo], np.float32)     # (M,2) normalized, on-mask pixel
    q_src = np.array([t_best[n] for n in todo], np.int64)
    o = _infer_tracks(model=model, video_model_rgb=frames, query_uv_norm=q_uv,
                      query_chunk_size=chunk, query_src_indices_global=q_src)
    nuv = o["tracks_uv_norm"].astype(np.float32)
    nvis = mask_consistency_vis(nuv, o["tracks_visibility"], inst_id[todo], masklets)
    improved = 0
    for i, n in enumerate(todo):
        if int(nvis[i].sum()) > int(vis[n].sum()):        # best-of: adopt only if more on-mask frames
            xyz[n] = o["tracks_xyz_ref0"][i]
            xyzl[n] = o["tracks_xyz_local"][i]
            uv[n] = nuv[i]
            conf[n] = o["tracks_confidence"][i]
            vis[n] = nvis[i]
            improved += 1
    return improved


def process_scene(inst_dir, keypoints_pkl, video_input_pkl, out_npz, model, chunk,
                  reanchor=True, active_anchor=True):
    """Build tracker query (SAM 3D camera projection) + run STV2 for one scene (model reused)."""
    with open(video_input_pkl, "rb") as f:
        video = pickle.load(f)
    insts = load_instances(inst_dir)
    scene = SceneReconstruction(instances=insts, scene_frame="gt_camera", flagged_ids=[])
    with open(keypoints_pkl, "rb") as f:
        keypoints = pickle.load(f)
    T, H, W = video.frames.shape[:3]
    frames = video.frames.astype(np.uint8)                 # ONE array -> VGGT geom cache hits in reanchor

    tq = build_tracker_query(keypoints, scene, video, drop_flagged=False)
    valid = tq.valid
    qv = tq.queries_2d[valid]
    query_uv_norm = np.stack([qv[:, 1] / W, qv[:, 2] / H], axis=1).astype(np.float32)
    query_src = qv[:, 0].astype(np.int64)
    index_valid = [tq.index[i] for i in np.nonzero(valid)[0]]

    sc = os.path.basename(os.path.dirname(out_npz)) or "scene"
    try:
        out = _infer_tracks(model=model, video_model_rgb=frames,
                            query_uv_norm=query_uv_norm, query_chunk_size=chunk,
                            query_src_indices_global=query_src)
    except (RuntimeError, ValueError) as e:
        # STV2 windowing edge case: when queries cluster at one frame, a later sliding window has
        # 0 new queries -> torch.randint(0,0) in SpaTrack.forward. Retry as a SINGLE window (all
        # queries in one window). T<=~32 DAVIS clips fit one window fine.
        pred = model[1]; old_sw = pred.S_wind
        pred.S_wind = int(frames.shape[0]) + 1
        _GEOM_CACHE.pop(id(frames), None)
        print(f"[stage4-stv2] {sc}: main pass retry single-window ({type(e).__name__}: {e})", flush=True)
        try:
            out = _infer_tracks(model=model, video_model_rgb=frames,
                                query_uv_norm=query_uv_norm, query_chunk_size=chunk,
                                query_src_indices_global=query_src)
        finally:
            pred.S_wind = old_sw
    kp_id = np.array([ix[0] for ix in index_valid], np.int64)
    inst_id = np.array([ix[1] for ix in index_valid], np.int64)
    part_id = np.array([(-1 if ix[2] is None else ix[2]) for ix in index_valid], np.int64)
    uv_tr = out["tracks_uv_norm"].astype(np.float32)
    vis_raw = out["tracks_visibility"]
    vis_mask = mask_consistency_vis(uv_tr, vis_raw, inst_id, video.masklets)
    xyz = out["tracks_xyz_ref0"].astype(np.float32)
    xyzl = out["tracks_xyz_local"].astype(np.float32)
    conf = out["tracks_confidence"].astype(np.float32)
    # active-anchor + reanchor are BEST-EFFORT enhancements over the (already good) main pass.
    # STV2 can choke on certain small/degenerate re-query batches (empty support window ->
    # "random_ from=0>=to=0" / queries.shape unpack); never let that lose the whole scene — the
    # in-place edits happen only AFTER _infer_tracks returns, so a caught exception leaves the
    # main-pass arrays intact and consistent.
    sc = os.path.basename(os.path.dirname(out_npz)) or "scene"
    n_active = 0
    if active_anchor:
        try:
            n_active = active_anchor_tracks(model, frames, xyz, xyzl, uv_tr, conf,
                                            vis_mask, inst_id, video.masklets, chunk)
        except Exception as e:
            print(f"[stage4-stv2] {sc}: active_anchor skipped ({type(e).__name__}: {e})", flush=True)
    gained = 0
    if reanchor:
        try:
            gained = reanchor_tracks(model, frames, xyz, xyzl, uv_tr, conf,
                                     vis_mask, query_src, inst_id, video.masklets, chunk)
        except Exception as e:
            print(f"[stage4-stv2] {sc}: reanchor skipped ({type(e).__name__}: {e})", flush=True)
    _GEOM_CACHE.pop(id(frames), None)                      # free this scene's VGGT geometry
    os.makedirs(os.path.dirname(out_npz) or ".", exist_ok=True)
    np.savez_compressed(
        out_npz,
        xyz_ref0=xyz, xyz_local=xyzl,
        visibility=vis_mask,                               # mask-corrected + re-anchor (downstream)
        visibility_d4rt=vis_raw,                           # raw tracker single-anchor (diagnostics)
        confidence=conf,
        query_t=query_src, query_uv_norm=query_uv_norm,
        uv_track=uv_tr,
        keypoint_id=kp_id, instance_id=inst_id, part_id=part_id,
    )
    print(f"[stage4-stv2] {os.path.basename(os.path.dirname(out_npz)) or out_npz}: "
          f"{valid.sum()}/{len(valid)} on-frame -> tracked {xyz.shape} "
          f"vis {vis_raw.mean():.2f}->{vis_mask.mean():.2f} "
          f"(raw->①active{n_active}②+{gained}f)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inst-dir")
    ap.add_argument("--keypoints")
    ap.add_argument("--video-input")
    ap.add_argument("--out")
    ap.add_argument("--batch-root", help="dir of scene_*/ each with inst_*.npz + pass1_keypoints.pkl "
                                         "+ video_input.pkl (batch: load STV2 once, tracks saved in-place)")
    ap.add_argument("--query-chunk-size", type=int, default=512)
    ap.add_argument("--no-fixed-cam", action="store_true",
                    help="run STV2's pycolmap BA (slower, refines cameras); default skips it (fixed_cam).")
    ap.add_argument("--no-active-anchor", action="store_true",
                    help="disable proactive best-frame re-anchoring (for ablation).")
    args = ap.parse_args()
    if args.no_fixed_cam:
        globals()["FIXED_CAM"] = False
    active = not args.no_active_anchor

    model = load_stv2()
    if args.batch_root:
        scene_dirs = sorted(d for d in glob.glob(os.path.join(args.batch_root, "scene_*"))
                            if os.path.exists(os.path.join(d, "pass1_keypoints.pkl")))
        print(f"[stage4-stv2] BATCH: {len(scene_dirs)} scenes under {args.batch_root} "
              f"(fixed_cam={FIXED_CAM} active_anchor={active})", flush=True)
        for sd in scene_dirs:
            try:
                process_scene(sd, os.path.join(sd, "pass1_keypoints.pkl"),
                              os.path.join(sd, "video_input.pkl"), os.path.join(sd, "tracks_d4rt.npz"),
                              model, args.query_chunk_size, active_anchor=active)
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"[stage4-stv2] {os.path.basename(sd)}: FAILED {type(e).__name__}: {e}", flush=True)
    else:
        if not all([args.inst_dir, args.keypoints, args.video_input, args.out]):
            ap.error("single-scene mode needs --inst-dir --keypoints --video-input --out (or --batch-root)")
        process_scene(args.inst_dir, args.keypoints, args.video_input, args.out, model,
                      args.query_chunk_size, active_anchor=active)


if __name__ == "__main__":
    main()
