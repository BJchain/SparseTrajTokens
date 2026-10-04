"""V3 Stage-③: STV2 DENSE FRONT-END tracker.

Runs BEFORE keypoint selection. For each instance, seeds a DENSE grid inside its SAM3 mask at
MULTIPLE frames across the sequence (so surfaces that only appear later — after rotation/occlusion —
get seeded at their first-visible frame), then tracks them all with SpaTrackerV2 in ONE call
(forward_stream bins queries by their frame t into sliding windows natively). Output = dense 3D
trajectories + visibility for the whole sequence, in STV2's world frame W_stv.

This is the V3 fix for "single-t* query is incomplete" (the occluded-part / rotating-rigid problem
that motivated V3): instead of projecting a few keypoints at one reference frame, we track EVERY
visible (grid-sampled) surface point across the whole clip.

Key STV2 settings (verified from third_party/SpaTrackerV2 code):
  - queries = (N,3) [t,x,y] with per-point query frame; multi-frame seeding in one call.
  - full_point=True  -> bypass STV2's per-query depth-confidence dropping (needed for dense).
  - fixed_cam=False  -> RUN pycolmap/pyceres BA (DEFAULT ON in V3): refine cameras. Needed for
      large-camera-motion data (e.g. Ego4D first-person); accuracy over speed (A800 + sharding).
  - query_no_BA=True -> mark user (dynamic) query points so they DON'T pollute the camera solve
      (camera is solved from STV2's internal static support points).
  - memory ∝ N×T -> chunk queries (<=chunk) sharing ONE VGGT geometry per scene.

Run in env SpaTrack2:
  TMPDIR=/data/.../tmp CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  <SpaTrack2-python> src/stage4/run_dense_frontend.py --batch-root <dir of scene_*/>
Output per instance: <scene>/dense_tracks_<iid>.npz
"""
from __future__ import annotations
import argparse, glob, os, pickle, sys
from pathlib import Path
import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_STV2 = os.path.join(_ROOT, "third_party", "SpaTrackerV2")
for p in (_ROOT, _STV2):
    if p not in sys.path:
        sys.path.insert(0, p)
import torch  # noqa: E402
# reuse the verified STV2 machinery
from src.stage4.run_stage4_stv2 import load_stv2, _vggt_geom  # noqa: E402


def _resize_mask(mask_hw: np.ndarray, Hn: int, Wn: int) -> np.ndarray:
    """nearest-neighbor resize a (H,W) bool mask to (Hn,Wn) — matches preprocess_image's pure
    resize (no crop for DAVIS 16:9 and square MOVi, verified)."""
    H, W = mask_hw.shape
    ys = np.clip((np.arange(Hn) * H / Hn).astype(int), 0, H - 1)
    xs = np.clip((np.arange(Wn) * W / Wn).astype(int), 0, W - 1)
    return mask_hw[ys][:, xs]


def _seed_dense_queries(masks: np.ndarray, Hn: int, Wn: int, stride: int, n_seed_frames: int,
                        min_pts: int = 8):
    """masks (T,H,W) bool. Seed a stride grid inside the mask at up to n_seed_frames evenly-spaced
    frames (that have enough mask). Returns queries (N,3)=[t,x_model,y_model] and per-point
    (seed_frame (N,), seed_uv_norm (N,2))."""
    T = masks.shape[0]
    have = [t for t in range(T) if masks[t].sum() > 0]
    if not have:
        return None
    # evenly-spaced seed frames among the frames that actually show the object
    if len(have) <= n_seed_frames:
        seed_frames = have
    else:
        seed_frames = [have[i] for i in np.linspace(0, len(have) - 1, n_seed_frames).astype(int)]
        seed_frames = sorted(set(seed_frames))
    gy, gx = np.meshgrid(np.arange(0, Hn, stride), np.arange(0, Wn, stride), indexing="ij")
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)   # (G,2) [x,y] model res
    q = []
    for f in seed_frames:
        mrs = _resize_mask(masks[f].astype(bool), Hn, Wn)
        inside = mrs[grid[:, 1].astype(int), grid[:, 0].astype(int)]
        pts = grid[inside]
        if len(pts) < min_pts:
            continue
        tcol = np.full((len(pts), 1), f, np.float32)
        q.append(np.concatenate([tcol, pts], axis=1))      # [t,x,y]
    if not q:
        return None
    Q = np.concatenate(q, axis=0)
    seed_frame = Q[:, 0].astype(np.int64)
    seed_uv_norm = Q[:, 1:3] / np.array([Wn, Hn], np.float32)
    return Q, seed_frame, seed_uv_norm


def _infer_dense(model, frames, queries_txy_model, chunk, ba: bool, gt_extrs=None):
    """queries_txy_model (N,3)=[t, x_model, y_model]. Returns world/cam/uv/vis/conf (N,T,*) + c2w,intrs.
    full_point=True (dense), fixed_cam=(not ba). c2w0 from first chunk fixes W_stv across chunks."""
    vggt, predictor = model
    vt, depth, intrs_g, extrs, unc, Hn, Wn = _vggt_geom(vggt, frames)
    T = vt.shape[0]
    # PATH 2 (MOVi-E, GT camera known): feed the GT extrinsics (c2w) and FIX the camera -> STV2 tracks
    # land in a GT-ORIENTED world (R,t pinned by GT), leaving only the VGGT-depth SCALE for Stage⑦ to
    # solve (1-DOF). Keep VGGT intrs+depth (their unprojection is mutually consistent). Path 1: VGGT
    # camera + BA as before.
    if gt_extrs is not None:
        extrs = np.asarray(gt_extrs, np.float32)[:T]
        fixed = True
    else:
        fixed = (not ba)
    N = int(queries_txy_model.shape[0])
    cs = max(2, int(chunk))
    n_chunks = max(1, (N + cs - 1) // cs)
    worlds, cams, uvs, viss, confs = [], [], [], [], []
    c2w0 = None; intrs_out = None
    for idx in np.array_split(np.arange(N), n_chunks):
        s, e = int(idx[0]), int(idx[-1]) + 1
        queries = queries_txy_model[s:e].astype(np.float32)          # (Nc,3) [t,x,y] model res
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            (c2w, io, _pm, _cd, tr3d, tr2d, vis, conf, _v) = predictor.forward(
                vt, depth=depth, intrs=intrs_g, extrs=extrs, queries=queries, fps=1,
                full_point=True, iters_track=4, query_no_BA=True, fixed_cam=fixed, stage=1,
                unc_metric=unc, support_frame=T - 1, replace_ratio=0.2)
        c2w = c2w.float().cpu().numpy()[:T]
        if c2w0 is None:
            c2w0 = c2w
            intrs_out = io.float().cpu().numpy()[:T] if hasattr(io, "float") else intrs_g
        tr3dc = tr3d.float().cpu().numpy()[:T, :, :3]
        tr2dc = tr2d.float().cpu().numpy()[:T, :, :2]
        visc = (vis.float().cpu().numpy()[:T, :, 0] > 0.5)
        confc = conf.float().cpu().numpy()[:T]
        if confc.ndim == 3:
            confc = confc[..., 0]
        world = np.einsum("tij,tnj->tni", c2w0[:, :3, :3], tr3dc) + c2w0[:, :3, 3][:, None]
        worlds.append(world); cams.append(tr3dc); uvs.append(tr2dc); viss.append(visc); confs.append(confc)
    world = np.concatenate(worlds, axis=1); cam = np.concatenate(cams, axis=1)
    uv = np.concatenate(uvs, axis=1) / np.array([Wn, Hn], float)
    vis = np.concatenate(viss, axis=1); conf = np.concatenate(confs, axis=1)
    return dict(
        xyz_world=world.transpose(1, 0, 2).astype(np.float32),      # (N,T,3) W_stv
        xyz_cam=cam.transpose(1, 0, 2).astype(np.float32),          # (N,T,3) camera frame
        uv_norm=uv.transpose(1, 0, 2).astype(np.float32),           # (N,T,2) model-res normalized
        vis=vis.transpose(1, 0), conf=conf.transpose(1, 0).astype(np.float32),
        c2w=c2w0.astype(np.float32), intrs=np.asarray(intrs_out, np.float32), Hn=Hn, Wn=Wn)


def _infer_dense_windowed(model, frames, masks, stride, n_seed, chunk, ba, gt_extrs,
                          win_len, overlap):
    """PART 1 v2 — ACTIVE HANDOFF (2026-07-10). v1 tracked each subwindow independently and stitched
    by 2D-NN matching; at low visibility the match rate was ~4% -> long trajectories never formed.
    v2 makes continuation DETERMINISTIC: window k+1's query set = CONTINUATION queries (each
    accumulated track alive in the overlap is re-seeded AT ITS OWN last-visible pixel -> identity
    inherited by construction, no matching) + fresh grid seeds for newly-appeared surfaces. Gauge
    between windows is aligned on the continuation pairs' 3D positions over the overlap
    (guarded: scale in [0.7,1.4] AND must reduce the overlap mismatch, else identity)."""
    from src.geometry import umeyama_similarity
    from src.stage4 import run_stage4_stv2 as _m
    T = frames.shape[0]
    starts = list(range(0, max(T - overlap, 1), win_len - overlap))
    wins = [(x, min(x + win_len, T)) for x in starts if x < T]
    if wins[-1][1] < T:
        wins.append((max(T - win_len, 0), T))
    _vt, _d, _i, _e, _u, Hn, Wn = _vggt_geom(model[0], frames)
    W = C = U = V = F = Q = None
    c2w0 = intrs0 = None
    prev_end = 0
    for (s, e) in wins:
        fresh = _seed_dense_queries(masks[s:e], Hn, Wn, stride, n_seed)
        if W is None:
            if fresh is None:
                continue
            Qf, sf, suv = fresh
            _m._GEOM_CACHE.clear()
            d = _infer_dense(model, frames[s:e], Qf, chunk, ba,
                             gt_extrs=(gt_extrs[s:e] if gt_extrs is not None else None))
            n = d["vis"].shape[0]
            W = np.zeros((n, T, 3), np.float32); C = np.zeros((n, T, 3), np.float32)
            U = np.zeros((n, T, 2), np.float32); V = np.zeros((n, T), bool)
            F = sf + s; Q = suv
            W[:, s:e] = d["xyz_world"]; C[:, s:e] = d["xyz_cam"]
            U[:, s:e] = d["uv_norm"];   V[:, s:e] = d["vis"]
            c2w0, intrs0 = d["c2w"], d["intrs"]
            prev_end = e
            continue
        # ---- ACTIVE HANDOFF: continuation queries from tracks alive in the overlap [s, prev_end) ----
        alive, qs = [], []
        for i in range(W.shape[0]):
            vseg = V[i, s:prev_end]
            if vseg.any():
                hf = s + int(np.nonzero(vseg)[0][-1])          # its own last-visible overlap frame
                u, vv = U[i, hf]
                qs.append([hf - s, u * Wn, vv * Hn]); alive.append(i)
        Qc = np.asarray(qs, np.float32) if qs else np.zeros((0, 3), np.float32)
        if fresh is not None:
            Qf, sf, suv = fresh
        else:
            Qf = np.zeros((0, 3), np.float32); sf = np.zeros(0, np.int64); suv = np.zeros((0, 2), np.float32)
        Qall = np.concatenate([Qc, Qf]) if len(Qc) or len(Qf) else None
        if Qall is None or len(Qall) == 0:
            prev_end = e
            continue
        _m._GEOM_CACHE.clear()
        d = _infer_dense(model, frames[s:e], Qall, chunk, ba,
                         gt_extrs=(gt_extrs[s:e] if gt_extrs is not None else None))
        nC = len(Qc)
        # gauge align on continuation pairs over the overlap (guarded)
        src, dst = [], []
        ov = prev_end - s
        for r, i in enumerate(alive):
            for k in range(ov):
                if d["vis"][r, k] and V[i, s + k]:
                    src.append(d["xyz_world"][r, k]); dst.append(W[i, s + k])
        if len(src) >= 8:
            src_a, dst_a = np.array(src), np.array(dst)
            sc, Rg, tg = umeyama_similarity(src_a, dst_a)
            e_b = float(np.median(np.linalg.norm(src_a - dst_a, axis=1)))
            e_a = float(np.median(np.linalg.norm((sc * (src_a @ Rg.T)) + tg - dst_a, axis=1)))
            if 0.7 <= sc <= 1.4 and e_a < e_b:
                Xb = d["xyz_world"].reshape(-1, 3)
                d["xyz_world"] = ((sc * (Xb @ Rg.T)) + tg).reshape(d["xyz_world"].shape).astype(np.float32)
        # write continuations (identity inherited)
        off = prev_end - s
        for r, i in enumerate(alive):
            W[i, prev_end:e] = d["xyz_world"][r, off:]; C[i, prev_end:e] = d["xyz_cam"][r, off:]
            U[i, prev_end:e] = d["uv_norm"][r, off:];   V[i, prev_end:e] = d["vis"][r, off:]
        # append fresh-seeded rows (new surfaces)
        m = d["vis"].shape[0] - nC
        if m > 0:
            Wn2 = np.zeros((m, T, 3), np.float32); Cn = np.zeros((m, T, 3), np.float32)
            Un = np.zeros((m, T, 2), np.float32); Vn = np.zeros((m, T), bool)
            Wn2[:, s:e] = d["xyz_world"][nC:]; Cn[:, s:e] = d["xyz_cam"][nC:]
            Un[:, s:e] = d["uv_norm"][nC:];   Vn[:, s:e] = d["vis"][nC:]
            W = np.concatenate([W, Wn2]); C = np.concatenate([C, Cn])
            U = np.concatenate([U, Un]);  V = np.concatenate([V, Vn])
            F = np.concatenate([F, sf + s]); Q = np.concatenate([Q, suv])
        prev_end = e
    if W is None:
        return None
    return dict(xyz_world=W, xyz_cam=C, uv_norm=U, vis=V, conf=V.astype(np.float32),
                c2w=c2w0, intrs=intrs0, Hn=Hn, Wn=Wn, seed_frame=F, seed_uv_norm=Q)


def process_scene(scene_dir, model, stride, n_seed_frames, chunk, ba):
    vip = os.path.join(scene_dir, "video_input.pkl")
    if not os.path.exists(vip):
        print(f"  [skip] {scene_dir}: no video_input.pkl", flush=True); return
    vid = pickle.load(open(vip, "rb"))
    frames = np.asarray(vid.frames)
    # Path 2 GT-camera feeding is OPT-IN (V3_PATH2_GTCAM=1) and currently DISABLED by default:
    # feeding raw GT c2w to STV2 leaves a ~150-160deg residual rotation vs W_gt (STV2/VGGT camera-axis
    # convention != kubric OpenCV), so the intended scale-only (R=I) align is INVALID. The general
    # 7-DOF Umeyama in Stage⑦ absorbs this rotation and works (recon 0.013, n=19). Re-enable once the
    # STV2<->kubric camera-convention flip is worked into the fed extrinsics.
    gt_extrs = getattr(vid, "camera_extrinsics", None) if os.environ.get("V3_PATH2_GTCAM") == "1" else None
    if gt_extrs is not None:
        print(f"  [Path2] GT camera present -> feeding GT extrinsics (fixed_cam, scale-only align)", flush=True)
    Hn = Wn = None
    for ml in vid.masklets:
        masks = np.asarray(ml.masks).astype(bool)
        # peek model res once
        if Hn is None:
            _vt, _d, _i, _e, _u, Hn, Wn = _vggt_geom(model[0], frames)
        # PART 1 opt-in: V3_DENSE_CHUNK="<win_len>:<overlap>" (e.g. "48:12") enables windowed
        # tracking for T > win_len. Unset (default) = single-call path -> other datasets untouched.
        wcfg = os.environ.get("V3_DENSE_CHUNK", "")
        T_all = frames.shape[0]
        try:
            if wcfg and T_all > int(wcfg.split(":")[0]):
                wl, ovl = (int(x) for x in wcfg.split(":"))
                d = _infer_dense_windowed(model, frames, masks, stride, n_seed_frames, chunk, ba,
                                          gt_extrs, wl, ovl)
                if d is None:
                    print(f"  inst {ml.instance_id}: no seedable frames -> skip", flush=True); continue
                Q = np.zeros((d["vis"].shape[0], 3), np.float32)   # (unused; kept for count print)
                seed_frame, seed_uv = d.pop("seed_frame"), d.pop("seed_uv_norm")
            else:
                seeded = _seed_dense_queries(masks, Hn, Wn, stride, n_seed_frames)
                if seeded is None:
                    print(f"  inst {ml.instance_id}: no seedable frames -> skip", flush=True); continue
                Q, seed_frame, seed_uv = seeded
                d = _infer_dense(model, frames, Q, chunk, ba, gt_extrs=gt_extrs)
        except Exception as ex:
            print(f"  inst {ml.instance_id}: FAILED {type(ex).__name__}: {str(ex)[:80]}", flush=True); continue
        # visible-count per frame (the V3 metric: co-visibility across time)
        vpf = d["vis"].sum(0)                                      # (T,)
        out = os.path.join(scene_dir, f"dense_tracks_{ml.instance_id}.npz")
        np.savez_compressed(out, instance_id=ml.instance_id, seed_frame=seed_frame,
                            seed_uv_norm=seed_uv, **d)
        print(f"  inst {ml.instance_id}: N={len(Q)} seeded on {len(set(seed_frame.tolist()))} frames "
              f"| visible/frame min={int(vpf.min())} med={int(np.median(vpf))} max={int(vpf.max())} "
              f"| mean vis={d['vis'].mean():.2f} -> {os.path.basename(out)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--scene-dir"); g.add_argument("--batch-root")
    ap.add_argument("--stride", type=int, default=6, help="grid stride in model-res px (density)")
    ap.add_argument("--n-seed-frames", type=int, default=8, help="how many frames to seed a grid at")
    ap.add_argument("--chunk", type=int, default=384, help="max queries per STV2 call (mem ∝ N×T)")
    ap.add_argument("--no-ba", action="store_true", help="disable pycolmap BA (fixed_cam); default BA ON")
    args = ap.parse_args()
    ba = not args.no_ba
    model = load_stv2()
    scenes = ([args.scene_dir] if args.scene_dir
              else sorted(glob.glob(os.path.join(args.batch_root, "scene_*"))))
    print(f"[dense-front] {len(scenes)} scene(s), stride={args.stride} n_seed={args.n_seed_frames} "
          f"chunk={args.chunk} BA={'ON' if ba else 'OFF'}", flush=True)
    for sd in scenes:
        print(f"[dense-front] {os.path.basename(sd)}", flush=True)
        global _GEOM_CACHE_CLEARED
        from src.stage4 import run_stage4_stv2 as _m
        _m._GEOM_CACHE.clear()               # per-scene geometry cache reset
        process_scene(sd, model, args.stride, args.n_seed_frames, args.chunk, ba)
    print("[dense-front] done", flush=True)


if __name__ == "__main__":
    main()
