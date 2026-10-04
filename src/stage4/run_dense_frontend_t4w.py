"""V3 Stage-③: Track4World DENSE FRONT-END tracker (contract-identical to the STV2 version,
src/stage4/run_dense_frontend.py — that file stays untouched).

T4W capability: frame-level dense-grid tracking only (no per-point query frames). So instead of
per-point [t,x,y] queries, we run a small number of SHARED PASSES: for each pass frame qf, ONE
model.infer(rgbs[:, qf:], tracking3d=True) yields dense flow_2d/flow_3d/visconf/camera_poses for
[qf, T). Pass frames = uniform every --pass-every (ADAPTIVE by default, 2026-07-29:
max(3, T//8) -> ~4-9 uniform passes even for short clips; the old fixed 20 gave T=24 only 2
passes) (always incl. 0) ∪ each instance's t* (inst_<iid>.npz reference_frame_index — MANDATORY
when the file exists at run time; in the V3 batch orchestration the front-end runs BEFORE
Stage 1, so t* coverage is topped up afterwards via --tstar-topup, see below) ∪ each instance's
first non-empty mask frame (greedy by how many instances first appear there, filling up to
max(MAX_PASSES, #mandatory)).

SEEDING DENSITY (ADAPTIVE by default, 2026-07-29): per instance the candidate-grid step is
derived from its LARGEST single-frame mask area at MODEL resolution, targeting ~--seed-target
points (default 200 — the ballpark the STV2 front-end gets from its model-res stride-6 grid),
with total accumulated tracks capped at --seed-max (default 2000) per instance. The old fixed
ORIGINAL-res stride under-seeded small-resolution data (MOVi 256px: stride 6 orig ~ 15 model px
-> N med 33 vs STV2 853). --stride <px> forces the legacy fixed original-res grid.

Per instance, per pass qf: candidate pixels = grid inside its mask@qf; candidates within one grid
step (model px) of an existing track that is visible at qf are dropped as duplicates; the rest are
gathered from the pass maps into NEW tracks covering [qf, T) (vis=False, values 0 before qf;
seed_frame=qf).

World gauge: pass0 (qf=0) defines the world (its frame-0 camera, via the official
transform_to_first_frame convention); each later pass is first expressed in its OWN world (its
qf-camera), then aligned to pass0's world by Umeyama over co-visible existing-track points in the
overlap — guarded like _infer_dense_windowed in the STV2 version (scale in [0.7,1.4] AND overlap
error must drop), else falls back to the rigid c2w composition inv(P0[0]) @ P0[qf].
(2026-07-29 fix: the pair-gathering pixel lookup used uv_norm*[sx,sy] instead of uv_norm*[Wm,Hm],
sampling corner pixels -> Umeyama almost never engaged and every pass silently fell back.)

--tstar-topup (A' 2026-07-29; the batch orchestrator runs it BETWEEN Stage 1 and Stage-⑦): for
each instance whose inst_<iid>.npz exists and whose existing dense_tracks_<iid>.npz has fewer
than --topup-min-vis visible tracks at its t*, run ONE extra pass with qf=t* and MERGE the new
tracks into the existing npz (same contract keys; c2w/intrs untouched). The topup pass is aligned
to the FILE's world by the same Umeyama+guard (rigid fallback = the file's stored c2w[qf]); if
the scene has NO co-visible existing tracks to align on at qf, the pass is skipped with a log
line (an unaligned merge would mix two world frames).

Output per instance <scene>/dense_tracks_<iid>.npz with EXACTLY the STV2-version keys:
  xyz_world (N,T,3) unified world | xyz_cam (N,T,3) per-frame camera | uv_norm (N,T,2) normalized
  by ORIGINAL frame W,H | vis (N,T) bool | conf (N,T) | c2w (T,4,4) | intrs (T,3,3) original-res
  pixels | seed_frame (N,) | seed_uv_norm (N,2).

Run in env track4world:
  TMPDIR=/tmp CUDA_VISIBLE_DEVICES=<free> \
  <track4world-python> src/stage4/run_dense_frontend_t4w.py --scene-dir <scene>/
"""
from __future__ import annotations
import argparse, glob, os, pickle, sys, time
import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import torch  # noqa: E402
from src.bakeoff2.adapters.run_t4w import load_t4w, _resize_frames  # noqa: E402  (also sets T4W sys.path)
from src.geometry import umeyama_similarity  # noqa: E402

MAX_PASSES = 10
VIS_THR = 0.5
MAX_ALIGN_PAIRS = 20000
SEED_TARGET = 200      # adaptive seeding: ~this many grid points per instance (largest mask frame)
SEED_MAX = 2000        # hard cap on accumulated tracks per instance
TOPUP_MIN_VIS = 30     # --tstar-topup: top up instances with fewer visible tracks at t*


def _resize_mask(mask_hw: np.ndarray, Hm: int, Wm: int) -> np.ndarray:
    """nearest-neighbor (H,W) bool -> (Hm,Wm); per-axis, matches cv2.resize's axis scaling."""
    H, W = mask_hw.shape
    ys = np.clip((np.arange(Hm) * H / Hm).astype(int), 0, H - 1)
    xs = np.clip((np.arange(Wm) * W / Wm).astype(int), 0, W - 1)
    return mask_hw[ys][:, xs]


def _adaptive_step(area_model_px: float, seed_target: int) -> int:
    """Grid step (model px) so the largest-mask frame yields ~seed_target points."""
    return max(1, int(round(np.sqrt(max(float(area_model_px), 1.0) / float(seed_target)))))


def _grid_candidates(mask_qf: np.ndarray, step_xy) -> np.ndarray:
    """(K,2) [x,y] model px on a step grid inside mask_qf (Hm,Wm) bool."""
    sxp, syp = step_xy
    gy, gx = np.meshgrid(np.arange(0, mask_qf.shape[0], syp),
                         np.arange(0, mask_qf.shape[1], sxp), indexing="ij")
    grid = np.stack([gx.ravel(), gy.ravel()], 1)
    return grid[mask_qf[grid[:, 1], grid[:, 0]]].astype(np.float32)


def _pick_pass_frames(masklets_first, T, pass_every, tstars=()):
    """uniform (incl 0) ∪ instance t* reference frames (MANDATORY: Stage-⑦ run_link_dense pairs
    dense tracks to mesh verts strictly at each instance's t*, and T4W tracks only cover
    [seed, T) forward — with no pass at t*, vis@t* can be 0 and the pairing collapses; this was
    the real root cause of the 2026-07-11 "few/no pairs" SE3 failures, NOT a uv scaling bug)
    ∪ instance first-non-empty frames (greedy by #instances, filling up to
    max(MAX_PASSES, #mandatory)). Frames >= T-1 dropped (T4W needs a >=2-frame clip); a t*=T-1
    is pulled to T-2 (that pass still covers frame T-1)."""
    chosen = {t for t in range(0, T, pass_every) if t <= T - 2}
    chosen.add(0)
    for ts in tstars:
        chosen.add(min(max(int(ts), 0), T - 2))
    cap = max(MAX_PASSES, len(chosen))
    cnt = {}
    for f0 in masklets_first.values():
        if f0 is not None and f0 <= T - 2:
            cnt[f0] = cnt.get(f0, 0) + 1
    extras = sorted((f for f in cnt if f not in chosen), key=lambda f: (-cnt[f], f))
    for f in extras:
        if len(chosen) >= cap:
            break
        chosen.add(f)
    return sorted(chosen)


def _t4w_pass(model, rgbs, qf, iters=4):
    """One shared pass: infer on rgbs[:, qf:]. Returns GPU tensors
    flow2d (Tc,2,Hm,Wm) abs model px | flow3d (Tc,Hm,Wm,3) per-frame CAMERA coords |
    conf (Tc,Hm,Wm) visconf product | poses (Tc,4,4) c2w | K_norm (3,3) normalized intrinsics.
    Includes the Tc<=2 time-dim-collapse defense and the visconf-on-CPU fix."""
    T = rgbs.shape[1]
    Hm, Wm = rgbs.shape[-2:]
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        output, _ = model.infer(rgbs[:, qf:], iters=iters, sw=None, is_training=False,
                                tracking3d=True, force_projection=True,
                                use_da3_focal=False, eval_dict=None)
    flow2d = output[1]["flow_2d"][0].float().cuda()
    flow3d = output[1]["flow_3d"][0].float().cuda()
    vconf = output[1]["visconf_maps_e"][0].float().cuda()   # may arrive on CPU
    poses = output[0]["camera_poses"].float().cuda()
    K_norm = output[0]["intrinsics"].float().cpu().numpy().reshape(-1, 3, 3)[0]
    pts0 = output[0]["points"][0, 0].float().cuda()
    del output
    torch.cuda.empty_cache()
    Tc = T - qf
    if vconf.dim() == 3:            # T<=2 branch of forward_sliding collapses the time dim
        vconf = vconf[None]
    if flow2d.shape[0] == Tc - 1:   # prepend the identity first-frame entry
        gy, gx = torch.meshgrid(torch.arange(Hm, device=flow2d.device),
                                torch.arange(Wm, device=flow2d.device), indexing="ij")
        grid = torch.stack([gx, gy], 0)[None].float()
        flow2d = torch.cat([grid, flow2d], 0)
        flow3d = torch.cat([pts0[None], flow3d], 0)
        vconf = torch.cat([torch.ones_like(vconf[:1]), vconf], 0)
    if poses.dim() == 2:
        poses = poses[None].expand(Tc, -1, -1).contiguous()
    conf = vconf[:, 0] * vconf[:, 1]
    return flow2d, flow3d, conf, poses, K_norm


def _gather(flow2d, flow3d, conf, trans, xi, yi):
    """Gather K pixel trajectories from pass maps. xi,yi (K,) long on cuda.
    Returns numpy uv (Tc,K,2) model px, cam (Tc,K,3), own-world (Tc,K,3), conf (Tc,K)."""
    tr2 = flow2d[:, :, yi, xi].permute(0, 2, 1)
    tr3 = flow3d[:, yi, xi, :]
    cf = conf[:, yi, xi]
    ones = torch.ones_like(tr3[..., :1])
    tr3w = torch.einsum("tij,tkj->tki", trans, torch.cat([tr3, ones], -1))[..., :3]
    return (tr2.cpu().numpy(), tr3.cpu().numpy(), tr3w.cpu().numpy(), cf.cpu().numpy())


def _gauge_pairs(states, flow2d, flow3d, conf, trans, qf, Wm, Hm):
    """Umeyama pair collection for a pass at qf: for every EXISTING track visible at qf (any
    instance), read the new pass's own-world trajectory at that track's qf model pixel and pair
    the co-visible frames over the overlap [qf,T). Returns src (own-world), dst (existing world).
    2026-07-29 fix: pixel lookup is uv_norm*[Wm,Hm] (model px); the old *[sx,sy] collapsed every
    lookup to the top-left corner, so Umeyama effectively never engaged."""
    src, dst = [], []
    for s in states:
        aliv = np.where(s["vis"][:, qf])[0]
        if len(aliv) == 0:
            continue
        exm = s["uv"][aliv, qf] * np.array([Wm, Hm], np.float32)         # model px
        xi = torch.from_numpy(np.clip(np.round(exm[:, 0]), 0, Wm - 1)).long().cuda()
        yi = torch.from_numpy(np.clip(np.round(exm[:, 1]), 0, Hm - 1)).long().cuda()
        _u, _c, w_own, cf = _gather(flow2d, flow3d, conf, trans, xi, yi)
        both = (cf > VIS_THR) & s["vis"][aliv, qf:].T                    # (Tc,K)
        tt, kk = np.nonzero(both)
        src.append(w_own[tt, kk]); dst.append(s["wld"][aliv[kk], qf + tt])
    src = np.concatenate(src) if src else np.zeros((0, 3), np.float32)
    dst = np.concatenate(dst) if dst else np.zeros((0, 3), np.float32)
    return src, dst


def _fit_gauge(src, dst, fallback):
    """Guarded Umeyama src->dst (scale in [0.7,1.4] AND overlap median error must drop), else the
    rigid fallback (s,R,t). Returns (s, R, t, note)."""
    sR, RR, tR = fallback
    note = "c2w-fallback"
    if len(src) > MAX_ALIGN_PAIRS:
        sel = np.linspace(0, len(src) - 1, MAX_ALIGN_PAIRS).astype(int)
        src, dst = src[sel], dst[sel]
    if len(src) >= 8:
        sc, Rg, tg = umeyama_similarity(src, dst)
        e_b = float(np.median(np.linalg.norm(src - dst, axis=1)))
        e_a = float(np.median(np.linalg.norm(sc * (src @ Rg.T) + tg - dst, axis=1)))
        if 0.7 <= sc <= 1.4 and e_a < e_b:
            sR, RR, tR = sc, Rg, tg
            note = f"umeyama(n={len(src)},s={sc:.3f},e{e_b:.3f}->{e_a:.3f})"
    return sR, RR, tR, note


def _new_tracks(flow2d, flow3d, conf, trans, pts, qf, T, Wm, Hm, gauge):
    """Seed len(pts) NEW tracks at pass qf: gather + apply the pass->world gauge. Returns the
    per-instance state increment dict (uv/cam/wld/vis/conf/sf/suv), full [0,T) arrays."""
    sR, RR, tR = gauge
    xi = torch.from_numpy(np.clip(np.round(pts[:, 0]), 0, Wm - 1)).long().cuda()
    yi = torch.from_numpy(np.clip(np.round(pts[:, 1]), 0, Hm - 1)).long().cuda()
    uv_m, cam3, w_own, cf = _gather(flow2d, flow3d, conf, trans, xi, yi)
    w_glob = sR * (w_own @ RR.T) + tR
    K = len(pts)
    uvF = np.zeros((K, T, 2), np.float32); camF = np.zeros((K, T, 3), np.float32)
    wldF = np.zeros((K, T, 3), np.float32); visF = np.zeros((K, T), bool)
    cfF = np.zeros((K, T), np.float32)
    uvF[:, qf:] = (uv_m / np.array([Wm, Hm], np.float32)).transpose(1, 0, 2)  # per-axis == orig-norm
    camF[:, qf:] = cam3.transpose(1, 0, 2)
    wldF[:, qf:] = w_glob.transpose(1, 0, 2).astype(np.float32)
    visF[:, qf:] = (cf > VIS_THR).T
    visF[:, qf] = True
    cfF[:, qf:] = cf.T
    return dict(uv=uvF, cam=camF, wld=wldF, vis=visF, conf=cfF,
                sf=np.full(K, qf, np.int64), suv=pts / np.array([Wm, Hm], np.float32))


def _append_state(s, inc):
    for k in ("uv", "cam", "wld", "vis", "conf", "sf", "suv"):
        s[k] = np.concatenate([s[k], inc[k]])


def _dedup_and_cap(pts, s, qf, Wm, Hm, dup_radius, seed_max):
    """Drop candidates within dup_radius model px of an existing track visible at qf; cap so the
    instance's accumulated track count stays <= seed_max."""
    if s is not None and len(pts):
        aliv = s["vis"][:, qf]
        if aliv.any():
            ex = s["uv"][aliv, qf] * np.array([Wm, Hm], np.float32)      # existing @ model px
            d = np.linalg.norm(pts[:, None] - ex[None], axis=-1).min(1)
            pts = pts[d >= dup_radius]
    room = seed_max - (s["vis"].shape[0] if s is not None else 0)
    if room <= 0:
        return pts[:0]
    if len(pts) > room:
        pts = pts[np.linspace(0, len(pts) - 1, room).astype(int)]
    return pts


def _steps_for(mm, stride, sx, sy, seed_target):
    """Per-instance grid step (model px). stride given -> legacy fixed ORIGINAL-px grid; else
    adaptive from the largest single-frame mask area (model res)."""
    if stride:
        return max(1, round(stride * sx)), max(1, round(stride * sy))
    st = _adaptive_step(int(mm.sum(axis=(1, 2)).max()), seed_target)
    return st, st


def process_scene(scene_dir, model, stride, pass_every, image_size, seed_target, seed_max):
    vip = os.path.join(scene_dir, "video_input.pkl")
    if not os.path.exists(vip):
        print(f"  [skip] {scene_dir}: no video_input.pkl", flush=True); return
    vid = pickle.load(open(vip, "rb"))
    frames = np.asarray(vid.frames)
    T, H, W = frames.shape[:3]
    fr, Hm, Wm = _resize_frames(frames, image_size)
    rgbs = torch.from_numpy(fr).permute(0, 3, 1, 2)[None].float().cuda()
    sx, sy = Wm / W, Hm / H
    pe = pass_every if pass_every else max(3, T // 8)        # ADAPTIVE default (2026-07-29)

    insts = []   # (iid, masks_model (T,Hm,Wm) bool)
    firsts, steps = {}, {}
    for ml in vid.masklets:
        masks = np.asarray(ml.masks).astype(bool)
        mm = np.stack([_resize_mask(masks[t], Hm, Wm) for t in range(T)])
        f0 = next((t for t in range(T) if mm[t].any()), None)
        insts.append((ml.instance_id, mm))
        firsts[ml.instance_id] = f0
        steps[ml.instance_id] = _steps_for(mm, stride, sx, sy, seed_target)
    # instance t* frames (Stage1 reference_frame_index) are MANDATORY pass frames when the inst
    # files already exist (re-runs); in the standard V3 orchestration Stage 1 runs later and t*
    # coverage is topped up by --tstar-topup instead.
    tstars = []
    for iid, _mm in insts:
        ipath = os.path.join(scene_dir, f"inst_{iid}.npz")
        if os.path.exists(ipath):
            try:
                tstars.append(int(np.load(ipath)["reference_frame_index"]))
            except Exception:
                pass
    passes = _pick_pass_frames(firsts, T, pe, tstars=tstars)
    print(f"  pass frames: {passes} (T={T}, model {Wm}x{Hm}, pass_every={pe}"
          f"{'(adaptive)' if not pass_every else ''}; incl {len(set(tstars))} t* frames)",
          flush=True)

    # per-instance accumulated state
    st = {iid: dict(uv=np.zeros((0, T, 2), np.float32), cam=np.zeros((0, T, 3), np.float32),
                    wld=np.zeros((0, T, 3), np.float32), vis=np.zeros((0, T), bool),
                    conf=np.zeros((0, T), np.float32), sf=np.zeros(0, np.int64),
                    suv=np.zeros((0, 2), np.float32)) for iid, _ in insts}

    P0 = None; c2w_out = None; intrs_out = None
    pass_times = []
    for qf in passes:
        # ---- collect per-instance NEW candidates first; skip infer if the pass seeds nothing ----
        cands = {}
        for iid, mm in insts:
            if not mm[qf].any():
                continue
            pts = _grid_candidates(mm[qf], steps[iid])
            pts = _dedup_and_cap(pts, st[iid], qf, Wm, Hm, max(steps[iid]), seed_max)
            if len(pts):
                cands[iid] = pts
        if not cands and qf != 0:                                        # pass0 always runs (world anchor)
            continue
        t0 = time.time()
        flow2d, flow3d, conf, poses, K_norm = _t4w_pass(model, rgbs, qf)
        trans = torch.linalg.inv(poses[0])[None] @ poses                 # camera-t -> own world (Tc,4,4)
        if qf == passes[0]:
            P0 = poses.clone()
            c2w_out = trans.cpu().numpy().astype(np.float32)             # world = pass0 frame-0 camera
            K_px = np.diag([W, H, 1.0]).astype(np.float32) @ K_norm      # normalized -> ORIGINAL px
            intrs_out = np.repeat(K_px[None], T, axis=0).astype(np.float32)
            gauge, align_note = (1.0, np.eye(3), np.zeros(3)), "identity(pass0)"
        else:
            # ---- gauge: Umeyama on co-visible existing-track points over the overlap [qf,T) ----
            src, dst = _gauge_pairs(st.values(), flow2d, flow3d, conf, trans, qf, Wm, Hm)
            M = (torch.linalg.inv(P0[0]) @ P0[qf]).cpu().numpy()         # rigid c2w fallback
            sR, RR, tR, align_note = _fit_gauge(src, dst, (1.0, M[:3, :3], M[:3, 3]))
            gauge = (sR, RR, tR)
        # ---- seed new tracks per instance from this pass ----
        n_new = 0
        for iid, pts in cands.items():
            _append_state(st[iid], _new_tracks(flow2d, flow3d, conf, trans, pts, qf, T, Wm, Hm, gauge))
            n_new += len(pts)
        dt = time.time() - t0
        pass_times.append((qf, dt))
        print(f"  pass qf={qf}: +{n_new} tracks over {len(cands)} inst | align {align_note} "
              f"| {dt:.1f}s", flush=True)
        del flow2d, flow3d, conf, poses, trans
        torch.cuda.empty_cache()

    # ---- save per instance ----
    for iid, _mm in insts:
        s = st[iid]
        if s["vis"].shape[0] == 0:
            print(f"  inst {iid}: no seedable pass frames -> skip", flush=True)
            continue
        vpf = s["vis"].sum(0)
        out = os.path.join(scene_dir, f"dense_tracks_{iid}.npz")
        np.savez_compressed(out, instance_id=iid, seed_frame=s["sf"], seed_uv_norm=s["suv"],
                            xyz_world=s["wld"], xyz_cam=s["cam"], uv_norm=s["uv"],
                            vis=s["vis"], conf=s["conf"], c2w=c2w_out, intrs=intrs_out,
                            Hn=Hm, Wn=Wm)
        print(f"  inst {iid}: N={s['vis'].shape[0]} seeded on {len(set(s['sf'].tolist()))} frames "
              f"| visible/frame min={int(vpf.min())} med={int(np.median(vpf))} max={int(vpf.max())} "
              f"| mean vis={s['vis'].mean():.2f} -> {os.path.basename(out)}", flush=True)
    print(f"  pass times: " + ", ".join(f"qf{q}={t:.1f}s" for q, t in pass_times), flush=True)


def _load_state(path):
    d = np.load(path)
    s = dict(uv=d["uv_norm"].astype(np.float32), cam=d["xyz_cam"].astype(np.float32),
             wld=d["xyz_world"].astype(np.float32), vis=d["vis"].astype(bool),
             conf=d["conf"].astype(np.float32), sf=d["seed_frame"].astype(np.int64),
             suv=d["seed_uv_norm"].astype(np.float32),
             c2w=d["c2w"].astype(np.float32), intrs=d["intrs"].astype(np.float32),
             Hn=int(d["Hn"]), Wn=int(d["Wn"]))
    return s


def topup_scene(scene_dir, model, stride, image_size, seed_target, seed_max, min_vis):
    """A' t* TOP-UP (runs AFTER Stage 1): for each instance with inst_<iid>.npz whose existing
    dense tracks have < min_vis visible tracks at its t*, run one pass at qf=t* and merge the new
    tracks into dense_tracks_<iid>.npz, aligned to the FILE's existing world."""
    vip = os.path.join(scene_dir, "video_input.pkl")
    if not os.path.exists(vip):
        print(f"  [skip] {scene_dir}: no video_input.pkl", flush=True); return
    vid = pickle.load(open(vip, "rb"))
    frames = np.asarray(vid.frames)
    T, H, W = frames.shape[:3]
    fr, Hm, Wm = _resize_frames(frames, image_size)
    sx, sy = Wm / W, Hm / H

    states, masks_by = {}, {}
    for ml in vid.masklets:
        masks_by[ml.instance_id] = np.asarray(ml.masks).astype(bool)
        p = os.path.join(scene_dir, f"dense_tracks_{ml.instance_id}.npz")
        if os.path.exists(p):
            states[ml.instance_id] = _load_state(p)
    if not states:
        print(f"  [skip] {os.path.basename(scene_dir)}: no existing dense_tracks_*.npz "
              f"(nothing to align a top-up pass to)", flush=True)
        return
    ref = next(iter(states.values()))
    if (ref["Hn"], ref["Wn"]) != (Hm, Wm):
        print(f"  [skip] {os.path.basename(scene_dir)}: model-res mismatch (existing "
              f"{ref['Wn']}x{ref['Hn']} vs current {Wm}x{Hm}) — rerun the front-end instead",
              flush=True)
        return
    c2w_file, intrs_file = ref["c2w"], ref["intrs"]

    # which instances need a t* top-up?
    needs = {}                                            # qf -> [(iid, tstar, n_before)]
    for ml in vid.masklets:
        iid = ml.instance_id
        ipath = os.path.join(scene_dir, f"inst_{iid}.npz")
        if not os.path.exists(ipath):
            continue                                      # no mesh (e.g. static-routed): no t*
        try:
            tstar = int(np.load(ipath)["reference_frame_index"])
        except Exception:
            continue
        tstar = min(max(tstar, 0), T - 1)
        s = states.get(iid)
        n_before = int(s["vis"][:, tstar].sum()) if s is not None else 0
        if n_before >= min_vis:
            continue
        qf = min(tstar, T - 2)                            # T4W needs a >=2-frame clip
        needs.setdefault(qf, []).append((iid, tstar, n_before))
    if not needs:
        print(f"  all instances already have >={min_vis} visible tracks at t* — no top-up",
              flush=True)
        return

    rgbs = torch.from_numpy(fr).permute(0, 3, 1, 2)[None].float().cuda()
    for qf in sorted(needs):
        cands = {}
        for iid, tstar, n_before in needs[qf]:
            mm_qf = _resize_mask(masks_by[iid][qf], Hm, Wm)
            if not mm_qf.any():
                print(f"  inst {iid}: empty mask at qf={qf} -> skip", flush=True)
                continue
            if stride:
                step_xy = (max(1, round(stride * sx)), max(1, round(stride * sy)))
            else:
                area_m = float(masks_by[iid].sum(axis=(1, 2)).max()) * sx * sy   # model-res approx
                stp = _adaptive_step(area_m, seed_target)
                step_xy = (stp, stp)
            pts = _grid_candidates(mm_qf, step_xy)
            pts = _dedup_and_cap(pts, states.get(iid), qf, Wm, Hm, max(step_xy), seed_max)
            if len(pts):
                cands[iid] = pts
        if not cands:
            continue
        t0 = time.time()
        flow2d, flow3d, conf, poses, _K = _t4w_pass(model, rgbs, qf)
        trans = torch.linalg.inv(poses[0])[None] @ poses
        src, dst = _gauge_pairs(states.values(), flow2d, flow3d, conf, trans, qf, Wm, Hm)
        if len(src) == 0:
            print(f"  pass qf={qf}: NO co-visible existing tracks to align on -> skipped "
                  f"(inst {sorted(cands)})", flush=True)
            del flow2d, flow3d, conf, poses, trans
            torch.cuda.empty_cache()
            continue
        M = np.asarray(c2w_file[qf], np.float64)          # file world: c2w[qf] == inv(P0[0])@P0[qf]
        sR, RR, tR, align_note = _fit_gauge(src, dst, (1.0, M[:3, :3], M[:3, 3]))
        for iid, tstar, n_before in needs[qf]:
            if iid not in cands:
                continue
            inc = _new_tracks(flow2d, flow3d, conf, trans, cands[iid], qf, T, Wm, Hm, (sR, RR, tR))
            s = states.get(iid)
            if s is None:                                 # instance had no file yet: create one
                s = dict(uv=np.zeros((0, T, 2), np.float32), cam=np.zeros((0, T, 3), np.float32),
                         wld=np.zeros((0, T, 3), np.float32), vis=np.zeros((0, T), bool),
                         conf=np.zeros((0, T), np.float32), sf=np.zeros(0, np.int64),
                         suv=np.zeros((0, 2), np.float32),
                         c2w=c2w_file, intrs=intrs_file, Hn=Hm, Wn=Wm)
                states[iid] = s
            _append_state(s, inc)
            out = os.path.join(scene_dir, f"dense_tracks_{iid}.npz")
            np.savez_compressed(out, instance_id=iid, seed_frame=s["sf"], seed_uv_norm=s["suv"],
                                xyz_world=s["wld"], xyz_cam=s["cam"], uv_norm=s["uv"],
                                vis=s["vis"], conf=s["conf"], c2w=s["c2w"], intrs=s["intrs"],
                                Hn=s["Hn"], Wn=s["Wn"])
            n_after = int(s["vis"][:, tstar].sum())
            print(f"  inst {iid}: t*={tstar} vis@t* {n_before} -> {n_after} "
                  f"(+{len(cands[iid])} tracks, N={s['vis'].shape[0]})", flush=True)
        print(f"  topup pass qf={qf}: align {align_note} | {time.time() - t0:.1f}s", flush=True)
        del flow2d, flow3d, conf, poses, trans
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--scene-dir"); g.add_argument("--batch-root")
    ap.add_argument("--stride", type=int, default=None,
                    help="FIXED grid stride in ORIGINAL-res px (legacy); default None = per-"
                         "instance adaptive step targeting --seed-target points")
    ap.add_argument("--pass-every", type=int, default=None,
                    help="uniform pass interval (frames); default None = adaptive max(3, T//8)")
    ap.add_argument("--seed-target", type=int, default=SEED_TARGET,
                    help="adaptive seeding: ~points per instance at its largest-mask frame")
    ap.add_argument("--seed-max", type=int, default=SEED_MAX,
                    help="hard cap on accumulated tracks per instance")
    ap.add_argument("--image-size", type=int, default=640, help="model max-dim (demo protocol)")
    ap.add_argument("--tstar-topup", action="store_true",
                    help="A' mode (run AFTER Stage 1): only add a t* pass for instances whose "
                         "existing dense tracks have <--topup-min-vis visible tracks at t*, "
                         "merging into the existing npz (same world)")
    ap.add_argument("--topup-min-vis", type=int, default=TOPUP_MIN_VIS,
                    help="t* visible-track threshold below which an instance is topped up")
    args = ap.parse_args()
    model = load_t4w()
    scenes = ([args.scene_dir] if args.scene_dir
              else sorted(glob.glob(os.path.join(args.batch_root, "scene_*"))))
    mode = "tstar-topup" if args.tstar_topup else "dense"
    print(f"[dense-front-t4w] {len(scenes)} scene(s), mode={mode} "
          f"stride={args.stride or 'adaptive'} pass_every={args.pass_every or 'adaptive'} "
          f"seed_target={args.seed_target} seed_max={args.seed_max} "
          f"image_size={args.image_size}", flush=True)
    for sd in scenes:
        print(f"[dense-front-t4w] {os.path.basename(sd)}", flush=True)
        if args.tstar_topup:
            topup_scene(sd, model, args.stride, args.image_size, args.seed_target,
                        args.seed_max, args.topup_min_vis)
        else:
            process_scene(sd, model, args.stride, args.pass_every, args.image_size,
                          args.seed_target, args.seed_max)
    print("[dense-front-t4w] done", flush=True)


if __name__ == "__main__":
    main()
