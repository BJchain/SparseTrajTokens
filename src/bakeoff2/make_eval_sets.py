"""Bake-off v2 — unified EVAL SETS (protocol locked with user 2026-07-10).

One eval npz per scene: the SAME physical points are scored for every tracker; each tracker may
use its best-capability querying to produce trajectories FOR these points (ceiling comparison).

  eval_<scene>.npz:
    frame_paths (T,) str | or frames (T,H,W,3) uint8 (MOVi, frames live inside scene.npz)
    gt_2d (N,T,2) px | gt_3d (N,T,3) world | gt_vis (N,T) bool     [PO/MOVi only]
    first_vis (N,)  each point's first visible frame (the natural query frame)
    query_xy (N,2)  pixel at first_vis
    K (T,3,3), E_w2c (T,4,4)                                       [PO/MOVi]
    masks (T,H,W) uint16 instance-id map [DAVIS: for on-mask proxy; PO/MOVi: aux]
    has_gt: 1/0    dataset: str

Point selection: N_PER=256/scene; stratified so >=25% of points have an ABSENCE run >=GAP(20)
frames inside the window followed by reappearance (drives the relock@20 metric). PO windows use
the SAME stride4x96 centered window as the pipeline adapter (comparable to our long-window work).

Envs: any python with numpy+PIL.
  python src/bakeoff2/make_eval_sets.py --out outputs/bakeoff2_eval
"""
from __future__ import annotations
import argparse, glob, json, os, pickle, sys
import numpy as np
from PIL import Image

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)
from src.stage0.po_to_videoinput import _window, _frame_ids  # noqa: E402

PO_SCENES = ["girl_egocentric", "seminar_h52_ego2", "cab_h_bench_ego2",
             "ani7_new_f", "character5_f", "animal1_"]
MOVI_SCENES = ["scene_00000", "scene_00010", "scene_00050", "scene_00100", "scene_00200"]
DAVIS_SCENES = ["cows", "camel", "breakdance", "dog", "horsejump-high"]
N_PER, GAP = 256, 20


def _absence_mask(vis):                                  # (N,T) -> (N,) has >=GAP absence then reappear
    out = np.zeros(len(vis), bool)
    for i, v in enumerate(vis):
        w = np.nonzero(v)[0]
        if len(w) < 2:
            continue
        gaps = np.diff(w)
        out[i] = bool((gaps > GAP).any())
    return out


def _pick(vis, n=N_PER):
    """Stratified pick: >=25% absence-points if available, rest by longest visibility."""
    ab = _absence_mask(vis)
    idx_ab = np.nonzero(ab)[0]
    n_ab = min(len(idx_ab), n // 4) if len(idx_ab) else 0
    sel_ab = idx_ab[np.linspace(0, len(idx_ab) - 1, max(n_ab, 1)).astype(int)][:n_ab] if n_ab else np.array([], int)
    rest = np.setdiff1d(np.argsort(-vis.sum(1)), sel_ab)[: n - n_ab]
    return np.concatenate([sel_ab, rest]).astype(int)


def build_po(out_dir):
    for s in PO_SCENES:
        sd = glob.glob(f"{_ROOT}/data/pointodyssey/*/{s}")[0]
        anno = np.load(os.path.join(sd, "anno.npz"))
        ids = [i for i in _frame_ids(sd) if i < anno["intrinsics"].shape[0]]
        sel = _window(ids, 96, 4)                        # same window as pipeline s4x96
        has3d = anno["trajs_3d"].ndim == 3
        g2 = anno["trajs_2d"][sel]
        g3 = anno["trajs_3d"][sel] if has3d else np.zeros(g2.shape[:2] + (3,), np.float32)
        gv = (anno["valids"][sel] & anno["visibs"][sel])
        vis = gv.T                                        # (N,T)
        alive = vis.any(1) & np.isfinite(g2.transpose(1, 0, 2)).all((1, 2))
        cand = np.nonzero(alive)[0]
        pick = cand[_pick(vis[cand])]
        first = vis[pick].argmax(1)
        q_xy = g2.transpose(1, 0, 2)[pick, first]
        np.savez_compressed(os.path.join(out_dir, f"eval_po_{s}.npz"),
            dataset="po", has_gt=1,
            frame_paths=np.array([os.path.join(sd, "rgbs", f"rgb_{t:05d}.jpg") for t in sel]),
            gt_2d=g2.transpose(1, 0, 2)[pick].astype(np.float32),
            gt_3d=g3.transpose(1, 0, 2)[pick].astype(np.float32),
            gt_vis=vis[pick], first_vis=first.astype(np.int64), query_xy=q_xy.astype(np.float32),
            K=anno["intrinsics"][sel].astype(np.float64), E_w2c=anno["extrinsics"][sel].astype(np.float64),
            has_3d=int(has3d))
        n_ab = int(_absence_mask(vis[pick]).sum())
        print(f"[po {s}] N={len(pick)} absence-pts={n_ab} T={len(sel)}", flush=True)


def build_movi(out_dir):
    from numpy.linalg import inv
    def q2R(q):
        w, x, y, z = q / np.linalg.norm(q)
        return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                         [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                         [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])
    for s in MOVI_SCENES:
        d = np.load(f"{_ROOT}/outputs/movi_scenes/{s}/scene.npz", allow_pickle=True)
        rgb, seg, dep = d["rgb"], d["segmentation"], d["depth_gt"]
        K, Ew2c = d["cam_intrinsics"], np.stack([inv(e) for e in inv(d["cam_extrinsics"])])  # keep w2c as stored? verify below
        Ew2c = d["cam_extrinsics"]                        # movie_loader stores OpenCV world->cam (pinned in memory)
        Ec2w = np.stack([inv(e) for e in Ew2c])
        T, H, W = seg.shape
        pos, quat = d["inst_positions"], d["inst_quaternions"]
        P2, P3, VIS = [], [], []
        for inst in range(pos.shape[0]):
            m0s = [t for t in range(T) if (seg[t] == inst + 1).sum() > 64]
            if not m0s:
                continue
            t0 = m0s[0]
            ys, xs = np.nonzero(seg[t0] == inst + 1)
            k = min(48, len(ys)); ss = np.linspace(0, len(ys) - 1, k).astype(int)
            ys, xs = ys[ss], xs[ss]
            z = dep[t0, ys, xs]
            pc = np.stack([(xs - K[t0][0, 2]) * z / K[t0][0, 0], (ys - K[t0][1, 2]) * z / K[t0][1, 1], z], 1)
            pw0 = (np.c_[pc, np.ones(len(pc))] @ Ec2w[t0].T)[:, :3]
            R0, p0 = q2R(quat[inst, t0]), pos[inst, t0]
            p_obj = (pw0 - p0) @ R0                       # object frame
            for t in range(T):
                Rt, pt = q2R(quat[inst, t]), pos[inst, t]
                pw = p_obj @ Rt.T + pt
                Xc = (np.c_[pw, np.ones(len(pw))] @ Ew2c[t].T)[:, :3]
                zz = Xc[:, 2]; u = K[t][0, 0] * Xc[:, 0] / zz + K[t][0, 2]; v = K[t][1, 1] * Xc[:, 1] / zz + K[t][1, 2]
                ib = (zz > 1e-6) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
                onm = np.zeros(len(pw), bool)
                ui, vi = np.clip(u, 0, W - 1).astype(int), np.clip(v, 0, H - 1).astype(int)
                onm[ib] = seg[t, vi[ib], ui[ib]] == inst + 1
                dz = np.abs(dep[t, vi, ui] - zz) < 0.05 * zz
                if t == 0:
                    p2 = np.zeros((len(pw), T, 2), np.float32); p3 = np.zeros((len(pw), T, 3), np.float32)
                    vv = np.zeros((len(pw), T), bool)
                p2[:, t] = np.stack([u, v], 1); p3[:, t] = pw; vv[:, t] = ib & onm & dz
            P2.append(p2); P3.append(p3); VIS.append(vv)
        P2, P3, VIS = np.concatenate(P2), np.concatenate(P3), np.concatenate(VIS)
        alive = VIS.any(1)
        cand = np.nonzero(alive)[0]
        pick = cand[_pick(VIS[cand], n=min(N_PER, len(cand)))]
        first = VIS[pick].argmax(1)
        np.savez_compressed(os.path.join(out_dir, f"eval_movi_{s}.npz"),
            dataset="movi", has_gt=1, has_3d=1, frames=rgb,
            gt_2d=P2[pick], gt_3d=P3[pick], gt_vis=VIS[pick],
            first_vis=first.astype(np.int64), query_xy=P2[pick, first].astype(np.float32),
            K=K.astype(np.float64), E_w2c=Ew2c.astype(np.float64))
        print(f"[movi {s}] N={len(pick)} absence-pts={int(_absence_mask(VIS[pick]).sum())} T={T}", flush=True)


def build_davis(out_dir):
    import SCHEMA  # noqa
    for s in DAVIS_SCENES:
        vip = f"{_ROOT}/outputs/davis_batch/scene_{s}/video_input.pkl"
        vid = pickle.load(open(vip, "rb"))
        frames = np.asarray(vid.frames); T, H, W = frames.shape[:3]
        seg = np.zeros((T, H, W), np.uint16)
        pts, fv = [], []
        for ml in vid.masklets:
            m = np.asarray(ml.masks).astype(bool)
            seg[m] = ml.instance_id + 1
            f0 = int(np.nonzero(m.reshape(T, -1).any(1))[0][0])
            ys, xs = np.nonzero(m[f0])
            k = min(40, len(ys)); ss = np.linspace(0, len(ys) - 1, k).astype(int)
            for y, x in zip(ys[ss], xs[ss]):
                pts.append([x, y]); fv.append(f0)
        pts = np.array(pts, np.float32)[:N_PER]; fv = np.array(fv, np.int64)[:N_PER]
        np.savez_compressed(os.path.join(out_dir, f"eval_davis_{s}.npz"),
            dataset="davis", has_gt=0, has_3d=0, frames=frames, masks=seg,
            first_vis=fv, query_xy=pts)
        print(f"[davis {s}] N={len(pts)} T={T}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="outputs/bakeoff2_eval")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    build_po(a.out); build_movi(a.out); build_davis(a.out)
    print("[eval-sets] done ->", a.out)
