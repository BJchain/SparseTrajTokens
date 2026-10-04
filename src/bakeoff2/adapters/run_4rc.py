"""Bake-off v2 adapter: 4RC (env 4RC).
Native protocol: the model ingests <=30 frames (CLI hard-downsamples, inference.py:76-78). If T>30
we select 30 frames covering ALL distinct first_vis frames, rest evenly spaced; each eval point's
first_vis is mapped to the nearest selected frame. Query strategy: 4RC only supports frame-level
DENSE-GRID queries (track_query_idx_list, arc.py:244) -> query frames = set of mapped first_vis
frames; each point's WORLD-3D trajectory is gathered from ITS query frame's dense track map at its
query pixel. p2d = that world track projected with the model's OWN predicted per-frame
intrinsic/extrinsic(c2w) -> monocular self-contained. No visibility head: vis = conf_track > thr.
NOTE frames are fed at multiple-of-14 resolution via direct resize (NO crop): load_images with
square inputs would crop 512x512 -> 504x378 and lose top/bottom eval points.
  <4RC-python> src/bakeoff2/adapters/run_4rc.py --eval-glob 'outputs/bakeoff2_eval/eval_*.npz'
"""
from __future__ import annotations
import argparse, glob, os, sys, time
import numpy as np
import torch
from PIL import Image

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_4RC = os.path.join(_ROOT, "third_party", "4RC")
for p in (_ROOT, _4RC):
    sys.path.insert(0, p)
from src.bakeoff2.adapters.common import load_eval, save_pred  # noqa: E402

NATIVE_T = 30  # 4RC native protocol: at most 30 frames per forward


def select_frames(T, first_vis, k=NATIVE_T):
    """Pick <=k frame indices: all distinct first_vis frames first, rest evenly spaced."""
    if T <= k:
        return np.arange(T)
    fv = np.unique(np.asarray(first_vis, int))
    if len(fv) > k:  # degenerate: more distinct query frames than budget
        fv = fv[np.linspace(0, len(fv) - 1, k, dtype=int)]
    sel = set(int(f) for f in fv)
    for i in np.linspace(0, T - 1, k, dtype=int):
        if len(sel) >= k:
            break
        sel.add(int(i))
    i = 0
    while len(sel) < k and i < T:  # top up if linspace collided
        sel.add(i)
        i += 1
    return np.array(sorted(sel))


def build_views(frames_sel, long_edge=512, patch=14):
    """(n,H,W,3) uint8 -> dust3r-style views, resized (not cropped) to multiples of 14."""
    from arc.dust3r.utils.image import ImgNorm
    H, W = frames_sel.shape[1:3]
    s = long_edge / max(H, W)
    Wp = max(patch, int(round(W * s / patch)) * patch)
    Hp = max(patch, int(round(H * s / patch)) * patch)
    views = []
    for i, f in enumerate(frames_sel):
        img = Image.fromarray(f).resize((Wp, Hp), Image.LANCZOS)
        views.append(dict(img=ImgNorm(img)[None], true_shape=np.int32([[Hp, Wp]]),
                          idx=i, instance=str(i)))
    return views, Hp, Wp


def run_one(model, device, ev, conf_thr):
    from arc.dust3r.inference_multiview import inference
    frames = ev["frames"]
    T, H, W = frames.shape[:3]
    first_vis = ev["first_vis"].astype(int)
    qxy = ev["query_xy"].astype(np.float32)
    N = len(qxy)

    sel = select_frames(T, first_vis)  # (n,) original frame indices fed to the model
    n = len(sel)
    # map each point's first_vis to nearest selected frame (local index in the 30f window)
    fv_local = np.abs(sel[None, :] - first_vis[:, None]).argmin(axis=1)  # (N,)
    qframes = sorted(set(fv_local.tolist()))  # dense-grid query frames (local)
    q_of = {q: i for i, q in enumerate(qframes)}
    qi = np.array([q_of[q] for q in fv_local])  # (N,) slot in track_multi's query dim

    views, Hp, Wp = build_views(frames[sel])
    qt = torch.tensor(qframes)
    for v in views:
        v["track_query_idx"] = qt

    t0 = time.time()
    out = inference(views, model, device, dtype="bf16-mixed", verbose=False,
                    use_center_as_anchor=False)
    dt = time.time() - t0
    preds = out["preds"]  # n per-frame dicts, CPU tensors

    sx, sy = Wp / W, Hp / H
    xp = np.clip(np.round(qxy[:, 0] * sx).astype(int), 0, Wp - 1)
    yp = np.clip(np.round(qxy[:, 1] * sy).astype(int), 0, Hp - 1)

    p3d = np.zeros((N, T, 3), np.float32)
    p2d = np.zeros((N, T, 2), np.float32)
    vis = np.zeros((N, T), bool)
    confs = np.zeros((N, n), np.float32)
    for j in range(n):
        t = int(sel[j])
        tm = preds[j]["track_multi"][0].float().numpy()       # (Q,Hp,Wp,3) world pos at frame j
        cm = preds[j]["conf_track_multi"][0].float().numpy()  # (Q,Hp,Wp)
        pts_w = tm[qi, yp, xp]                                # (N,3)
        confs[:, j] = cm[qi, yp, xp]
        p3d[:, t] = pts_w
        # project with the model's own predicted camera (monocular self-contained)
        c2w = preds[j]["extrinsic"].float().numpy()           # (4,4)
        K = preds[j]["intrinsic"].float().numpy()             # (3,3) @ processing res
        w2c = np.linalg.inv(c2w)
        Xc = pts_w @ w2c[:3, :3].T + w2c[:3, 3]
        z = np.where(np.abs(Xc[:, 2:3]) < 1e-9, 1e-9, Xc[:, 2:3])
        uv = (Xc @ K.T)[:, :2] / z
        p2d[:, t] = uv / np.array([sx, sy], np.float32)       # back to ORIGINAL pixels
    vis[:, sel] = confs > conf_thr

    pct = {f"p{q}": round(float(np.percentile(confs, q)), 3) for q in (5, 25, 50, 75, 95)}
    print(f"    conf pct {pct} | vis-rate {vis[:, sel].mean():.3f} | "
          f"{n}f/{len(qframes)}q | infer {dt:.1f}s", flush=True)
    meta = {"frames_idx": [int(t) for t in sel], "native_30f": True,
            "vis_source": f"conf>{conf_thr}", "conf_thr": conf_thr, "conf_pct": pct,
            "proc_res": [int(Hp), int(Wp)], "query_frames_local": qframes,
            "query": "per-point first_vis mapped to nearest selected frame; dense-grid gather",
            "infer_s": round(dt, 1)}
    return p2d, p3d, vis, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-glob", required=True)
    ap.add_argument("--out", default="outputs/bakeoff2_pred")
    ap.add_argument("--ckpt", default=os.path.join(_4RC, "checkpoints", "4RC"))
    # expp1 conf; calibrated on smoke scene 0: occluded pop concentrates <~10 (p25=8.9) vs
    # visible p25=139 -> knee at ~10 (P0.90/R0.96 vs gt_vis there)
    ap.add_argument("--conf-thr", type=float, default=10.0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from arc.models.arc import Arc
    model = Arc.from_pretrained(args.ckpt).to(device).eval()
    for ep in sorted(glob.glob(args.eval_glob)):
        ev = load_eval(ep)
        try:
            p2d, p3d, vis, meta = run_one(model, device, ev, args.conf_thr)
        except Exception as ex:
            print(f"[4rc] {os.path.basename(ep)}: FAILED {type(ex).__name__}: {str(ex)[:120]}",
                  flush=True)
            continue
        save_pred(args.out, ep, "4rc", p2d, p3d, vis, has_3d=1, meta=meta)
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
