"""Bake-off v2 adapter: TAPIP3D (env tapip3d), UNIFIED MONOCULAR protocol.
Front-end: its OWN MegaSAM(+MoGe) run on RGB only (no eval K/E/GT depth ever fed).
Queries: per-point [t, xyz_world] at each eval point's first_vis frame, backprojected via its
OWN estimated depth/intrinsics/extrinsics (mirrors utils/inference_utils.get_grid_queries).
Outputs: coords (T,N,3) world -> p3d; p2d re-projected with its own intr/extr, rescaled to
ORIGINAL pixels ((res-1)/(res-1) convention, same as inference.py intrinsics rescale).

Run (from stage0_4, REQUIRED env: TMPDIR=/tmp CUDA_HOME=/usr/local/cuda-12.4):
  TMPDIR=/tmp CUDA_HOME=/usr/local/cuda-12.4 CUDA_VISIBLE_DEVICES=6 \
    <tapip3d-python> src/bakeoff2/adapters/run_tapip3d.py --eval-glob 'outputs/bakeoff2_eval/eval_*.npz'
"""
from __future__ import annotations
import argparse, glob, logging, os, sys, time
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_TAPIP = os.path.join(_ROOT, "third_party", "TAPIP3D")
for p in (_ROOT, _TAPIP):
    sys.path.insert(0, p)
from src.bakeoff2.adapters.common import load_eval, save_pred  # noqa: E402

os.environ.setdefault("TMPDIR", "/tmp")
import torch  # noqa: E402
import cv2  # noqa: E402
from annotation.megasam import MegaSAMAnnotator  # noqa: E402
from datasets.data_ops import _filter_one_depth  # noqa: E402
from utils.inference_utils import load_model, inference, resize_depth_bilinear  # noqa: E402

logging.basicConfig(level=logging.INFO)
CKPT = os.path.join(_TAPIP, "checkpoints", "tapip3d_final.pth")
MEGASAM_SCRIPT = os.path.join(_TAPIP, "third_party", "megasam", "inference.py")


def _sample_depth(depth_t: np.ndarray, x: float, y: float, max_r: int = 12) -> float:
    """Depth at (x,y); if invalid (0), nearest valid depth in a growing window."""
    H, W = depth_t.shape
    j, i = int(round(x)), int(round(y))
    j = min(max(j, 0), W - 1); i = min(max(i, 0), H - 1)
    if depth_t[i, j] > 0:
        return float(depth_t[i, j])
    for r in range(1, max_r + 1):
        win = depth_t[max(i - r, 0):i + r + 1, max(j - r, 0):j + r + 1]
        vals = win[win > 0]
        if len(vals):
            return float(np.median(vals))
    return float(depth_t[depth_t > 0].mean()) if (depth_t > 0).any() else 1.0


def _build_queries(first_vis, query_xy_inf, depths, intrinsics, extrinsics):
    """[t, xyz_world] per point, backprojecting via TAPIP3D's own geom (numpy)."""
    N = len(first_vis)
    q = np.zeros((N, 4), np.float32)
    for n in range(N):
        t = int(first_vis[n]); x, y = float(query_xy_inf[n, 0]), float(query_xy_inf[n, 1])
        d = _sample_depth(depths[t], x, y)
        ray = np.linalg.inv(intrinsics[t]) @ np.array([x, y, 1.0])
        cam = ray * d
        world = (np.linalg.inv(extrinsics[t]) @ np.array([*cam, 1.0]))[:3]
        q[n] = [t, *world]
    return q


def _project(coords_w, intrinsics, extrinsics):
    """coords_w (T,N,3) world -> pixels (T,N,2) at inference res (own intr/extr)."""
    T = len(coords_w)
    out = np.zeros(coords_w.shape[:2] + (2,), np.float32)
    for t in range(T):
        homo = np.concatenate([coords_w[t], np.ones_like(coords_w[t][:, :1])], axis=1)
        cam = (extrinsics[t] @ homo.T).T[:, :3]
        pix = (intrinsics[t] @ cam.T).T
        out[t] = pix[:, :2] / np.clip(pix[:, 2:3], 1e-6, None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-glob", required=True)
    ap.add_argument("--out", default="outputs/bakeoff2_pred")
    ap.add_argument("--resolution-factor", type=int, default=2)
    ap.add_argument("--depth-model", default="moge")
    ap.add_argument("--num-iters", type=int, default=6)
    ap.add_argument("--support-grid-size", type=int, default=16)
    args = ap.parse_args()
    device = "cuda"

    model = load_model(CKPT)
    model.to(device)
    inf_res = (int(model.image_size[0] * np.sqrt(args.resolution_factor)),
               int(model.image_size[1] * np.sqrt(args.resolution_factor)))
    model.set_image_size(inf_res)
    print(f"[tapip3d] model loaded, inference_res={inf_res}", flush=True)

    for ep in sorted(glob.glob(args.eval_glob)):
        t0 = time.time()
        ev = load_eval(ep)
        frames = ev["frames"]; T, H, W = frames.shape[:3]

        # 1) monocular front-end: MegaSAM(+depth model) on RGB ONLY (fairness rule)
        megasam = MegaSAMAnnotator(script_path=MEGASAM_SCRIPT, depth_model=args.depth_model,
                                   resolution=inf_res[0] * inf_res[1])
        try:
            depths, intrinsics, extrinsics = megasam.process_video(
                frames, gt_intrinsics=None, return_raw_depths=True)
        except Exception as ex:
            print(f"[tapip3d] {os.path.basename(ep)}: MegaSAM FAILED {type(ex).__name__}: {str(ex)[:120]}", flush=True)
            continue
        t_mega = time.time() - t0

        # 2) rescale intrinsics + resize video/depth to inference res (mirrors inference.py)
        intrinsics = intrinsics.copy()
        intrinsics[:, 0, :] *= (inf_res[1] - 1) / (W - 1)
        intrinsics[:, 1, :] *= (inf_res[0] - 1) / (H - 1)
        video_r = np.stack([cv2.resize(f, (inf_res[1], inf_res[0]), interpolation=cv2.INTER_LINEAR)
                            for f in frames])
        depths_r = np.stack([resize_depth_bilinear(d, (inf_res[1], inf_res[0])) for d in depths])
        depths_r = np.stack([_filter_one_depth(d, 0.08, 15, k) for d, k in zip(depths_r, intrinsics)])

        # 3) queries: eval pixel -> inference-res pixel -> backproject via OWN depth at first_vis
        q_xy_inf = ev["query_xy"].astype(np.float64).copy()
        q_xy_inf[:, 0] *= (inf_res[1] - 1) / (W - 1)
        q_xy_inf[:, 1] *= (inf_res[0] - 1) / (H - 1)
        queries = _build_queries(ev["first_vis"], q_xy_inf, depths_r, intrinsics, extrinsics)

        video_t = (torch.from_numpy(video_r).permute(0, 3, 1, 2).float() / 255.0).to(device)
        depths_t = torch.from_numpy(depths_r).float().to(device)
        intr_t = torch.from_numpy(intrinsics).float().to(device)
        extr_t = torch.from_numpy(extrinsics).float().to(device)
        query_t = torch.from_numpy(queries).float().to(device)

        # 4) track
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                coords, visibs = inference(model=model, video=video_t, depths=depths_t,
                                           intrinsics=intr_t, extrinsics=extr_t,
                                           query_point=query_t, num_iters=args.num_iters,
                                           grid_size=args.support_grid_size)
        except Exception as ex:
            print(f"[tapip3d] {os.path.basename(ep)}: tracker FAILED {type(ex).__name__}: {str(ex)[:120]}", flush=True)
            continue
        coords = coords.float().cpu().numpy()        # (T,N,3) world
        visibs = visibs.cpu().numpy()                # (T,N)

        # 5) p2d: project with OWN intr/extr, rescale inference-res px -> ORIGINAL px
        p2d_inf = _project(coords, intrinsics, extrinsics)
        p2d = p2d_inf.copy()
        p2d[..., 0] *= (W - 1) / (inf_res[1] - 1)
        p2d[..., 1] *= (H - 1) / (inf_res[0] - 1)

        save_pred(args.out, ep, "tapip3d",
                  p2d.transpose(1, 0, 2), coords.transpose(1, 0, 2), visibs.T, has_3d=1,
                  meta={"front_end": "megasam-monocular", "depth_model": args.depth_model,
                        "inference_res": list(inf_res), "num_iters": args.num_iters,
                        "support_grid_size": args.support_grid_size,
                        "query": "per-point first_vis, own-depth backprojection"})
        print(f"[tapip3d] {os.path.basename(ep)}: megasam {t_mega:.0f}s, total {time.time()-t0:.0f}s", flush=True)
        del video_t, depths_t, coords, visibs
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
