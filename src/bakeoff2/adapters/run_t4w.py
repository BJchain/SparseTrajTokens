"""Bake-off v2 adapter: Track4World (env track4world).
Capability note: T4W has NO per-point query API — it is frame-level dense-grid tracking only.
Best-capability protocol (mirrors its official evaluation/track/eval.py): group eval points by
first_vis, run model.infer(rgbs[:, qf:], tracking3d=True) once per unique query frame, then gather
the dense flow_2d / flow_3d / visconf maps at each point's (rounded) query pixel.
3D: flow_3d is per-frame CAMERA coords; converted to one world frame (= camera of the earliest
query frame) via predicted c2w poses, exactly like the official --world_eval path
(transform_to_first_frame). Later-qf chunks are re-anchored through the base run's poses
(per-run monocular scale => minor cross-chunk inconsistency; only affects late-appearing points).
MONOCULAR ONLY: RGB in, nothing else. Non-metric (relative) scale.
  <track4world-python> src/bakeoff2/adapters/run_t4w.py --eval-glob 'outputs/bakeoff2_eval/eval_*.npz'
"""
from __future__ import annotations
import argparse, glob, json, os, sys, time
from types import SimpleNamespace

import numpy as np
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_T4W = os.path.join(_ROOT, "third_party", "Track4World")
for p in (_ROOT, _T4W):
    sys.path.insert(0, p)
from src.bakeoff2.adapters.common import load_eval, save_pred  # noqa: E402


def load_t4w(ckpt=None, config=None):
    cwd = os.getcwd()
    os.chdir(_T4W)  # demo/model use repo-relative paths
    try:
        from demo import load_model
        args = SimpleNamespace(
            ckpt_init=ckpt or os.path.join(_T4W, "checkpoints", "track4world_da3.pth"),
            coordinate="world_depthanythingv3",
            use_original_backbone=False,
            metric_scale=False,
        )
        with open(config or os.path.join(_T4W, "track4world", "config", "eval", "v1.json")) as f:
            cfg = json.load(f)
        model = load_model(args, cfg)
    finally:
        os.chdir(cwd)
    return model


def _resize_frames(frames, image_size):
    """Resize (T,H,W,3) uint8 to max-dim=image_size with H,W multiples of 64 (demo protocol)."""
    import cv2
    T, H, W = frames.shape[:3]
    s = image_size / max(H, W)
    Hm, Wm = (int(H * s) // 64) * 64, (int(W * s) // 64) * 64
    out = np.stack([cv2.resize(f, (Wm, Hm), interpolation=cv2.INTER_LINEAR) for f in frames])
    return out, Hm, Wm


def track_one(model, ev, image_size, iters, vis_thr):
    frames = ev["frames"]
    T, H, W = frames.shape[:3]
    first_vis = ev["first_vis"].astype(int)
    query_xy = ev["query_xy"].astype(np.float32)  # (N,2) ORIGINAL px
    N = len(query_xy)

    fr, Hm, Wm = _resize_frames(frames, image_size)
    rgbs = torch.from_numpy(fr).permute(0, 3, 1, 2)[None].float().cuda()  # (1,T,3,Hm,Wm)
    sx, sy = Wm / W, Hm / H

    # defaults: static at query pixel, invisible, no 3D
    p2d = np.tile(query_xy[:, None, :], (1, T, 1)).astype(np.float32)
    p3d = np.zeros((N, T, 3), np.float32)
    vis = np.zeros((N, T), bool)

    uq = sorted(np.unique(first_vis).tolist())
    base_qf, P_base = uq[0], None
    for qf in uq:
        idxs = np.where(first_vis == qf)[0]
        vis[idxs, qf] = True  # visible at own query frame by definition
        if qf >= T - 1:
            continue
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            output, _ = model.infer(rgbs[:, qf:], iters=iters, sw=None, is_training=False,
                                    tracking3d=True, force_projection=True,
                                    use_da3_focal=False, eval_dict=None)
        flow2d = output[1]["flow_2d"][0].float().cuda()        # (Tc,2,Hm,Wm) abs px @ model res
        flow3d = output[1]["flow_3d"][0].float().cuda()        # (Tc,Hm,Wm,3) camera-t coords
        vconf = output[1]["visconf_maps_e"][0].float().cuda()  # (Tc,2,Hm,Wm)
        poses = output[0]["camera_poses"].float().cuda()       # (Tc,4,4) c2w
        pts0 = output[0]["points"][0, 0].float().cuda()        # (Hm,Wm,3) clip frame-0 geometry
        del output
        torch.cuda.empty_cache()

        # Defense for T4W's T<=2 branch (forward_sliding line ~1554): a 2-frame clip returns
        # pair-flow with the time dim COLLAPSED (only the target frame). Restore the time axis
        # and prepend the identity first-frame entry (2D=pixel grid, 3D=clip frame-0 point map).
        Tc = T - qf
        if vconf.dim() == 3:
            vconf = vconf[None]
        if flow2d.shape[0] == Tc - 1:
            gy, gx = torch.meshgrid(torch.arange(Hm, device=flow2d.device),
                                    torch.arange(Wm, device=flow2d.device), indexing="ij")
            grid = torch.stack([gx, gy], 0)[None].float()          # (1,2,Hm,Wm)
            flow2d = torch.cat([grid, flow2d], 0)
            flow3d = torch.cat([pts0[None], flow3d], 0)
            vconf = torch.cat([torch.ones_like(vconf[:1]), vconf], 0)
        if poses.dim() == 2:
            poses = poses[None].expand(Tc, -1, -1).contiguous()
        if qf == base_qf:
            P_base = poses.clone()

        # nearest-grid lookup at query pixels (model res)
        xi = torch.from_numpy(np.clip(np.round(query_xy[idxs, 0] * sx), 0, Wm - 1)).long().cuda()
        yi = torch.from_numpy(np.clip(np.round(query_xy[idxs, 1] * sy), 0, Hm - 1)).long().cuda()
        tr2 = flow2d[:, :, yi, xi].permute(0, 2, 1)     # (Tc,K,2)
        tr3 = flow3d[:, yi, xi, :]                      # (Tc,K,3)
        vc = vconf[:, :, yi, xi]                        # (Tc,2,K)

        # camera-t -> world (= chunk-start camera frame), official transform_to_first_frame
        trans = torch.linalg.inv(poses[0])[None] @ poses            # (Tc,4,4)
        ones = torch.ones_like(tr3[..., :1])
        tr3w = torch.einsum("tij,tkj->tki", trans, torch.cat([tr3, ones], -1))[..., :3]
        if qf != base_qf and P_base is not None:
            # re-anchor chunk-start camera frame into the base run's world frame
            M = torch.linalg.inv(P_base[0]) @ P_base[qf - base_qf]  # (4,4)
            tr3w = torch.einsum("ij,tkj->tki", M, torch.cat([tr3w, ones], -1))[..., :3]

        p2d[idxs, qf:] = (tr2.cpu().numpy() / np.array([sx, sy], np.float32)).transpose(1, 0, 2)
        p3d[idxs, qf:] = tr3w.cpu().numpy().transpose(1, 0, 2)
        vis[idxs, qf:] = ((vc[:, 0] * vc[:, 1]) > vis_thr).cpu().numpy().T
        vis[idxs, qf] = True
    return p2d, p3d, vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-glob", required=True)
    ap.add_argument("--out", default="outputs/bakeoff2_pred")
    ap.add_argument("--image-size", type=int, default=640)
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--vis-thr", type=float, default=0.5)
    args = ap.parse_args()
    model = load_t4w()
    for ep in sorted(glob.glob(args.eval_glob)):
        ev = load_eval(ep)
        t0 = time.time()
        try:
            p2d, p3d, vis = track_one(model, ev, args.image_size, args.iters, args.vis_thr)
        except Exception as ex:
            import traceback
            tb = traceback.extract_tb(ex.__traceback__)[-1]
            print(f"[t4w] {os.path.basename(ep)}: FAILED {type(ex).__name__}: {str(ex)[:120]} "
                  f"(at {os.path.basename(tb.filename)}:{tb.lineno})", flush=True)
            torch.cuda.empty_cache()
            continue
        save_pred(args.out, ep, "t4w", p2d, p3d, vis, has_3d=1, meta={
            "coordinate": "world_depthanythingv3",
            "query": "frame-level dense grid only; grouped by first_vis, per-group infer(rgbs[qf:]), "
                     "nearest-grid gather at rounded query pixel",
            "world_frame": "camera of earliest query frame (transform_to_first_frame, official world_eval); "
                           "later chunks re-anchored via base-run c2w poses (per-run scale caveat)",
            "vis": "model visconf[0]*visconf[1] > %.2f (official eval rule); query frame forced True; "
                   "frames before first_vis: vis=False, p2d=query px static, p3d=0" % args.vis_thr,
            "scale": "relative (non-metric), use_da3_focal=False per official track eval",
            "image_size": args.image_size, "iters": args.iters,
            "runtime_s": round(time.time() - t0, 1),
        })
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
