"""Bake-off v2 adapter: D4RT (env d4rt). Ceiling strategy: native per-point (u,v,t_src) queries at
each eval point's first_vis frame; anchor-clip windowing handles T>48 internally.
  <d4rt-python> src/bakeoff2/adapters/run_d4rt.py --eval-glob '...'
"""
from __future__ import annotations
import argparse, glob, os, sys
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
# D4RT's repo has its own top-level `src` package (src.core) — do NOT import stage0_4's src
# package here or it shadows it (the V2-era SCHEMA-shim lesson). Load our helpers by PATH.
import importlib.util as _ilu
def _by_path(name, path):
    sp = _ilu.spec_from_file_location(name, path); m = _ilu.module_from_spec(sp)
    sp.loader.exec_module(m); return m
common = _by_path("bo2_common", os.path.join(_ROOT, "src", "bakeoff2", "adapters", "common.py"))
load_eval, save_pred = common.load_eval, common.save_pred
_d4 = _by_path("run_stage4_d4rt", os.path.join(_ROOT, "src", "stage4", "run_stage4_d4rt.py"))
load_d4rt, _infer_tracks = _d4.load_d4rt, _d4._infer_tracks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-glob", required=True)
    ap.add_argument("--out", default="outputs/bakeoff2_pred")
    ap.add_argument("--chunk", type=int, default=256)
    args = ap.parse_args()
    model = load_d4rt()
    for ep in sorted(glob.glob(args.eval_glob)):
        ev = load_eval(ep)
        frames = ev["frames"]; T, H, W = frames.shape[:3]
        quv = np.stack([ev["query_xy"][:, 0] / W, ev["query_xy"][:, 1] / H], 1).astype(np.float32)
        try:
            out = _infer_tracks(model=model, video_model_rgb=frames.astype(np.uint8),
                                query_uv_norm=quv, query_chunk_size=args.chunk,
                                query_src_indices_global=ev["first_vis"].astype(np.int64))
        except Exception as ex:
            print(f"[d4rt] {os.path.basename(ep)}: FAILED {type(ex).__name__}: {str(ex)[:90]}", flush=True)
            continue
        p2d = out["tracks_uv_norm"] * np.array([W, H], np.float32)
        save_pred(args.out, ep, "d4rt", p2d, out["tracks_xyz_ref0"], out["tracks_visibility"],
                  has_3d=1, meta={"query": "per-point (u,v,t_src)", "world": "frame0-cam"})


if __name__ == "__main__":
    main()
