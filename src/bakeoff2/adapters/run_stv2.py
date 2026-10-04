"""Bake-off v2 adapter: SpaTrackerV2 (env SpaTrack2).
Ceiling strategy: native per-point [t,x,y] query at each eval point's first_vis frame; one call over
the FULL window (STV2 offline handles T<=512 with internal sliding windows); BA ON.
  <SpaTrack2-python> src/bakeoff2/adapters/run_stv2.py --eval-glob 'outputs/bakeoff2_eval/eval_*.npz'
"""
from __future__ import annotations
import argparse, glob, os, sys
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
for p in (_ROOT, os.path.join(_ROOT, "third_party", "SpaTrackerV2")):
    sys.path.insert(0, p)
from src.bakeoff2.adapters.common import load_eval, save_pred  # noqa: E402
from src.stage4.run_stage4_stv2 import load_stv2  # noqa: E402
from src.stage4.run_dense_frontend import _infer_dense  # noqa: E402
from src.stage4 import run_stage4_stv2 as _m  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-glob", required=True)
    ap.add_argument("--out", default="outputs/bakeoff2_pred")
    ap.add_argument("--chunk", type=int, default=384)
    args = ap.parse_args()
    model = load_stv2()
    for ep in sorted(glob.glob(args.eval_glob)):
        ev = load_eval(ep)
        frames = ev["frames"]; T, H, W = frames.shape[:3]
        _m._GEOM_CACHE.clear()
        from src.stage4.run_dense_frontend import _vggt_geom
        _vt, _d, _i, _e, _u, Hn, Wn = _vggt_geom(model[0], frames)
        q = np.stack([ev["first_vis"].astype(np.float32),
                      ev["query_xy"][:, 0] * Wn / W,
                      ev["query_xy"][:, 1] * Hn / H], axis=1).astype(np.float32)
        try:
            d = _infer_dense(model, frames, q, args.chunk, ba=True, gt_extrs=None)
        except Exception as ex:
            print(f"[stv2] {os.path.basename(ep)}: FAILED {type(ex).__name__}: {str(ex)[:90]}", flush=True)
            continue
        p2d = d["uv_norm"] * np.array([W, H], np.float32)      # (N,T,2) orig px
        save_pred(args.out, ep, "stv2", p2d, d["xyz_world"], d["vis"], has_3d=1,
                  meta={"ba": True, "query": "per-point first_vis"})
        _m._GEOM_CACHE.clear()


if __name__ == "__main__":
    main()
