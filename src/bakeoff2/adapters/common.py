"""Bake-off v2 adapter contract.
Input:  eval_*.npz (see make_eval_sets.py). frames come either as 'frames' array or 'frame_paths'.
Output: pred_<evalstem>__<tracker>.npz {p2d (N,T,2) ORIGINAL pixels, p3d (N,T,3) tracker world
        (zeros if none, has_3d=0), vis (N,T) bool, meta json string}
Each adapter may use its BEST query strategy for the given eval points (ceiling protocol)."""
from __future__ import annotations
import json, os
import numpy as np
from PIL import Image


def load_eval(path):
    d = np.load(path, allow_pickle=True)
    if "frames" in d.files:
        frames = d["frames"]
    else:
        frames = np.stack([np.asarray(Image.open(p).convert("RGB")) for p in d["frame_paths"]])
    out = {k: d[k] for k in d.files if k not in ("frames",)}
    out["frames"] = frames.astype(np.uint8)
    return out


def save_pred(out_dir, eval_path, tracker, p2d, p3d, vis, has_3d, meta=None):
    stem = os.path.splitext(os.path.basename(eval_path))[0]
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, f"pred_{stem}__{tracker}.npz"),
                        p2d=np.asarray(p2d, np.float32), p3d=np.asarray(p3d, np.float32),
                        vis=np.asarray(vis, bool), has_3d=int(has_3d),
                        meta=json.dumps(meta or {}))
    print(f"[{tracker}] {stem}: pred saved (N={len(p2d)}, T={p2d.shape[1]})", flush=True)
