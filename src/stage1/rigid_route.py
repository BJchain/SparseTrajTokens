"""RIGID ROUTING decision (2026-08-13) — the "static + rigid routing" upgrade.

Extends the V3 STATIC routing (`run_stage1.py`: masklets the Stage-0-pre VLM judged
motion=='static' skip SAM 3D entirely). Here the same treatment is extended to RIGID instances:
a rigid object that passes both gates below skips Stage 1 (SAM 3D mesh) AND Stage 2a (PartSAM),
and its final representation is built at finalize straight from the dense tracks — k representative
points + a per-frame SE(3), exactly the fields the mesh path would have produced.

WHY (measured on stereo4d_v3 / stereo4d_p2):
  - rigid = 30.3% / 31.5% of the mesh-bearing instances
  - Stage 1 + Stage 2a = 96.9% of the whole chain's wall clock (2a alone 85.3%)
  - a rigid instance runs that entire mesh chain to produce a median of **4 keypoints**
  - `final_repr`'s rigid entry contains NO mesh — only `ref_positions` (those 4 points) + SE(3)
  - PartSAM's per-instance cost is ~constant (it resamples every mesh to 100k surface points),
    so the instance-count share is a fair proxy for the 85.3% component
=> ~30% of the heaviest compute buys 4 points that the dense tracks already contain.

TWO GATES (both must pass; every failure mode routes to the SAFE side = keep the mesh path):
  1. SINGLE PRESENCE SEGMENT. A multi-segment instance has structurally broken cross-segment track
     identity and is re-anchored by the V3 mesh registry — no mesh, no bridge. Measured 7.0% (v3) /
     14.7% (p2) of rigid instances. Feasible here because `presence_segments` reads ONLY the
     masklet masks, which are a Stage-0 product — the decision does not need Stage 1 or Stage ⑦.
  2. QWEN **AND** THE MESH-FREE RIGID GATE AGREE. We have no measurement of Qwen's rigid/non-rigid
     accuracy (MOVi/dt4d have no VLM; Stereo4D/DAVIS have no GT type), and routing makes that call
     structural — a wrong "rigid" yields no mesh, no parts and a single SE(3), with no mesh/parts
     left to notice the error with. So Qwen's verdict must be seconded by
     `stage2b.conservative_rigid_gate` run on the instance's RAW DENSE TRACKS (that function needs
     only P and vis — no mesh, no association). The gate is known trigger-happy (12.7% false
     non-rigid on MOVi's all-rigid GT), which is exactly the right direction for a one-sided veto:
     it costs some of the speedup and never routes a non-rigid object by mistake.

ANCHOR FRAME: the mesh path anchors the residual at t* (inst_*.npz reference_frame_index), which
does not exist yet at this point in the chain. We use the max-visibility frame instead — the same
substitute `stage2b.part_se3_traj` already makes for its own anchor a*.

Produces per scene (ADDITIVE — nothing is renamed, absent file = every consumer keeps its current
behavior bit-for-bit):
  rigid_route.json   {"instances": {iid: {routed, reason, qwen_type, gate_type, n_segments,
                     n_tracks, anchor, e_smooth_*, ...}}, "n_routed": N}

Pure numpy/scipy (env sam3, same as 2b-early).
  python src/stage1/rigid_route.py --batch-root <root> --tau-rigid 0.05 --min-kp 6
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_S2 = os.path.join(_ROOT, "src", "stage2")
if _S2 not in sys.path:
    sys.path.insert(0, _S2)
import stage2b  # noqa: E402  (REUSED AS-IS — conservative_rigid_gate needs only P and vis)
import SCHEMA  # noqa: E402,F401  (registers Masklet/VideoInput for video_input.pkl unpickle)
from src.registry import presence_segments  # noqa: E402  (SAME function ⑦ uses for segments)
from src.stage2.run_stage2b_dense import _inmask_tstar  # noqa: E402  (SAME in-mask gate as 2b-early)

MIN_TRACKS = 20        # below this the gate cannot judge -> do NOT route (safe side)
MAX_TRACKS = 8000      # safety cap (uniform subsample) for the O(N) robust fits — same as 2b-early
MAX_HOLE = 2           # presence_segments hole tolerance — SAME value ⑦ uses (run_link_dense.py)


def _route_instance(sd, iid, m, tau_rigid, min_kp, knn=8):
    """Both gates for one masklet. Returns the per-instance record (routed True/False + evidence).
    Every unusable input returns routed=False — routing is opt-in, never a fallback."""
    rec = {"qwen_type": getattr(m, "object_type", None),
           "qwen_motion": getattr(m, "motion", None), "concept": getattr(m, "concept", None)}
    if rec["qwen_motion"] == "static":
        rec.update(routed=False, reason="static (already static-routed)")
        return rec
    if rec["qwen_type"] != "rigid":
        rec.update(routed=False, reason=f"qwen says {rec['qwen_type']!r}, not rigid")
        return rec

    masks = np.asarray(m.masks).astype(bool)
    # ---- GATE 1: single presence segment (masklet-only, so available before Stage 1) ----
    segs = presence_segments(masks.any(axis=(1, 2)), max_hole=MAX_HOLE)
    rec["n_segments"] = len(segs)
    if len(segs) != 1:
        rec.update(routed=False, reason=f"{len(segs)} presence segments (mesh registry needed)")
        return rec

    dp = os.path.join(sd, f"dense_tracks_{iid}.npz")
    if not os.path.exists(dp):
        rec.update(routed=False, reason="no dense_tracks (nothing to build a representation from)")
        return rec
    d = np.load(dp)
    xyz = d["xyz_world"].astype(np.float64)                       # (N,T,3) W_track
    vis = d["vis"].astype(bool) & np.isfinite(xyz).all(2)         # (N,T)
    uv = d["uv_norm"].astype(np.float32)
    T = vis.shape[1]
    if T < 2:
        rec.update(routed=False, reason=f"T={T} (no motion to judge)")
        return rec

    # ---- anchor = max-visibility frame (t* does not exist yet; see module docstring) ----
    anchor = int(np.argmax(vis.sum(0)))
    inm = _inmask_tstar(uv[:, anchor], masks, anchor)
    keep = np.nonzero(vis[:, anchor] & inm)[0]
    rec.update(anchor=anchor, n_tracks_total=int(vis.shape[0]),
               n_vis_anchor=int(vis[:, anchor].sum()), n_tracks=int(len(keep)))
    if len(keep) < MIN_TRACKS:
        rec.update(routed=False, reason=f"only {len(keep)} in-mask tracks at anchor (<{MIN_TRACKS})")
        return rec
    if len(keep) > MAX_TRACKS:
        keep = keep[np.linspace(0, len(keep) - 1, MAX_TRACKS).astype(int)]

    # ---- GATE 2: the mesh-free conservative rigid gate must SECOND Qwen's verdict ----
    order = [anchor] + [f for f in range(T) if f != anchor]       # rest frame = anchor (frame reorder only)
    P = np.nan_to_num(xyz[keep][:, order])
    V = vis[keep][:, order]
    g = stage2b.conservative_rigid_gate(P, V, tau_rigid=tau_rigid, min_kp=min_kp, knn=knn)
    rec.update(gate_type=g["object_type"], K_motion=int(g["K_motion"]), n_high=int(g["n_high"]),
               e_smooth_median=float(g["e_smooth_median"]), e_smooth_p90=float(g["e_smooth_p90"]),
               audit_max_resid=float(g["audit_max_resid"]), bbox_diag=float(g["bbox_diag"]),
               tau_rigid=tau_rigid, min_kp=min_kp)
    if g["object_type"] != "rigid":
        rec.update(routed=False,
                   reason=f"gate disagrees with qwen (gate={g['object_type']}, "
                          f"K_motion={g['K_motion']}, n_high={g['n_high']})")
        return rec
    rec.update(routed=True, reason="qwen=rigid AND gate=rigid AND single segment")
    return rec


def process_scene(sd, tau_rigid, min_kp):
    vip = os.path.join(sd, "video_input.pkl")
    if not os.path.exists(vip):
        print(f"  [skip] {os.path.basename(sd)}: no video_input.pkl", flush=True)
        return
    with open(vip, "rb") as f:
        video = pickle.load(f)
    out = {}
    for m in video.masklets:
        iid = int(m.instance_id)
        try:
            out[str(iid)] = _route_instance(sd, iid, m, tau_rigid, min_kp)
        except Exception as e:
            import traceback
            traceback.print_exc()
            out[str(iid)] = {"routed": False, "reason": f"ERROR {type(e).__name__}: {e}"}
    n_routed = sum(1 for r in out.values() if r.get("routed"))
    n_rigid = sum(1 for r in out.values() if r.get("qwen_type") == "rigid"
                  and r.get("qwen_motion") != "static")
    json.dump({"scene": os.path.basename(sd), "tau_rigid": tau_rigid, "min_kp": min_kp,
               "n_routed": n_routed, "n_qwen_rigid": n_rigid, "instances": out},
              open(os.path.join(sd, "rigid_route.json"), "w"), indent=2)
    print(f"  {os.path.basename(sd)}: routed {n_routed}/{n_rigid} qwen-rigid "
          f"({len(out)} masklets) -> rigid_route.json", flush=True)
    return n_routed, n_rigid, len(out)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--scene-dir")
    g.add_argument("--batch-root")
    ap.add_argument("--tau-rigid", type=float, default=0.05)   # same V3-recalibrated values as 2b-early
    ap.add_argument("--min-kp", type=int, default=6)
    args = ap.parse_args()
    scenes = ([args.scene_dir] if args.scene_dir
              else sorted(glob.glob(os.path.join(args.batch_root, "scene_*"))))
    print(f"[rigid-route] {len(scenes)} scene(s) tau_rigid={args.tau_rigid} min_kp={args.min_kp}",
          flush=True)
    tot_r = tot_q = tot_m = 0
    for sd in scenes:
        try:
            res = process_scene(sd, args.tau_rigid, args.min_kp)
            if res:
                tot_r += res[0]; tot_q += res[1]; tot_m += res[2]
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[rigid-route] {os.path.basename(sd)}: FAILED {type(e).__name__}: {e}", flush=True)
    print(f"[rigid-route] DONE: routed {tot_r} / {tot_q} qwen-rigid / {tot_m} masklets "
          f"({tot_r / max(1, tot_q):.1%} of qwen-rigid passed both gates)", flush=True)


if __name__ == "__main__":
    main()
