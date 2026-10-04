"""Run Stage 2b (conservative rigid-gate classification + per-keypoint residual motion-part
clustering) + §4 SE(3) over a batch, PERSIST per-scene results for Stage 3 pass2 / §4, and grade
against MOVi-E GT. Pure numpy (no model).

Per scene writes (consumed by Stage 3 pass2 + §4):
  stage2b.npz   keypoint-aligned (same order as tracks_d4rt.npz):
                  keypoint_id, instance_id, part_id (semantic),
                  motion_part  (INSTANCE-LOCAL int: 0 = base, 1.. = moving parts; read together
                                with instance_id), e_smooth (per-kp residual = motion saliency for
                                pass2 allocation), object_type (per-kp str), tau_rigid,
                  se3_R_<iid> / se3_t_<iid> / se3_ok_<iid>  for each RIGID instance (§4 global SE(3))
  stage2b_meta.json   per-instance {object_type, K_motion, n_high, audit_max_resid}

MOVi-E is ALL rigid -> §4 validation: the recovered rotation ANGLE frame0->t (per-frame Kabsch) is
invariant to the D4RT-vs-GT frame alignment (conjugation preserves angle), so it is directly
comparable to GT's |R_gt(t) R_gt(0)^T| angle.

  python src/stage2/run_stage2b_batch.py --batch-root outputs/m1_batch \
      --scenes-root outputs/gt_scenes --out outputs/m1_runs/stage2b_eval.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import stage2b  # noqa: E402


def _quat_wxyz_to_R(q):
    w, x, y, z = q / (np.linalg.norm(q) + 1e-12)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _load_vlm_types(sd):
    """instance_id -> object_type from Stage 0-pre, via the masklets in <scene>/video_input.pkl.
    Returns {} when there's no pickle or no VLM types (MOVi-E / dt4d / PartNet) -> gate fallback."""
    p = os.path.join(sd, "video_input.pkl")
    if not os.path.exists(p):
        return {}
    try:
        import pickle
        root = os.path.dirname(os.path.dirname(_HERE))   # .../stage0_4 (for src.stage0.schema)
        if root not in sys.path:
            sys.path.insert(0, root)
        with open(p, "rb") as f:
            vi = pickle.load(f)
        return {int(m.instance_id): m.object_type for m in vi.masklets
                if getattr(m, "object_type", None) in ("rigid", "non-rigid")}
    except Exception as e:
        print(f"  [warn] could not read VLM types from {p}: {e}")
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-root", default="outputs/m1_batch")
    ap.add_argument("--scenes-root", default="outputs/gt_scenes")
    ap.add_argument("--tau-rigid", type=float, default=0.05)   # conservative rigid-gate floor (§3.0)
    ap.add_argument("--min-kp", type=int, default=4)
    ap.add_argument("--out", default="outputs/m1_runs/stage2b_eval.json")
                    # 2026-08-15 起 PURE-2a 是【默认】(motion-part 已于 2026-08-13 退场,见 memory
                    # project_motion_part_retired)。--no-motion-groups 保留为向后兼容的 no-op,
                    # 想跑回旧的运动臂用 --motion-groups。
    ap.add_argument("--no-motion-groups", action="store_true", default=True,
                    help="PURE-2a arm (2026-08-13): motion_part is written as ALL ZEROS and "
                         "motion_part_source='2a'. ⚠️ Skipping stage2b_early.json alone is NOT "
                         "enough — stage2b_instance() below runs unconditionally and its "
                         "res['motion_part'] is the LEGACY keypoint-level residual clustering "
                         "(which fragments heavily: 100%% of dt4d/PO non-rigid instances get >=2 "
                         "parts). Without this explicit zeroing the 'pure 2a' arm would silently "
                         "be the old motion clustering. object_type resolution is UNCHANGED.")
    ap.add_argument("--motion-groups", dest="no_motion_groups", action="store_false",
                    help="退回 2026-08-13 之前的运动臂(motion-part 参与部件结构)。仅用于\n"
                         "复现旧产物/对照实验;实测 62.3%% 的非刚体实例 motion_part 恒为 0。")
    args = ap.parse_args()

    records = []
    for sd in sorted(glob.glob(os.path.join(args.batch_root, "scene_*"))):
        tp = os.path.join(sd, "tracks_d4rt.npz")
        if not os.path.exists(tp):
            continue
        name = os.path.basename(sd)
        tr = np.load(tp)
        xyz, vis = tr["xyz_ref0"], tr["visibility"].astype(bool)
        inst, pid = tr["instance_id"], tr["part_id"]
        kpid = tr["keypoint_id"] if "keypoint_id" in tr else np.arange(len(inst))
        gt_path = os.path.join(args.scenes_root, name, "scene.npz")   # MOVi-E GT (None for dt4d)
        gt_quat = (np.load(gt_path, allow_pickle=True)["inst_quaternions"]
                   if os.path.exists(gt_path) else None)
        vlm_types = _load_vlm_types(sd)               # Stage 0-pre verdict (authoritative if present)
        # Stage-2b-early sem->motion map (2026-07-30 restructure): when present, the motion
        # structure was ALREADY computed on the dense trajectories (by the SAME stage2b functions,
        # earlier in the chain — see run_stage2b_dense.py); this runner then only MATERIALIZES it
        # keypoint-aligned into stage2b.npz (contract unchanged). The keypoint-level analysis below
        # still runs UNCHANGED (e_smooth + gate audit source). Instances without an early entry
        # keep the legacy keypoint-clustering result — zero behavior change on old data.
        early = {}
        ep = os.path.join(sd, "stage2b_early.json")
        if not args.no_motion_groups and os.path.exists(ep):
            try:
                early = json.load(open(ep)).get("instances", {})
            except Exception as e:
                print(f"  [warn] unreadable {ep}: {e}")

        # per-scene keypoint-aligned outputs (Stage 3 pass2 consumes these)
        K_tot = len(inst)
        motion_part = -np.ones(K_tot, int)          # instance-local: 0 = base, 1.. = moving parts
        e_smooth = np.full(K_tot, np.nan, np.float32)
        otype = np.array(["unscored"] * K_tot, dtype=object)
        se3_store, meta = {}, []                     # whole_se3 per rigid instance; per-instance meta

        for iid in np.unique(inst):
            m = inst == iid
            if int(m.sum()) < stage2b.MIN_PART_KP:
                continue
            idx = np.nonzero(m)[0]
            P, V, PID = xyz[m], vis[m], pid[m]
            # drop frames that are all-NaN (tracker invalid)
            finite = np.isfinite(P).all(axis=2)              # (K,T)
            V = V & finite
            P = np.nan_to_num(P)
            res = stage2b.stage2b_instance(P, PID, V, tau_rigid=args.tau_rigid, min_kp=args.min_kp,
                                           vlm_type=vlm_types.get(int(iid)))
            e = early.get(str(int(iid)))
            if e is not None:                       # materialize the 2b-early result (see above)
                # OBJECT TYPE stays the LEGACY resolution (VLM authoritative > keypoint gate) —
                # 2b-early's dense gate is AUDIT-ONLY. Measured (MOVi 5-scene, all-rigid GT,
                # 2026-07-30): the dense gate flips 8/48 rigid instances to non-rigid on coherent
                # tracker-3D noise (e_p90 up to 0.25 bbox), and no tau separates that from true
                # non-rigid motion (p90 ~0.09); the rigid path must stay undisturbed. Only the
                # MOTION-PART STRUCTURE of non-rigid-typed instances comes from the dense map.
                if res["object_type"] != "rigid":
                    emap = {int(s): int(v) for s, v in e.get("map", {}).items()}
                    mp2 = np.array([emap.get(int(p), 0) for p in PID], int)
                    res["motion_part"] = mp2
                    res["K_motion"] = int(len(np.unique(mp2)))
                    res["n_moving"] = res["K_motion"] - 1
                res["motion_part_source"] = "2b_early"
                res["dense_gate_object_type"] = e.get("gate_object_type")
            if args.no_motion_groups:
                # PURE-2a arm: there is exactly ONE motion part (the whole object). The part
                # structure downstream comes from `part_id` (Stage-2a semantic), which is already
                # in this npz. object_type is deliberately NOT touched — its resolution (VLM
                # authoritative > keypoint gate) is orthogonal to the motion-part decomposition.
                res["motion_part"] = np.zeros(len(idx), int)
                res["K_motion"], res["n_moving"] = 1, 0
                res["motion_part_source"] = "2a"
            motion_part[idx] = res["motion_part"]
            e_smooth[idx] = res["e_smooth"]
            otype[idx] = res["object_type"]
            rec = {"scene": name, "instance_id": int(iid), "K_semantic": res["K_semantic"],
                   "K_motion": res["K_motion"], "object_type": res["object_type"],
                   "type_source": res["type_source"], "gate_object_type": res["gate_object_type"],
                   "type_agrees": res["type_agrees"],
                   "motion_part_source": res.get("motion_part_source", "kp_cluster"),
                   "dense_gate_object_type": res.get("dense_gate_object_type"),
                   "n_high": res["n_high"], "audit_max_resid": res["gate_audit_max_resid"]}
            if res["object_type"] == "rigid" and "whole_se3_R" in res:
                se3_store[f"se3_R_{int(iid)}"] = res["whole_se3_R"].astype(np.float32)
                se3_store[f"se3_t_{int(iid)}"] = res["whole_se3_t"].astype(np.float32)
                se3_store[f"se3_ok_{int(iid)}"] = res["whole_se3_ok"]
            # §4 rotation-angle validation vs GT (rigid only; needs MOVi-E GT quaternions)
            if res["object_type"] == "rigid" and "whole_se3_R" in res and gt_quat is not None:
                R, ok = res["whole_se3_R"], res["whole_se3_ok"]
                T = R.shape[0]
                Rg0 = _quat_wxyz_to_R(gt_quat[iid, 0])
                errs = []
                for f in range(T):
                    if not ok[f]:
                        continue
                    a_rec = stage2b._rot_angle_deg(R[f])
                    a_gt = stage2b._rot_angle_deg(_quat_wxyz_to_R(gt_quat[iid, f]) @ Rg0.T)
                    errs.append(abs(a_rec - a_gt))
                if errs:
                    rec["se3_angle_err_mean"] = float(np.mean(errs))
                    rec["se3_angle_err_max"] = float(np.max(errs))
                    rec["gt_total_rot_deg"] = float(stage2b._rot_angle_deg(
                        _quat_wxyz_to_R(gt_quat[iid, T - 1]) @ Rg0.T))
            meta.append({"instance_id": int(iid), "object_type": res["object_type"],
                         "type_source": res["type_source"], "type_agrees": res["type_agrees"],
                         "motion_part_source": res.get("motion_part_source", "kp_cluster"),
                         "K_motion": int(res["K_motion"]), "n_high": int(res["n_high"]),
                         "audit_max_resid": float(res["gate_audit_max_resid"])})
            records.append(rec)

        # persist per-scene Stage 2b result for Stage 3 pass2 / §4
        np.savez_compressed(os.path.join(sd, "stage2b.npz"),
                            keypoint_id=kpid, instance_id=inst, part_id=pid,
                            motion_part=motion_part, e_smooth=e_smooth,
                            object_type=otype.astype(str), tau_rigid=args.tau_rigid, **se3_store)
        json.dump({"scene": name, "tau_rigid": args.tau_rigid, "min_kp": args.min_kp,
                   "instances": meta}, open(os.path.join(sd, "stage2b_meta.json"), "w"), indent=2)

    # ---- aggregate ----
    n = len(records)
    rigid = [r for r in records if r["object_type"] == "rigid"]
    nonrigid = [r for r in records if r["object_type"] == "non-rigid"]
    errs = np.array([r["se3_angle_err_mean"] for r in rigid if "se3_angle_err_mean" in r])
    ksem = np.array([r["K_semantic"] for r in records])
    kmot = np.array([r["K_motion"] for r in records])
    n_vlm = sum(1 for r in records if r.get("type_source") == "vlm")
    n_disagree = sum(1 for r in records if not r.get("type_agrees", True))
    summary = {
        "n_objects": n,
        "rigid_rate": len(rigid) / n if n else 0.0,          # MOVi-E all rigid -> want ~1.0
        "n_non_rigid_falsepos": len(nonrigid),
        "n_type_from_vlm": n_vlm,                            # Stage 0-pre authoritative count
        "n_vlm_gate_disagree": n_disagree,                  # VLM said X but residual gate said other
        "frac_type_from_vlm": n_vlm / n if n else 0.0,
        "K_semantic_median": float(np.median(ksem)),
        "K_motion_median": float(np.median(kmot)),
        "K_motion_dist": {int(k): int((kmot == k).sum()) for k in np.unique(kmot)},
        "se3_angle_err_deg_median": float(np.median(errs)) if len(errs) else None,
        "se3_angle_err_deg_p90": float(np.percentile(errs, 90)) if len(errs) else None,
        "tau_rigid": args.tau_rigid, "min_kp": args.min_kp,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"summary": summary, "records": records}, open(args.out, "w"), indent=2)
    print("=== STAGE 2b + §4 EVAL (MOVi-E, all rigid) ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
