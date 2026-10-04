"""Stage 3 pass2 (v2): prune pass1 candidates to the final MLLM keypoint subset (REFERENCE §5.0 B).

Consumes the persisted Stage 4 + Stage 2b results (keypoint-aligned, keyed by keypoint_id = pass1
id, so pass2 ⊆ pass1 is automatic, B4):
    tracks_d4rt.npz : xyz_ref0 (K,T,3), visibility (K,T), keypoint_id, instance_id
    stage2b.npz     : motion_part (instance-local), e_smooth, object_type (per kp)

Per instance:
  - importance(kp) = trajectory MOTION MAGNITUDE = max_t ||p(t)-p(0)|| / bbox  (§5.6 -> B3, from the
    REAL tracker trajectory, not FK).
  - VISIBILITY-AWARE: selection first restricts to well-tracked candidates (co-visible with frame 0
    in >= VIS_FLOOR of frames), so the chosen points actually define a usable trajectory / per-frame
    SE(3) — otherwise spatially-spread-but-occluded points leave the rigid SE(3) all-identity. Relax
    to the most-visible points only if too few clear the bar.
  - rigid     : lock <= 4 keypoints, chosen (among visible) by farthest-point sampling for max
                spatial spread (best SE(3) conditioning; importance-biased seed).            (B1)
  - non-rigid : per motion part — base (label 0) anchors the dominant body, each moving part gets a
                floor + extra scaled by its motion complexity; points picked by importance-weighted
                FPS = motion priority + spatial de-redundancy.                (B1/B2/B3)
  - non-rigid, PURE-2a arm (--no-motion-groups, 2026-08-13): ONE keypoint per SEMANTIC part,
                chosen as the MEDOID of that part's well-tracked points. Why the budget shrinks:
                the non-rigid branch stores RAW per-keypoint trajectories and never solves a
                per-part SE(3) (see stage4_finalize), so the old 2-6 points/part — which existed to
                condition an SE(3) — buys nothing here. One point per part covers the whole body
                (median 7 semantic parts) at 0.72x the old token cost and with better spread than
                the status quo (whose median non-rigid instance kept only 4 points in total).

pass2 ⊆ pass1 and every pass1 keypoint already has a trajectory (tracker ran on all of pass1), so
there are NO gaps here; gap_warnings is emitted (empty) to keep the B4 contract explicit.

  - PHANTOM GATE (2026-08-16): a candidate never observed in ANY frame (vis_frac == 0) is no longer
                selectable. Such points only ever entered through the three padding fall-backs, and
                they carry an all-zero placeholder trajectory: an empty track in the token stream,
                and (via finalize's ref_positions) a fake keypoint at the WORLD ORIGIN. Measured:
                377 of 2600 selected points (14.5%). A part with nothing observable gets no
                representative; an INSTANCE with nothing observable is dropped and recorded as
                dropped="no_observable_keypoint" (all of them are stage-⑦ zero-link instances).

Output per scene:
  pass2.npz       selected_keypoint_id, instance_id, motion_part, part_id, importance,
                  object_type (per kp). `part_id` = Stage-2a SEMANTIC part (ADDITIVE, 2026-08-13);
                  it is FORCED TO 0 for rigid instances — a rigid object has no motion-meaningful
                  part decomposition, and letting it inherit PartSAM's 7-10 semantic parts would
                  push meaningless structure into the token stream.
  pass2_meta.json per-instance {object_type, n_pass1, n_pass2, n_moving_parts, n_sel_parts, n_seen,
                  bbox, bbox_degenerate, gap_warnings} — plus, for instances that produced NO row,
                  {n_pass2: 0, dropped: "few_keypoints"|"no_observable_keypoint"} so a missing
                  instance is always explained rather than silently absent (2026-08-16).
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

N_RIGID = 4            # B1: rigid locked to <= 4 (SE(3) needs only a few)
N_BASE = 4            # (legacy; superseded for non-rigid by the distributed floor below)
N_PART_FLOOR = 2      # B2: each DETECTED moving part's floor
N_PART_CAP = 6        # B2: per-part cap
# NON-RIGID DISTRIBUTED BASE FLOOR. Non-rigid deformation is often DISTRIBUTED (whole-body soft
# motion), not a discrete "moving part"; when Stage 2b detects no moving part (K_motion==1) the base
# part is the whole object, and tying its budget to N_BASE=4 under-represented it (a non-rigid object
# then got the SAME 4 points as a rigid one). So the base part now gets a floor that samples the
# surface densely enough to carry per-point trajectories, scaled mildly with the base's point count.
NONRIGID_BASE_FRAC = 0.08   # base budget = this fraction of the base part's pass1 points, ...
NONRIGID_BASE_MIN = 8       # ... floored here (>> N_RIGID, so deformation is captured even at K_motion==1) ...
NONRIGID_BASE_MAX = 16      # ... and capped here (MLLM token budget).
NONRIGID_MIN = 4            # a non-rigid object keeps >= this many points; if too few clear VIS_FLOOR,
                            # pad by visibility to reach it (legacy N_BASE floor) so it never drops out.
VIS_FLOOR = 0.5       # selected points must be co-visible-with-frame-0 in >= this frac of frames,
#                       so their trajectories / the rigid SE(3) are actually well-defined.
BBOX_MIN = 1e-4       # 米。小于此视为"这个实例没有可测的尺度"(见 instance_bbox)
# 幽灵点闸(2026-08-16):vis_frac == 0 的关键点【一帧都没被观测到】—— 它的轨迹整条是 0 占位,
# 送进 token 就是一条空轨迹(纯浪费 token),还会被 finalize 当成 ref_positions 写成世界原点的
# 假点。它们进不了 vis_frac >= VIS_FLOOR 的候选池,只能从三条兜底路径漏进来(rigid 的 pad 松弛、
# per-part 的"至少留一个"、实例级 floor_pad),实测占最终选点的 14.5%(377/2600)。
# 现在三条兜底路径都只在【有观测】的点里挑;一个 part 全无观测就不给代表点,一个实例全无观测
# 就整个不写入 pass2(reason=no_observable_keypoint,记进 pass2_meta.json 供审计)。


def instance_bbox(P, vis):
    """实例尺寸(米)= 锚点帧上【可见点】的 extent。返回 (bbox, anchor, degenerate)。

    ⚠️ 2026-08-16 修:旧口径是 `extent(P[:, 0])`,对【全部 pass1 关键点】在帧 0 的坐标取范围。
    P 里没被 ⑦ 链接上的关键点整行是 (0,0,0) 占位,而自 2026-08-08 起 xyz_ref0 落在 W_track
    (Stereo4D 是公制 GT 世界,原点 = 首帧相机),物体离原点 10-25 m —— 于是"实例尺寸"被算成
    了"物体到相机的距离"。实测 395 个动态实例:bbox 被撑大中位 2.68x / p90 7.05x;帧 0 一个
    有效点都没有时 extent=0 -> bbox=1e-9 -> importance 爆到 1e9~1e10(17-19% 的实例)。
    W_mesh 时代原点贴着物体,所以这个 bug 一直没显形。

    新口径:锚点帧 a* = 可见点最多的那一帧(与 stage2b.part_se3_traj 的锚点规则同源),只取该帧
    【可见】的点算 extent;退化则取逐帧可见 extent 的最大值;仍然退化(全片没有任何一帧有 >=2
    个可见点,实测 75/3396)则返回 degenerate=True,由调用方把 importance 整体置 0 —— 一个没有
    尺度的实例上,"位移/尺寸"没有意义,给 0 比给 1e9 诚实。"""
    a = int(np.argmax(vis.sum(0)))
    if int(vis[:, a].sum()) >= 2:
        q = P[vis[:, a], a]
        bb = float(np.linalg.norm(q.max(0) - q.min(0)))
        if bb >= BBOX_MIN:
            return bb, a, False
    ext = [float(np.linalg.norm(P[vis[:, t], t].max(0) - P[vis[:, t], t].min(0)))
           for t in range(vis.shape[1]) if int(vis[:, t].sum()) >= 2]
    bb = max(ext) if ext else 0.0
    return (bb, a, False) if bb >= BBOX_MIN else (BBOX_MIN, a, True)


def trajectory_stats(P, vis):
    """Per-keypoint motion magnitude AND visibility fraction.

    DE-ANCHORED from frame 0 (long-window fix, 2026-07-09): the old stats required co-visibility
    with FRAME 0 (vis_frac=0 for any point unseen at frame 0), which on long windows / egocentric
    sweeps deprioritized perfectly good points that appear later. Now: vis_frac = the point's OWN
    visibility fraction; importance = max displacement from its own FIRST VISIBLE frame (measured
    over its visible frames). Short windows where everything is visible at frame 0 are unchanged.

    Normalization = `instance_bbox` (2026-08-16 fix; see there). Returns
    (importance, vis_frac, bbox, degenerate)."""
    bbox, _a, degen = instance_bbox(P, vis)
    K, T = vis.shape
    first = np.where(vis.any(1), vis.argmax(1), 0)       # (K,) each point's first visible frame
    ref = P[np.arange(K), first]                          # (K,3) its own anchor position
    d = np.where(vis, np.linalg.norm(P - ref[:, None, :], axis=2), 0.0)
    imp = np.zeros(K) if degen else d.max(axis=1) / bbox
    return imp, vis.mean(axis=1), bbox, degen


def _weighted_fps(pts, weight, k):
    """Importance-biased farthest-point sampling: seed at max weight, then repeatedly add the point
    maximizing (min-dist-to-selected) * (1+weight). Gives spatial spread (de-redundancy) while
    favouring high-motion points. Returns selected LOCAL indices (<= k)."""
    n = len(pts)
    if n <= k:
        return np.arange(n)
    first = int(np.argmax(weight))
    sel = [first]
    min_d = np.linalg.norm(pts - pts[first], axis=1)
    while len(sel) < k:
        score = min_d * (1.0 + weight)
        nxt = int(np.argmax(score))
        if nxt in sel:
            break
        sel.append(nxt)
        np.minimum(min_d, np.linalg.norm(pts - pts[nxt], axis=1), out=min_d)
    return np.asarray(sel, int)


def _select(pos, imp, vis_frac, k, pad=True):
    """Pick up to k keypoints that are WELL-TRACKED (visible), spatially spread, and high-motion.
    Restrict to well-visible candidates (vis_frac >= VIS_FLOOR), then importance-weighted FPS for
    spread + motion priority.
    `pad=True` (rigid): if too few clear the bar, RELAX to the top-k by visibility (so SE(3) still
    has ~k anchors). `pad=False` (non-rigid): NEVER fill the budget with low-visibility points — cap
    at the well-tracked count. A poorly-tracked point adds no signal for the MLLM's per-point
    trajectory and only costs a token, so a non-rigid instance keeps only its reliably-tracked
    points (fewer, clean) rather than padding with noise.

    PHANTOM GATE (2026-08-16): a point with vis_frac == 0 was never observed in ANY frame, so it
    carries no trajectory at all — the relaxation path used to pad the rigid budget with such
    points. Every path below now draws only from `seen` (vis_frac > 0); an instance with nothing
    seen returns an EMPTY selection (the caller drops it, honestly, instead of emitting 4 points
    that are all the world origin)."""
    seen = np.nonzero(vis_frac > 0)[0]
    pool = np.nonzero(vis_frac >= VIS_FLOOR)[0]
    if not pad:                                         # non-rigid: well-tracked only, no padding
        if len(pool) == 0:
            return np.array([], int)
        return pool[_weighted_fps(pos[pool], imp[pool], min(k, len(pool)))]
    if len(seen) == 0:
        return np.array([], int)
    if len(seen) <= k:
        return seen
    if len(pool) < k:                                   # relax: take the most-visible SEEN points
        pool = seen[np.argsort(vis_frac[seen])[::-1][:max(k, 1)]]
    pick = _weighted_fps(pos[pool], imp[pool], k)
    return pool[pick]


def _medoid(pos, vis_frac):
    """ONE representative point for a part: the MEDOID (min total distance to the other candidates)
    among the WELL-TRACKED points.

    Deliberately NOT argmax(importance) / _weighted_fps(k=1): those seed on the highest-motion
    point, i.e. exactly the noise-sensitive extreme. With a budget of 1 the representative should be
    the part's spatial centre among points that actually carry a usable trajectory.

    PER-PART FALLBACK: when nothing clears VIS_FLOOR, keep the single most-visible point so the part
    does not vanish from the representation. At a budget of 1 this costs one token (the old 2-6
    budget is what made padding-with-noise expensive). PHANTOM GATE (2026-08-16): that fallback is
    restricted to points actually OBSERVED somewhere (vis_frac > 0) — it used to hand back an
    all-zero placeholder point for parts nothing was ever seen on. Returns -1 = this part has no
    observable point, so it gets NO representative (honest: the part is not in the data)."""
    pool = np.nonzero(vis_frac >= VIS_FLOOR)[0]
    if len(pool) == 0:
        seen = np.nonzero(vis_frac > 0)[0]
        return int(seen[np.argmax(vis_frac[seen])]) if len(seen) else -1
    if len(pool) == 1:
        return int(pool[0])
    if len(pool) > 1024:                       # guard: exact medoid is O(n^2)
        pool = pool[np.argsort(vis_frac[pool])[::-1][:1024]]
    Q = pos[pool]
    d = np.linalg.norm(Q[:, None, :] - Q[None, :, :], axis=2).sum(1)
    return int(pool[int(np.argmin(d))])


def pass2_instance(P, vis, ids, motion_part, part_id, object_type, pure_2a=False):
    """Select the final keypoint subset for one instance. Returns dict with selected ids (global),
    their importance, and the per-part budget breakdown."""
    imp, vis_frac, bbox, bbox_degen = trajectory_stats(P, vis)
    pos0 = P[:, 0]
    sel_local = []
    budget = {}
    if object_type == "rigid":
        sel_local = list(_select(pos0, imp, vis_frac, min(N_RIGID, len(ids))))
        budget["rigid"] = len(sel_local)
    elif pure_2a:
        # PURE-2a: one medoid keypoint per SEMANTIC part (see module docstring).
        for lab in np.unique(part_id):
            if lab < 0:
                continue
            idx = np.nonzero(part_id == lab)[0]
            if len(idx) == 0:
                continue
            j = _medoid(pos0[idx], vis_frac[idx])
            if j < 0:                       # 该 part 全无观测 -> 不给代表点(见 _medoid)
                continue
            sel_local.append(int(idx[j]))
            budget[int(lab)] = 1
    else:
        for lab in np.unique(motion_part):
            idx = np.nonzero(motion_part == lab)[0]
            if lab == 0:
                # non-rigid base = DISTRIBUTED FLOOR (captures whole-body deformation even when no
                # discrete moving part was detected); >> rigid's 4. _select does importance-weighted
                # FPS, so these points spread over the surface.
                nb = int(np.clip(round(NONRIGID_BASE_FRAC * len(idx)),
                                 NONRIGID_BASE_MIN, NONRIGID_BASE_MAX))
                n = min(nb, len(idx))
            else:                                       # moving part: floor + extra by complexity
                complexity = float(np.median(imp[idx]))         # how much this part moves / bbox
                n = int(np.clip(N_PART_FLOOR + round(4 * min(1.0, complexity)),
                                N_PART_FLOOR, N_PART_CAP))
                n = min(n, len(idx))
            pick = _select(pos0[idx], imp[idx], vis_frac[idx], n, pad=False)   # non-rigid: no low-vis padding
            # PER-PART MINIMUM (2026-07-30, user-approved compromise): every motion part keeps at
            # least ONE point — when nothing clears VIS_FLOOR, anchor the part with its most
            # visible point instead of leaving it trajectory-less. The instance-level NONRIGID_MIN
            # below is unchanged; this is the same visibility-ranked padding applied per part (k=1).
            # PHANTOM GATE (2026-08-16): only among points actually observed somewhere.
            if len(pick) == 0 and len(idx) > 0:
                seen_i = np.nonzero(vis_frac[idx] > 0)[0]
                if len(seen_i):
                    pick = np.array([int(seen_i[np.argmax(vis_frac[idx][seen_i])])])
            sel_local.extend(idx[pick].tolist())
            budget[int(lab)] = int(len(pick))
    if object_type != "rigid" and len(set(sel_local)) < NONRIGID_MIN:
        # instance MINIMUM (BOTH non-rigid paths): a non-rigid object keeps >= NONRIGID_MIN points.
        # If too few points cleared VIS_FLOOR (so pad=False returned <4, or the instance has fewer
        # than 4 semantic parts on the pure-2a path), fall back to the legacy N_BASE floor — pad by
        # visibility (highest first) to reach 4 — so the object still has a usable trajectory anchor
        # and never drops out of the 4D representation entirely.
        # PHANTOM GATE (2026-08-16): pad only with OBSERVED points. An instance whose points were
        # never seen at all pads with nothing and ends up empty -> process_scene drops it.
        have = set(sel_local)
        seen = np.nonzero(vis_frac > 0)[0]
        for j in seen[np.argsort(vis_frac[seen])[::-1]]:
            if int(j) not in have:
                have.add(int(j)); sel_local.append(int(j))
            if len(have) >= min(NONRIGID_MIN, len(ids), len(seen)):
                break
        budget["floor_pad"] = len(set(sel_local))
    sel_local = sorted(set(sel_local))
    # RIGID -> single part BY DEFINITION (2026-08-13). Before the pure-2a change this was implicit:
    # Stage-2b-early collapsed every rigid instance's sem->motion map to {all -> 0}. With part_id
    # now carrying the Stage-2a semantic label, a rigid instance would otherwise inherit PartSAM's
    # 7-10 parts — meaningless structure in the token stream.
    pid_out = ([0] * len(sel_local) if object_type == "rigid"
               else [int(x) for x in part_id[sel_local]])
    return {"selected_ids": ids[sel_local].tolist(),
            "selected_local": sel_local,
            "importance": imp[sel_local].tolist(),
            "motion_part": motion_part[sel_local].tolist(),
            "part_id": pid_out,
            "budget": budget, "n_pass1": len(ids),
            "bbox": float(bbox), "bbox_degenerate": bool(bbox_degen),
            "n_seen": int((vis_frac > 0).sum())}


def process_scene(sd, pure_2a=False):
    tr = np.load(os.path.join(sd, "tracks_d4rt.npz"))
    s2 = np.load(os.path.join(sd, "stage2b.npz"), allow_pickle=True)
    xyz = np.nan_to_num(tr["xyz_ref0"]); vis = tr["visibility"].astype(bool)
    vis = vis & np.isfinite(tr["xyz_ref0"]).all(2)
    kpid, inst = tr["keypoint_id"], tr["instance_id"]
    mpart = s2["motion_part"]; otype = s2["object_type"].astype(str)
    # Stage-2a SEMANTIC part, keypoint-aligned (stage2b.npz has carried it all along; pass2 simply
    # never passed it on — which is why final_repr/tokens only ever saw motion_part).
    spart = s2["part_id"] if "part_id" in s2.files else tr["part_id"]

    sel_id, sel_inst, sel_mp, sel_pid, sel_imp, sel_ot = [], [], [], [], [], []
    meta = []
    for iid in np.unique(inst):
        m = inst == iid
        ot = str(otype[m][0]) if int(m.sum()) else "unscored"
        # DROPPED-INSTANCE LEDGER (2026-08-16). Both exits used to be SILENT `continue`s; an
        # instance that leaves no row in pass2.npz simply never appears in final_repr and nothing
        # says why. Record it instead — finalize copies these into final_repr_meta.json.
        if int(m.sum()) < 3:
            meta.append({"instance_id": int(iid), "object_type": ot, "n_pass1": int(m.sum()),
                         "n_pass2": 0, "dropped": "few_keypoints"})
            continue
        r = pass2_instance(xyz[m], vis[m], kpid[m], mpart[m], spart[m], ot, pure_2a=pure_2a)
        if not r["selected_ids"]:
            # 幽灵点闸的必然出口:这个实例【一个关键点都没有被观测到】(⑦ 零链接,或链接上的轨迹
            # 从没落进自己的 mask)。旧行为是兜底凑 4 个全零点,写成世界原点上的假物体。
            meta.append({"instance_id": int(iid), "object_type": ot, "n_pass1": r["n_pass1"],
                         "n_pass2": 0, "n_seen": r["n_seen"],
                         "dropped": "no_observable_keypoint"})
            print(f"[pass2] {os.path.basename(sd)}: inst {int(iid)} ({ot}): "
                  f"{r['n_pass1']} 个 pass1 关键点全程无观测 -> DROP no_observable_keypoint",
                  flush=True)
            continue
        sel_id.extend(r["selected_ids"]); sel_imp.extend(r["importance"])
        sel_mp.extend(r["motion_part"]); sel_pid.extend(r["part_id"])
        sel_inst.extend([int(iid)] * len(r["selected_ids"]))
        sel_ot.extend([ot] * len(r["selected_ids"]))
        # budget keys: "rigid" / "floor_pad" (str) or a part label (int). Only INT keys are parts —
        # the old expression counted "floor_pad" as a moving part.
        part_keys = [k for k in r["budget"] if isinstance(k, int)]
        n_moving = 0 if pure_2a else len([k for k in part_keys if k != 0])
        meta.append({"instance_id": int(iid), "object_type": ot, "n_pass1": r["n_pass1"],
                     "n_pass2": len(r["selected_ids"]), "n_moving_parts": int(n_moving),
                     "n_sel_parts": len(set(r["part_id"])),
                     "n_seen": r["n_seen"], "bbox": round(r["bbox"], 4),
                     "bbox_degenerate": r["bbox_degenerate"],
                     "budget": r["budget"], "gap_warnings": []})

    np.savez_compressed(os.path.join(sd, "pass2.npz"),
                        selected_keypoint_id=np.array(sel_id, int), instance_id=np.array(sel_inst, int),
                        motion_part=np.array(sel_mp, int), part_id=np.array(sel_pid, int),
                        importance=np.array(sel_imp, np.float32),
                        object_type=np.array(sel_ot, dtype=str))
    json.dump({"scene": os.path.basename(sd), "instances": meta},
              open(os.path.join(sd, "pass2_meta.json"), "w"), indent=2)
    n_kp = len(sel_id)
    return len(meta), n_kp


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--batch-root")
    g.add_argument("--scene-dir")
                    # 2026-08-15 起 PURE-2a 是【默认】(motion-part 已于 2026-08-13 退场,见 memory
                    # project_motion_part_retired)。--no-motion-groups 保留为向后兼容的 no-op,
                    # 想跑回旧的运动臂用 --motion-groups。
    ap.add_argument("--no-motion-groups", action="store_true", default=True,
                    help="PURE-2a arm (2026-08-13): non-rigid instances get ONE medoid keypoint "
                         "per SEMANTIC part instead of the motion-part budget. rigid is unchanged "
                         "(<=4 points for SE(3)) and its part_id is forced to 0 either way.")
    ap.add_argument("--motion-groups", dest="no_motion_groups", action="store_false",
                    help="退回 2026-08-13 之前的运动臂(motion-part 参与部件结构)。仅用于\n"
                         "复现旧产物/对照实验;实测 62.3%% 的非刚体实例 motion_part 恒为 0。")
    args = ap.parse_args()
    sds = ([args.scene_dir] if args.scene_dir else
           sorted(d for d in glob.glob(os.path.join(args.batch_root, "scene_*"))
                  if os.path.exists(os.path.join(d, "stage2b.npz"))))
    print(f"[pass2] BATCH: {len(sds)} scenes")
    tot_obj = tot_kp = 0
    for sd in sds:
        try:
            no, nk = process_scene(sd, pure_2a=args.no_motion_groups)
            tot_obj += no; tot_kp += nk
            print(f"[pass2] {os.path.basename(sd)}: {no} instances -> {nk} final keypoints", flush=True)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[pass2] {os.path.basename(sd)}: FAILED {type(e).__name__}: {e}", flush=True)
    print(f"[pass2] DONE: {tot_obj} instances, {tot_kp} final keypoints total")


if __name__ == "__main__":
    main()
