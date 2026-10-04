"""§4 post-processing -> the FINAL v2 sparse 4D representation (REFERENCE §4 / §5.0).

Per instance, over its pass2-selected keypoints (the points the MLLM will see):
  - rigid     : per-frame Kabsch over the selected keypoints -> global SE(3) trajectory R(t),t(t).
                Stored as ref_positions (frame 0) + SE(3); the full trajectory = R(t)@p* + t(t).
  - non-rigid : keep the per-keypoint trajectories (selected points) as-is.

Consumes the persisted artifacts (all keyed by keypoint_id):
    pass2.npz        selected_keypoint_id, instance_id, motion_part, part_id, importance,
                     object_type   (part_id = Stage-2a SEMANTIC part, ADDITIVE 2026-08-13; absent
                     on pre-2026-08-13 products -> the field is simply omitted from final_repr)
    tracks_d4rt.npz  xyz_ref0 (K,T,3), visibility (K,T), keypoint_id

Doctrine note: SE(3) is computed from the SELECTED points only (§5.0 "rigid: 选中点轨迹 → 逐帧
Kabsch") — consistent with exactly the points the MLLM receives. pass2 is visibility-aware (it picks
well-tracked points), so the selected points are co-visible enough to define a per-frame SE(3); no
all-keypoint fallback is needed. A genuinely untracked object (whose keypoints are almost never
co-visible) yields se3_ok=False frames — flagged honestly, not papered over. A reconstruction
residual (R(t)@p*+t(t) vs tracked) is reported as a sanity probe: small => SE(3) faithful.

REFERENCE POSITIONS (rigid, 2026-08-16 fix): `ref_positions` used to be `P_sel[:, rf]` verbatim,
which writes a (0,0,0) PLACEHOLDER for any selected point that has no observation at the reference
frame — and since 2026-08-08 the frame is W_track (metric, origin = first camera), so consumers
reconstructing R(t)@ref + t(t) got a fake keypoint 12-17 m away at the world origin. Now such rows
are back-projected into the rest frame through the solved SE(3), or dropped. See `_rest_positions`.

Output per scene:
  final_repr.pkl   list[dict] per instance (see _finalize_instance; + "label" = Stage 0 text
                   label from video_input.pkl Masklet.concept — Qwen VL or dataset GT)
  final_repr_meta.json  per-instance summary (counts, rigid recon residual, label) + scene "frame".
                   ADDITIVE 2026-08-16: rows with "dropped" = instances deliberately NOT in
                   final_repr.pkl ("no_observable_keypoint" from pass2 / "few_keypoints" /
                   "no_valid_reference_position"), so a missing object is always explained;
                   plus per-instance n_ref_backprojected / n_ref_dropped when they are non-zero.

COORDINATE FRAME: everything here (dynamic trajectories/SE(3) AND static representative points)
lives in W_track — the dense front-end's own world frame, which is METRIC on Stereo4D (the GT world)
and the tracker world elsewhere. It is scene-consistent by construction, which the old target
(W_mesh) was not: see the OUTPUT FRAME note in src/stage4/run_link_dense.py. Recorded per scene as
final_repr_meta.json["frame"]; "W_mesh" there means a pre-2026-08-08 product.
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
_S2 = os.path.join(_HERE, "..", "stage2")
if _S2 not in sys.path:
    sys.path.insert(0, _S2)
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import stage2b  # noqa: E402  (kabsch / part_se3_traj — pure numpy, no D4RT)
from src.registry import se3_compose, se3_inv, _kabsch  # noqa: E402
from src.stage2.run_stage2b_dense import _inmask_tstar  # noqa: E402  (SAME in-mask track gate the
#                                       rigid-route decision used — routed branch must see the same
#                                       track set the verdict was made on)
import SCHEMA  # noqa: E402,F401  (registers Masklet/VideoInput so video_input.pkl unpickles)


def _load_masklet_info(sd):
    """instance_id -> {label, object_type, motion} from video_input.pkl masklets.
    getattr defaults keep OLD pickles (no motion field) loading: motion None == dynamic.
    Tolerant: returns {} if video_input.pkl is missing/unreadable (labels then stay None)."""
    try:
        with open(os.path.join(sd, "video_input.pkl"), "rb") as f:
            video = pickle.load(f)
        return {int(m.instance_id): {"label": m.concept,
                                     "object_type": getattr(m, "object_type", None),
                                     "motion": getattr(m, "motion", None)}
                for m in video.masklets}
    except Exception as e:
        print(f"[finalize] {os.path.basename(sd)}: labels unavailable "
              f"({type(e).__name__}: {e})", flush=True)
        return {}


def _fps_idx(pts, k):
    """Deterministic farthest-point sampling: start at the point farthest from the median
    center, then greedy max-min. Returns the selected ROW INDICES (<=k)."""
    pts = np.asarray(pts, np.float64)
    n = len(pts)
    if n <= k:
        return np.arange(n)
    c = np.median(pts, 0)
    idx = [int(np.argmax(np.linalg.norm(pts - c, axis=1)))]
    d = np.linalg.norm(pts - pts[idx[0]], axis=1)
    while len(idx) < k:
        j = int(np.argmax(d))
        idx.append(j)
        d = np.minimum(d, np.linalg.norm(pts - pts[j], axis=1))
    return np.asarray(idx, int)


def _fps(pts, k):
    """FPS returning the selected rows (thin wrapper over _fps_idx; static branch uses this)."""
    pts = np.asarray(pts, np.float64)
    return pts[_fps_idx(pts, k)]


def _static_instance_repr(sd, iid, info, scene_T, k=4):
    """STATIC branch (V3 routing): a static instance has dense tracks but NO mesh / keypoints /
    SE(3). Its final representation = k representative 3D points (per-track median of the visible
    positions, FPS-k), mapped by scene_T into the SAME coordinate frame as the dynamic instances.
    Stillness is expressed by the FORM of the entry (points only, no trajectories/SE3) — no new
    field names.

    scene_T is the IDENTITY under the current contract (⑦ track_frame=='W_track'): the dynamic
    instances' xyz_ref0 and these dense xyz_world are both already in W_track, so nothing to do.
    It is ⑦'s pooled W_track->W_mesh fit only for pre-2026-08-08 products (xyz_ref0 in W_mesh).
    Graceful None when the dense tracks are missing/empty (finalize must not hard-depend on
    dense_tracks_*.npz: the lane thins them after chunk completion)."""
    dp = os.path.join(sd, f"dense_tracks_{iid}.npz")
    if not os.path.exists(dp):
        print(f"[finalize] {os.path.basename(sd)}: static inst {iid} "
              f"({info.get('label')}): dense_tracks missing -> skipped", flush=True)
        return None
    d = np.load(dp)
    xyz = d["xyz_world"].astype(np.float64)                    # (N,T,3) W_t4w
    vis = d["vis"].astype(bool) & np.isfinite(xyz).all(2)      # (N,T)
    nvis = vis.sum(1)
    keep = np.nonzero(nvis >= 3)[0]                            # prefer stable tracks
    if len(keep) < k:
        keep = np.nonzero(nvis >= 1)[0]
    if len(keep) == 0:
        print(f"[finalize] {os.path.basename(sd)}: static inst {iid} "
              f"({info.get('label')}): no visible tracks -> skipped", flush=True)
        return None
    med = np.stack([np.median(xyz[i][vis[i]], axis=0) for i in keep])   # (n,3) W_t4w
    med = med @ scene_T[:3, :3].T + scene_T[:3, 3]             # -> unified scene frame
    pts = _fps(med, k)
    return {"instance_id": int(iid), "object_type": info.get("object_type"),
            "label": info.get("label"), "ref_positions": pts.astype(np.float32)}


def _pass2_dropped(sd):
    """pass2_meta.json 里带 `dropped` 的实例记录(2026-08-16)。pass2 会剔除【一个关键点都没有被
    观测到】的实例(全部是 ⑦ 零链接的那批),旧行为是兜底凑 4 个全零点写成世界原点上的假物体。
    剔除本身是对的,但不能静默 —— 把原因带进 final_repr_meta.json。
    文件缺失/不可读 -> 空列表(老产物、或 pass2 早于本次改动)。"""
    p = os.path.join(sd, "pass2_meta.json")
    if not os.path.exists(p):
        return []
    try:
        with open(p) as f:
            return [r for r in json.load(f).get("instances", []) if r.get("dropped")]
    except Exception as e:
        print(f"[finalize] {p}: unreadable ({e}) -> 剔除名单不可用", flush=True)
        return []


def _routed_info(sd):
    """(routed instance_ids, {iid: anchor frame}) from rigid_route.json (src/stage1/rigid_route.py).
    Missing/unreadable -> empty, i.e. nothing routed and this whole branch is inert."""
    p = os.path.join(sd, "rigid_route.json")
    if not os.path.exists(p):
        return set(), {}
    try:
        with open(p) as f:
            recs = json.load(f).get("instances", {})
        ids = {int(k) for k, v in recs.items() if v.get("routed")}
        return ids, {int(k): recs[k].get("anchor") for k in recs if int(k) in ids}
    except Exception as e:
        print(f"[finalize] {p}: unreadable ({e}) -> nothing rigid-routed", flush=True)
        return set(), {}


def _load_masks(sd, ids):
    """{iid: (T,H,W) bool} for just `ids` — the routed branch needs the masklet masks for the
    in-mask track gate. Loaded once per scene, only for the routed instances (masks are big)."""
    if not ids:
        return {}
    try:
        with open(os.path.join(sd, "video_input.pkl"), "rb") as f:
            video = pickle.load(f)
        return {int(m.instance_id): np.asarray(m.masks).astype(bool)
                for m in video.masklets if int(m.instance_id) in ids}
    except Exception as e:
        print(f"[finalize] {os.path.basename(sd)}: masks unavailable ({type(e).__name__}: {e})",
              flush=True)
        return {}


def _routed_rigid_repr(sd, iid, info, masks, scene_T, anchor=None, k=4):
    """RIGID ROUTING branch (2026-08-13). A routed rigid instance has dense tracks but NO mesh /
    keypoints / parts (Stage 1 + Stage 2a were skipped for it — see src/stage1/rigid_route.py).

    The representation is built from the tracks and then goes through the SAME `_finalize_instance`
    the mesh path uses, so the SE(3) solver, the honesty gates and the reconstruction residual are
    literally the same code — only the point SOURCE differs. That is the whole point: for a rigid
    instance the mesh path never contributed anything but the 3-4 points it sampled.

    Track filter = the one rigid_route.py gated on (in-mask at the anchor, visible, finite), so the
    points come from exactly the track set the rigid verdict was made on. SE(3) is solved over ALL
    of them (time-diverse anchors at zero token cost — same as the mesh path's `P_all`); the k
    stored `ref_positions` are an FPS subset for spatial spread.

    ⚠️ KNOWN, ACCEPTED LOSS: these points lie only on the VISIBLE surface, whereas mesh-sampled
    keypoints can also cover the t*-occluded side. SE(3) is unaffected (it was only ever solved
    from tracked points); the 'shape sketch' the MLLM receives is slightly poorer."""
    dp = os.path.join(sd, f"dense_tracks_{iid}.npz")
    if not os.path.exists(dp):
        print(f"[finalize] {os.path.basename(sd)}: routed inst {iid} "
              f"({info.get('label')}): dense_tracks missing -> skipped", flush=True)
        return None
    d = np.load(dp)
    xyz = d["xyz_world"].astype(np.float64)                    # (N,T,3) W_track
    vis = d["vis"].astype(bool) & np.isfinite(xyz).all(2)      # (N,T)
    T = vis.shape[1]
    if anchor is None or not (0 <= int(anchor) < T):
        anchor = int(np.argmax(vis.sum(0)))                    # same rule rigid_route.py uses
    anchor = int(anchor)
    m = masks.get(iid)
    inm = (_inmask_tstar(d["uv_norm"][:, anchor].astype(np.float32), m, anchor)
           if m is not None else np.ones(len(vis), bool))
    keep = np.nonzero(vis[:, anchor] & inm)[0]
    if len(keep) < 3:                                          # Kabsch needs >=3 non-collinear
        print(f"[finalize] {os.path.basename(sd)}: routed inst {iid} "
              f"({info.get('label')}): only {len(keep)} usable tracks -> skipped", flush=True)
        return None
    # Map into the unified scene frame BEFORE solving, so the SE(3) comes out in that frame
    # directly — no conjugation of an already-solved pose (identity under the current contract).
    P = xyz[keep] @ scene_T[:3, :3].T + scene_T[:3, 3]
    V = vis[keep]
    sel = _fps_idx(P[:, anchor], min(k, len(keep)))             # spatial spread at the anchor
    bbox = float(np.linalg.norm(P[:, anchor].max(0) - P[:, anchor].min(0))) + 1e-9
    disp = np.where(V, np.linalg.norm(P - P[:, anchor:anchor + 1], axis=2), 0.0)
    imp = (disp.max(axis=1) / bbox)[sel]                        # same definition as pass2's importance
    z = np.zeros(len(sel), int)
    r = _finalize_instance(iid, "rigid", P[sel], V[sel], keep[sel], z, imp.astype(np.float32),
                           P_all=P, vis_all=V, seg_info=None, part_id=z)
    if r is None:                          # 参考位置全不可信(见 _rest_positions ③)
        print(f"[finalize] {os.path.basename(sd)}: routed inst {iid} "
              f"({info.get('label')}): 无可信参考位置 -> skipped", flush=True)
        return None
    r["repr_source"] = "routed_rigid"      # self-documenting: points came from tracks, not a mesh
    return r


def _seg_local_solve(Pf, vf, s, e):
    """Per-segment SE(3): part_se3_traj on the segment's frame slice. Returns dict with local
    poses (relative to the segment's own anchor frame), a segment-LOCAL recon residual
    (normalized by the anchor-frame keypoint extent), or None when nothing is solvable."""
    Pl, vl = Pf[:, s:e + 1], vf[:, s:e + 1]
    if not vl.any():
        return None
    Rl, tl, okl, rfl = stage2b.part_se3_traj(Pl, vl)
    if not okl.any():
        return None
    anchor = s + int(rfl)
    sel_a = vf[:, anchor]
    pts = Pf[sel_a, anchor] if sel_a.any() else Pf[:, anchor]
    bb = float(np.linalg.norm(pts.max(0) - pts.min(0))) + 1e-9
    recon = np.einsum("tij,mj->tmi", Rl, Pf[:, anchor]) + tl[:, None, :]     # (Tl,M,3)
    msk = (vl & vf[:, anchor:anchor + 1]).T & okl[:, None]
    err = np.linalg.norm(recon - Pl.transpose(1, 0, 2), axis=2)
    resid = float(np.median(err[msk]) / bb) if (msk.any() and bb > 1e-4) else float("nan")
    # segment usable = solvable AND self-consistent (same >1.0-bbox honesty gate as the global one)
    gate_ok = bool(resid == resid and resid <= 1.0)
    return dict(R=Rl, t=tl, ok=okl, anchor=anchor, s=s, e=e, resid=resid, gate_ok=gate_ok)


def _pose_mesh_to_frame(f_target, segs, locs, per7):
    """pose(mesh -> frame f_target): the ⑦ re-capture pose gives mesh->(⑦ anchor); the segment's
    local solve moves it to f_target. Returns (R, t) or None. The t* segment's re-capture is the
    identity at t* by construction (the mesh IS the object at t* in W_mesh)."""
    k = next((i for i, (s, e) in enumerate(segs) if s <= f_target <= e), None)
    if k is None or locs[k] is None or per7[k] is None or not per7[k].get("registered"):
        return None
    m7, loc = per7[k], locs[k]
    Rr, tr = np.asarray(m7["reg_R"], float), np.asarray(m7["reg_t"], float)
    pm = se3_inv(Rr, tr)                              # mesh ≈ Rr@p(f7)+tr  ->  pose(mesh->f7)
    f7 = int(m7["anchor"])
    s = segs[k][0]
    if f7 != loc["anchor"]:                           # move mesh->f7 to mesh->anchor
        f7l = f7 - s
        if not (0 <= f7l < len(loc["ok"]) and loc["ok"][f7l]):
            return None
        pm = se3_compose(*se3_inv(loc["R"][f7l], loc["t"][f7l]), *pm)
    if f_target == loc["anchor"]:
        return pm
    fl = f_target - s
    if not (0 <= fl < len(loc["ok"]) and loc["ok"][fl]):
        return None
    return se3_compose(loc["R"][fl], loc["t"][fl], *pm)


def _apply_segments(Pf, vf, base_R, base_t, base_ok, base_rf, seg_info, bbox):
    """Multi-presence SE(3) assembly (V3 mesh registry, 2026-07-11). STRICTLY ADDITIVE over the
    global (baseline) solve: base ok-frames are kept verbatim; each presence segment is solved on
    its own slice and bridged onto the global reference by (in order) the base pose at the
    segment's anchor, co-visible-keypoint Kabsch (mesh-linked kps shared across segments), or the
    ⑦ mesh re-capture poses. Segments that fail registration/bridging stay honestly undefined —
    as do all absent frames (vis is masklet-gated upstream, so nothing is solvable there anyway).
    Returns (R, t, ok, rest, seg_report, T_present) or None when nothing beyond base is possible."""
    segs = [tuple(map(int, x)) for x in seg_info["segments"]]
    per7 = list(seg_info.get("per_seg", []))
    if len(per7) != len(segs):
        per7 = [None] * len(segs)
    T = base_ok.shape[0]
    T_present = int(sum(e - s + 1 for s, e in segs))
    locs = [_seg_local_solve(Pf, vf, s, e) for s, e in segs]
    report = []
    base_usable = bool(base_ok.any())
    if base_usable:
        R, t, ok, rest = base_R.copy(), base_t.copy(), base_ok.copy(), int(base_rf)
        ref_k = None
    else:                                             # pure-segment mode: best segment anchors all
        cand = [i for i, l in enumerate(locs) if l is not None and l["gate_ok"]]
        if not cand:
            return None
        ref_k = max(cand, key=lambda i: int(locs[i]["ok"].sum()))
        rest = locs[ref_k]["anchor"]
        R = np.tile(np.eye(3), (T, 1, 1)); t = np.zeros((T, 3)); ok = np.zeros(T, bool)
        loc = locs[ref_k]
        s, e = segs[ref_k]
        for fl in np.nonzero(loc["ok"])[0]:
            R[s + fl], t[s + fl] = loc["R"][fl], loc["t"][fl]
            ok[s + fl] = True
    for k, ((s, e), loc) in enumerate(zip(segs, locs)):
        rec = {"range": [s, e], "n_frames": e - s + 1,
               "registered": bool(per7[k].get("registered")) if per7[k] else None,
               "inlier_frac": per7[k].get("inlier_frac") if per7[k] else None,
               "n_kp_hooked": per7[k].get("n_kp") if per7[k] else None}
        if loc is None:
            rec.update(solved=False, bridge=None, n_ok=0, resid_local=None)
            report.append(rec); continue
        rec.update(solved=True, anchor=loc["anchor"], resid_local=round(loc["resid"], 4)
                   if loc["resid"] == loc["resid"] else None, n_ok=int(loc["ok"].sum()))
        if not loc["gate_ok"]:
            rec.update(bridge="dropped:high_resid")
            report.append(rec); continue
        if k == ref_k:
            rec.update(bridge="reference")
            report.append(rec); continue
        a_k = loc["anchor"]
        B, how = None, None
        if ok[a_k]:                                   # base/ref pose already covers the anchor
            B, how = (R[a_k].copy(), t[a_k].copy()), "chain"
        if B is None:                                 # mesh-linked kps shared across segments
            sel = vf[:, rest] & vf[:, a_k]
            if int(sel.sum()) >= 3:
                Rb, tb = _kabsch(Pf[sel, rest], Pf[sel, a_k])
                bres = float(np.median(np.linalg.norm(
                    Pf[sel, rest] @ Rb.T + tb - Pf[sel, a_k], axis=1)))
                if bres <= 0.25 * bbox:
                    B, how = (Rb, tb), f"covis(n={int(sel.sum())})"
        if B is None:                                 # mesh bridge via ⑦ re-capture poses
            pm_k = _pose_mesh_to_frame(a_k, segs, locs, per7)
            pm_r = _pose_mesh_to_frame(rest, segs, locs, per7)
            if pm_k is not None and pm_r is not None:
                B, how = se3_compose(*pm_k, *se3_inv(*pm_r)), "mesh"
        rec["bridge"] = how or "unbridged"
        if B is None:
            report.append(rec); continue
        n_add = 0
        for fl in np.nonzero(loc["ok"])[0]:
            f = s + int(fl)
            if not ok[f]:                             # additive: never touch base ok frames
                R[f], t[f] = se3_compose(loc["R"][fl], loc["t"][fl], B[0], B[1])
                ok[f] = True
                n_add += 1
        rec["n_added"] = n_add
        report.append(rec)
    if not base_usable:                               # presence-relative honesty gate (was 0.1*T)
        if ok.sum() < max(3, int(0.1 * T_present)):
            return None
    return R, t, ok, rest, report, T_present


def _rest_positions(P_sel, vis_sel, R, t, ok, rf):
    """rigid 的逐点参考系位置 + 保留掩码。返回 (ref (M',3) float32, keep (M,) bool, n_backproj)。

    ⚠️ 2026-08-16 修。旧写法是 `ref = P_sel[:, rf]` 一把梭:pass2 选中的点只保证 vis_frac >= 0.5,
    **不保证在 rf 那一帧有观测**,没观测的行 xyz 就是 (0,0,0) 占位。自 2026-08-08 起坐标落在
    W_track(公制 GT 世界,原点 = 首帧相机),于是 tokenizer 的 `_rigid_world_traj` 用
    R(t)@ref + t(t) 重建时,把这一行算成"世界原点在该位姿下的像"——一个离物体 12-17 m 的假点,
    并且进 `compute_scene_norm` 撑大整个场景的归一化尺度。实测 rigid ref 行 106/388 = 27.3% 为零。
    **刚体条目里没有任何逐点有效性信号**(只有逐帧的 se3_ok),消费方无从自查,这是它比非刚体
    (有 visibility 可查)危险的地方。

    三级规则:
      ① rf 帧有观测            -> 直接取 P_sel[m, rf](与旧行为逐位相同)
      ② 否则用**已解出的 SE(3)** 把该点自己的观测反投影回参考系:
             p_rest = median_{f: vis & ok} R[f]^T (p(f) - t[f])
         这正是位姿的定义式 p(f) = R[f] p_rest + t[f] 的逆解;取中位数是为了对逐帧位姿误差稳健。
         实测修复后该点落在到其余参考点质心 1.78 m(中位)/ 4.36 m(p90)处 = 在物体身上。
      ③ 都不行(该点没有任何"既可见又解出了位姿"的帧)-> 这一行没有可信的参考位置,**整行剔除**。

    `ok` 传【解算器原始输出】而不是过完诚实门的那份:门决定的是"要不要对外发布这个 SE(3)",
    而不是"这个位姿能不能用来反算点的静止位置"。位姿再差,反投影出来的点也在物体附近,而占位
    行是确定错的(在世界原点)。真到了一帧都没解出来的地步,走 ③ 剔除。

    注意残差与诚实门**不受本函数影响**:残差 mask 要求与 rf 共可见,②③ 覆盖的行本来就被掩掉。"""
    M, T = vis_sel.shape
    have = vis_sel[:, rf]
    ref = P_sel[:, rf].astype(np.float64).copy()
    keep = have.copy()
    n_bp = 0
    for m in np.nonzero(~have)[0]:
        usable = np.nonzero(vis_sel[m] & ok)[0]
        if len(usable) == 0:
            continue                                  # ③ 无解 -> keep 保持 False,整行剔除
        bp = np.stack([R[f].T @ (P_sel[m, f] - t[f]) for f in usable])
        ref[m] = np.median(bp, axis=0)                # ② 反投影回参考系
        keep[m] = True
        n_bp += 1
    return ref[keep].astype(np.float32), keep, n_bp


def _finalize_instance(iid, otype, P_sel, vis_sel, kp_ids, motion_part, importance,
                       P_all=None, vis_all=None, seg_info=None, part_id=None):
    """Build the final representation dict for one instance + a reconstruction residual (rigid).
    SE(3): solved over ALL of the instance's pass1 keypoints when provided (long-window fix,
    2026-07-09): pass2's tiny rigid budget (~4 pts) starves the chained solver on long windows —
    all tracked keypoints give time-diverse anchors at ZERO token cost (only pass2-selected points
    are stored). Falls back to the selected points when P_all is None."""
    ref = P_sel[:, 0].astype(np.float32)                     # (M,3) reference-frame positions
    out = {"instance_id": int(iid), "object_type": otype, "keypoint_ids": kp_ids.tolist(),
           "ref_positions": ref, "motion_part": motion_part.astype(int),
           "importance": importance.astype(np.float32)}
    # ADDITIVE (2026-08-13): Stage-2a SEMANTIC part per selected keypoint. `motion_part` is kept
    # (consumers: QA adapters, viz, training serializer) but is all-zeros on the pure-2a arm, so
    # `part_id` is the field that actually carries part structure there. Rigid instances always
    # carry part_id == 0 (forced in pass2 — a rigid object has no motion-meaningful decomposition).
    if part_id is not None:
        out["part_id"] = np.asarray(part_id).astype(int)
    if otype == "rigid":
        Pf = P_sel if P_all is None else P_all
        vf = vis_sel if vis_all is None else vis_all
        R, t, ok, rf = stage2b.part_se3_traj(Pf, vf)         # all-keypoint, chained (see stage2b)
        # NORMALIZATION FIX (2026-07-11): bbox must be the INSTANCE's extent, not the selected
        # points' spread — pass2 can select as few as 2 near-coincident points, making a
        # selected-point bbox ~0.05 that inflates resid past the >1.0 gate even for an excellent
        # pose (measured: err/inst-bbox 0.02 killed as resid 1.12). Use all tracked keypoints
        # visible at the anchor frame; fall back to the selected spread only if degenerate.
        vf_rf = vf[:, rf] if vf.ndim == 2 else vf
        pts_rf = Pf[vf_rf, rf] if vf_rf.any() else Pf[:, rf]
        bbox = float(np.linalg.norm(pts_rf.max(0) - pts_rf.min(0))) + 1e-9
        if bbox < 1e-4:                                      # 退化兜底:选中点在帧 0 的跨度
            bbox = float(np.linalg.norm(P_sel[:, 0].max(0) - P_sel[:, 0].min(0))) + 1e-9
        # QUALITY GATE (2026-07-09): the chained solver can stitch a "pose" for a nearly-untracked
        # instance (vis ~2%) from a few weak links — meaningless (and its ~coincident selected
        # points give bbox~0 -> recon explodes to 1e7). Honest rule: enough solved frames AND real
        # overall visibility AND a non-degenerate reference, else se3 stays undefined.
        T_ = ok.shape[0]
        ok_raw = ok.copy()                                   # 解算器原始输出,供 _rest_positions 用
        if ok.sum() < max(3, int(0.1 * T_)) or bbox < 1e-4:
            ok = np.zeros_like(ok)
        # 逐点参考位置(2026-08-16):rf 帧无观测的行改用 SE(3) 反投影,无解则整行剔除。见 _rest_positions。
        ref, keep, n_bp = _rest_positions(P_sel, vis_sel, R, t, ok_raw, rf)
        if not keep.any():
            print(f"[finalize] inst {iid}: 选中点在参考帧 {rf} 全无观测且无法反投影 -> 剔除该实例",
                  flush=True)
            return None
        if not keep.all():                                   # 同步收缩所有逐点字段
            P_sel = P_sel[keep]; vis_sel = vis_sel[keep]
            out["keypoint_ids"] = [k for k, kp in zip(out["keypoint_ids"], keep) if kp]
            out["motion_part"] = out["motion_part"][keep]
            out["importance"] = out["importance"][keep]
            if "part_id" in out:
                out["part_id"] = out["part_id"][keep]
            out["n_ref_dropped"] = int((~keep).sum())
        if n_bp:
            out["n_ref_backprojected"] = int(n_bp)
        out["ref_positions"] = ref
        if rf != 0:
            out["ref_frame"] = int(rf)
        out["se3_R"] = R.astype(np.float32); out["se3_t"] = t.astype(np.float32); out["se3_ok"] = ok
        # reconstruction residual: R(t)@p* + t(t) vs tracked position, over valid frames
        recon = np.einsum("tij,mj->tmi", R, ref) + t[:, None, :]   # (T,M,3)
        m = (vis_sel & vis_sel[:, rf:rf+1]).T & ok[:, None]       # (T,M) co-visible-with-ref & frame ok
        err = np.linalg.norm(recon - P_sel.transpose(1, 0, 2), axis=2)  # (T,M)
        resid = float(np.median(err[m]) / bbox) if m.any() else float("nan")
        # SELF-DIAGNOSING GATE: a pose that cannot reconstruct its own selected points (resid > 1
        # bbox, or no co-visible frame to even measure) is not a usable SE(3) — mark undefined.
        if resid != resid or resid > 1.0:
            out["se3_ok"] = ok = np.zeros_like(ok)
            resid = float("nan")
        out["recon_resid_norm"] = resid
        # --- V3 multi-presence (2026-07-11): per-segment SE(3) + mesh bridging. Single-segment
        # instances take the path above UNCHANGED. Strictly additive: base ok frames are kept;
        # only frames the global solve left undefined can gain a segment-bridged pose. ---
        if seg_info is not None and len(seg_info.get("segments", [])) > 1:
            aug = _apply_segments(Pf, vf, R, t, ok, rf, seg_info, bbox)
            if aug is not None:
                R2, t2, ok2, rest, seg_report, T_present = aug
                pure_seg = not ok.any()               # base was unusable -> re-anchored at `rest`
                out["se3_R"] = R2.astype(np.float32); out["se3_t"] = t2.astype(np.float32)
                out["se3_ok"] = ok2
                out["segments"] = seg_report
                out["T_present"] = T_present
                if pure_seg:
                    # 同 _rest_positions(2026-08-16):re-anchor 到 `rest` 时同样不能直接取
                    # P_sel[:, rest] —— 在 rest 帧没观测的行会是 (0,0,0) 占位。这里用**分段拼接
                    # 后**的位姿 (R2,t2,ok2) 反投影;无解的行整行剔除。
                    ref2, keep2, n_bp2 = _rest_positions(P_sel, vis_sel, R2, t2, ok2, rest)
                    if not keep2.any():
                        print(f"[finalize] inst {iid}: 纯分段模式下参考帧 {rest} 全无观测且无法"
                              f"反投影 -> 剔除该实例", flush=True)
                        return None
                    if not keep2.all():
                        P_sel = P_sel[keep2]; vis_sel = vis_sel[keep2]
                        out["keypoint_ids"] = [k for k, kp in zip(out["keypoint_ids"], keep2) if kp]
                        out["motion_part"] = out["motion_part"][keep2]
                        out["importance"] = out["importance"][keep2]
                        if "part_id" in out:
                            out["part_id"] = out["part_id"][keep2]
                        out["n_ref_dropped"] = out.get("n_ref_dropped", 0) + int((~keep2).sum())
                    if n_bp2:
                        out["n_ref_backprojected"] = out.get("n_ref_backprojected", 0) + int(n_bp2)
                    out["ref_positions"] = ref = ref2
                    if rest != 0:
                        out["ref_frame"] = int(rest)
                    elif "ref_frame" in out:
                        del out["ref_frame"]
                    rs = [sr["resid_local"] for sr in seg_report
                          if sr.get("resid_local") is not None and sr.get("n_added", 0) +
                          (1 if sr.get("bridge") == "reference" else 0) > 0]
                    out["recon_resid_norm"] = resid = (float(np.median(rs)) if rs
                                                       else float("nan"))
            else:                                     # nothing solvable beyond base
                out["segments"] = [{"range": [int(a), int(b)]}
                                   for a, b in seg_info["segments"]]
    else:
        out["trajectories"] = P_sel.astype(np.float32)       # (M,T,3) per-keypoint trajectory
        out["visibility"] = vis_sel
        out["recon_resid_norm"] = None
    return out


def process_scene(sd, static_skip=True, rigid_route=True):
    minfo = _load_masklet_info(sd)
    static_ids = (sorted(iid for iid, v in minfo.items() if v["motion"] == "static")
                  if static_skip else [])
    # RIGID ROUTING (2026-08-13): instances Stage 1/2a skipped. Static wins if somehow both apply.
    routed_ids, routed_anchor = (_routed_info(sd) if rigid_route else (set(), {}))
    routed_ids = {i for i in routed_ids if i in minfo} - set(static_ids)
    p2_path = os.path.join(sd, "pass2.npz")
    tr_path = os.path.join(sd, "tracks_d4rt.npz")
    have_dyn = os.path.exists(p2_path) and os.path.exists(tr_path)
    if not have_dyn and not static_ids and not routed_ids:
        print(f"[finalize] {os.path.basename(sd)}: no pass2/tracks and no static/routed instances "
              f"-> skipped", flush=True)
        return None

    reprs, meta = [], []
    scene_T = None
    track_frame = None
    if have_dyn:
        p2 = np.load(p2_path, allow_pickle=True)
        tr = np.load(tr_path)
        xyz = np.nan_to_num(tr["xyz_ref0"]); vis = tr["visibility"].astype(bool)
        vis = vis & np.isfinite(tr["xyz_ref0"]).all(2)
        row_of = {int(k): i for i, k in enumerate(tr["keypoint_id"])}  # keypoint_id -> tracks row
        seg_all = json.loads(str(tr["segments"])) if "segments" in tr.files else {}
        # UNIFIED SCENE FRAME. Current contract: ⑦ emits xyz_ref0 in W_track (the dense front-end's
        # own world — metric W_gt on Stereo4D), which is exactly the frame the static branch's
        # dense xyz_world is already in -> identity. Pre-2026-08-08 products have xyz_ref0 in the
        # per-instance W_mesh and still need ⑦'s pooled scene_T to drag static points over there.
        track_frame = str(tr["track_frame"]) if "track_frame" in tr.files else "W_mesh"
        if track_frame == "W_track":
            scene_T = np.eye(4)
        elif "scene_T" in tr.files:                          # legacy: W_t4w -> W_mesh (⑦)
            scene_T = np.asarray(tr["scene_T"], np.float64)

        sel_id = p2["selected_keypoint_id"]; sel_inst = p2["instance_id"]
        sel_mp = p2["motion_part"]; sel_imp = p2["importance"]; sel_ot = p2["object_type"].astype(str)
        sel_pid = p2["part_id"] if "part_id" in p2.files else None   # ADDITIVE; None on old products

        for iid in np.unique(sel_inst):
            if int(iid) in static_ids:
                continue        # static routing: point-based entry below, even if a mesh exists
            m = sel_inst == iid
            ids = sel_id[m]
            rows = np.array([row_of[int(k)] for k in ids])
            otype = sel_ot[m][0]
            arows = np.nonzero(tr["instance_id"] == iid)[0]  # ALL tracked keypoints of this instance
            seg_info = seg_all.get(str(int(iid)))
            r = _finalize_instance(iid, otype, xyz[rows], vis[rows], ids, sel_mp[m], sel_imp[m],
                                   P_all=xyz[arows], vis_all=vis[arows], seg_info=seg_info,
                                   part_id=(sel_pid[m] if sel_pid is not None else None))
            if r is None:            # 参考位置全不可信(见 _rest_positions ③)-> 留痕剔除
                meta.append({"instance_id": int(iid), "object_type": str(otype),
                             "label": minfo.get(int(iid), {}).get("label"),
                             "n_keypoints": 0, "dropped": "no_valid_reference_position"})
                continue
            # Stage 0 text label (Qwen VL / dataset GT), for MLLM
            r["label"] = minfo.get(int(iid), {}).get("label")
            reprs.append(r)
            mrec = {"instance_id": int(iid), "label": r["label"],
                    "object_type": otype, "n_keypoints": int(len(r["keypoint_ids"])),
                    "recon_resid_norm": r["recon_resid_norm"],
                    "se3_undefined": bool(otype == "rigid" and float(r["se3_ok"].mean()) == 0.0)}
            for k in ("n_ref_dropped", "n_ref_backprojected"):
                if k in r:
                    mrec[k] = r[k]
            if seg_info is not None:                         # ADDITIVE meta field (V3 multi-presence)
                mrec["segments"] = r.get("segments",
                                         [{"range": [int(a), int(b)]} for a, b in seg_info["segments"]])
                mrec["n_segments"] = len(seg_info["segments"])
                if otype == "rigid":
                    mrec["se3_ok_frames"] = int(r["se3_ok"].sum())
            meta.append(mrec)

    # --- STATIC branch (V3 routing): 4 representative points per static instance, in the SAME
    # unified scene frame as the dynamic instances. Identity when there is nothing to map: either
    # the current contract (both sides already in W_track) or an all-static scene (no tracks_d4rt
    # at all — the dense xyz_world IS the unified frame, so this is correct, not a fallback). Only
    # a LEGACY product whose ⑦ failed to solve scene_T is a real degradation ('t4w_raw'). ---
    frame_note = None
    if (static_ids or routed_ids) and scene_T is None:
        scene_T = np.eye(4)
        frame_note = "t4w_raw" if have_dyn else "W_track"
        if frame_note == "t4w_raw":
            print(f"[finalize] {os.path.basename(sd)}: legacy product with no scene_T -> static "
                  f"points stay in t4w_raw frame while dynamic ones are in W_mesh (MIXED — "
                  f"re-run stage 4 to fix)", flush=True)
    for iid in static_ids:
        r = _static_instance_repr(sd, iid, minfo[iid], scene_T)
        if r is None:
            continue
        reprs.append(r)
        mrec = {"instance_id": int(iid), "label": r["label"],
                "object_type": r["object_type"],
                "n_keypoints": int(len(r["ref_positions"]))}
        if frame_note:
            mrec["frame"] = frame_note
        meta.append(mrec)

    # --- ROUTED-RIGID branch (2026-08-13): mesh-free rigid instances. An instance that DID get a
    # mesh anyway (stage 1 run with --no-rigid-route, or a stale product) is already in `reprs` via
    # the dynamic loop above and is skipped here — the mesh path's points are strictly better. ---
    done_dyn = {int(x) for x in np.unique(sel_inst)} if have_dyn else set()
    routed_todo = sorted(routed_ids - done_dyn)
    rmasks = _load_masks(sd, set(routed_todo))
    for iid in routed_todo:
        r = _routed_rigid_repr(sd, iid, minfo[iid], rmasks, scene_T,
                               anchor=routed_anchor.get(iid))
        if r is None:
            continue
        r["label"] = minfo.get(int(iid), {}).get("label")
        reprs.append(r)
        mrec = {"instance_id": int(iid), "label": r["label"], "object_type": "rigid",
                "n_keypoints": int(len(r["ref_positions"])),
                "recon_resid_norm": r["recon_resid_norm"], "repr_source": "routed_rigid",
                "se3_undefined": bool(float(r["se3_ok"].mean()) == 0.0)}
        if frame_note:
            mrec["frame"] = frame_note
        meta.append(mrec)

    # DROPPED-INSTANCE LEDGER (2026-08-16): carry pass2's drop reasons into the file consumers
    # actually read next to final_repr.pkl, so "this object is not in the representation" always
    # has a recorded WHY (never a silent absence). Static/routed instances have their own paths and
    # are excluded — an instance dropped by pass2 may still legitimately appear via those.
    got = {int(r["instance_id"]) for r in reprs}
    for prec in _pass2_dropped(sd):
        iid = int(prec["instance_id"])
        if iid in got or iid in static_ids or iid in routed_ids:
            continue
        meta.append({"instance_id": iid, "label": minfo.get(iid, {}).get("label"),
                     "object_type": prec.get("object_type"), "n_keypoints": 0,
                     "n_pass1": prec.get("n_pass1"), "dropped": prec["dropped"]})

    if not reprs:
        # 每个实例都被剔除,且**每一次剔除都有记录** -> 这是一个已解释的空场景:照常落盘(空 list
        # + 完整 meta)。必须落盘,不能 return None,有两个理由:
        #   ① 陈旧产物:重跑一个已有产物的 scene 时,不写就等于把【旧的 final_repr.pkl 原地留下】,
        #      内容还是修复前那批世界原点假点 —— 与 project_stale_dense_tracks_bug 同一形态的坑。
        #   ② 队列:stereo4d_queue.sh 的 done 判据就是 final_repr.pkl 存在。不写 = 这个 clip 被
        #      反复重新认领到 MAXTRY 再进 gave_up,白烧两遍 GPU 全链。
        # 反过来,**没有任何剔除记录**的空场景说明上游就没产出(stage1/⑦ 挂了之类),那属于失败,
        # 必须保持 return None —— 写了它就等于把一次失败标成完成,把失败从队列里藏起来。
        if any(m.get("dropped") for m in meta):
            print(f"[finalize] {os.path.basename(sd)}: 所有实例均被剔除(见 final_repr_meta.json "
                  f"的 dropped 记录)-> 落盘空表示", flush=True)
        else:
            print(f"[finalize] {os.path.basename(sd)}: nothing to finalize -> skipped", flush=True)
            return None
    with open(os.path.join(sd, "final_repr.pkl"), "wb") as f:
        pickle.dump(reprs, f)
    json.dump({"scene": os.path.basename(sd), "frame": track_frame or "W_track",
               "instances": meta},
              open(os.path.join(sd, "final_repr_meta.json"), "w"), indent=2)
    n_rigid = sum(1 for r in reprs if r["object_type"] == "rigid")
    return len(reprs), n_rigid, [m.get("recon_resid_norm") for m in meta
                                 if m.get("recon_resid_norm") is not None]


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--batch-root")
    g.add_argument("--scene-dir")
    ap.add_argument("--no-static-skip", action="store_true",
                    help="escape hatch: ignore Masklet.motion and finalize every instance through "
                         "the full keypoint path (V3 static routing off)")
    ap.add_argument("--no-rigid-route", action="store_true",
                    help="escape hatch: ignore rigid_route.json — routed rigid instances then get "
                         "NO entry at all (their meshes were never built), so use this only "
                         "together with a Stage-1 re-run that also had --no-rigid-route")
    args = ap.parse_args()
    static_skip = not args.no_static_skip
    rigid_route = not args.no_rigid_route
    # a scene is finalizable with pass2.npz (dynamic path) OR video_input.pkl (may carry
    # static-routed / rigid-routed instances whose entries need no pass2); process_scene sorts it out.
    sds = ([args.scene_dir] if args.scene_dir else
           sorted(d for d in glob.glob(os.path.join(args.batch_root, "scene_*"))
                  if os.path.exists(os.path.join(d, "pass2.npz"))
                  or ((static_skip or rigid_route)
                      and os.path.exists(os.path.join(d, "video_input.pkl")))))
    print(f"[finalize] BATCH: {len(sds)} scenes")
    tot_inst = tot_rigid = 0; resids = []
    for sd in sds:
        try:
            res = process_scene(sd, static_skip=static_skip, rigid_route=rigid_route)
            if res is None:
                continue
            ni, nr, rs = res
            tot_inst += ni; tot_rigid += nr; resids += rs
            print(f"[finalize] {os.path.basename(sd)}: {ni} instances ({nr} rigid)", flush=True)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[finalize] {os.path.basename(sd)}: FAILED {type(e).__name__}: {e}", flush=True)
    rr = np.array([r for r in resids if r == r])     # drop nan
    print(f"[finalize] DONE: {tot_inst} instances, {tot_rigid} rigid; "
          f"rigid recon residual median {np.median(rr):.4f} (n={len(rr)})" if len(rr) else
          f"[finalize] DONE: {tot_inst} instances, {tot_rigid} rigid")


if __name__ == "__main__":
    main()
