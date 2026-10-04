"""Stage 2b-EARLY (2026-07-30 restructure): dense-trajectory motion analysis BEFORE pass1.

New stage order:  t4w -> stage1 -> 2a -> **2b-early** -> pass1 -> ⑦ -> pass2 -> finalize.

WHY: pass1 used to sample keypoints per SEMANTIC part (PartSAM over-segments -> on DAVIS v4
non-rigid instances 53% of semantic parts ended up with zero selected points: pigeonhole + low
visibility), and Stage 2b could only cluster motion AFTER the tracker had linked those keypoints.
The dense front-end (t4w) already tracks the whole visible surface, so the motion structure is
knowable BEFORE keypoints exist. This stage associates the dense trajectories to the Stage-2a
semantic parts at t* and runs the EXISTING Stage-2b analysis on them, producing a
semantic-part -> motion-part map that pass1 then uses as its sampling unit.

REUSE CONTRACT (user-mandated): NO Stage-2b algorithm is re-implemented here. The motion analysis
is `stage2b.stage2b_instance(P, part_id, vis, tau_rigid, min_kp, vlm_type)` — the exact function
the late 2b batch runner calls — fed with associated dense trajectories instead of keypoint
trajectories. This file only adds the SEAM: (a) dense-track -> semantic-part association at t*
(projection pairing via the SHARED 4x-fix projection `src.geometry.sam3d_project_norm`, the same
code Stage-⑦ uses), (b) a frame reorder so the rigid-fit rest frame is t* (dense tracks are not
all visible at frame 0; stage2b anchors its residual at index 0), (c) a majority-vote reduction
of per-track motion labels to per-semantic-part labels, (d) persistence.

Consumes per scene:
  dense_tracks_<iid>.npz (t4w: xyz_world W_t4w frame-invariant, uv_norm, vis)  [analysis input]
  inst_<iid>.npz         (mesh verts + t* + SAM3D own camera)                  [association]
  parts_<iid>.npz        (Stage-2a per-vertex semantic part labels)            [association]
  video_input.pkl        (masks: t* in-mask gate; masklet object_type/motion)  [gates]
Produces per scene (ADDITIVE — nothing downstream is renamed):
  stage2b_early.json     per-instance {object_type, map: {sem: motion}, gate stats, ...}
                         consumed by pass1 (sampling unit), ⑦ (propagation peer tier) and the
                         late 2b materializer (keypoint-aligned stage2b.npz motion_part).
                         NOTE: `object_type` here is THIS stage's resolution (VLM > dense gate)
                         and is used only for the sampling-time rigid collapse (map all->0). The
                         FINAL object_type keeps the legacy resolution (VLM > keypoint gate) at
                         the materializer — the dense gate proved trigger-happy on coherent
                         tracker-3D noise (MOVi all-rigid GT: 8/48 false non-rigid, e_p90 up to
                         0.25 bbox, unseparable from true non-rigid p90 ~0.09), so it is kept as
                         AUDIT evidence only.
  stage2b_early.npz      per-track calibration arrays (instance_id, sem_part, motion_part,
                         e_smooth) for threshold-recalibration evidence.

Instances SKIPPED (honestly absent from the json -> every consumer falls back to its legacy
per-semantic-part behavior): static-routed (Masklet.motion=='static' or no mesh), missing
parts/dense files, or too few associated tracks for the analysis to be meaningful.

Pure numpy/scipy (env sam3, same as the late 2b runner).
  python src/stage2/run_stage2b_dense.py --batch-root <root> --tau-rigid 0.05 --min-kp 6
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
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import stage2b  # noqa: E402  (REUSED AS-IS — the analysis itself)
import SCHEMA  # noqa: E402,F401  (registers Masklet/VideoInput for video_input.pkl unpickle)
from src.geometry import sam3d_project_norm  # noqa: E402  (SHARED 4x-fix projection, same as ⑦)
from src.stage3.stage3_pass1 import compute_front_facing  # noqa: E402  (REUSED AS-IS)

PIX_R = 0.03          # association radius (normalized px) — same default as ⑦'s t* pixel pairing
MASK_TOL = 4          # t* in-mask dilation (px) — same tolerance as ⑦'s _point_in_mask
MIN_ASSOC = 20        # below this many associated tracks the analysis is not meaningful -> skip
MAX_TRACKS = 8000     # safety cap (uniform subsample) for the O(N) robust fits
# 语义 part 的运动拆分判据(2026-08-09,见 _split_parts_by_motion):一个语义 part 内的次要运动
# 标签,需 track 数 >= SPLIT_MIN_TRACKS 且占比 >= SPLIT_MIN_FRAC 才算"显著"、才拆。
# 0.10~0.30 扫过:0.15/3 在两条路线上都收回大部分损失(≥2运动部件 20.2%->36.9% / 28.7%->40.0%),
# 再放宽到 0.10 只多 3.6 个点却更容易把噪声当运动。
SPLIT_MIN_FRAC = 0.15
SPLIT_MIN_TRACKS = 3


def _masklet_info(scene_dir):
    """iid -> {object_type, motion, masks} from video_input.pkl (tolerant)."""
    p = os.path.join(scene_dir, "video_input.pkl")
    if not os.path.exists(p):
        return {}
    try:
        with open(p, "rb") as f:
            vid = pickle.load(f)
        return {int(m.instance_id): {
            "object_type": getattr(m, "object_type", None),
            "motion": getattr(m, "motion", None),
            "masks": np.asarray(m.masks).astype(bool)} for m in vid.masklets}
    except Exception as e:
        print(f"  [warn] could not read {p}: {e}")
        return {}


def _inmask_tstar(uvn, masks, t, tol=MASK_TOL):
    """Vectorized ⑦-style in-mask gate: (n,2) normalized pixels inside the (tol-dilated) instance
    mask at frame t. Same box-dilation semantics as run_link_dense._point_in_mask."""
    if masks is None or not (0 <= t < masks.shape[0]):
        return np.zeros(len(uvn), bool)
    from scipy.ndimage import binary_dilation
    Hh, Ww = masks.shape[1:]
    dil = binary_dilation(masks[t], structure=np.ones((2 * tol + 1, 2 * tol + 1), bool))
    x = np.clip(np.round(uvn[:, 0] * Ww).astype(int), 0, Ww - 1)
    y = np.clip(np.round(uvn[:, 1] * Hh).astype(int), 0, Hh - 1)
    return dil[y, x]


def associate_tracks(dense, inst, part_labels, masks, W, H, pix_r=PIX_R):
    """SEAM (a): dense tracks -> semantic parts at t* by projection pairing.
    Mesh side: FRONT-FACING vertices only (compute_front_facing — a back-facing vertex projects
    inside the silhouette onto the front surface, so pairing against it would mislabel the track),
    projected with the SHARED 4x-fix SAM3D camera round-trip. Track side: t*-visible, finite,
    in-mask. Pair = nearest projected vertex within pix_r (cKDTree).
    Returns (track_indices (n,), sem_labels (n,), vert_indices (n,), diag dict)."""
    from scipy.spatial import cKDTree
    tstar = int(inst["reference_frame_index"])
    dw = dense["xyz_world"]
    duv = dense["uv_norm"].astype(np.float32)
    dvis = dense["vis"].astype(bool)
    vt = dvis[:, tstar] & np.isfinite(dw[:, tstar]).all(1)
    ti = np.nonzero(vt)[0]
    diag = {"tstar": tstar, "n_tracks": int(dvis.shape[0]), "n_tstar_vis": int(len(ti))}
    if len(ti) == 0:
        return np.empty(0, int), np.empty(0, int), np.empty(0, int), diag
    inm = _inmask_tstar(duv[ti, tstar], masks, tstar)
    ti = ti[inm]
    diag["n_inmask"] = int(len(ti))
    if len(ti) == 0:
        return np.empty(0, int), np.empty(0, int), np.empty(0, int), diag
    verts = inst["verts"].astype(np.float64)
    front = compute_front_facing(verts, inst["faces"],
                                 inst["object_to_scene"] if "object_to_scene" in inst else None,
                                 inst["object_pose_cam"] if "object_pose_cam" in inst else None)
    vpx, okv = sam3d_project_norm(verts, inst, W, H)
    lab = np.asarray(part_labels).reshape(-1)
    vi = np.nonzero(front & okv & (lab >= 0))[0]
    diag["n_front_verts"] = int(len(vi))
    if len(vi) == 0:
        return np.empty(0, int), np.empty(0, int), np.empty(0, int), diag
    d, j = cKDTree(vpx[vi]).query(duv[ti, tstar], k=1)
    keep = d < pix_r
    diag["n_assoc"] = int(keep.sum())
    # 第三个返回值是配对到的 MESH 顶点索引:_split_parts_by_motion 拿它当"带运动标签的种子",
    # 才能把 track 级的运动结构传播回顶点(否则只剩每 part 一个多数票标签)。
    return ti[keep], lab[vi[j[keep]]].astype(int), vi[j[keep]].astype(int), diag


def _majority_map(sem_lab, motion_lab):
    """SEAM (c): per-track motion labels -> per-semantic-part motion label by majority vote,
    then remapped to contiguous labels with base (0) preserved. PartSAM over-segments, so a
    semantic part is (sub-)part-sized and votes cleanly.

    ⚠️ 保留仅为 rigid 分支/退化路径复用;主路径已换成 _split_parts_by_motion(见其 docstring)。"""
    raw = {}
    for s in np.unique(sem_lab):
        labs, cnt = np.unique(motion_lab[sem_lab == s], return_counts=True)
        raw[int(s)] = int(labs[np.argmax(cnt)])
    # contiguous remap; `used` sorted so base 0 stays 0 when present. If NO part voted base
    # (whole object moving), the largest moving cluster (label 1, clusters are size-sorted)
    # becomes the new base 0 — pass2 then treats it as the dominant part, which it is.
    used = sorted(set(raw.values()))
    remap = {v: i for i, v in enumerate(used)}
    return {s: remap[v] for s, v in raw.items()}


def _split_parts_by_motion(part_labels, verts, sem_lab, motion_lab, vert_idx,
                           min_frac=SPLIT_MIN_FRAC, min_tracks=SPLIT_MIN_TRACKS):
    """把一个语义 part 按其内部 track 的运动标签【拆开】,而不是投多数票。

    为什么(2026-08-09):运动边界和语义边界本就不对齐——手臂可能横跨几个语义 part,一个语义
    part 也可能同时含手臂和躯干。_majority_map 每个语义 part 只能投出一个运动标签,把运动结构
    抹平了:实测"≥2 个运动部件"的实例占比,从 track 层面的 47.6%(TripoSR)/45.0%(SAM3D)
    塌到 20.2%/28.7%。而 pass1 正是按运动部件采样的,final_repr 的 motion_part 中位因此只有 1。

    做法:一个语义 part 内若有 >=2 个【显著】运动标签(各自 track 数 >= min_tracks 且占该 part
    的比例 >= min_frac),就把它拆成对应的几个子 part;拆分用该 part 内【带 track 标签的顶点】
    当种子,对该 part 的其余顶点做 3D 最近邻传播。不显著就退回多数票(与旧行为一致)。
    判据 0.15/3 实测把 ≥2 的占比拉回 36.9%/40.0%,即收回 61%/69% 的损失。

    返回 (新的 per-vertex part 标签, {新part id: 运动part id})。下游契约不变:pass1 用新标签写
    kp.part_id,stage5/⑦ 仍旧拿 kp.part_id 查这个 map —— 只是现在是一对一,无损。"""
    from scipy.spatial import cKDTree
    part_labels = np.asarray(part_labels).reshape(-1)
    verts = np.asarray(verts, float)
    out = np.full(len(part_labels), -1, np.int64)
    raw, nxt, n_split = {}, 0, 0
    for s in np.unique(part_labels):
        if s < 0:
            continue
        vsel = np.nonzero(part_labels == s)[0]
        sub = sem_lab == s
        mlab, vidx = motion_lab[sub], vert_idx[sub]
        if len(mlab) == 0:                        # 该 part 没有 track 证据 -> 归 base
            out[vsel] = nxt; raw[nxt] = 0; nxt += 1
            continue
        u, c = np.unique(mlab, return_counts=True)
        keep = [int(uu) for uu, cc in zip(u, c)
                if cc >= min_tracks and cc / len(mlab) >= min_frac]
        seed = np.isin(mlab, keep)
        if len(keep) <= 1 or seed.sum() < 2 or len(np.unique(vidx[seed])) < 2:
            out[vsel] = nxt; raw[nxt] = int(u[np.argmax(c)]); nxt += 1   # 不拆:多数票
            continue
        sv, sm = vidx[seed], mlab[seed]
        _, nn = cKDTree(verts[sv]).query(verts[vsel])                    # 最近种子顶点
        assigned = sm[nn]
        n_split += 1
        for uu in sorted(set(keep)):
            hit = vsel[assigned == uu]
            if len(hit) == 0:
                continue
            out[hit] = nxt; raw[nxt] = uu; nxt += 1
    # 与 _majority_map 相同的连续化:base(0) 若存在则仍是 0;若无 part 归 base,最大的运动簇
    # (标签 1,簇按大小排序)成为新的 base —— pass2 靠 lab==0 认 base part。
    used = sorted(set(raw.values()))
    remap = {v: i for i, v in enumerate(used)}
    return out, {k: remap[v] for k, v in raw.items()}, n_split


def process_scene(sd, tau_rigid, min_kp, knn=8):
    vip = os.path.join(sd, "video_input.pkl")
    if not os.path.exists(vip):
        print(f"  [skip] {sd}: no video_input.pkl", flush=True)
        return
    minfo = _masklet_info(sd)
    # frame size from any masklet (avoid loading full frames)
    W = H = None
    for v in minfo.values():
        H, W = v["masks"].shape[1:]
        break
    instances, npz_rows = {}, []
    for ipath in sorted(glob.glob(os.path.join(sd, "inst_*.npz"))):
        iid = int(os.path.splitext(os.path.basename(ipath))[0].replace("inst_", ""))
        info = minfo.get(iid, {})
        if info.get("motion") == "static":                 # static routing: no motion analysis
            print(f"  inst {iid}: static-routed -> skip", flush=True)
            continue
        pp = os.path.join(sd, f"parts_{iid}.npz")
        dp = os.path.join(sd, f"dense_tracks_{iid}.npz")
        if not (os.path.exists(pp) and os.path.exists(dp)):
            print(f"  inst {iid}: missing parts/dense -> skip (pass1 falls back to semantic)",
                  flush=True)
            continue
        _inz = np.load(ipath)
        inst = {k: _inz[k] for k in _inz.files}
        parts = np.load(pp)
        dense = np.load(dp)
        tr_idx, sem_lab, vert_idx, diag = associate_tracks(dense, inst, parts["part_labels"],
                                                          info.get("masks"), W, H)
        if len(tr_idx) < MIN_ASSOC:
            print(f"  inst {iid}: only {len(tr_idx)} associated tracks (<{MIN_ASSOC}) -> skip "
                  f"(pass1 falls back to semantic) | {diag}", flush=True)
            continue
        if len(tr_idx) > MAX_TRACKS:
            sel = np.linspace(0, len(tr_idx) - 1, MAX_TRACKS).astype(int)
            tr_idx, sem_lab, vert_idx = tr_idx[sel], sem_lab[sel], vert_idx[sel]
        # SEAM (b): rest frame must be t* (all associated tracks are t*-visible by construction;
        # stage2b anchors its per-frame rigid fits at index 0). Pure frame REORDER — the same
        # trajectories, so the reused analysis itself is untouched.
        T = dense["vis"].shape[1]
        tstar = diag["tstar"]
        order = [tstar] + [f for f in range(T) if f != tstar]
        P = np.nan_to_num(dense["xyz_world"][tr_idx][:, order].astype(np.float64))
        V = (dense["vis"][tr_idx][:, order].astype(bool)
             & np.isfinite(dense["xyz_world"][tr_idx][:, order]).all(2))
        # === REUSED AS-IS: the exact Stage-2b analysis (VLM-authoritative type + conservative
        # gate + per-track residual clustering), on dense trajectories instead of keypoints. ===
        res = stage2b.stage2b_instance(P, sem_lab, V, tau_rigid=tau_rigid, min_kp=min_kp,
                                       knn=knn, vlm_type=info.get("object_type"))
        # rigid 一律塌成单个部件(与旧行为一致);non-rigid 走【拆分】而非多数票。
        if res["object_type"] == "rigid":
            vlab = np.asarray(parts["part_labels"]).reshape(-1).copy()
            smap = {int(s): 0 for s in np.unique(vlab) if s >= 0}
            n_split = 0
        else:
            vlab, smap, n_split = _split_parts_by_motion(
                parts["part_labels"], inst["verts"], sem_lab, res["motion_part"], vert_idx)
            np.savez_compressed(os.path.join(sd, f"parts_motion_{iid}.npz"),
                                part_labels=vlab.astype(np.int64),
                                n_parts=np.int64(len({int(x) for x in vlab if x >= 0})),
                                n_split=np.int64(n_split))
        k_sem_motion = len(set(smap.values()))
        e = res["e_smooth"]
        rec = {"object_type": res["object_type"], "type_source": res["type_source"],
               "gate_object_type": res["gate_object_type"], "type_agrees": res["type_agrees"],
               "K_motion_tracks": int(res["K_motion"]), "K_motion_sem": int(k_sem_motion),
               "n_high": int(res["n_high"]),
               "map": {str(s): int(v) for s, v in smap.items()},
               "n_sem_parts_mapped": len(smap), "n_split_parts": int(n_split),
               "e_smooth_median": float(np.median(e)), "e_smooth_p90": float(np.percentile(e, 90)),
               "e_smooth_max": float(e.max()), "bbox_diag": float(res["bbox_diag"]),
               "assoc": diag, "tau_rigid": tau_rigid, "min_kp": min_kp}
        instances[str(iid)] = rec
        npz_rows.append((np.full(len(tr_idx), iid, np.int64), tr_idx.astype(np.int64),
                         sem_lab.astype(np.int64), res["motion_part"].astype(np.int64),
                         e.astype(np.float32)))
        print(f"  inst {iid}: {diag['n_tstar_vis']} t*-vis -> {len(tr_idx)} assoc | "
              f"type={res['object_type']}({res['type_source']}) gate={res['gate_object_type']} "
              f"K_motion tracks={res['K_motion']} sem={k_sem_motion} split={n_split} "
              f"e_med={rec['e_smooth_median']:.4f} p90={rec['e_smooth_p90']:.4f}", flush=True)
    if not instances:
        print(f"  [skip] {sd}: no analyzable instance (no json written)", flush=True)
        return
    json.dump({"scene": os.path.basename(sd), "tau_rigid": tau_rigid, "min_kp": min_kp,
               "instances": instances},
              open(os.path.join(sd, "stage2b_early.json"), "w"), indent=2)
    np.savez_compressed(os.path.join(sd, "stage2b_early.npz"),
                        instance_id=np.concatenate([r[0] for r in npz_rows]),
                        track_idx=np.concatenate([r[1] for r in npz_rows]),
                        sem_part=np.concatenate([r[2] for r in npz_rows]),
                        motion_part=np.concatenate([r[3] for r in npz_rows]),
                        e_smooth=np.concatenate([r[4] for r in npz_rows]),
                        tau_rigid=tau_rigid, min_kp=min_kp)
    print(f"  -> stage2b_early.json ({len(instances)} instances) + stage2b_early.npz", flush=True)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--scene-dir")
    g.add_argument("--batch-root")
    ap.add_argument("--tau-rigid", type=float, default=0.05)
    ap.add_argument("--min-kp", type=int, default=6)
    args = ap.parse_args()
    scenes = ([args.scene_dir] if args.scene_dir
              else sorted(glob.glob(os.path.join(args.batch_root, "scene_*"))))
    print(f"[2b-early] {len(scenes)} scene(s) tau_rigid={args.tau_rigid} min_kp={args.min_kp}",
          flush=True)
    for sd in scenes:
        print(f"[2b-early] {os.path.basename(sd)}", flush=True)
        try:
            process_scene(sd, args.tau_rigid, args.min_kp)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[2b-early] {os.path.basename(sd)}: FAILED {type(e).__name__}: {e}", flush=True)
    print("[2b-early] done", flush=True)


if __name__ == "__main__":
    main()
