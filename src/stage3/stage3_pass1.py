"""Stage 3 pass1 (v2): per-semantic-part keypoint candidate sampling.

REFERENCE §5.0 A: FPS (max spatial coverage) + curvature weighting (corners/joints get
more points), adaptive density per part until the A3 coverage criterion is met:
    max(surface_point -> nearest keypoint) < alpha * part_bbox_diag      (relative scale)
with a per-part soft cap. Runs on every instance regardless of rigid/non-rigid (type only
affects Stage 4 post-processing). Output = list[SCHEMA.Keypoint] (part_id = SEMANTIC part
id from Stage 2a) + per-part coverage_ok probe.

Pure numpy + trimesh (curvature); no GPU / external model. Keypoints are mesh vertices
(dense SAM 3D meshes -> vertices are a fine surface sample), expressed in the scene frame.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from SCHEMA import Keypoint  # noqa: E402


def compute_front_facing(verts_scene: np.ndarray, faces: np.ndarray,
                         object_to_scene, object_pose_cam) -> np.ndarray:
    """Per-vertex VISIBILITY mask at the reference frame t* (SAM 3D camera), via back-face
    culling: a vertex is potentially visible iff its surface normal faces the camera
    (n·viewdir < 0) AND it is in front of the camera (z > 0).

    WHY: keypoints are mesh vertices; the tracker (Stage 4) seeds a query at the pixel a vertex
    projects to and tracks that pixel over time. A BACK-FACING vertex still projects INSIDE the
    silhouette (onto the front surface), so it passes the on-mask gate but the pixel images a
    DIFFERENT (front) surface — tracking it yields a trajectory mis-attributed to this vertex's
    part. Restricting keypoint sampling to front-facing vertices removes that fabrication. This
    is GT-free and self-contained (uses only the stored mesh + SAM 3D pose). It captures
    self-occlusion (the back of the object); INTER-object occlusion is handled separately by the
    Stage 4 on-mask gate (a vertex occluded by another instance projects outside this instance's
    mask). Concave self-occlusion is not modeled (minor for DAVIS-class objects).

    Returns (V,) bool; falls back to all-True if normals/transforms are unavailable.
    NOTE: vertex normals are computed in PURE NUMPY (area-weighted face normals) so this works in
    any env — the orchestrator runs pass1 under env `sam3`, which has no trimesh (a trimesh-based
    version would silently fall back to all-True, making the whole filter a no-op).
    """
    try:
        verts_scene = np.asarray(verts_scene, float)
        faces = np.asarray(faces)
        if object_to_scene is None or object_pose_cam is None or len(faces) == 0:
            return np.ones(len(verts_scene), bool)
        # scene -> SAM 3D camera (in Path 1 this is identity: scene frame == camera frame)
        M = np.asarray(object_pose_cam, float) @ np.linalg.inv(np.asarray(object_to_scene, float))
        v_cam = (np.c_[verts_scene, np.ones(len(verts_scene))] @ M.T)[:, :3]
        # area-weighted vertex normals: face cross product (|.| = 2*area) summed onto vertices
        f0, f1, f2 = faces[:, 0], faces[:, 1], faces[:, 2]
        fn = np.cross(verts_scene[f1] - verts_scene[f0], verts_scene[f2] - verts_scene[f0])
        n_scene = np.zeros_like(verts_scene)
        np.add.at(n_scene, f0, fn); np.add.at(n_scene, f1, fn); np.add.at(n_scene, f2, fn)
        n_cam = n_scene @ M[:3, :3].T                 # sign-preserving under rotation+uniform scale
        front = (n_cam * v_cam).sum(1) < 0            # normal points back toward the camera origin
        in_front = v_cam[:, 2] > 1e-9
        vis = front & in_front
        return vis if vis.any() else np.ones(len(verts_scene), bool)  # degenerate -> don't drop all
    except Exception:
        return np.ones(len(verts_scene), bool)


try:                                   # 2026-08-15:曲率是否真的可用,必须让产物能自证
    import trimesh as _tm               # 生产 env 曾经没装它,_vertex_curvature 静默回退成全 0,
    _HAS_TRIMESH = True                 # 关键点少 ~30% 且没有任何报错(见 memory
except Exception:                       # project_pass1_env_curvature)
    _HAS_TRIMESH = False


def _vertex_curvature(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Cheap per-vertex curvature proxy in [0,1]: normalized angle defect (2pi - sum of
    incident face corner angles). Corners/cones -> high; flat -> ~0. Falls back to zeros."""
    try:
        import trimesh
        m = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        defect = np.abs(trimesh.curvature.vertex_defects(m))  # (V,)
        if not np.isfinite(defect).any() or defect.max() <= 0:
            return np.zeros(len(verts))
        d = np.nan_to_num(defect)
        return d / (np.percentile(d, 99) + 1e-9)
    except Exception:
        return np.zeros(len(verts))


def _fps_with_coverage(pts: np.ndarray, weight: np.ndarray, thresh: float,
                       max_pts: int, min_pts: int = 1):
    """Greedy farthest-point sampling on `pts`, biased by `weight` (higher -> picked
    sooner). Maintains min-distance to the selected set incrementally; stops when the
    farthest remaining point is within `thresh` (A3 coverage) or `max_pts` reached.
    Returns (selected_indices, coverage_radius)."""
    n = len(pts)
    if n == 0:
        return np.empty((0,), int), 0.0
    # seed at the highest-weight point (a corner if curvature available, else point 0)
    first = int(np.argmax(weight)) if weight.max() > 0 else 0
    selected = [first]
    min_d = np.linalg.norm(pts - pts[first], axis=1)
    while True:
        score = min_d * (1.0 + weight)          # curvature-biased coverage score
        nxt = int(np.argmax(score))
        cov = float(min_d.max())                # current coverage radius (true distance)
        if len(selected) >= min_pts and (cov < thresh or len(selected) >= max_pts):
            break
        selected.append(nxt)
        np.minimum(min_d, np.linalg.norm(pts - pts[nxt], axis=1), out=min_d)
    return np.asarray(selected, int), float(min_d.max())


def sample_pass1_keypoints(verts: np.ndarray, faces: np.ndarray, part_labels: np.ndarray,
                           instance_id: int, *, visible: "Optional[np.ndarray]" = None,
                           min_visible: int = 3, alpha: float = 0.10,
                           max_pts_per_part: int = 64, curvature: bool = True,
                           start_kp_id: int = 0,
                           group_labels: "Optional[np.ndarray]" = None):
    """Sample pass1 candidate keypoints for one instance, RESTRICTED to the visible (front-facing)
    surface (see compute_front_facing). Each SAMPLING GROUP is sampled only over its visible
    vertices; a group with fewer than `min_visible` visible vertices is unobservable at t* — it
    gets NO keypoints here and is reported in `occluded_parts` (candidate for multi-frame seeding,
    relevant for non-rigid objects; rigid objects carry occluded parts via the body SE(3)).

    SAMPLING-UNIT SEAM (2026-07-30 restructure): `group_labels` (V,) optionally overrides WHAT a
    "part" is for sampling purposes — per MOTION part (Stage-2b-early sem->motion map applied to
    the vertex labels) instead of per semantic part. The sampling ALGORITHM itself (curvature-
    weighted FPS + A3 adaptive coverage + per-part cap) is UNCHANGED, merely applied per group;
    `group_labels=None` (legacy/no 2b-early product) is bit-identical to the old per-semantic-part
    behavior. Keypoint.part_id ALWAYS stays the SEMANTIC part id of the chosen vertex (downstream
    ⑦ propagation / stage2b alignment key), whatever the grouping.

    Returns (keypoints, coverage_ok: dict[group_id, bool], occluded_parts: list[int group_id],
             next_kp_id: int)."""
    verts = np.asarray(verts, float)
    part_labels = np.asarray(part_labels).reshape(-1)
    groups = part_labels if group_labels is None else np.asarray(group_labels).reshape(-1)
    if visible is None:
        visible = np.ones(len(verts), bool)
    visible = np.asarray(visible, bool).reshape(-1)
    curv = _vertex_curvature(verts, faces) if curvature else np.zeros(len(verts))

    keypoints: list[Keypoint] = []
    coverage_ok: dict[int, bool] = {}
    occluded_parts: list[int] = []
    kp_id = start_kp_id
    for p in np.unique(groups):
        if p < 0:
            continue
        vidx = np.nonzero((groups == p) & visible)[0]         # VISIBLE vertices of this group only
        if len(vidx) < min_visible:                           # group not observable at t*
            coverage_ok[int(p)] = False
            occluded_parts.append(int(p))
            continue
        pts = verts[vidx]
        bbox_diag = float(np.linalg.norm(pts.max(0) - pts.min(0)))
        if group_labels is None:
            thresh, cap = alpha * bbox_diag, max_pts_per_part
        else:
            # DENSITY-PRESERVING seam: a motion group merges several semantic parts, so its bbox
            # diag is much larger — reusing alpha*group_diag would thin the candidate pool ~5x
            # (measured breakdance-flare 267->57) and starve pass2's visibility-aware selection.
            # Instead the group's coverage threshold/cap are DERIVED from its constituent semantic
            # parts (median part diag; cap summed), so the sampling DENSITY stays exactly the
            # legacy per-semantic-part density — only the allocation UNIT changes. The sampling
            # algorithm (_fps_with_coverage) is untouched.
            sem_in = np.unique(part_labels[vidx])
            diags = []
            for s_ in sem_in:
                spts = verts[vidx[part_labels[vidx] == s_]]
                if len(spts) >= min_visible:
                    diags.append(float(np.linalg.norm(spts.max(0) - spts.min(0))))
            thresh = alpha * (float(np.median(diags)) if diags else bbox_diag)
            cap = max_pts_per_part * max(1, len(sem_in))
        sel_local, cov = _fps_with_coverage(pts, curv[vidx], thresh, cap)
        coverage_ok[int(p)] = bool(bbox_diag > 0 and cov < thresh)
        for li in sel_local:
            keypoints.append(Keypoint(keypoint_id=kp_id, instance_id=int(instance_id),
                                      position=pts[li].astype(float),
                                      part_id=int(part_labels[vidx[li]])))
            kp_id += 1
    return keypoints, coverage_ok, occluded_parts, kp_id


def _main():
    import argparse, glob, json, pickle
    ap = argparse.ArgumentParser()
    ap.add_argument("--inst-dir", help="dir with inst_*.npz (single-scene)")
    ap.add_argument("--parts-dir", help="dir with parts_*.npz (single-scene; default = inst-dir)")
    ap.add_argument("--out", help="output keypoints pickle (single-scene)")
    ap.add_argument("--batch-root", help="dir of scene_*/ each with inst_*.npz + parts_*.npz "
                                         "(batch: pass1_keypoints.pkl saved in-place per scene)")
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--max-pts-per-part", type=int, default=64)
    ap.add_argument("--no-curvature", action="store_true",
                    help="关掉曲率加权(退回均匀 FPS)。**这不是性能开关,是 A/B 手柄** —— "
                         "2026-08-15 之前生产 env(sam3)没装 trimesh,_vertex_curvature 静默回退成"
                         "全 0,所有生产批次实际都跑在这个模式下;装上 trimesh 后默认恢复加权。"
                         "本 flag 用来复现旧产物、以及做曲率开/关的对照实验。")
    ap.add_argument("--front-facing-only", action="store_true",
                    help="V2 legacy: restrict pass1 keypoints to the t* front-facing surface. "
                         "DEFAULT is V3 whole-mesh sampling (occluded-at-t* parts get keypoints too, "
                         "linked via dense-track propagation in Stage⑦) — the main V3 upgrade.")
                    # 2026-08-15 起 PURE-2a 是【默认】(motion-part 已于 2026-08-13 退场,见 memory
                    # project_motion_part_retired)。--no-motion-groups 保留为向后兼容的 no-op,
                    # 想跑回旧的运动臂用 --motion-groups。
    ap.add_argument("--no-motion-groups", action="store_true", default=True,
                    help="PURE-2a arm: ignore BOTH stage2b_early.json (sem->motion map) AND "
                         "parts_motion_<iid>.npz (the motion-SPLIT vertex labels), so sampling AND "
                         "kp.part_id are the untouched Stage-2a semantic parts. Fixed 2026-08-13 — "
                         "before that this flag only dropped the map, leaving the split labels in "
                         "place, so the 'legacy' arm was NOT pure 2a.")
    ap.add_argument("--motion-groups", dest="no_motion_groups", action="store_false",
                    help="退回 2026-08-13 之前的运动臂(motion-part 参与部件结构)。仅用于\n"
                         "复现旧产物/对照实验;实测 62.3%% 的非刚体实例 motion_part 恒为 0。")
    args = ap.parse_args()

    def run_one(inst_dir, parts_dir, out_pkl):
        # Stage-2b-early sem->motion map (2026-07-30 restructure): when present, the SAMPLING UNIT
        # becomes the MOTION part (map applied to the per-vertex semantic labels). Absent map /
        # absent instance entry / --no-motion-groups -> legacy per-semantic-part sampling (logged).
        early = {}
        ep = os.path.join(inst_dir, "stage2b_early.json")
        if not args.no_motion_groups and os.path.exists(ep):
            try:
                early = {int(k): {int(s): int(v) for s, v in rec.get("map", {}).items()}
                         for k, rec in json.load(open(ep)).get("instances", {}).items()}
            except Exception as e:
                print(f"[pass1] {ep}: unreadable ({e}) -> semantic fallback", flush=True)
        elif not args.no_motion_groups:
            print(f"[pass1] {os.path.basename(inst_dir)}: no stage2b_early.json -> legacy "
                  f"per-SEMANTIC-part sampling", flush=True)
        all_kp, cov_report, kp_id = [], {}, 0
        for ip in sorted(glob.glob(os.path.join(inst_dir, "inst_*.npz"))):
            iid = int(os.path.splitext(os.path.basename(ip))[0].replace("inst_", ""))
            pp = os.path.join(parts_dir, f"parts_{iid}.npz")
            if not os.path.exists(pp):
                continue
            di, dp = np.load(ip), np.load(pp)
            if not args.front_facing_only:
                # V3 DEFAULT: whole-mesh sampling — the dense front-end tracks every visible surface
                # across ALL frames, and Stage⑦ links each keypoint (incl. t*-occluded) via propagation.
                visible = np.ones(len(di["verts"]), bool)
            else:
                # V2 legacy: visibility (front-facing) from the SAM 3D camera at t* — keypoints only on
                # the observable surface (see compute_front_facing).
                visible = compute_front_facing(
                    di["verts"], di["faces"],
                    di["object_to_scene"] if "object_to_scene" in di else None,
                    di["object_pose_cam"] if "object_pose_cam" in di else None)
            part_labels = np.asarray(dp["part_labels"]).reshape(-1)
            # 2b-early 若把某些语义 part 按运动拆开了(parts_motion_<iid>.npz),就用拆分后的标签:
            # 它才是 stage2b_early.json 里 `map` 的 key,也是 kp.part_id 该写的值(⑦/stage5 拿
            # kp.part_id 查那个 map)。文件不存在 = 没拆过 -> 用 2a 的原始标签,行为不变。
            # ⚠️ --no-motion-groups 必须【也】跳过这一步(2026-08-13 修):在此之前该 flag 只清空
            # `early`,拆分后的顶点标签照读不误,所以"legacy 臂"拿到的是【被运动拆分过的】语义
            # part,不是纯 2a —— 任何以它为对照的 A/B 都是脏的。
            lab_src = "2a"
            pm = os.path.join(parts_dir, f"parts_motion_{iid}.npz")
            if not args.no_motion_groups and os.path.exists(pm):
                pml = np.asarray(np.load(pm)["part_labels"]).reshape(-1)
                if len(pml) == len(part_labels):
                    part_labels = pml
                    lab_src = "2b_split"
                else:
                    print(f"[pass1] inst {iid}: parts_motion 顶点数 {len(pml)} != "
                          f"{len(part_labels)} — 忽略,用 2a 原始标签", flush=True)
            emap = early.get(iid)
            if emap is not None:
                group_labels = np.array([emap.get(int(p), 0) if p >= 0 else -1
                                         for p in part_labels])
                unit = "motion"
            else:
                group_labels, unit = None, "semantic"
                if early:                                  # map exists but not for this instance
                    print(f"[pass1] inst {iid}: no 2b-early entry -> semantic fallback", flush=True)
            kps, cov, occluded, kp_id = sample_pass1_keypoints(
                di["verts"], di["faces"], part_labels, iid, visible=visible,
                alpha=args.alpha, max_pts_per_part=args.max_pts_per_part, start_kp_id=kp_id,
                group_labels=group_labels, curvature=not args.no_curvature)
            all_kp.extend(kps)
            cov_report[iid] = {"n_parts": int(dp["n_parts"]), "n_keypoints": len(kps),
                               "sampling_unit": unit, "part_labels_src": lab_src,
                               "n_groups": int(len(np.unique(group_labels[group_labels >= 0])))
                               if group_labels is not None else int(dp["n_parts"]),
                               "curvature": (not args.no_curvature) and _HAS_TRIMESH,
                               "n_visible_verts": int(visible.sum()), "n_verts": int(len(visible)),
                               "occluded_parts": occluded,
                               "coverage_ok": cov,
                               "coverage_ok_frac": float(np.mean(list(cov.values()))) if cov else 1.0}
        os.makedirs(os.path.dirname(out_pkl) or ".", exist_ok=True)
        with open(out_pkl, "wb") as f:
            pickle.dump(all_kp, f)
        json.dump(cov_report, open(out_pkl.replace(".pkl", "_coverage.json"), "w"), indent=2)
        return all_kp, cov_report

    if args.batch_root:
        scene_dirs = sorted(d for d in glob.glob(os.path.join(args.batch_root, "scene_*"))
                            if glob.glob(os.path.join(d, "inst_*.npz")))
        print(f"[pass1] BATCH: {len(scene_dirs)} scenes under {args.batch_root}")
        for sd in scene_dirs:
            try:
                kps, _ = run_one(sd, sd, os.path.join(sd, "pass1_keypoints.pkl"))
                print(f"[pass1] {os.path.basename(sd)}: {len(kps)} keypoints", flush=True)
            except Exception as e:
                print(f"[pass1] {os.path.basename(sd)}: FAILED {type(e).__name__}: {e}", flush=True)
    else:
        if not args.inst_dir or not args.out:
            ap.error("single-scene mode needs --inst-dir and --out (or use --batch-root)")
        kps, _ = run_one(args.inst_dir, args.parts_dir or args.inst_dir, args.out)
        print(f"[pass1] {len(kps)} total keypoints -> {args.out}")


if __name__ == "__main__":
    _main()
