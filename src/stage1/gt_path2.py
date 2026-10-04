"""把 Stereo4D 的 GT 相机/尺度补进 VideoInput,让 Stage 1 走 **Path 2**(mesh 落在 W_gt)。

背景(2026-08-09):`SCHEMA.compose_scene` 的分派条件只有一句
`if video.camera_extrinsics is not None: -> _compose_path2_gt_camera`。Stereo4D 明明有 GT
外参(`gt_stereo4d.npz::extrs_w2c`),但 stage0 只喂了内参(`run_stage0.py --intrinsics`),
所以一直退到 Path 1 —— 每个实例的 mesh 各自躺在 SAM3D 自己的假想相机系里,W_mesh 不统一。
Path 2 用 `world_from_cam @ M_fix @ cam_from_obj` 把每个 mesh 放进【同一个】GT 世界系,
顺带用 GT 几何做 placement(方向取 GT 相机过 mask 质心的射线,距离取 f*W_perp/px),绕开
SAM3D 假想相机的 FOV 失配(schema.py 里记着实测 tangential error ∝ 像面偏心度, r≈0.79)。

**注入是纯内存的**,不改 `video_input.pkl`。原因:`run_link_dense.py` 和
`run_dense_frontend.py` 都有 `camera_extrinsics is not None` 的分支(靠 env `V3_PATH2_GTCAM`
兜着),落盘会让"改 stage1"变成"改全链路",不可控。

三个字段怎么来:

* `camera_extrinsics` (T,4,4) = `extrs_w2c` 求逆(Path 2 要的是 world_from_cam)。

* `gt_object_frontal_extent` {iid: float} —— **注意是 frontal(正面/剪影)口径,不是
  `gt_object_extent`**。2026-08-10 修正:一开始喂的是 `gt_object_extent`,而 Path 2 的除数
  `placed_ext` 是 mesh 的【整体】bbox 对角 √(w²+h²+d²),我给的 C 却是剪影对角 √(w²+h²)——
  差了沿视线的厚度那一维,于是每个 mesh 被按 perp/full 缩小、距离同比例拉近(实测中位
  0.906、p10 0.681,**随物体扁平度变,不是能除掉的常数**)。改喂 frontal 后 Path 2 的除数
  换成 mesh 自己【垂直于视线】的跨度,两边同口径,`rng = f*frontal/w_pix` 精确等于剪影
  隐含的深度,而且**完全不需要估物体厚度**(Stereo4D 的立体深度噪声很大,估不准)。
  下面这段标定说的就是这个 C:229 个实例实测三种:
      A 直接取 GT 点云 bbox 对角(max-min):被 Stereo4D 的立体深度噪声撑大,
        A/C 中位 1.272、p90 3.893 —— 深度跨度 / 最大横向跨度中位居然到 1.46,不合理。
      B 稳健百分位跨度(p95-p05):B/A 中位 0.715,仍散(B/C p90 1.85)。
      C **mask 像素 bbox 对角 x GT 轨迹中位深度 / f**:最稳。跨帧变异系数中位 0.192,
        且 t* 那一帧的值 ≈ 跨帧中位(比值中位 0.971)。
  取 C,并且取【跨帧中位】(t* 在 stage1 内部才定,注入时还不知道;C 本来就近似帧无关)。

* `gt_object_poses` {iid: (T,4,4)} —— **只用于诊断打分**(`_grade_against_gt` 出
  position_error / scale_ratio)。R=I,t = 该帧可见 GT 轨迹的【中位】世界坐标。注意它是
  可见表面的中位、不是体心,对人这类物体有 ~0.1m 的系统偏置,所以只能横向比较不同配置,
  不能当绝对精度。V3 链路里 `flagged` 不影响任何下游取舍(只有 V2 的 build_tracker_query
  会 drop_flagged),纯记录。
"""
from __future__ import annotations

import glob
import os

import numpy as np

MIN_TRACKS_PER_FRAME = 10      # 一帧至少这么多可见 GT 轨迹才拿来估深度/中心
MIN_MASK_PX = 200              # 与 run_stage1_fast 的 t* 门槛一致


def _c2w_from_w2c(w2c: np.ndarray) -> np.ndarray:
    """(T,3,4) world->cam  ->  (T,4,4) cam->world。"""
    T = w2c.shape[0]
    out = np.zeros((T, 4, 4), np.float64)
    R, t = w2c[:, :, :3].astype(np.float64), w2c[:, :, 3].astype(np.float64)
    Rt = np.transpose(R, (0, 2, 1))
    out[:, :3, :3] = Rt
    out[:, :3, 3] = -np.einsum("tij,tj->ti", Rt, t)
    out[:, 3, 3] = 1.0
    return out


def _fill_gaps(vals: np.ndarray, have: np.ndarray) -> np.ndarray:
    """把 have=False 的帧用【最近的有数据帧】补上(诊断量,不参与几何)。全空则返回 None。"""
    idx = np.nonzero(have)[0]
    if len(idx) == 0:
        return None
    near = idx[np.abs(np.arange(len(have))[:, None] - idx[None, :]).argmin(1)]
    return vals[near]


def inject_gt_path2(video, scene_dir: str) -> dict | None:
    """就地给 `video` 补上 Path 2 所需的 GT 字段。返回一份日志用的摘要;场景没有 GT
    (缺 gt_stereo4d.npz)或形状对不上就返回 None,调用方照旧走 Path 1。"""
    gtp = os.path.join(scene_dir, "gt_stereo4d.npz")
    if not os.path.exists(gtp):
        return None
    gt = np.load(gtp, allow_pickle=True)
    if "extrs_w2c" not in gt.files:
        return None
    w2c = np.asarray(gt["extrs_w2c"])
    T = int(np.asarray(video.frames).shape[0])
    if w2c.ndim != 3 or w2c.shape[0] != T or w2c.shape[1:] != (3, 4):
        print(f"  [path2] {os.path.basename(scene_dir)}: extrs_w2c {w2c.shape} 与 T={T} 对不上"
              f" —— 放弃注入,退回 Path 1", flush=True)
        return None

    video.camera_extrinsics = _c2w_from_w2c(w2c)

    masks_by = {int(m.instance_id): np.asarray(m.masks).astype(bool) for m in video.masklets}
    extent, poses, n_from_tracks = {}, {}, 0
    for dp in sorted(glob.glob(os.path.join(scene_dir, "dense_tracks_*.npz"))):
        iid = int(os.path.basename(dp).split("_")[-1].split(".")[0])
        masks = masks_by.get(iid)
        if masks is None or masks.shape[0] != T:
            continue
        d = np.load(dp)
        vis = np.asarray(d["vis"]).astype(bool)                 # (N,T)
        xyz_cam = np.asarray(d["xyz_cam"], np.float64)          # (N,T,3)
        xyz_world = np.asarray(d["xyz_world"], np.float64)      # (N,T,3)
        K = np.asarray(d["intrs"], np.float64)                  # (T,3,3)
        if vis.shape[1] != T:
            continue
        areas = masks.reshape(T, -1).sum(1)
        ext_per_frame, ctr, have = [], np.zeros((T, 3)), np.zeros(T, bool)
        for t in range(T):
            sel = vis[:, t]
            if sel.sum() < MIN_TRACKS_PER_FRAME:
                continue
            ctr[t] = np.median(xyz_world[:, t][sel], axis=0)
            have[t] = True
            if areas[t] < MIN_MASK_PX:
                continue
            ys, xs = np.nonzero(masks[t])
            px = float(np.hypot(xs.max() - xs.min(), ys.max() - ys.min()))
            f = 0.5 * (K[t][0, 0] + K[t][1, 1])
            z = float(np.median(xyz_cam[:, t][sel][:, 2]))
            if px > 1.0 and f > 1e-6 and z > 1e-6:
                ext_per_frame.append(px * z / f)                 # 定义 C,见模块 docstring
        if ext_per_frame:
            extent[iid] = float(np.median(ext_per_frame))
            n_from_tracks += 1
        filled = _fill_gaps(ctr, have)
        if filled is not None:
            P = np.tile(np.eye(4), (T, 1, 1))
            P[:, :3, 3] = filled
            poses[iid] = P

    # 走 FRONTAL 口径,不是 gt_object_extent(见模块 docstring)。`gt_object_extent` 保持 None:
    # 它的契约是"物体【整体】3D AABB 对角线"(Kubric/MOVi 给的就是那个),我们估不出那个量,
    # 假装能估会让 _grade_against_gt 的 scale_ratio 变成恒等于 1 的自证式数字。
    video.gt_object_frontal_extent = extent or None
    video.gt_object_poses = poses or None
    n_inst = len(video.masklets)
    print(f"  [path2] {os.path.basename(scene_dir)}: 注入 GT 外参 (T={T}), "
          f"gt_object_frontal_extent {len(extent)}/{n_inst} 个实例, "
          f"诊断用 gt_object_poses {len(poses)} 个", flush=True)
    return {"T": T, "n_masklets": n_inst, "n_extent": len(extent), "n_poses": len(poses),
            "extent_median": (float(np.median(list(extent.values()))) if extent else None)}
