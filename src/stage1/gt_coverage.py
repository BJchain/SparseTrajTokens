r"""t\* 选帧时用的"该实例在这一帧有多少条可见 GT 稠密轨迹"。

**为什么要它**:⑦(`run_link_dense`)是**严格在 t\* 配对**的 —— 把 mesh 顶点投到 t\* 像素、
和 dense track 的 t\* 像素配对,再解相似变换。t\* 那一帧没有 GT 轨迹的实例,不管 mesh 多好、
mask 多干净,都必然 0 链接。而选帧打分器只看 mask 面积/完整度/清晰度,完全不知道这件事。

实测(`stereo4d_fast50`,110 实例):**22.7% 的实例 t\* 处可见 GT 轨迹 < 30 条、11.8% 直接是
0**;同时"该实例最好那一帧"的覆盖中位是 399 条,`t*/best` 比值中位只有 0.72 —— 也就是说
覆盖是**有的**,只是选帧没往那边选。两条 SAM3D 臂上这个比例是 29.1% / 28.2%。

**两级地板(2026-08-17)**:主地板 30 不变;只有当**全片没有任何一帧达到 30** 时,才退到兜底
地板 10 再筛一次(而不是像以前那样直接放弃约束)。这一条专治"⑦ 零链接"—— 实测三批 1329 个
动态实例里 40 个(3.0%)是 ⑦ 零链接,其中 33 个正是栽在"放弃约束"上。完整依据见
`eligible_frames` 的说明(为什么不能平铺降阈、为什么地板取 10)。

用法(stage1 的两个 runner 共用;fast-path 的 t\* 是 mask 面积 argmax,SAM3D 路是打分器,
两边都只需要把不合格的帧从候选里去掉):

    cov  = load_gt_track_coverage(scene_dir)          # {iid: (T,) 每帧可见轨迹数}
    elig = eligible_frames(cov, iid, T)               # (T,) bool,或 None = 不施加约束
"""
from __future__ import annotations

import glob
import os

import numpy as np

# ⑦ 在 t* 解的是 7-DOF 相似变换,点太少解不稳;这个门槛和 fast-path 三级位姿里 R2 的
# `len(Pgt) >= 30`、以及本项目一直用的"gtpts<30"失败桶保持一致。
MIN_GT_AT_TSTAR = 30
# 二级地板(2026-08-17,用户拍板 30->10):只在"全片没有任何一帧达到 30"时才启用,详见
# eligible_frames 的说明。设 0 可关掉兜底、回到 2026-08-17 之前的行为(够不到 30 就完全弃权)。
FALLBACK_GT_AT_TSTAR = 10


def load_gt_track_coverage(scene_dir: str) -> dict[int, np.ndarray]:
    """读场景目录下所有 `dense_tracks_<iid>.npz` 的 `vis`,返回 {iid: (T,) int 每帧可见数}。
    没有 dense track 的实例不出现在字典里(= 不施加约束)。"""
    cov: dict[int, np.ndarray] = {}
    for p in sorted(glob.glob(os.path.join(scene_dir, "dense_tracks_*.npz"))):
        try:
            iid = int(os.path.basename(p).split("_")[-1].split(".")[0])
            vis = np.asarray(np.load(p)["vis"]).astype(bool)      # (N,T)
        except Exception as e:                                     # 坏文件不该拖垮 stage1
            print(f"  [gt-cov] 跳过 {os.path.basename(p)}: {type(e).__name__}: {e}", flush=True)
            continue
        if vis.ndim == 2:
            cov[iid] = vis.sum(0).astype(np.int64)
    return cov


def eligible_frames(cov: dict[int, np.ndarray] | None, instance_id: int, T: int,
                    min_tracks: int = MIN_GT_AT_TSTAR,
                    fallback_tracks: int = FALLBACK_GT_AT_TSTAR) -> np.ndarray | None:
    r"""(T,) bool:该实例在哪些帧有 >= `min_tracks` 条可见 GT 轨迹。**两级地板**,见下。

    返回 **None 表示"不施加约束"**,有三种情况都归到这里,都是诚实退回而不是硬失败:
      * 场景根本没有 GT 稠密轨迹(DAVIS/MOVi 等)   -> cov 为空
      * 这个实例没有 dense track 文件(被 min-tracks 丢掉了)
      * 这个实例**任何一帧**都达不到 `fallback_tracks`(连兜底地板都够不着)

    ⚠️ 二级兜底(2026-08-17)。在此之前,"没有任何一帧达到 30"直接返回 None = **整个放弃约束**,
    于是 t\* 纯按 mask 质量选,极可能落在一条 GT 轨迹都不可见的帧上 —— ⑦ 严格在 t\* 配对,
    那就是结构性的 0 链接,该实例的全部关键点拿到零轨迹,最后被 finalize 整个剔除。
    实测(ab_sam3d + s4d_1k_0816 + stereo4d_viz50,1329 个动态实例):⑦ 零链接 40 个(3.0%),
    其中 **33 个正是栽在这条"放弃约束"的路径上**(全片最高覆盖只有 6~29 条)。

    为什么是"两级"而不是直接把 30 降下来:平铺降阈会**拖累健康的多数派**。实测门当前生效的
    1231 个实例里,645 个(52%)存在 [10,30) 覆盖的帧;把地板一路降到 10,这些帧就进了候选集
    (占其候选集中位 3%、p90 33%),打分器可能给一个本来稳拿 >=30 覆盖的实例挑走一个只有十几条
    的帧 —— 用多数派的匹配率(门生效组 0.722)去换少数派(弃权组 0.159),不划算。
    两级的语义是:**能满足 30 的实例约束一分不松**,只有"全片都够不到 30"的那批才退到 10。

    地板取 10 的依据:⑦ 的 `_solve_T` 每条可见轨迹最多产 1 个配对(轨迹 -> 最近 mesh 顶点),
    而 `run_link_dense.MIN_PAIRS = 8`。所以地板必须 >8 才有余量 —— 取 8 的话 RANSAC 内点数
    几乎必然掉到 8 以下,相似变换永远被判 `bad`。10 救回 27/33(地板 8 是 31/33,但那 4 个多出来
    的实例本来就解不稳)。
    """
    if not cov:
        return None
    c = cov.get(int(instance_id))
    if c is None or len(c) != T:
        return None
    c = np.asarray(c)
    ok = c >= int(min_tracks)
    if ok.any():
        return ok
    fb = int(fallback_tracks or 0)
    if 0 < fb < int(min_tracks):
        ok2 = c >= fb
        if ok2.any():
            print(f"  [gt-cov] inst {instance_id}: 全片最高覆盖 {int(c.max())} 条 < {min_tracks},"
                  f"退到兜底地板 {fb}(合格帧 {int(ok2.sum())}/{T})", flush=True)
            return ok2
    return None
