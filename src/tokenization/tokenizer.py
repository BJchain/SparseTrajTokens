"""生产 tokenizer (v4):final_repr.pkl (+ gt_stereo4d.npz) -> SceneTokens。

v4(2026-08-20 定案):默认曲线族 = **事件切段三次 B 样条**(curve="bspline",splitB:逐事件帧
切段 C⁰ 角点 + 段内三次 + 阶退化 B 档,`curve_fit_v3.fit_run_split`);v3 的 PL 保留为对照臂
(curve="pl",路径逐位未动)。两族共用:D1 时间参数化 / D2 误差预算 α·bbox / D3 事件检测 /
D4 可见性 run / 场景归一化 / 静态分支 / 低运动双闸。回归基线见 PROBE_A2_RESULT_20260820.md。

以下为 v3 原头注:

已拍板决定(不改):
- 轨迹 = 分段线性 PL(复用 curve_fit),D2 自适应分段(budget=α·bbox_diag)、
  D3 事件对齐 knot(反转/冲击帧当段边界,复用 events)、D4 可见性感知(只在可见 run 拟合,
  遮挡缺口出 occlusion_gaps flag,不脑补)。α 默认 0.02(探针 A 膝点)。
- T2 速度/加速度不存导数:由重建轨迹走有限差分(探针证明解析导数更差)。
- object_type∈{rigid,non-rigid};kind 由 final_repr 实际形态判(有 se3_R=rigid、
  有 trajectories=nonrigid、都没有=static)。
- 静态 → static_points(≤4)直出,不拟合 PL / 不出 SE3。
- 场景级归一化:center=全场景可见点质心、scale=各向同性最大半径;所有坐标归一到单位球内,
  多物体同一归一化系(保关系);(center,scale) 存 SCENE header 供逆回 metric。
- rigid SE(3) 分段:旋转用 rotvec(log(R), so3)PL、平移用归一化 t' PL(见下)。

纯 CPU (numpy/scipy)。
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from typing import List, Optional

import numpy as np
from scipy.spatial.transform import Rotation

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import curve_fit as cf          # noqa: E402
import curve_fit_v3 as cf3      # noqa: E402
import events as ev             # noqa: E402
from schema import (            # noqa: E402
    KeypointTrack, ObjectTokens, PLSegment, SceneTokens, SE3Track, SplineRun,
)

_SRC = os.path.dirname(_HERE)
if os.path.dirname(_SRC) not in sys.path:
    sys.path.insert(0, os.path.dirname(_SRC))
from src.geometry import static_point_residual_px  # noqa: E402  (与生产前端同一份可观测性判据)

DEFAULT_ALPHA = 0.02
DEFAULT_CURVE = "bspline"       # v4 生产默认;"pl" = v3 对照臂(逐位保留)
DEFAULT_RIGID_MODE = "joint"    # 刚体路径(2026-08-22,§2.2):
                                #   "joint" = 旋转/平移**共节点**联合拟合,误差预算落在
                                #             **重建 keypoint 空间**(定案规格)
                                #   "split" = 旧口径对照臂:两流各自独立拟合、各用各的相对预算
                                #             (旋转的容差在弧度空间,与任务无关;且喂的是会翻轴的
                                #              绝对 rotvec —— 实测 MOVi 25% 实例有 2π 级跳变)
RIGID_PARAM = "centroid"        # 2026-09-14:刚体参数化以**参考帧质心**为旋转中心。
                                #   ref_norm = (ref − mean(ref))/scale = 物体自身形状(质心为原点);
                                #   trans    = 归一化**质心轨迹** c(t) = R(t)·mean(ref) + t(t)(与 keypoint 均值精确一致)。
                                #   旧口径 "origin" 以世界原点为旋转中心(trans = 归一化 se3_t):物体自转一点点,t 就要补偿一大段,
                                #   token 侧只读平移流时位移失配 82%(B 集 Gate 0 实测),车辆几乎全被剔除。两种口径重建出的
                                #   keypoint 点集完全相同(p = R·ref_norm + trans),只是分解不同;旧 pkl 无 rigid_param 属性 = "origin"。
RIGID_REL_CAP = None            # 刚体系数相对上限 n ≤ max(4, ceil(rel_cap·m)),防近插值在帧间过冲
RIGID_ROBUST = False            # 刚体判据改稳健档:keypoint p95 ≤ ε 且 max ≤ 3ε(§9 备选)
RIGID_INTERIOR_MULT = None      # 帧间过冲上限 = 该倍数 × ε(判据只在数据点上判,管不住帧间)。
                                # **实测不划算,默认关**:S4D 200 场景上 5ε 档把达标率 94.5%→79.3%、
                                # p90 误差 0.020→0.071,只把节点污染(>0.5bbox)从 2.11% 降到 1.92%。
                                # 真正治住极端值的是 _cubic_by_tokens 的评分修复(max 125.9→41.3,零代价)。
                                # 留作旋钮:S4D 刚体的重尾过冲根子在上游 SE(3) 噪声(23% 实例 >10%bbox)
MOTION_THRESH = 0.05            # 世界位移闸(米)。**单独用它是不够的** —— 见下。
RESID_TAU_ABS_PX = 1.0          # 可观测性闸的绝对下限(与 gt_dense_frontend 同值)
RESID_ONIMG_MIN = 0.8           # 坐标系自检:投影落在画面内的比例低于此 -> 判定相机与 W_track
                                # 不同系,本闸整个弃权(保守:不降级),绝不拿错配的相机做判断


def _scene_resid_thr(scene_dir):
    """本场景的可观测性阈值(px)。优先复用 gt_dense_frontend 已算好并记在 gt_dense_stats.json
    里的自适应阈值(= max(绝对下限, 2×本场景残差地板)),保证前端与 tokenizer 口径**同源**;
    没有该文件(非 GT 前端的根)则退回绝对下限。"""
    p = os.path.join(scene_dir, "gt_dense_stats.json")
    if os.path.exists(p):
        try:
            import json
            v = json.load(open(p)).get("resid_thr_px")
            if v:
                return float(v)
        except Exception:
            pass
    return RESID_TAU_ABS_PX


def _obs_resid_px(traj, vis, K, w2c, W, H):
    """实例的运动可观测性残差(px)= 逐 keypoint 拟合单个静止世界点后的重投影残差中位数。
    返回 None 表示"无法判断"(相机缺失/帧数不符/坐标系自检不过/可见帧太少)-> 调用方须弃权。

    为什么需要它:世界位移/速度这类判据在 Stereo4D GT 上被米级深度噪声污染(静止的树世界
    "速度" 0.24m/s),而本判据不涉及深度,同一棵树只有 0.40px。反过来,原地往复/摆动的运动
    世界位移≈0(LS 斜率与 std 都盲),本判据看得见(实测 person 5.2px)。两者互补,取 AND。
    """
    if K is None or w2c is None:
        return None
    w2c = np.asarray(w2c, float)
    if w2c.ndim != 3 or w2c.shape[0] != traj.shape[1]:
        return None
    K = np.asarray(K, float)
    if W is None or H is None:
        W, H = 2.0 * float(K[0, 2]), 2.0 * float(K[1, 2])
    R, tv = w2c[:, :, :3], w2c[:, :, 3]
    cam = np.einsum("tij,mtj->mti", R, np.nan_to_num(traj)) + tv[None]
    z = cam[..., 2]
    zs = np.where(np.abs(z) > 1e-9, z, 1.0)
    uv = np.stack([K[0, 0] * cam[..., 0] / zs + K[0, 2],
                   K[1, 1] * cam[..., 1] / zs + K[1, 2]], -1)
    onimg = (z > 1e-6) & (uv[..., 0] >= 0) & (uv[..., 0] < W) & (uv[..., 1] >= 0) & (uv[..., 1] < H)
    nv = int(vis.sum())
    if nv == 0 or float((onimg & vis).sum()) / nv < RESID_ONIMG_MIN:
        return None                                   # 坐标系不匹配 -> 弃权,不降级
    X0 = np.array([np.median(np.nan_to_num(traj[m][vis[m]]), axis=0) if vis[m].any()
                   else np.zeros(3) for m in range(traj.shape[0])])
    r = static_point_residual_px(uv, vis & onimg, w2c, K, X0=X0)
    r = r[np.isfinite(r)]
    return float(np.median(r)) if len(r) else None


def _load_imsize(scene_dir):
    """(W,H)。GT npz 带 imw/imh 就用它,否则 None(由 K 的主点推)。"""
    import glob as _glob
    for p in ([os.path.join(scene_dir, n) for n in ("gt_stereo4d.npz", "gt_movi.npz")]
              + sorted(_glob.glob(os.path.join(scene_dir, "gt_*.npz")))):
        if not os.path.exists(p):
            continue
        try:
            d = np.load(p, allow_pickle=True)
        except Exception:
            continue
        if "imw" in d.files and "imh" in d.files:
            return float(d["imw"]), float(d["imh"])
        return None, None
    return None, None


def _is_lowmotion(traj, vis, mag, K, w2c, imw, imh, resid_thr):
    """两闸取 AND 才判"无运动可编码"。两个闸是互补的,缺一不可:
      · 世界位移闸 mag:看得见【沿视线方向】的平动 —— 那恰是可观测性判据的盲区;
        但它被深度噪声污染(静止的树 mag 超阈)。
      · 可观测性闸 resid:对深度噪声免疫;但看不见沿视线的运动。
    任一闸"无法判断"(None)一律弃权 -> 保留为动态。**单侧安全**:宁可多花 token 编码噪声,
    也绝不静默删掉真运动(后者不可逆且下游无从察觉)。
    """
    if mag is None:
        return True          # 无任何观测(没有 >=2 可见帧的 keypoint):没有运动可编码,交给
                             # _encode_static 的 never_visible 诚实分支(0 点,不脑补)。保持旧行为。
    if mag > MOTION_THRESH:
        return False
    resid = _obs_resid_px(traj, vis, K, w2c, imw, imh)
    return resid is not None and resid < resid_thr


# ================================================================== 工具
def _kind_of(inst: dict) -> str:
    """kind 由实际形态判定(决定 #3)。object_type(Qwen)不参与分流。"""
    if "se3_R" in inst:
        return "rigid"
    if "trajectories" in inst:
        return "nonrigid"
    return "static"


def _valid_ref_rows(inst: dict) -> np.ndarray:
    """(M,) bool:rigid 的哪些 ref_positions 行是真实坐标而不是 (0,0,0) 占位。

    存量产物守卫(2026-08-16)。2026-08-16 之前的 finalize 把"选中点在参考帧没有观测"直接写成
    (0,0,0),而坐标系是 W_track(公制,原点 = 首帧相机):R(t)@0 + t(t) 会重建出一个离物体
    12-17 m 的假关键点,并把 `compute_scene_norm` 的场景尺度一起撑大。实测旧批次 rigid ref 行
    27.3% 中招。修好的产物里这个掩码恒为全 True(= no-op),它只为旧 scene 目录 / 已生成的
    token 数据兜底。判据安全:W_track 原点是首帧相机位置,真实表面点落在那儿的概率为 0。"""
    ref = np.asarray(inst["ref_positions"], float)
    return np.abs(ref).sum(1) > 0


def _rigid_world_traj(inst: dict):
    """rigid: 从 SE(3) 重建逐 keypoint 世界轨迹 (M,T,3) + vis (M,T)。约定同探针 A。
    占位行按 `_valid_ref_rows` 剔除(旧产物守卫);全是占位则返回空轨迹 = 无观测。"""
    keep = _valid_ref_rows(inst)
    ref = np.asarray(inst["ref_positions"], float)[keep]      # (M',3)
    R = np.asarray(inst["se3_R"], float)                # (T,3,3)
    t = np.asarray(inst["se3_t"], float)                # (T,3)
    traj = (np.einsum("tij,mj->tmi", R, ref) + t[:, None, :]).transpose(1, 0, 2)
    ok = np.asarray(inst["se3_ok"], bool)
    vis = np.broadcast_to(ok[None, :], traj.shape[:2]).copy()
    return traj.astype(float), vis


def _instance_motion_mag(traj, vis):
    """median over tracks of ||per-track world-pos std over visible frames||。<None if 无>=2 可见帧>。"""
    mags = []
    for m in range(traj.shape[0]):
        v = vis[m]
        if v.sum() >= 2:
            mags.append(float(np.linalg.norm(traj[m, v].std(axis=0))))
    return float(np.median(mags)) if mags else None


def _cloud_diag(P):
    """点云各向同性尺度(bbox 对角线)。"""
    if len(P) < 2:
        return 0.0
    return float(np.linalg.norm(P.max(0) - P.min(0)))


def _knot_type_at(frame, rev_set, imp_set, is_endpoint):
    if is_endpoint:
        return "endpoint"
    if frame in imp_set:
        return "impact"
    if frame in rev_set:
        return "reversal"
    return "smooth"


def _pl_to_segment(pl, T, rev_set, imp_set, f0, f1):
    """把 curve_fit.fit_pl_events 的 {vf,vp} 包成一个 PLSegment(逐顶点 knot_type)。"""
    vf, vp = pl["vf"], pl["vp"]
    kts = []
    for j, f in enumerate(vf):
        is_end = (j == 0) or (j == len(vf) - 1)
        kts.append(_knot_type_at(int(f), rev_set, imp_set, is_end))
    denom = max(T - 1, 1)
    return PLSegment(u_start=f0 / denom, u_end=f1 / denom,
                     vframes=np.asarray(vf, int), verts=np.asarray(vp, float),
                     knot_types=kts)


def _fit_stream_runs(frames_all, pos_all, vis_all, budget, T, event_frames_fn):
    """对一个 (可见性分段) 流做 D3/D4 PL:每个可见 run 出一个 PLSegment;缺口出 occlusion_gaps。
    event_frames_fn(fr,pp) -> (rev_list, imp_list) 在该 run 上算 knot 先验。"""
    runs = ev.visible_runs(vis_all)
    segments, gaps = [], []
    denom = max(T - 1, 1)
    for (a, b) in runs:
        fr = frames_all[a:b + 1]
        pp = pos_all[a:b + 1]
        f0, f1 = int(fr[0]), int(fr[-1])
        if len(fr) >= 3:
            rev, imp = event_frames_fn(fr, pp)
        else:
            rev, imp = [], []
        knots = sorted(set(rev) | set(imp))
        pl = cf.fit_pl_events(fr, pp, budget, knots)
        segments.append(_pl_to_segment(pl, T, set(rev), set(imp), f0, f1))
    # 遮挡缺口(相邻 run 之间)
    for (a0, b0), (a1, b1) in zip(runs[:-1], runs[1:]):
        gaps.append((float(frames_all[b0]) / denom, float(frames_all[a1]) / denom))
    vis_summary = dict(n_visible=int(vis_all.sum()), T=int(len(vis_all)),
                       frac=float(vis_all.mean()), n_runs=len(runs))
    return segments, gaps, vis_summary


def _finite_or_none(x):
    """None/NaN/inf 一律 None —— 宁可标"没有质量信息",也不发一个假的 0。"""
    if x is None:
        return None
    v = float(x)
    return v if np.isfinite(v) else None


def _unwrap_rotvec(phi):
    """把逐帧旋转向量展开成**连续**路径。

    同一个旋转的所有旋转向量表示是 {(θ+2πk)·n};`as_rotvec` 只返回 θ∈[0,π] 的那一个,
    于是物体转过 π 时轴会整个翻向、信号跳 2π。样条穿过这种阶跃就是灾难 —— 实测 MOVi
    25.0% 的刚体实例中招,最坏一处物体一帧没转(0.000 rad)而信号跳了 6.283 rad。
    这里逐帧在候选 {(θ+2πk)·n} 里挑离上一帧最近的那个,取得的旋转完全相同,只是表示连续。
    """
    out = np.array(phi, dtype=float, copy=True)
    if len(out) < 2:
        return out
    two_pi = 2.0 * np.pi
    ks = np.arange(-4, 5)
    for i in range(1, len(out)):
        prev = out[i - 1]
        nrm = float(np.linalg.norm(out[i]))
        if nrm < 1e-9:                      # 恒等旋转:轴任意,取沿上一帧方向最近的 2π 倍数
            pn = float(np.linalg.norm(prev))
            out[i] = (prev / pn) * (two_pi * round(pn / two_pi)) if pn > 1e-9 else out[i] * 0.0
            continue
        axis = out[i] / nrm
        cands = (nrm + two_pi * ks)[:, None] * axis[None, :]
        out[i] = cands[int(np.argmin(np.linalg.norm(cands - prev, axis=1)))]
    return out


def _kp_criterion(ref_norm, R0, rad, budget, robust=False, interior_mult=None):
    """§2.2 的误差预算:判在**重建 keypoint 空间**,而不是各流自己的空间。

    联合拟合的 P 每行是 [平移(3), rad·φ(3)],φ 是相对 run 首帧姿态 R0 的旋转向量。
    还原成 keypoint:p_j(u) = R0·exp(φ(u))·ref_j + trans(u);判据 =
    max_{u,j} ‖拟合出来的 p_j − 原值 p_j‖ ≤ ε(与非刚体同源同单位)。返回 (ok, err_fn)。
    """
    def _pts(V):
        V = np.atleast_2d(np.asarray(V, float))
        Rm = np.einsum("ij,tjk->tik", R0,
                       Rotation.from_rotvec(V[:, 3:] / rad).as_matrix())
        return np.einsum("tij,mj->tmi", Rm, ref_norm) + V[:, :3][:, None, :]

    def _dist(tck, u, P):
        return np.linalg.norm(_pts(cf3._eval(tck, u, 0)) - _pts(P), axis=-1).max(axis=1)

    def err(tck, u, P):
        return float(np.max(_dist(tck, u, P)))

    def _interior_ok(tck, u, P):
        """帧间约束:半帧处的曲线不得偏离"相邻两帧的线性内插"超过 interior_mult·ε。
        判据只在数据点上判时,样条可以在帧上达标却在帧间飙到 100× bbox(节点恰落在帧间,
        于是**垃圾直接进 token**)。这条把"误差上界"从数据点扩展到整条曲线。"""
        if interior_mult is None or len(u) < 2:
            return True
        um = 0.5 * (np.asarray(u, float)[:-1] + np.asarray(u, float)[1:])
        tgt = _pts(P)
        lin = 0.5 * (tgt[:-1] + tgt[1:])
        mid = _pts(cf3._eval(tck, um, 0))
        return float(np.max(np.linalg.norm(mid - lin, axis=-1))) <= interior_mult * budget + 1e-12

    def ok_max(tck, u, P):
        return err(tck, u, P) <= budget + 1e-12 and _interior_ok(tck, u, P)

    def ok_robust(tck, u, P):   # §9 稳健档:不为单帧噪声把样条逼成近插值
        e = _dist(tck, u, P)
        return (np.percentile(e, 95) <= budget + 1e-12) and (e.max() <= 3 * budget + 1e-12) \
            and _interior_ok(tck, u, P)

    def score(tck, u, P):
        """兜底排序:数据点误差与帧间过冲都按各自预算归一,取较大者。越小越好。"""
        a = err(tck, u, P) / max(budget, 1e-12)
        if interior_mult is None or len(u) < 2:
            return a
        um = 0.5 * (np.asarray(u, float)[:-1] + np.asarray(u, float)[1:])
        tgt = _pts(P)
        mid = _pts(cf3._eval(tck, um, 0))
        b = float(np.max(np.linalg.norm(mid - 0.5 * (tgt[:-1] + tgt[1:]), axis=-1)))
        return max(a, b / max(interior_mult * budget, 1e-12))

    return (ok_robust if robust else ok_max), err, score


def _spline_run_from_fit(run, T, rev_set, imp_set, cols=None, val_scale=1.0, rot_ref=None):
    """cf3.fit_run / fit_run_split 的结果 -> SplineRun。逐节点 knot_type:run 首尾=endpoint、
    段边界(事件帧,两侧共位双节点)=impact/reversal(impact 优先,同 _knot_type_at)、其余 smooth。

    cols/val_scale(§2.2 联合拟合用):从 6 维联合解里取出一条流。cols=(0,1,2) 取平移、
    (3,4,5) 取旋转并乘 val_scale=1/rad 换回弧度。两条流因此**共享同一组节点**。"""
    parts = run.get("parts", [run])
    fnodes, nodes, kts, degs, tcks = [], [], [], [], []
    n_parts = len(parts)
    for j, part in enumerate(parts):
        fp = np.asarray(part["fpos"], float)
        pp = np.asarray(part["p"], float)
        if cols is not None:
            pp = pp[:, list(cols)] * val_scale
        k = int(part["k"])
        if part["tck"] is not None:
            t, c, kk = part["tck"]
            sel = list(cols) if cols is not None else range(len(c))
            tcks.append((np.asarray(t, float),
                         [np.asarray(c[d], float) * val_scale for d in sel], int(kk),
                         int(part["f0"]), int(part["f1"])))
        for i in range(len(fp)):
            is_run_end = (j == 0 and i == 0) or (j == n_parts - 1 and i == len(fp) - 1)
            is_boundary = (i == 0 and j > 0) or (i == len(fp) - 1 and j < n_parts - 1)
            if is_boundary:
                f = int(round(fp[i]))
                kt = "impact" if f in imp_set else ("reversal" if f in rev_set else "smooth")
            elif is_run_end:
                kt = "endpoint"
            else:
                f = int(round(fp[i]))
                kt = _knot_type_at(f, rev_set, imp_set, False) if abs(fp[i] - round(fp[i])) < 1e-9 else "smooth"
            fnodes.append(float(fp[i])); nodes.append(pp[i]); kts.append(kt); degs.append(k)
    denom = max(T - 1, 1)
    return SplineRun(
        u_start=run["f0"] / denom, u_end=run["f1"] / denom,
        fnodes=np.asarray(fnodes, float), nodes=np.asarray(nodes, float),
        knot_types=kts, degrees=np.asarray(degs, int), tcks=tcks,
        budget_miss=bool(run.get("budget_miss", False)), max_err=float(run.get("max_err", 0.0)),
        rot_ref=(None if rot_ref is None else np.asarray(rot_ref, float)))


def _fit_stream_runs_spline(frames_all, pos_all, vis_all, budget, T, event_frames_fn):
    """split-B 样条版流拟合(v4 生产):每个可见 run 一个 SplineRun;缺口 occlusion_gaps 与
    vis_summary 逻辑与 PL 版逐位相同(独立实现以保 PL 路径零改动)。"""
    runs = ev.visible_runs(vis_all)
    segments, gaps = [], []
    denom = max(T - 1, 1)
    for (a, b) in runs:
        fr = frames_all[a:b + 1]
        pp = pos_all[a:b + 1]
        if len(fr) >= 3:
            rev, imp = event_frames_fn(fr, pp)
        else:
            rev, imp = [], []
        run = cf3.fit_run_split(fr, pp, budget, sorted(set(rev) | set(imp)), mode="B")
        segments.append(_spline_run_from_fit(run, T, set(rev), set(imp)))
    for (a0, b0), (a1, b1) in zip(runs[:-1], runs[1:]):
        gaps.append((float(frames_all[b0]) / denom, float(frames_all[a1]) / denom))
    vis_summary = dict(n_visible=int(vis_all.sum()), T=int(len(vis_all)),
                       frac=float(vis_all.mean()), n_runs=len(runs))
    return segments, gaps, vis_summary


def _stream_fitter(curve):
    if curve == "pl":
        return _fit_stream_runs
    if curve == "bspline":
        return _fit_stream_runs_spline
    raise ValueError(f"unknown curve family: {curve}")


# ================================================================== 归一化
def compute_scene_norm(reprs):
    """场景级归一化:center=全场景可见点质心、scale=各向同性最大半径(点到质心)。
    可见点 = 动态实例可见轨迹点 + 静态实例 ref_positions。"""
    pts = []
    for inst in reprs:
        kind = _kind_of(inst)
        if kind == "static":
            rp = inst.get("ref_positions")
            if rp is not None:
                pts.append(np.asarray(rp, float))
        elif kind == "nonrigid":
            traj = np.asarray(inst["trajectories"], float)
            vis = np.asarray(inst["visibility"], bool)
            if vis.any():
                pts.append(traj[vis])
        else:  # rigid
            traj, vis = _rigid_world_traj(inst)
            if vis.any():
                pts.append(traj[vis])
    if not pts:
        return np.zeros(3), 1.0
    allpts = np.concatenate([p.reshape(-1, 3) for p in pts], axis=0)
    center = allpts.mean(0)
    rad = np.linalg.norm(allpts - center, axis=1)
    scale = float(rad.max()) if len(rad) else 1.0
    if scale < 1e-8:
        scale = 1.0
    return center.astype(float), scale


# ================================================================== 每实例编码
def _static_points_world(inst):
    """静态实例的 ≤4 代表点(世界系)。
    - 真静态(无 traj/无 se3):直接用 ref_positions(已验证 0% 退化)。
    - 被 drop_lowmotion 降级的动态实例:ref_positions 常是 [0,0,0] 占位(失败/全遮挡重建),
      改用实际可见帧的代表点(每 keypoint 可见位置均值);全程不可见 → 返回空(诚实,0 点)。
    返回 (n<=4, 3) world;never_visible bool。"""
    kind = _kind_of(inst)
    if kind == "static":
        return np.asarray(inst["ref_positions"], float)[:4], False
    if kind == "nonrigid":
        traj = np.asarray(inst["trajectories"], float)
        vis = np.asarray(inst["visibility"], bool)
    else:
        traj, vis = _rigid_world_traj(inst)
    pts = []
    for m in range(traj.shape[0]):
        if vis[m].any():
            pts.append(traj[m, vis[m]].mean(0))
        if len(pts) >= 4:
            break
    if not pts:
        return np.zeros((0, 3)), True         # 全程不可见:无观测,不脑补
    return np.asarray(pts, float), False


def _part_labels(inst, n):
    """逐 keypoint 的部件标签(送进 token 的那个)。优先 `part_id`(Stage-2a 语义部件,
    2026-08-13 新增),回退 `motion_part`(Stage-2b 运动分解;纯 2a 臂上它恒为 0)。
    两者都没有(旧产物)则全 0。刚体的 part_id 在 pass2 已强制为 0。"""
    mp = inst.get("part_id")
    if mp is None:
        mp = inst.get("motion_part")
    if mp is None:
        return np.zeros(n, int)
    return np.asarray(mp).reshape(-1).astype(int)


def _encode_static(inst, center, scale):
    rp, never_vis = _static_points_world(inst)
    static_pts = (rp - center) / scale if len(rp) else rp
    mp = inst.get("part_id")
    if mp is None:
        mp = inst.get("motion_part")   # 静态条目两者皆无 -> 保持 parts=[] / n_parts=0 的旧语义
    parts = sorted(set(int(x) for x in np.asarray(mp).ravel())) if mp is not None else []
    summ = dict(n_parts=len(parts), part_ids=parts)
    if never_vis:
        summ["never_visible"] = True          # 诚实 flag:无观测占位实例
    return ObjectTokens(
        instance_id=int(inst["instance_id"]), label=str(inst.get("label", "?")),
        object_type=str(inst["object_type"]), kind="static",
        motion_part_summary=summ,
        static_points=static_pts.astype(float))


def _encode_nonrigid(inst, center, scale, alpha, T, curve=DEFAULT_CURVE):
    traj = np.asarray(inst["trajectories"], float)     # (M,Tn,3) world
    vis = np.asarray(inst["visibility"], bool)         # (M,Tn)
    Tn = traj.shape[1]
    traj_n = (traj - center) / scale                   # 归一化
    # object bbox_diag(归一化系,所有可见点)
    allvis = traj_n[vis]
    bd = _cloud_diag(allvis) if len(allvis) >= 2 else 0.0
    budget = alpha * bd if bd > 0 else 0.0
    kp_ids = inst.get("keypoint_ids", list(range(traj.shape[0])))
    mp = _part_labels(inst, traj.shape[0])
    frames_all = np.arange(Tn)
    tracks = []
    for m in range(traj.shape[0]):
        def ev_fn(fr, pp, _bd=bd):
            rev = ev.detect_reversals(fr, pp, _bd) if _bd > 0 else []
            imp = ev.detect_impacts(fr, pp, _bd) if _bd > 0 else []
            return rev, imp
        segs, gaps, vsum = _stream_fitter(curve)(frames_all, traj_n[m], vis[m], budget, T, ev_fn)
        tracks.append(KeypointTrack(
            keypoint_id=int(kp_ids[m]), part_id=int(mp[m]),
            pl_segments=segs, occlusion_gaps=gaps, visibility_summary=vsum))
    parts = sorted(set(int(x) for x in mp.ravel()))
    return ObjectTokens(
        instance_id=int(inst["instance_id"]), label=str(inst.get("label", "?")),
        object_type=str(inst["object_type"]), kind="nonrigid",
        motion_part_summary=dict(n_parts=len(parts), part_ids=parts),
        keypoint_tracks=tracks)


def _fit_rigid_joint(frames_all, R, trans, ok, ref_norm, budget, T, ev_fn,
                     rel_cap=None, robust=False, interior_mult=None, max_coef=None):
    """§2.2 刚体路径:旋转 + 平移**共节点**联合拟合,误差预算落在**重建 keypoint 空间**。

    与旧口径("split")的三点差别:
    1. 旋转不再喂**绝对** rotvec,而是 φ(u)=log(R_{f0}ᵀR(u)) —— 相对 run 首帧、并做 2π 展开,
       信号连续。旧口径喂的绝对 rotvec 在 π 附近翻轴,样条是在穿越 2π 阶跃。
    2. 两条流拼成一条 6 维曲线一起拟合(P=[平移, rad·φ]),因此**共用同一组节点**。
       rad = ref 点云最大半径,把弧度换算成长度,使 6 维里两半量纲一致、s 的残差和 ≈ keypoint 平方误差。
    3. 达标判据 = max_{u,j}‖R0·exp(φ(u))·ref_j + trans(u) − 原值‖ ≤ ε,ε=α·bbox_diag
       (与非刚体同源同单位)。旧口径是两流各判各的、旋转的容差还在弧度空间 —— 既误报也漏报。
    返回 (rot_segments, trans_segments, occlusion_gaps, visibility_summary)。
    """
    rad = float(np.linalg.norm(ref_norm, axis=1).max()) if len(ref_norm) else 0.0
    rad = rad if rad > 1e-9 else 1.0
    runs = ev.visible_runs(ok)
    rot_segs, trans_segs, gaps = [], [], []
    denom = max(T - 1, 1)
    for (a, b) in runs:
        fr = frames_all[a:b + 1]
        R0 = R[fr[0]]
        phi = _unwrap_rotvec(
            Rotation.from_matrix(np.einsum("ij,tjk->tik", R0.T, R[fr])).as_rotvec())
        P6 = np.concatenate([trans[fr], rad * phi], axis=1)
        rev, imp = ev_fn(fr, None) if len(fr) >= 3 else ([], [])
        crit, errf, scorer = _kp_criterion(ref_norm, R0, rad, budget, robust=robust,
                                           interior_mult=interior_mult)
        run = cf3.fit_run_split(fr, P6, budget, sorted(set(rev) | set(imp)),
                                mode="B", criterion=crit, err_fn=errf, rel_cap=rel_cap,
                                search=("scan" if interior_mult is not None else "bisect"),
                                score=scorer, **({"M": int(max_coef)} if max_coef else {}))
        rs, ims = set(rev), set(imp)
        trans_segs.append(_spline_run_from_fit(run, T, rs, ims, cols=(0, 1, 2)))
        rot_segs.append(_spline_run_from_fit(
            run, T, rs, ims, cols=(3, 4, 5), val_scale=1.0 / rad,
            rot_ref=Rotation.from_matrix(R0).as_rotvec()))
    for (a0, b0), (a1, b1) in zip(runs[:-1], runs[1:]):
        gaps.append((float(frames_all[b0]) / denom, float(frames_all[a1]) / denom))
    vsum = dict(n_visible=int(ok.sum()), T=int(len(ok)),
                frac=float(ok.mean()), n_runs=len(runs))
    return rot_segs, trans_segs, gaps, vsum


CAMERA_MAX_COEF = 48              # 相机 run 的系数上限(物体默认 12):长路径 + 10 cm 绝对预算下 12 个系数会 budget_miss
CAMERA_ABS_BUDGET_M = 0.10        # 相机平移流的绝对误差上界(米)。相机路径可长达 50 m 而场景尺度只有 1~2 m,
                                  # 按 α·bbox 的相对预算会放出 ~1 m 的光心误差(实测 max 1.03 m);相机 GT 无深度噪声,
                                  # 用绝对预算 10 cm(远低于 QA 的 0.5 m 门与 1.3× 选项间距)。


def _encode_rigid(inst, center, scale, alpha, T, curve=DEFAULT_CURVE,
                  rigid_mode=DEFAULT_RIGID_MODE, abs_budget_m=None, max_coef=None):
    R = np.asarray(inst["se3_R"], float)               # (Tn,3,3)
    t = np.asarray(inst["se3_t"], float)               # (Tn,3)
    keep = _valid_ref_rows(inst)                       # 旧产物占位行守卫,见 _valid_ref_rows
    if not keep.any():                                 # 全是占位:没有几何可编码,走诚实空条目
        return _encode_static(inst, center, scale)
    ref = np.asarray(inst["ref_positions"], float)[keep]   # (M',3) 参考帧世界系坐标
    ok = np.asarray(inst["se3_ok"], bool)              # (Tn,)
    Tn = R.shape[0]
    c0 = ref.mean(0)                                   # 参考帧质心(世界系,米)
    ref_norm = (ref - c0) / scale                      # 物体自身形状(质心为原点);p' = R@ref_norm + trans(见 RIGID_PARAM)
    cen = np.einsum("tij,j->ti", R, c0) + t            # 质心轨迹 c(t) = R(t)·c0 + t(t) = keypoint 均值(精确)
    trans = (cen - center) / scale                     # (Tn,3) 归一化质心轨迹
    # rotvec(so3):仅在 se3_ok 帧有效,无效帧填 0(会被可见性 run 排除)
    rotvec = np.zeros((Tn, 3))
    if ok.any():
        rotvec[ok] = Rotation.from_matrix(R[ok]).as_rotvec()
    # 尺度:平移用重建 keypoint 世界 bbox(归一化系);旋转用 rotvec 云 diag(相对预算, D2)
    world_n = np.einsum("tij,mj->tmi", R, ref_norm) + trans[:, None, :]
    world_vis = world_n[ok] if ok.any() else world_n
    trans_bd = _cloud_diag(world_vis.reshape(-1, 3))
    rot_bd = _cloud_diag(rotvec[ok]) if ok.sum() >= 2 else 0.0
    trans_budget = alpha * trans_bd if trans_bd > 0 else 0.0
    rot_budget = alpha * rot_bd if rot_bd > 0 else 0.0
    if abs_budget_m is not None:                       # 绝对预算(米 → 归一化);相机用
        trans_budget = min(trans_budget, abs_budget_m / scale) if trans_budget > 0 else abs_budget_m / scale
    frames_all = np.arange(Tn)
    # 事件 knot:在物体质心(归一化世界)轨迹上算,rot/trans 共用
    centroid = world_n.mean(axis=1)                    # (Tn,3)

    def ev_fn(fr, pp, _bd=trans_bd, _cen=centroid):
        cc = _cen[fr]
        rev = ev.detect_reversals(fr, cc, _bd) if _bd > 0 else []
        imp = ev.detect_impacts(fr, cc, _bd) if _bd > 0 else []
        return rev, imp

    if curve == "bspline" and rigid_mode == "joint":
        # §2.2 定案:共节点联合拟合 + keypoint 空间预算(rot_budget/rotvec 在这条路上不参与)
        rot_segs, trans_segs, gaps, vsum = _fit_rigid_joint(
            frames_all, R, trans, ok, ref_norm, trans_budget, T, ev_fn,
            rel_cap=RIGID_REL_CAP, robust=RIGID_ROBUST,
            interior_mult=RIGID_INTERIOR_MULT, max_coef=max_coef)
    else:
        fit = _stream_fitter(curve)         # 旧口径对照臂:两流独立、各用各的相对预算
        trans_segs, gaps, vsum = fit(frames_all, trans, ok, trans_budget, T, ev_fn)
        rot_segs, _, _ = fit(frames_all, rotvec, ok, rot_budget, T, ev_fn)
    # keypoint_ids / part_id 是按【原始】行数对齐的,占位行剔除后要一起收缩,否则身份错位
    kp_ids = np.asarray(inst.get("keypoint_ids", list(range(len(keep)))))[keep]
    mp = _part_labels(inst, len(keep))[keep]
    se3 = SE3Track(
        ref_norm=ref_norm.astype(float),
        keypoint_ids=[int(x) for x in kp_ids],
        part_ids=[int(x) for x in mp.ravel()],
        rot_segments=rot_segs, trans_segments=trans_segs,
        occlusion_gaps=gaps, visibility_summary=vsum)
    parts = sorted(set(int(x) for x in mp.ravel()))
    return ObjectTokens(
        instance_id=int(inst["instance_id"]), label=str(inst.get("label", "?")),
        object_type=str(inst["object_type"]), kind="rigid",
        motion_part_summary=dict(n_parts=len(parts), part_ids=parts),
        se3_track=se3, recon_resid_norm=_finite_or_none(inst.get("recon_resid_norm")))


# ================================================================== 顶层
def _load_camera(scene_dir):
    # 通用 GT 相机读取:优先 gt_stereo4d.npz,回退 gt_movi.npz(或任意 gt_*.npz)。
    # 三者共享字段契约:intrinsics_K/extrs_w2c/frame_indices/convention。
    npz = None
    for name in ("gt_stereo4d.npz", "gt_movi.npz"):
        p = os.path.join(scene_dir, name)
        if os.path.exists(p):
            npz = p
            break
    if npz is None:
        import glob as _glob
        cands = sorted(_glob.glob(os.path.join(scene_dir, "gt_*.npz")))
        npz = cands[0] if cands else None
    if npz is None or not os.path.exists(npz):
        return None, None, None, None
    d = np.load(npz, allow_pickle=True)
    K = np.asarray(d["intrinsics_K"], float) if "intrinsics_K" in d.files else None
    w2c = np.asarray(d["extrs_w2c"], float) if "extrs_w2c" in d.files else None
    conv = str(d["convention"]) if "convention" in d.files else "unknown"
    T = int(len(d["frame_indices"])) if "frame_indices" in d.files else None
    return K, w2c, conv, T


MOVI_FPS = 12.0   # Kubric MOVi-E 固定 12 fps × 24 帧(数据集常量,npz 里没有 timestamps 字段)


def _load_frame_times(scene_dir, T):
    """读该场景的绝对时间轴,返回 (times_s (T,) clip起点=0, fps) 或 (None, None)。

    **绝不用帧号冒充秒**(v0.3 §1.2-5:时间轴 join 失败 → 隔离名单,禁止静默降级)——
    拿不到就返回 None,下游 token 的 has_tsec=0,而不是悄悄写一个假秒数。
    Stereo4D 走 gt_stereo4d.npz 的 timestamps_us(实核 506/506 场景都有);
    MOVi 无时间戳字段,用数据集常量 12 fps 推。"""
    p_s4d = os.path.join(scene_dir, "gt_stereo4d.npz")
    if os.path.exists(p_s4d):
        d = np.load(p_s4d, allow_pickle=True)
        if "timestamps_us" in d.files:
            ts = np.asarray(d["timestamps_us"], np.float64).reshape(-1)
            if ts.size == T and T >= 1:
                t = (ts - ts[0]) / 1e6
                fps = float((T - 1) / t[-1]) if (T > 1 and t[-1] > 0) else None
                return t.astype(np.float32), fps
        meta = os.path.join(scene_dir, "stereo4d_meta.json")     # 退路:meta 的 fps_effective
        if os.path.exists(meta):
            try:
                fps = float(json.load(open(meta)).get("fps_effective") or 0)
            except Exception:
                fps = 0.0
            if fps > 0:
                return (np.arange(T, dtype=np.float32) / fps), fps
        return None, None
    if os.path.exists(os.path.join(scene_dir, "gt_movi.npz")):
        return (np.arange(T, dtype=np.float32) / MOVI_FPS), MOVI_FPS
    return None, None


CAMERA_IID = 1_000_000            # 相机伪实例 id(与真实 instance_id 不冲突)
CAMERA_LABEL = "camera"
# 相机刚体的合成参考点(相机坐标系,米):不含 (0,0,0)(会被 _valid_ref_rows 当占位行剔掉)。
# 平移流 = 相机光心 C(t) 的归一化坐标;旋转流 = R_c2w(t) 相对首帧的旋转向量;四个参考点只为
# 让"共节点联合拟合"的 keypoint 空间预算有定义(跨度 ~1 m → 误差上界 α·bbox)。
# 2026-09-14:四点**均值为 0**(±0.5 各轴)→ 质心参数化下 trans 流 = 光心 C(t) 本身(精确);旧四点均值 (0.125,0.125,0.375)
# 会让平移流偏离光心 |c0|·转角 量级(20° 时 ~0.14 m)。
CAMERA_REF_POINTS = np.array([[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, -0.5, 0.0]])


def camera_pseudo_inst(w2c):
    """把 GT 相机外参 (T,3,4)/(T,4,4) world->cam 变成一个"刚体实例"dict(与 final_repr 刚体同契约):
    se3_R = R_c2w = Rᵀ,se3_t = C = -Rᵀ t(光心,世界系米制),se3_ok 全 True。"""
    E = np.asarray(w2c, float)
    Tn = E.shape[0]
    R_c2w = np.transpose(E[:, :3, :3], (0, 2, 1))                  # (T,3,3)
    C = -np.einsum("tij,tj->ti", R_c2w, E[:, :3, 3])                # (T,3)
    return dict(instance_id=CAMERA_IID, label=CAMERA_LABEL, object_type="rigid",
                se3_R=R_c2w, se3_t=C, se3_ok=np.ones(Tn, bool),
                ref_positions=CAMERA_REF_POINTS.copy(), keypoint_ids=np.arange(4),
                part_id=np.zeros(4, int), recon_resid_norm=0.0)     # GT 位姿:残差 0 是正确值


def tokenize_scene(scene_dir, alpha=DEFAULT_ALPHA, drop_lowmotion=True,
                   curve=DEFAULT_CURVE, rigid_mode=DEFAULT_RIGID_MODE,
                   camera=False) -> SceneTokens:
    """把一个 scene_* 目录 tokenize 成 SceneTokens。
    drop_lowmotion=True: 动态实例在【世界位移小 AND 图像上无可观测运动】时退化为静态 4 点,
    避免把噪声当运动拟合(与生产 gt_dense_frontend 的单向门同一套信号、同一个阈值来源)。

    2026-08-13 改动:此前只看世界位移(mag<=0.05m)。单独用它两头都错 ——
      · 假阴:Stereo4D GT 的米级深度噪声让静止的树 mag 超阈,于是被当运动拟合(实测一个
        全静态场景仍有 3 棵树被编成 nonrigid,花掉 63 个 PL 顶点);
      · 假阳:原地往复/摆动的人 mag≈0.037m 低于阈值,被静默压成 4 个静态点(实测这类
        person×9/horse×2 图像残差 5.2px,是真运动)。
    改成两闸取 AND 后严格更保守(只会更少降级),假阳那一头被根治;假阴那一头由前端的
    单向门在上游解决(Qwen 判静的实例根本不会走到这里)。"""
    reprs = pickle.load(open(os.path.join(scene_dir, "final_repr.pkl"), "rb"))
    K, w2c, conv, Tcam = _load_camera(scene_dir)
    imw, imh = _load_imsize(scene_dir)
    resid_thr = _scene_resid_thr(scene_dir)
    # 场景帧数 T:优先相机(总是可得),否则任意动态实例的时间长度
    T = Tcam
    if T is None:
        for inst in reprs:
            if "trajectories" in inst:
                T = int(np.asarray(inst["trajectories"]).shape[1]); break
            if "se3_R" in inst:
                T = int(np.asarray(inst["se3_R"]).shape[0]); break
    if T is None:
        T = 1
    center, scale = compute_scene_norm(reprs)

    objects: List[ObjectTokens] = []
    for inst in reprs:
        kind = _kind_of(inst)
        if kind == "static":
            objects.append(_encode_static(inst, center, scale))
            continue
        # 动态:先判"真运动";低于阈值退化为静态(取 ref_positions)
        if kind == "nonrigid":
            traj = np.asarray(inst["trajectories"], float)
            vis = np.asarray(inst["visibility"], bool)
        else:
            traj, vis = _rigid_world_traj(inst)
        mag = _instance_motion_mag(traj, vis)
        if drop_lowmotion and _is_lowmotion(traj, vis, mag, K, w2c, imw, imh, resid_thr):
            objects.append(_encode_static(inst, center, scale))
            continue
        if kind == "nonrigid":
            objects.append(_encode_nonrigid(inst, center, scale, alpha, T, curve))
        else:
            objects.append(_encode_rigid(inst, center, scale, alpha, T, curve, rigid_mode))

    # 2026-09-13:相机作为一个刚体物体组进 token(用户拍板放开"相机不进 token"红线)。
    # 追加在物体列表**末尾**:已有物体的 group 序号/part 编号不变;不过低运动闸(静止相机也保留
    # 为刚体组,结构统一);归一化仍只按物体点算(相机可在单位球外,量级 >1 是正常的)。
    if camera and w2c is not None and w2c.shape[0] == T:
        objects.append(_encode_rigid(camera_pseudo_inst(w2c), center, scale, alpha, T, curve, rigid_mode,
                                     abs_budget_m=CAMERA_ABS_BUDGET_M, max_coef=CAMERA_MAX_COEF))
    times_s, fps = _load_frame_times(scene_dir, T)
    st = SceneTokens(
        scene_id=os.path.basename(scene_dir), T=T, frame_convention=(conv or "unknown"),
        norm_center=center, norm_scale=scale, camera_K=K, camera_w2c=w2c, objects=objects,
        curve_family=curve, frame_times_s=times_s, fps_effective=fps, rigid_param=RIGID_PARAM)
    st.has_camera_object = bool(camera and w2c is not None and w2c.shape[0] == T)
    return st


# ================================================================== 逆重建(round-trip)
def _reconstruct_one(seg, frames) -> np.ndarray:
    """单个 PLSegment / SplineRun 在整数帧上的重建(段外 nan)。"""
    if isinstance(seg, SplineRun):
        from scipy.interpolate import splev
        out = np.full((len(frames), 3), np.nan)
        if not seg.tcks:                              # 单点 run
            f = int(round(seg.fnodes[0]))
            m = frames == f
            out[m] = seg.nodes[0]
            return out
        for (t, c, k, f0, f1) in seg.tcks:
            m = (frames >= f0) & (frames <= f1)
            if not m.any():
                continue
            u = (frames[m] - f0) / max(f1 - f0, 1)
            out[m] = np.stack(splev(u, (t, c, k)), axis=1)
        return out
    pl = dict(vf=seg.vframes, vp=seg.verts)
    return cf.pl_reconstruct_frames(pl, frames)


def reconstruct_keypoint(kt: KeypointTrack, T: int) -> np.ndarray:
    """从 KeypointTrack 反重建归一化世界轨迹 (T,3);遮挡缺口为 nan(诚实,不脑补)。"""
    out = np.full((T, 3), np.nan)
    frames = np.arange(T)
    for seg in kt.pl_segments:
        rec = _reconstruct_one(seg, frames)
        m = ~np.isnan(rec[:, 0])
        out[m] = rec[m]
    return out


def _reconstruct_stream(segs, T: int) -> np.ndarray:
    out = np.full((T, 3), np.nan)
    frames = np.arange(T)
    for seg in segs:
        rec = _reconstruct_one(seg, frames)
        m = ~np.isnan(rec[:, 0])
        out[m] = rec[m]
    return out


def _reconstruct_rotmats(segs, T: int) -> np.ndarray:
    """旋转流 → 逐帧旋转矩阵 (T,3,3),无效帧 nan。
    segment 带 rot_ref(§2.2 联合拟合)时 nodes 存的是相对量:R = exp(rot_ref)·exp(φ);
    rot_ref=None 时 nodes 即绝对旋转向量(旧口径),自动兼容。"""
    out = np.full((T, 3, 3), np.nan)
    frames = np.arange(T)
    for seg in segs:
        rec = _reconstruct_one(seg, frames)
        m = ~np.isnan(rec[:, 0])
        if not m.any():
            continue
        Rm = Rotation.from_rotvec(rec[m]).as_matrix()
        rr = getattr(seg, "rot_ref", None)
        if rr is not None:
            Rm = np.einsum("ij,tjk->tik", Rotation.from_rotvec(np.asarray(rr, float)).as_matrix(), Rm)
        out[m] = Rm
    return out


def reconstruct_rigid(se: SE3Track, T: int) -> np.ndarray:
    """从 SE3Track 反重建逐 keypoint 归一化世界轨迹 (M,T,3);无效帧 nan。"""
    Rall = _reconstruct_rotmats(se.rot_segments, T)    # (T,3,3)
    trans = _reconstruct_stream(se.trans_segments, T)  # (T,3)
    ref = se.ref_norm                                  # (M,3)
    M = len(ref)
    out = np.full((M, T, 3), np.nan)
    for f in range(T):
        if np.isnan(Rall[f, 0, 0]) or np.isnan(trans[f, 0]):
            continue
        out[:, f, :] = (Rall[f] @ ref.T).T + trans[f]
    return out


def denormalize(pts_norm: np.ndarray, center: np.ndarray, scale: float) -> np.ndarray:
    """归一化坐标 -> metric 世界坐标。"""
    return np.asarray(pts_norm) * scale + center
