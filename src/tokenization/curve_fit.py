"""贝塞尔 / 分段线性 拟合与解析求导核心 (纯 numpy, 探针 A/C 共用).

约定:
- 一条 keypoint 轨迹在 *可见段(run)* 内拟合;遮挡缺口不拟合、不脑补 (方案 D4)。
- 时间参数化 (D1):段 [f0,f1] 内 u=(f-f0)/(f1-f0)∈[0,1];速度对帧的导数 = dP/du / (f1-f0)。
- 自适应阶数+分段+误差上界 (D2):max_u ||curve-traj|| < budget 为止。
- 事件对齐 knot (D3):在真值轨迹的反转/冲击帧放段边界 (由调用方传入)。
"""
from __future__ import annotations
import numpy as np
from scipy.special import comb


# ---------------------------------------------------------------- Bézier 基础
def _bernstein(n: int, u: np.ndarray) -> np.ndarray:
    """(len(u), n+1) Bernstein 基矩阵, 阶数 n。"""
    u = np.asarray(u, float)
    i = np.arange(n + 1)
    # B_{i,n}(u) = C(n,i) u^i (1-u)^(n-i)
    coeff = comb(n, i)                                   # (n+1,)
    return coeff[None, :] * (u[:, None] ** i[None, :]) * ((1 - u)[:, None] ** (n - i)[None, :])


def fit_bezier_ctrl(u: np.ndarray, P: np.ndarray, degree: int):
    """最小二乘拟合一段 degree 阶贝塞尔到 (u_i, P_i)。返回 (ctrl[degree+1,3], max_err)。"""
    B = _bernstein(degree, u)                            # (N, degree+1)
    ctrl, *_ = np.linalg.lstsq(B, P, rcond=None)        # (degree+1, 3)
    recon = B @ ctrl
    err = float(np.max(np.linalg.norm(recon - P, axis=1))) if len(P) else 0.0
    return ctrl, err


def eval_bezier(ctrl: np.ndarray, u: np.ndarray) -> np.ndarray:
    n = len(ctrl) - 1
    return _bernstein(n, np.asarray(u, float)) @ ctrl


def bezier_deriv_ctrl(ctrl: np.ndarray) -> np.ndarray:
    """贝塞尔导数(对 u)的控制点:阶数降 1。d/du B = n*(P_{i+1}-P_i)。"""
    n = len(ctrl) - 1
    if n < 1:
        return np.zeros((1, ctrl.shape[1]))
    return n * (ctrl[1:] - ctrl[:-1])


# ---------------------------------------------------------------- 自适应贝塞尔拟合
def _fit_run_segment(frames, P, budget, max_degree, depth, max_depth, out):
    """把帧区间 [frames[0],frames[-1]] 内的点拟合成 1+ 段贝塞尔, 追加到 out。
    frames: (n,) int 连续或含事件切分后的子段; P: (n,3)。"""
    n = len(frames)
    f0, f1 = int(frames[0]), int(frames[-1])
    if n == 1 or f1 == f0:
        out.append(dict(f0=f0, f1=f0, ctrl=P[:1].copy(), knot="point"))
        return
    u = (frames - f0) / (f1 - f0)
    cap = min(max_degree, n - 1)
    best = None
    for deg in range(1, cap + 1):
        ctrl, err = fit_bezier_ctrl(u, P, deg)
        best = (ctrl, err, deg)
        if err <= budget:
            out.append(dict(f0=f0, f1=f1, ctrl=ctrl, knot="smooth"))
            return
    # 到最高阶仍不达标 -> 在最坏误差帧处二分(D2 增段), 限深度
    if depth >= max_depth or n < 4:
        out.append(dict(f0=f0, f1=f1, ctrl=best[0], knot="smooth"))  # 尽力而为
        return
    ctrl = best[0]
    recon = eval_bezier(ctrl, u)
    split = int(np.argmax(np.linalg.norm(recon - P, axis=1)))
    split = min(max(split, 1), n - 2)                    # 保证两侧各>=2点
    _fit_run_segment(frames[:split + 1], P[:split + 1], budget, max_degree, depth + 1, max_depth, out)
    _fit_run_segment(frames[split:], P[split:], budget, max_degree, depth + 1, max_depth, out)


def adaptive_bezier(frames, P, budget, event_frames, max_degree=6, max_depth=4):
    """一个可见 run 的自适应分段贝塞尔。event_frames: run 内真值事件帧 (D3 knot 先验)。
    返回 segments 列表 [{f0,f1,ctrl,knot}]。"""
    frames = np.asarray(frames, int)
    P = np.asarray(P, float)
    # D3: 先按事件帧切成子段, 每子段内再做 D2 自适应
    evs = sorted(int(e) for e in event_frames if frames[0] < e < frames[-1])
    bounds = [frames[0]] + evs + [frames[-1]]
    out = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        sel = (frames >= a) & (frames <= b)
        fr, pp = frames[sel], P[sel]
        if len(fr) == 0:
            continue
        knot_here = "event" if a in evs else None
        pre = len(out)
        _fit_run_segment(fr, pp, budget, max_degree, 0, max_depth, out)
        if knot_here and pre < len(out):
            out[pre]["knot"] = "event"                   # 段起点是事件 knot
    return out


def n_control_points(segments) -> int:
    """token 计数:控制点总数 (段数×每段(阶+1))。单点段计 1。"""
    return int(sum(len(s["ctrl"]) for s in segments))


def bezier_reconstruct_frames(segments, frames):
    """在给定整数帧上重建位置。frames 必须落在 segments 覆盖范围内。返回 (len(frames),3)。"""
    frames = np.asarray(frames, int)
    out = np.full((len(frames), 3), np.nan)
    for i, f in enumerate(frames):
        seg = None
        for s in segments:                               # 取包含 f 的段(边界归前段)
            if s["f0"] <= f <= s["f1"]:
                seg = s
                break
        if seg is None:
            continue
        if seg["f1"] == seg["f0"]:
            out[i] = seg["ctrl"][0]
        else:
            u = (f - seg["f0"]) / (seg["f1"] - seg["f0"])
            out[i] = eval_bezier(seg["ctrl"], np.array([u]))[0]
    return out


def bezier_kinematics_analytic(segments, frames):
    """解析求导得 speed/accel(对帧), 在整数帧上。返回 speed(len,), accel(len,), 均可能 nan。"""
    frames = np.asarray(frames, int)
    speed = np.full(len(frames), np.nan)
    accel = np.full(len(frames), np.nan)
    for i, f in enumerate(frames):
        seg = None
        for s in segments:
            if s["f0"] <= f <= s["f1"]:
                seg = s
                break
        if seg is None or seg["f1"] == seg["f0"]:
            continue
        span = seg["f1"] - seg["f0"]
        u = np.array([(f - seg["f0"]) / span])
        d1 = bezier_deriv_ctrl(seg["ctrl"])
        d2 = bezier_deriv_ctrl(d1)
        vel = eval_bezier(d1, u)[0] / span               # dP/dframe
        acc = eval_bezier(d2, u)[0] / (span ** 2)
        speed[i] = np.linalg.norm(vel)
        accel[i] = np.linalg.norm(acc)
    return speed, accel


# ---------------------------------------------------------------- 分段线性 baseline (探针 C)
def _pl_add_vertices(frames, P, target_n=None, budget=None):
    """自顶向下 Douglas-Peucker:从两端点起, 反复加入最大偏差帧。
    target_n: 顶点数上限; budget: 最大偏差阈值。二者满足其一即停。返回顶点帧索引(排序)。"""
    n = len(frames)
    if n <= 2:
        return list(range(n))
    keep = [0, n - 1]
    while True:
        keep_sorted = sorted(keep)
        max_dev, max_j = -1.0, -1
        for a, b in zip(keep_sorted[:-1], keep_sorted[1:]):
            if b - a < 2:
                continue
            fa, fb = frames[a], frames[b]
            for j in range(a + 1, b):
                if fb == fa:
                    p_interp = P[a]
                else:
                    t = (frames[j] - fa) / (fb - fa)
                    p_interp = P[a] * (1 - t) + P[b] * t
                dev = np.linalg.norm(P[j] - p_interp)
                if dev > max_dev:
                    max_dev, max_j = dev, j
        if max_j < 0:
            break
        if target_n is not None and len(keep) >= target_n:
            break
        if budget is not None and max_dev <= budget:
            break
        keep.append(max_j)
        if len(keep) >= n:
            break
    return sorted(keep)


def fit_pl(frames, P, target_n=None, budget=None):
    """返回 PL 表示 {vf:(顶点帧数组), vp:(顶点位置)} 。"""
    frames = np.asarray(frames, int)
    P = np.asarray(P, float)
    idx = _pl_add_vertices(frames, P, target_n=target_n, budget=budget)
    return dict(vf=frames[idx], vp=P[idx])


def pl_reconstruct_frames(pl, frames):
    frames = np.asarray(frames, int)
    vf, vp = pl["vf"], pl["vp"]
    out = np.full((len(frames), 3), np.nan)
    for i, f in enumerate(frames):
        if f < vf[0] or f > vf[-1]:
            continue
        k = np.searchsorted(vf, f)
        if vf[k] == f:
            out[i] = vp[k]
        else:
            a, b = k - 1, k
            t = (f - vf[a]) / (vf[b] - vf[a])
            out[i] = vp[a] * (1 - t) + vp[b] * t
    return out


def pl_n_tokens(pl) -> int:
    return int(len(pl["vf"]))


# ---------------------------------------------------------------- 事件对齐 PL (D3, 生产 tokenizer 用)
def fit_pl_events(frames, P, budget, event_frames):
    """一个可见 run 内的事件对齐分段线性 (D2 误差上界 + D3 事件 knot)。

    与 fit_pl 的区别:先在真值事件帧(反转/冲击)处强制切段边界,每子段内再做
    Douglas-Peucker 自适应加点到 budget。事件帧被强制成顶点 → 只会 *增加* 顶点,
    重建保真度 >= 纯 DP(fit_pl)。返回 {vf:(顶点帧), vp:(顶点位置)}。
    """
    frames = np.asarray(frames, int)
    P = np.asarray(P, float)
    if len(frames) <= 2:
        return dict(vf=frames.copy(), vp=P.copy())
    evs = sorted({int(e) for e in event_frames if frames[0] < e < frames[-1]})
    bounds = [int(frames[0])] + evs + [int(frames[-1])]
    keep = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        sel = (frames >= a) & (frames <= b)
        fr, pp = frames[sel], P[sel]
        if len(fr) == 0:
            continue
        idx = _pl_add_vertices(fr, pp, budget=budget)     # DP 到 budget
        keep.extend(int(fr[j]) for j in idx)
    keep = sorted(set(keep))
    fmap = {int(f): p for f, p in zip(frames, P)}
    vf = np.asarray(keep, int)
    vp = np.asarray([fmap[f] for f in keep], float)
    return dict(vf=vf, vp=vp)
