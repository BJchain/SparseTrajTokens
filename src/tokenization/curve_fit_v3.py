"""v0.3 拟合层(实验层;生产 tokenizer.py / curve_fit.py 尚未切换,探针 A″ 先用它出数)。

方案原文:references/traj_fitting_tokenization_v0.3.md §2。本文件实现:
- 三次 B 样条(k=3)每可见 run 一条;时间参数化 u=(f-f0)/(f1-f0)(D1,与 PL / 对照实验同约定)
- 节点放置 = 残差驱动:`splprep` 平滑因子 s 二分到 max‖B(u_i)-P_i‖ ≤ ε(FITPACK 在残差最大处局部加节点)
- 事件处理(2026-08-20 探针 A″ 定案)= **`fit_run_split`:在事件帧把 run 切段、逐段拟合、共享端点 = C⁰ 角点**
  (merge_adjacent=False,逐事件帧切)。`fit_run(event_frames=…)` 里"事件帧并入节点 + 固定节点重解"的路径
  保留作对照臂:FITPACK 不允许内节点重数>1,单重节点保持 C² 救不了反转;`make_lsq_spline` 三重节点数值失稳。
- 阶退化档位 **B**(2026-08-20 拍板,§2.7):m≥5 一律三次;m<5 取满足 ε 的最低阶
  (k=1 两系数 → k=2 三系数 → 插值全部 m 点);含事件帧的 run 不降阶。档位 A(m<5 一律插值全点)
  保留为对照臂。
- 系数上限 M/run;触顶保留最好解并置 `budget_miss`(不静默交白卷)
- **Greville 节点态**:τ_i = 相邻 k 个节点的均值;在 τ_i 处解析求 p / p' / p''(§2.1-4)
- 重建 / 解析运动学 / token 计数(token = 系数个数 = 节点态个数,与历次实验同口径)

全部纯 numpy/scipy,逐位可复现。本文件**不**做归一化、不读 final_repr —— 那是 tokenizer 的事。
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.interpolate import make_lsq_spline, splev, splprep

import curve_fit as cf

warnings.filterwarnings("ignore", category=RuntimeWarning)

DEFAULT_M = 12          # 系数上限 / run(§7)
_BISECT_ITERS = 30      # s 二分次数(对照实验用 48;30 次已到 1e-9 相对精度)


# ---------------------------------------------------------------- 基础:tck 工具
def _eval(tck, u, der=0):
    """在 u 上求值(或 der 阶导数),返回 (len(u),3)。der > k 时导数恒 0。"""
    t, c, k = tck
    u = np.asarray(u, float)
    if der > k:
        return np.zeros((len(u), len(c)))
    vals = splev(u, (t, c, k), der=der)
    return np.stack(vals, axis=1)


def _max_err(tck, u, P):
    rec = _eval(tck, u, 0)
    return float(np.max(np.linalg.norm(rec - P, axis=1)))


def _err_vec(tck, u, P):
    return np.linalg.norm(_eval(tck, u, 0) - P, axis=1)


def make_criterion(budget, robust=False):
    """达标判据。robust=False:max ≤ ε(与 PL/DP、探针 A、对照实验同口径)。
    robust=True:p95 ≤ ε 且 max ≤ 3ε —— 不为单帧深度噪声追到近插值(§9 备选)。"""
    if not robust:
        return lambda tck, u, P: _max_err(tck, u, P) <= budget + 1e-12
    def ok(tck, u, P):
        e = _err_vec(tck, u, P)
        return (np.percentile(e, 95) <= budget + 1e-12) and (e.max() <= 3 * budget + 1e-12)
    return ok


def _n_coef(tck):
    t, c, k = tck
    return int(len(t) - k - 1)


def _tck_bezier(u, P, k):
    """k 阶 Bézier 的最小二乘(Bernstein LSQ)= 无内节点的 k 阶 B 样条。n = k+1。
    当 k = m-1 时就是插值(过全部点)。"""
    ctrl, _ = cf.fit_bezier_ctrl(u, P, k)                 # (k+1,3)
    t = np.array([0.0] * (k + 1) + [1.0] * (k + 1))
    c = [ctrl[:, d].copy() for d in range(P.shape[1])]
    return (t, c, k)


def _lsq_fixed(u, P, k, t_int):
    """固定节点 LSQ(make_lsq_spline):允许内节点重数 ≤k(FITPACK 的 curfit/parcur 要求内节点严格递增,
    因此 splprep(task=-1) 做不了角点)。要求 Schoenberg–Whitney;失败返回 None。"""
    try:
        t_full = np.concatenate([np.zeros(k + 1), np.asarray(sorted(t_int), float), np.ones(k + 1)])
        if len(t_full) - k - 1 > len(u):
            return None
        spl = make_lsq_spline(np.asarray(u, float), np.asarray(P, float), t_full, k=k)
    except Exception:
        return None
    c = np.asarray(spl.c, float)
    if not np.all(np.isfinite(c)):
        return None
    return (np.asarray(spl.t, float), [c[:, d] for d in range(c.shape[1])], int(k))


def _splprep(u, P, k, s=None, task=0, t_int=None):
    """splprep 薄封装:显式 u、三轴共享节点。失败返回 None。
    task=-1(固定节点重解)时 scipy≥1.16 要的是**完整**节点向量(两端各 k+1 个重节点 + 内节点),
    这里由内节点 t_int 拼出;系数数 n=len(t)-k-1 不得超过点数 m(否则 LSQ 欠定 / S–W 不满足)。"""
    try:
        if task == -1:
            t_full = np.concatenate([np.zeros(k + 1), np.asarray(sorted(t_int), float), np.ones(k + 1)])
            if len(t_full) - k - 1 > len(u):
                return None
            tck, _ = splprep([P[:, d] for d in range(P.shape[1])], u=u, k=k, task=-1, t=t_full)
        else:
            tck, _ = splprep([P[:, d] for d in range(P.shape[1])], u=u, k=k, s=s)
    except Exception:
        return None
    t, c, kk = tck
    return (np.asarray(t, float), [np.asarray(ci, float) for ci in c], int(kk))


def _interior(tck):
    t, _, k = tck
    return t[k + 1: len(t) - k - 1]


# ---------------------------------------------------------------- 三次:s 二分到 ε / 到系数上限
def _cubic_by_scan(u, P, ok, n_steps=28, score=None):
    """s 从大到小扫,返回**第一个**满足 ok 的解(= 最省系数的达标解)。

    为什么不能用 _cubic_by_budget 的二分:二分假定"s=0(插值)一定达标",于是
    `if not ok(r0): return r0` —— 一旦判据在 s=0 处不成立(比如加了帧间过冲约束,
    近插值解恰恰是最糟的那个),二分会**直接把最糟的解交出去**。扫描没有这个假设。

    都不达标时的兜底:有 score 就返回**评分最优**的候选(score 越小越好),没有才退回 s=0。
    这条很关键 —— 兜底成 s=0 等于"救不了就交最坏的",实测会让过冲约束完全失效。"""
    scale = max(float(np.sum(np.var(P, axis=0))) * len(P), 1e-12)
    cands = []
    for i in range(n_steps):
        r = _splprep(u, P, 3, s=scale * 4.0 * (0.5 ** i))
        if r is None:
            continue
        if ok(r, u, P):
            return r
        cands.append(r)
    if score is not None and cands:
        return min(cands, key=lambda r: score(r, u, P))
    return _splprep(u, P, 3, s=0.0)


def _cubic_by_budget(u, P, ok, search="bisect", score=None):
    """s 二分:取满足判据 ok(tck,u,P) 的**最大** s(最省系数)。失败返回 None。
    search="scan" 改用 _cubic_by_scan(判据可能在 s=0 处不成立时必须用它)。"""
    if search == "scan":
        return _cubic_by_scan(u, P, ok, score=score)
    scale = max(float(np.sum(np.var(P, axis=0))) * len(P), 1e-12)
    s_hi = scale * 4.0
    r0 = _splprep(u, P, 3, s=0.0)
    if r0 is None or not ok(r0, u, P):
        return r0                                   # 插值都够不到(数值异常);交给调用方判 budget_miss
    r1 = _splprep(u, P, 3, s=s_hi)
    if r1 is not None and ok(r1, u, P):
        return r1
    lo, hi, best = 0.0, s_hi, r0
    for _ in range(_BISECT_ITERS):
        mid = 0.5 * (lo + hi)
        r = _splprep(u, P, 3, s=mid)
        if r is not None and ok(r, u, P):
            best, lo = r, mid
        else:
            hi = mid
    return best


def _cubic_by_tokens(u, P, n_max, score=None):
    """s 二分:取系数数 ≤ n_max 的**最小** s(拟合最好)。

    score 非空时改为**扫描 + 评分**:在所有 n ≤ n_max 的候选里取评分最优的那个。
    为什么必须这样(2026-08-22):原版只按"n 达标 + s 最小"选,完全不看解好不好 ——
    它会把上一步刚选出的合格解换成一个在数据点上更准、却在**帧之间飙到 100× bbox**
    的病态解(FITPACK 在中等 s 处的节点会挤成一堆)。系数上限本是省 token 的旋钮,
    不该让它有权把解换坏。"""
    if score is not None:
        scale = max(float(np.sum(np.var(P, axis=0))) * len(P), 1e-12)
        best = None
        for i in range(28):
            r = _splprep(u, P, 3, s=scale * 4.0 * (0.5 ** i))
            if r is None or _n_coef(r) > n_max:
                continue
            if best is None or score(r, u, P) < score(best, u, P):
                best = r
        return best
    scale = max(float(np.sum(np.var(P, axis=0))) * len(P), 1e-12)
    s_hi = scale * 4.0
    r0 = _splprep(u, P, 3, s=0.0)
    if r0 is not None and _n_coef(r0) <= n_max:
        return r0
    r1 = _splprep(u, P, 3, s=s_hi)
    if r1 is None or _n_coef(r1) > n_max:
        return r1
    lo, hi, best = 0.0, s_hi, r1
    for _ in range(_BISECT_ITERS):
        mid = 0.5 * (lo + hi)
        r = _splprep(u, P, 3, s=mid)
        if r is not None and _n_coef(r) <= n_max:
            best, hi = r, mid
        else:
            lo = mid
    return best


def _add_event_knots(tck, u, P, event_u, ok, mult=1, frame_du=None):
    """把事件帧的 u 并入内节点(重数 mult:1=C²、2=C¹、3=C⁰ 角点),task=-1 重解 LSQ。
    返回 (tck, n_added, dropped)。
    mult>1 时先把距事件 ≤1 帧的既有节点挪走(否则相邻节点把数据点夹空,Schoenberg–Whitney 不满足),
    失败再退回 mult=1;仍失败(或重解不达标)→ 放弃事件节点,dropped=True。"""
    if not len(event_u):
        return tck, 0, False
    t_int0 = list(_interior(tck))
    evs = [e for e in event_u if 0.0 < e < 1.0]
    if not evs:
        return tck, 0, False
    tol = (frame_du or 0.0) * 1.0 + 1e-9
    for mu in ([mult, 1] if mult > 1 else [1]):
        t_int = [x for x in t_int0 if all(abs(e - x) > tol for e in evs)] if mu > 1 else list(t_int0)
        add = []
        for e in evs:
            have = sum(1 for x in t_int if abs(e - x) <= 1e-9)
            add += [e] * max(0, mu - have)
        if not add:
            continue
        r = _lsq_fixed(u, P, 3, sorted(t_int + add))
        if r is not None and ok(r, u, P):
            return r, len(add), False
    return tck, 0, True


# ---------------------------------------------------------------- 主入口
def fit_run(frames: Sequence[int], P: np.ndarray, budget: float, event_frames: Sequence[int] = (),
            mode: str = "B", M: int = DEFAULT_M, robust: bool = False, rel_cap: Optional[float] = None,
            event_mult: int = 1, criterion=None, err_fn=None, search: str = "bisect",
            score=None) -> Dict:
    """拟合一个可见 run。返回 run 字典:
      f0,f1,m          帧区间 / 帧数
      k, tck, n        阶 / (t,c,k) / 系数数(= token 数)
      degree_rule      'cubic' | 'line' | 'quad' | 'interp' | 'point'
      tau, fpos        Greville 横坐标 (n,) / 对应的(浮点)帧位置 (n,)
      p, v_u, a_u      节点态 (n,3):位置 / 对 u 的一阶二阶导(调用方按 Δt 换算)
      max_err, budget_miss, event_knots_added, event_knots_dropped
    mode='B'(定案):m<5 取满足 budget 的最低阶;mode='A':m<5 一律插值全点。
    robust:判据 p95≤ε 且 max≤3ε(默认 max≤ε);rel_cap:系数数相对上限 n ≤ max(4, ceil(rel_cap·m))
    (防近插值在帧间过冲);event_mult:事件节点重数(3 = C⁰ 角点)。

    criterion / err_fn(2026-08-22 加,为 §2.2 刚体联合拟合):把"达标"与"误差"判在**别的空间**。
    默认 None = 在拟合点自身空间判 max‖B(u)−P‖(现役行为,逐位不变)。刚体联合拟合时 P 是 6 维
    的 [平移, 半径·旋转向量],直接判 6 维距离没有物理意义 —— 传入的判据把它换算成
    **重建 keypoint 的最大误差**再比预算。签名与 make_criterion 一致:ok(tck, u, P) -> bool;
    err_fn(tck, u, P) -> float。P 为任意维;本函数其余部分对维数通用。"""
    frames = np.asarray(frames, int)
    P = np.asarray(P, float)
    m = len(frames)
    f0, f1 = int(frames[0]), int(frames[-1])
    out = dict(f0=f0, f1=f1, m=m, budget=float(budget), event_knots_added=0,
               event_knots_dropped=False, budget_miss=False)
    D = int(P.shape[1]) if P.ndim == 2 else 3          # 维数通用(刚体联合拟合 D=6)
    if m == 1 or f1 == f0:
        out.update(k=0, tck=None, n=1, degree_rule="point", tau=np.zeros(1), fpos=np.array([float(f0)]),
                   p=P[:1].copy(), v_u=np.zeros((1, D)), a_u=np.zeros((1, D)),
                   v_f=np.zeros((1, D)), a_f=np.zeros((1, D)), max_err=0.0)
        return out
    u = (frames - f0) / float(f1 - f0)
    frame_du = 1.0 / float(f1 - f0)
    ev_raw = sorted({(float(e) - f0) / float(f1 - f0) for e in event_frames if f0 < int(e) < f1})
    ev_u = []                                              # 相邻(≤1.5 帧)事件合并成一个
    for e in ev_raw:
        if ev_u and (e - ev_u[-1][-1]) <= 1.5 * frame_du:
            ev_u[-1].append(e)
        else:
            ev_u.append([e])
    ev_u = [float(np.mean(g)) for g in ev_u]
    ok = criterion if criterion is not None else make_criterion(budget, robust)
    n_cap = M if rel_cap is None else min(M, max(4, int(np.ceil(rel_cap * m))))

    tck = None
    rule = None
    if m >= 5:
        tck = _cubic_by_budget(u, P, ok, search=search, score=score)
        rule = "cubic"
        if tck is None:                                   # 数值异常兜底:插值全点
            tck = _tck_bezier(u, P, min(3, m - 1)); rule = "interp"
        else:
            if _n_coef(tck) > n_cap:                       # 系数上限:保留 ≤cap 内最好的解,置 budget_miss
                tck2 = _cubic_by_tokens(u, P, n_cap, score=score)
                if tck2 is not None:
                    tck = tck2
            tck, n_add, dropped = _add_event_knots(tck, u, P, ev_u, ok, mult=event_mult, frame_du=frame_du)
            out["event_knots_added"] = n_add
            out["event_knots_dropped"] = dropped
            out["budget_miss"] = not ok(tck, u, P)
    else:
        # m ∈ {2,3,4}:三次拟不了(scipy 要求 m>k)。
        if mode == "B" and not ev_u:
            cand = _tck_bezier(u, P, 1)                    # 直线(LSQ),两系数
            if ok(cand, u, P):
                tck, rule = cand, "line"
            elif m >= 3:
                cand = _tck_bezier(u, P, 2)                # 二次 Bézier,三系数
                if ok(cand, u, P):
                    tck, rule = cand, "quad"
        if tck is None:                                    # A 档 / 含事件 / 低阶不达标:插值全点
            tck, rule = _tck_bezier(u, P, m - 1), "interp"

    t, c, k = tck
    n = _n_coef(tck)
    tau = np.array([t[i + 1: i + k + 1].mean() if k >= 1 else t[i] for i in range(n)])
    span = float(f1 - f0)
    v_u, a_u = _eval(tck, tau, 1), _eval(tck, tau, 2)
    out.update(k=int(k), tck=tck, n=int(n), degree_rule=rule, tau=tau,
               fpos=f0 + tau * span,
               p=_eval(tck, tau, 0), v_u=v_u, a_u=a_u, v_f=v_u / span, a_f=a_u / span ** 2,
               max_err=(err_fn(tck, u, P) if err_fn is not None else _max_err(tck, u, P)))
    return out


def fit_run_split(frames: Sequence[int], P: np.ndarray, budget: float, event_frames: Sequence[int] = (),
                  merge_adjacent: bool = False, **kw) -> Dict:
    """**生产定案入口(2026-08-20)**:在事件帧把 run **切开分段拟合**(相邻段共享事件帧端点 →
    C⁰ 角点),每段各自走 fit_run(无内部事件)。默认 merge_adjacent=False = 探针 A″ 的 splitB_split_all
    臂(逐事件帧切;合并相邻事件会丢 ~1/3 反转 recall:0.745 vs 0.834)。返回复合 run(parts 列表;
    节点态按段拼接;共享端点两侧各一个节点,位置同、速度不同 —— 这就是角点)。token = 各段系数数之和。"""
    frames = np.asarray(frames, int)
    P = np.asarray(P, float)
    f0, f1 = int(frames[0]), int(frames[-1])
    evs = sorted({int(e) for e in event_frames if f0 < int(e) < f1})
    merged = []                                            # 相邻(≤1 帧)事件合并(merge_adjacent=False 则逐帧切)
    for e in evs:
        if merge_adjacent and merged and e - merged[-1] <= 1:
            continue
        merged.append(e)
    if not merged or len(frames) < 3:
        r = fit_run(frames, P, budget, (), **kw)
        return r
    bounds = [f0] + merged + [f1]
    parts = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        sel = (frames >= a) & (frames <= b)
        parts.append(fit_run(frames[sel], P[sel], budget, (), **kw))
    out = dict(f0=f0, f1=f1, m=int(len(frames)), budget=float(budget), parts=parts, tck=None, k=3,
               degree_rule="split", n=int(sum(p["n"] for p in parts)),
               event_knots_added=len(merged), event_knots_dropped=False,
               budget_miss=any(p["budget_miss"] for p in parts),
               max_err=max(p["max_err"] for p in parts),
               tau=np.concatenate([p["tau"] for p in parts]),
               fpos=np.concatenate([p["fpos"] for p in parts]),
               p=np.concatenate([p["p"] for p in parts]),
               v_f=np.concatenate([p["v_f"] for p in parts]),
               a_f=np.concatenate([p["a_f"] for p in parts]))
    return out


# ---------------------------------------------------------------- 重建 / 运动学 / 计数
def reconstruct_frames(run: Dict, frames) -> np.ndarray:
    """在整数帧上重建位置;run 外为 nan。复合 run(split)逐段填充。"""
    frames = np.asarray(frames, float)
    if "parts" in run:
        out = np.full((len(frames), 3), np.nan)
        for part in run["parts"]:
            r = reconstruct_frames(part, frames)
            m = ~np.isnan(r[:, 0])
            out[m] = r[m]
        return out
    out = np.full((len(frames), 3), np.nan)
    f0, f1 = run["f0"], run["f1"]
    msk = (frames >= f0) & (frames <= f1)
    if not msk.any():
        return out
    if run["tck"] is None:
        out[msk] = run["p"][0]
        return out
    u = (frames[msk] - f0) / max(f1 - f0, 1)
    out[msk] = _eval(run["tck"], u, 0)
    return out


def kinematics_analytic_frames(run: Dict, frames):
    """解析导数在整数帧上的 speed / accel(对帧的导数;与 events.finite_diff_kin 同单位)。"""
    frames = np.asarray(frames, float)
    if "parts" in run:
        sp = np.full(len(frames), np.nan); ac = np.full(len(frames), np.nan)
        for part in run["parts"]:
            s1, a1 = kinematics_analytic_frames(part, frames)
            m = ~np.isnan(s1)
            sp[m] = s1[m]; ac[m] = a1[m]
        return sp, ac
    sp = np.full(len(frames), np.nan)
    ac = np.full(len(frames), np.nan)
    f0, f1 = run["f0"], run["f1"]
    msk = (frames >= f0) & (frames <= f1)
    if not msk.any() or run["tck"] is None:
        return sp, ac
    span = max(f1 - f0, 1)
    u = (frames[msk] - f0) / span
    v = _eval(run["tck"], u, 1) / span
    a = _eval(run["tck"], u, 2) / (span ** 2)
    sp[msk] = np.linalg.norm(v, axis=1)
    ac[msk] = np.linalg.norm(a, axis=1)
    return sp, ac


def node_kinematics(run: Dict):
    """节点态的解析速度 / 加速度(对帧的导数,(n,3))。"""
    return run["v_f"], run["a_f"]


def n_tokens(run: Dict) -> int:
    return int(run["n"])


# ---------------------------------------------------------------- 可逆性自检(§10)
def roundtrip_from_nodes(run: Dict, frames) -> Optional[np.ndarray]:
    """用节点态 (τ_i, p_i) 反解控制点再重建:验证"节点态 ↔ 控制多边形"可逆。
    做法:Greville 配置矩阵 B_j(τ_i) 求逆(Schoenberg–Whitney 保证非奇异)。返回重建或 None。"""
    if "parts" in run:
        frames = np.asarray(frames, float)
        out = np.full((len(frames), 3), np.nan)
        for part in run["parts"]:
            r = roundtrip_from_nodes(part, frames)
            if r is None:
                continue
            m = ~np.isnan(r[:, 0]); out[m] = r[m]
        return out
    if run["tck"] is None:
        return None
    t, c, k = run["tck"]
    n = run["n"]
    tau = run["tau"]
    # 配置矩阵:逐基函数求值
    A = np.zeros((n, n))
    for j in range(n):
        e = np.zeros(n); e[j] = 1.0
        A[:, j] = splev(tau, (t, e, k))
    try:
        ctrl = np.linalg.solve(A, run["p"])
    except np.linalg.LinAlgError:
        return None
    tck2 = (t, [ctrl[:, d] for d in range(3)], k)
    frames = np.asarray(frames, float)
    u = (frames - run["f0"]) / max(run["f1"] - run["f0"], 1)
    msk = (u >= 0) & (u <= 1)
    out = np.full((len(frames), 3), np.nan)
    out[msk] = _eval(tck2, u[msk], 0)
    return out
