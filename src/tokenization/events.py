"""事件检测 (T3 反转 / T5 冲击) + 运动学 (T2) + 事件匹配。

科学纪律:GT 与各表示(贝塞尔重建 / PL 重建)喂给 **完全相同** 的检测器/运动学管线,
只输入不同(真值位置 vs 重建位置)。检测器阈值如何调不影响公平性 —— 比的是
"重建轨迹是否让答案存活",而非检测器绝对精度。
所有量只在 *可见 run* 内计算;不跨遮挡缺口做差分。
"""
from __future__ import annotations
import numpy as np


def visible_runs(vis: np.ndarray):
    """vis: (T,) bool -> [(start,end_inclusive), ...] 连续可见段。"""
    runs = []
    T = len(vis)
    i = 0
    while i < T:
        if vis[i]:
            j = i
            while j + 1 < T and vis[j + 1]:
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1
    return runs


def finite_diff_kin(frames, pos):
    """在一个 run 内(整数帧连续)对位置做有限差分。
    速度 v[k] = (pos[k+1]-pos[k-1])/2 (中心差分), 端点单侧。加速度同理二阶。
    返回 speed(len,), accel(len,) 与 frames 对齐, 端点为 nan。"""
    n = len(frames)
    speed = np.full(n, np.nan)
    accel = np.full(n, np.nan)
    if n >= 2:
        vel = np.full((n, 3), np.nan)
        vel[1:-1] = (pos[2:] - pos[:-2]) / 2.0
        vel[0] = pos[1] - pos[0]
        vel[-1] = pos[-1] - pos[-2]
        speed = np.linalg.norm(vel, axis=1)
        if n >= 3:
            acc = np.full((n, 3), np.nan)
            acc[1:-1] = pos[2:] - 2 * pos[1:-1] + pos[:-2]
            accel = np.linalg.norm(acc, axis=1)
    return speed, accel


def detect_reversals(frames, pos, bbox_diag):
    """方向反转 (T3):连续速度向量夹角>90° (点积<0) 且两速度模均超噪声地板。
    返回反转事件帧列表(取后一速度所在帧)。"""
    n = len(frames)
    if n < 3:
        return []
    vel = pos[1:] - pos[:-1]                              # (n-1,3) 帧间位移
    sp = np.linalg.norm(vel, axis=1)
    floor = max(0.02 * bbox_diag, 0.1 * np.median(sp[sp > 0]) if np.any(sp > 0) else 0.0)
    ev = []
    for k in range(1, len(vel)):
        if sp[k] > floor and sp[k - 1] > floor:
            if np.dot(vel[k], vel[k - 1]) < 0:
                ev.append(int(frames[k]))                # 反转发生在 frames[k]
    return ev


def detect_impacts(frames, pos, bbox_diag):
    """冲击/碰撞 (T5):加速度模的局部尖峰, 超过 (mean+2std) 且超地板。返回尖峰帧列表。"""
    n = len(frames)
    if n < 3:
        return []
    _, accel = finite_diff_kin(frames, pos)
    a = accel.copy()
    valid = ~np.isnan(a)
    if valid.sum() < 3:
        return []
    av = a[valid]
    thr = max(av.mean() + 2 * av.std(), 0.05 * bbox_diag)
    ev = []
    for k in range(1, n - 1):
        if np.isnan(a[k]):
            continue
        # 局部极大 且 超阈
        left = a[k - 1] if not np.isnan(a[k - 1]) else -np.inf
        right = a[k + 1] if not np.isnan(a[k + 1]) else -np.inf
        if a[k] >= left and a[k] >= right and a[k] > thr:
            ev.append(int(frames[k]))
    return ev


def match_events(gt_ev, pred_ev, tol=1):
    """按时间贪心匹配 (±tol 帧)。返回 (tp, n_gt, n_pred)。"""
    gt = sorted(gt_ev)
    pred = sorted(pred_ev)
    used = [False] * len(pred)
    tp = 0
    for g in gt:
        for i, p in enumerate(pred):
            if not used[i] and abs(p - g) <= tol:
                used[i] = True
                tp += 1
                break
    return tp, len(gt), len(pred)
