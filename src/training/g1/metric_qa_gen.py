"""米制 / 绝对时间 / 相对深度 / 部件级 4D QA 生成器(2026-09-12;"token 独有信息才能答的题")。

动机:P1 实测现有事件 QA 8 帧 + 文本先验就能答(rgb 67.4 vs traj 67.9),token 独有的内容——
米制世界坐标、绝对秒、部件结构——没有任何一类题在要。本生成器照 MLLM-4D(arXiv 2603.00515)
附录 8.2 的物理求解式出题,并加上我们对 Stereo4D 深度噪声的可观测性防线(v5 那套)。

数据源(每场景):final_repr.pkl(keypoint 轨迹=token 内容)+ tokens pkl(frame_times_s/fps)
+ gt_stereo4d.npz(GT 相机,只用于噪声分层/冻结相机门/接地措辞/相机族)+ dense_tracks_<iid>.npz
(800 点 GT 世界轨迹,作答案 GT 与 keypoint 质心的一致性核验:答案必须**从 token 侧可复现**)。

任务族(hallucination_type="metric_4d";除相机族外全部只依赖世界系物体轨迹 = token 可答):
  obj_abs_displacement   物体在 [i,j] 的世界系位移(米,4 选 1 数值)      ← MLLM-4D Object Absolute Distance
  obj_speed              物体在 [i,j] 的平均速率(m/s,数值)= 位移 / 绝对时间差(米制 + 绝对时间)
  obj_displacement_compare  X 与 Y 谁在 [i,j] 移动得更远(2 选 1)          ← 我们:比较题不需要尺度算术
  obj_pair_distance_change  X 与 Y 的距离在 [i,j] 变近/变远/基本不变(3 选 1)← MLLM-4D 物-相机相对距离的物-物版
  obj_nearest_to         在第 i 帧,X 和 Y 谁离 Z 更近(2 选 1,Z 可为静态物)  ← "相对深度"的物体相对版
  obj_start_time         X 大约在第几秒开始移动(数值,秒;整段 clip 不给窗口)  ← 绝对时间
  part_motion_max        [i,j] 内 X 的哪个运动部件相对物体质心动得最多(≤4 选 1,部件编号 = token 块里的 [part k])
相机族(needs_camera=True,**相机位姿不进 token(用户红线)→ token 不可答**,只作对照/与 MLLM-4D 对表;单独落盘):
  camera_abs_distance / camera_relative_direction / obj_camera_relative_distance / obj_camera_abs_distance

深度噪声防线(全部显式):
  noise(z) 分层(来自 08-13 实测静态点最大半径中位):z<5m 0.05 / 5-10 0.25 / 10-20 0.53 / 20-40 0.90 / >40 不出题
  "有运动"类答案要求:位移 ≥ max(MIN_DISP, 3·noise) 且冻结相机投影位移 ≥ FROZEN_PX(纯深度方向运动 → 不出题)
  一致性:keypoint 质心(token 侧)与 dense 质心(GT 侧)的位移/位置必须吻合,否则不出题(答案要能从 token 复现)
选项:数值题按"目标名次均匀抽 + 相邻 ≥1.3× 等比"造干扰项(09-09 T-a 名次偏置的教训);类别题全局每类等量。

用法:python -m g1.metric_qa_gen --batch-root data/scenes --tokens data/tokens_v4td \
        --out outputs/metric_qa [--workers 32] [--limit N]
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import pickle
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(_HERE, "..", ".."), os.path.join(_HERE, "..", "..", "tokenization"), os.path.join(_HERE, "..")):
    p = os.path.abspath(p)
    if p not in sys.path:
        sys.path.insert(0, p)
import schema  # noqa: F401,E402
from g1.t8_gen import _centroid_traj  # noqa: E402
from g1.time_qa_gen import _load_camera, _frozen_disp_px, _motion_duration_s, _fmt_s, _seed  # noqa: E402

LETTERS = ("A", "B", "C", "D")
HT = "metric_4d"
FROZEN_PX = 10.0          # 冻结相机可观测门(与 v5 同源)
MIN_DISP = 0.5            # "有运动"类位移下限(米)
MIN_GAP_FRAMES = 3        # 帧对最小间隔
MIN_DT_S = 0.5            # 速率题最小时间差
NOISE_K = 3.0             # 阈值 = NOISE_K × noise(z)
DENSE_MIN_PTS = 20        # dense 质心至少这么多可见点
MAX_PER_FAMILY_CLIP = 2   # 每族每 clip 上限
MAX_PER_CLIP = 16
RATIO_STEP = (1.3, 1.45)  # 数值干扰项相邻等比步长区间
HUMAN = ("person", "man", "woman", "child", "boy", "girl", "people", "pedestrian", "dancer", "player", "kid", "baby", "lady", "guy", "worker", "runner", "skater", "cyclist", "rider")
SPEED_CAP_HUMAN = 6.0     # m/s;超过 = 深度噪声灌爆的位移(冻结相机门只证"有横向运动",管不住径向幅值)
SPEED_CAP_ANY = 30.0
PART_REL_SPEED_CAP = 3.0  # 部件相对质心的速率上限(m/s)
PART_MIN_DT = 0.3         # 部件题最小时间差(秒)
NEAREST_MIN_M = 0.5       # "谁更近"的近者距离下限(排除挎包/帽子这类附着物的文本先验)


def noise_m(z):
    """按相机距离分层的 GT 深度噪声尺度(米);>40m 视为不可测(返回 inf)。"""
    if not np.isfinite(z) or z <= 0:
        return np.inf
    if z < 5: return 0.05
    if z < 10: return 0.25
    if z < 20: return 0.53
    if z < 40: return 0.90
    return np.inf


def cam_z(w2c, X, t):
    E = w2c[t]; return float((E[:3, :3] @ np.asarray(X, float) + E[:3, 3])[2])


def cam_center(w2c, t):
    E = w2c[t]; R, tv = E[:3, :3], E[:3, 3]; return -R.T @ tv


def proj_x(K, w2c, X, t):
    E = w2c[t]; xc = E[:3, :3] @ np.asarray(X, float) + E[:3, 3]
    if xc[2] <= 1e-6: return None
    uv = (K[t] if K.ndim == 3 else K) @ xc; return float(uv[0] / xc[2])


def fmt_num(v, unit):
    if unit == "m/s":
        return f"{v:.2f}" if v < 1 else f"{v:.1f}"
    return f"{v:.2f}" if v < 1 else f"{v:.1f}"


def numeric_options(v, unit, rng):
    """目标名次 r∈{1..4} 均匀抽;r-1 个更小、4-r 个更大的值,相邻等比 U(1.3,1.45)。返回 (opts, ans) 或 None。"""
    for _ in range(6):
        r = int(rng.randint(1, 5))
        vals = [v]; x = v
        for _ in range(r - 1):
            x = x / rng.uniform(*RATIO_STEP); vals.insert(0, x)
        x = v
        for _ in range(4 - r):
            x = x * rng.uniform(*RATIO_STEP); vals.append(x)
        txt = [fmt_num(x, unit) for x in vals]
        if len(set(txt)) == 4 and all(float(t) > 0 for t in txt):
            order = rng.permutation(4)
            opts = {LETTERS[k]: txt[order[k]] for k in range(4)}
            ans = LETTERS[int(np.nonzero(order == r - 1)[0][0])]
            return opts, ans, r
    return None


def cat_options(correct, others, rng):
    items = [correct] + list(others); order = rng.permutation(len(items))
    opts = {LETTERS[k]: items[order[k]] for k in range(len(items))}
    return opts, LETTERS[int(np.nonzero(order == 0)[0][0])]


# ------------------------------------------------------------------ 实例表示
class Inst:
    __slots__ = ("iid", "label", "kind", "c_kp", "vis", "c_dense", "vis_dense", "dense", "dense_vis", "parts", "P", "V", "pid")

    def __init__(self, iid, label, kind):
        self.iid, self.label, self.kind = iid, label, kind
        self.c_kp = self.vis = self.c_dense = self.vis_dense = self.dense = self.dense_vis = None
        self.parts = None; self.P = self.V = self.pid = None

    def pos(self, t):
        """作答 GT 用的位置:优先 dense 质心,回退 keypoint 质心;不可见 → None。"""
        if self.c_dense is not None and self.vis_dense[t]:
            return self.c_dense[t]
        if self.c_kp is not None and self.vis[t]:
            return self.c_kp[t]
        return None

    def consistent(self, t, z):
        """token 侧(keypoint)与 GT 侧(dense)在 t 帧位置吻合(否则答案无法从 token 复现)。"""
        if self.c_dense is None or self.c_kp is None:
            return self.c_kp is not None and self.vis[t]
        if not (self.vis[t] and self.vis_dense[t]):
            return False
        return float(np.linalg.norm(self.c_kp[t] - self.c_dense[t])) <= max(1.0, 0.15 * max(z, 1.0))


def load_scene(sd, tok_path):
    sc = pickle.load(open(tok_path, "rb"))
    times = np.asarray(getattr(sc, "frame_times_s", None), float) if getattr(sc, "frame_times_s", None) is not None else None
    if times is None or len(times) < 4:
        return None, "no_time_axis"
    reprs = pickle.load(open(os.path.join(sd, "final_repr.pkl"), "rb"))
    K, w2c = _load_camera(sd)
    if K is None:
        return None, "no_camera"
    T = len(times)
    if w2c.shape[0] != T:
        return None, "T_mismatch"
    insts = []
    for inst in reprs:
        iid = int(inst["instance_id"]); label = str(inst.get("label") or "object"); ot = inst.get("object_type")
        traj = inst.get("trajectories"); has_traj = traj is not None and np.asarray(traj).size > 0
        has_se3 = inst.get("se3_ok") is not None and np.asarray(inst.get("se3_ok")).any()
        if has_traj or has_se3:
            c, vis = _centroid_traj(inst)
            if c is None or c.shape[0] != T or vis.sum() < 2:
                continue
            it = Inst(iid, label, "nonrigid" if ot == "non-rigid" else "rigid"); it.c_kp, it.vis = c, vis
            dp = os.path.join(sd, f"dense_tracks_{iid}.npz")
            if os.path.exists(dp):
                try:
                    d = np.load(dp); xyz = np.asarray(d["xyz_world"], float); dv = np.asarray(d["vis"], bool)
                    if xyz.shape[1] == T:
                        cd = np.full((T, 3), np.nan); vd = np.zeros(T, bool)
                        for t in range(T):
                            m = dv[:, t] & np.isfinite(xyz[:, t]).all(1)
                            if m.sum() >= DENSE_MIN_PTS:
                                cd[t] = np.median(xyz[m, t], 0); vd[t] = True
                        it.c_dense, it.vis_dense, it.dense, it.dense_vis = cd, vd, xyz, dv
                except Exception:
                    pass
            if has_traj and ot == "non-rigid":
                it.P = np.asarray(traj, float); it.V = np.asarray(inst["visibility"], bool)
                pid = inst.get("part_id"); it.pid = np.asarray(pid).reshape(-1).astype(int) if pid is not None else None
            insts.append(it)
        else:
            rp = inst.get("ref_positions")
            if rp is None: continue
            rp = np.asarray(rp, float).reshape(-1, 3); rp = rp[np.abs(rp).sum(1) > 0]
            if len(rp) == 0: continue
            it = Inst(iid, label, "static"); c = np.repeat(np.median(rp, 0)[None], T, 0)
            it.c_kp, it.vis = c, np.ones(T, bool); insts.append(it)
    return dict(sc=sc, times=times, fps=float(sc.fps_effective), T=T, K=K, w2c=w2c, insts=insts,
                vid=os.path.basename(sd)), None


def descriptors(S, t):
    """接地措辞:label 唯一 → "the X";恰两个同 label 且在 t 帧像面 x 分得开 → "the left/right X";其余不可指代。"""
    by = defaultdict(list)
    for it in S["insts"]: by[it.label].append(it)
    out = {}
    W = 2 * float((S["K"][0] if S["K"].ndim == 3 else S["K"])[0, 2])
    for lbl, L in by.items():
        if len(L) == 1:
            out[L[0].iid] = f"the {lbl}"
        elif len(L) == 2:
            xs = []
            for it in L:
                p = it.pos(t); xs.append(proj_x(S["K"], S["w2c"], p, t) if p is not None else None)
            if None in xs or abs(xs[0] - xs[1]) < 0.15 * W: continue
            a, b = (0, 1) if xs[0] < xs[1] else (1, 0)
            out[L[a].iid] = f"the left {lbl}"; out[L[b].iid] = f"the right {lbl}"
    return out


def names_ok(S, desc, iids, frames):
    """left/right 措辞在问句涉及的每一帧都要与 frame 0 的命名一致;唯一 label 恒 True。"""
    for iid in iids:
        if not desc[iid].startswith("the left ") and not desc[iid].startswith("the right "):
            continue
        for t in frames:
            dt = S["_desc_cache"].get(t)
            if dt is None:
                dt = descriptors(S, t); S["_desc_cache"][t] = dt
            if dt.get(iid) != desc[iid]:
                return False
    return True


def frame_pairs(vis_mask, T, rng, n=4):
    idx = np.where(vis_mask)[0]
    if len(idx) < 2: return []
    pairs = set()
    for _ in range(n * 6):
        i, j = sorted(rng.choice(idx, 2, replace=False))
        if j - i >= MIN_GAP_FRAMES: pairs.add((int(i), int(j)))
        if len(pairs) >= n: break
    return sorted(pairs)


def when(S, i, j):
    return (f"frame {i} and frame {j} of {S['T']} (t = {_fmt_s(S['times'][i])} s and t = {_fmt_s(S['times'][j])} s)")


def mk(S, subtype, q, opts, ans, fr, iids, payload):
    h = hashlib.md5((S["vid"] + subtype + q + json.dumps(opts, sort_keys=True)).encode()).hexdigest()[:8]
    return dict(id=f"{S['vid']}_{subtype}_{h}", video_id=S["vid"], hallucination_type=HT, event_subtype=subtype,
                question=q, options=opts, answer=ans, frame_range=[int(fr[0]), int(fr[1])],
                involved_inst_ids=[int(x) for x in iids], gt_payload=payload)


# ------------------------------------------------------------------ 各任务族
def disp_facts(S, it, i, j):
    """物体在 [i,j] 的位移事实 + 全部门;不过门返回 (None, reason)。"""
    pi, pj = it.pos(i), it.pos(j)
    if pi is None or pj is None: return None, "invisible"
    zi, zj = cam_z(S["w2c"], pi, i), cam_z(S["w2c"], pj, j)
    nz = max(noise_m(zi), noise_m(zj))
    if not np.isfinite(nz): return None, "too_far"
    # 位移题只要求位移(而非逐帧位置)在 token 侧与 GT 侧一致:keypoint 与 dense 质心的公共偏移在位移里相消
    if it.c_kp is not None and not (it.vis[i] and it.vis[j]): return None, "kp_invisible"
    d = float(np.linalg.norm(pj - pi))
    if it.c_kp is not None and it.c_dense is not None:
        dk = float(np.linalg.norm(it.c_kp[j] - it.c_kp[i]))
        if abs(dk - d) > max(0.3, 0.3 * d): return None, "disp_mismatch"
    fz = _frozen_disp_px(S["K"], S["w2c"], pi, pj, i)
    # 径向(沿视线)幅值可信性:深度噪声正沿视线,冻结相机门证不了它。有 dense 点时用"像面尺度变化 ≈ z_i/z_j"核验;
    # 没 dense 时要求径向分量不超过横向分量。
    C = cam_center(S["w2c"], i); nhat = pi - C; nhat = nhat / max(np.linalg.norm(nhat), 1e-9)
    delta = pj - pi; d_rad = abs(float(delta @ nhat)); d_lat = float(np.sqrt(max(d * d - d_rad * d_rad, 0.0)))
    verified = False
    if it.dense is not None:
        m = it.dense_vis[:, i] & it.dense_vis[:, j] & np.isfinite(it.dense[:, i]).all(1) & np.isfinite(it.dense[:, j]).all(1)
        if m.sum() >= DENSE_MIN_PTS:
            def spread(t):
                pts = it.dense[m, t]; E = S["w2c"][t]; xc = pts @ E[:3, :3].T + E[:3, 3]
                ok = xc[:, 2] > 1e-6
                if ok.sum() < DENSE_MIN_PTS: return None
                Kt = S["K"][t] if S["K"].ndim == 3 else S["K"]; uv = (xc[ok] @ Kt.T); uv = uv[:, :2] / uv[:, 2:3]
                return float(np.median(np.linalg.norm(uv - np.median(uv, 0), axis=1)))
            si, sj = spread(i), spread(j)
            if si is not None and sj is not None and si >= 2 and sj >= 2:
                r_exp = zi / max(zj, 1e-6)
                if abs(np.log(r_exp)) > np.log(1.15) and abs(np.log(sj / si) - np.log(r_exp)) > np.log(1.3):
                    return None, "radial_scale_mismatch"
                verified = True
    if not verified and d_rad > max(d_lat, NOISE_K * nz):
        return None, "radial_unverifiable"
    dt = float(S["times"][j] - S["times"][i])
    cap = SPEED_CAP_HUMAN if any(h in it.label.lower() for h in HUMAN) else SPEED_CAP_ANY
    if dt > 0 and d / dt > cap: return None, "speed_cap"
    return dict(d=d, z=max(zi, zj), noise=nz, fz=fz, pi=pi, pj=pj, d_rad=d_rad, d_lat=d_lat), None


def fam_obj_displacement(S, desc, rng, rej):
    out = []
    for it in S["insts"]:
        if it.kind == "static" or it.iid not in desc: continue
        for (i, j) in frame_pairs(it.vis, S["T"], rng):
            f, why = disp_facts(S, it, i, j)
            if f is None: rej[why] += 1; continue
            if not names_ok(S, desc, [it.iid], (i, j)): rej["name_swap"] += 1; continue
            if f["d"] < max(MIN_DISP, NOISE_K * f["noise"]): rej["disp_below_gate"] += 1; continue
            if f["fz"] is None or f["fz"] < FROZEN_PX: rej["frozen_px"] += 1; continue
            q = f"In the world frame (ignoring camera motion), approximately how far (in meters) did {desc[it.iid]} move between {when(S, i, j)}?"
            r = numeric_options(f["d"], "m", rng)
            if r is None: continue
            opts, ans, rank = r
            out.append(mk(S, "obj_abs_displacement", q, opts, ans, (i, j), [it.iid],
                          dict(value_m=round(f["d"], 3), obj_dist_m=round(f["z"], 2), noise_m=f["noise"], frozen_px=round(f["fz"], 1), answer_rank=rank, needs_camera=False,
                               pos_i=[round(float(x), 2) for x in f["pi"]], pos_j=[round(float(x), 2) for x in f["pj"]], obj=desc[it.iid])))
            dt = S["times"][j] - S["times"][i]
            if dt >= MIN_DT_S:
                v = f["d"] / dt
                q2 = f"In the world frame (ignoring camera motion), what was the average speed (in meters per second) of {desc[it.iid]} between {when(S, i, j)}?"
                r2 = numeric_options(v, "m/s", rng)
                if r2 is not None:
                    opts, ans, rank = r2
                    out.append(mk(S, "obj_speed", q2, opts, ans, (i, j), [it.iid],
                                  dict(value_mps=round(v, 3), disp_m=round(f["d"], 3), dt_s=round(float(dt), 3), obj_dist_m=round(f["z"], 2), noise_m=f["noise"], frozen_px=round(f["fz"], 1), answer_rank=rank, needs_camera=False,
                                       pos_i=[round(float(x), 2) for x in f["pi"]], pos_j=[round(float(x), 2) for x in f["pj"]], obj=desc[it.iid])))
            if sum(1 for o in out if o["involved_inst_ids"] == [it.iid] and o["event_subtype"] == "obj_abs_displacement") >= MAX_PER_FAMILY_CLIP: break
    return out


def fam_compare(S, desc, rng, rej):
    out = []
    dyn = [it for it in S["insts"] if it.kind != "static" and it.iid in desc]
    for a in range(len(dyn)):
        for b in range(a + 1, len(dyn)):
            X, Y = dyn[a], dyn[b]
            for (i, j) in frame_pairs(X.vis & Y.vis, S["T"], rng, n=3):
                fx, wx = disp_facts(S, X, i, j); fy, wy = disp_facts(S, Y, i, j)
                if fx is None or fy is None: rej[wx or wy] += 1; continue
                if not names_ok(S, desc, [X.iid, Y.iid], (i, j)): rej["name_swap"] += 1; continue
                gate = max(MIN_DISP, NOISE_K * max(fx["noise"], fy["noise"]))
                big, small = (X, Y) if fx["d"] >= fy["d"] else (Y, X); fb, fs = (fx, fy) if fx["d"] >= fy["d"] else (fy, fx)
                if fb["d"] < 1.5 * fs["d"] or fb["d"] - fs["d"] < gate: rej["compare_margin"] += 1; continue
                if fb["fz"] is None or fb["fz"] < FROZEN_PX: rej["frozen_px"] += 1; continue
                q = f"In the world frame (ignoring camera motion), which object moved a longer distance between {when(S, i, j)}?"
                opts, ans = cat_options(desc[big.iid], [desc[small.iid]], rng)
                out.append(mk(S, "obj_displacement_compare", q, opts, ans, (i, j), [X.iid, Y.iid],
                              dict(disp_m={desc[X.iid]: round(fx["d"], 3), desc[Y.iid]: round(fy["d"], 3)}, ratio=round(fb["d"] / max(fs["d"], 1e-6), 2), needs_camera=False, winner=desc[big.iid])))
                break
    return out


def fam_pair_distance(S, desc, rng, rej):
    out = []
    dyn = [it for it in S["insts"] if it.kind != "static" and it.iid in desc]
    others = [it for it in S["insts"] if it.iid in desc]
    cls_cnt = Counter()
    for X in dyn:
        for Y in others:
            if Y.iid == X.iid: continue
            if Y.kind != "static" and Y.iid < X.iid: continue      # 动-动对只算一次
            for (i, j) in frame_pairs(X.vis & Y.vis, S["T"], rng, n=3):
                pxi, pxj, pyi, pyj = X.pos(i), X.pos(j), Y.pos(i), Y.pos(j)
                if any(p is None for p in (pxi, pxj, pyi, pyj)): rej["invisible"] += 1; continue
                if not names_ok(S, desc, [X.iid, Y.iid], (i, j)): rej["name_swap"] += 1; continue
                zs = [cam_z(S["w2c"], p, t) for p, t in ((pxi, i), (pxj, j), (pyi, i), (pyj, j))]
                nz = sum(noise_m(z) for z in zs[:2]) / 2 + sum(noise_m(z) for z in zs[2:]) / 2
                if not np.isfinite(nz): rej["too_far"] += 1; continue
                if not (X.consistent(i, zs[0]) and X.consistent(j, zs[1]) and Y.consistent(i, zs[2]) and Y.consistent(j, zs[3])): rej["kp_dense_mismatch"] += 1; continue
                d0, d1 = float(np.linalg.norm(pxi - pyi)), float(np.linalg.norm(pxj - pyj)); delta = d1 - d0
                tau = max(0.5, NOISE_K * nz)
                if delta >= tau: cls = "farther apart"
                elif delta <= -tau: cls = "closer together"
                elif abs(delta) <= 0.3 * tau: cls = "about the same distance apart"
                else: rej["pair_ambiguous"] += 1; continue
                if cls != "about the same distance apart":
                    fz = _frozen_disp_px(S["K"], S["w2c"], pxi, pxj, i)
                    if fz is None or fz < FROZEN_PX: rej["frozen_px"] += 1; continue
                q = f"In the world frame (ignoring camera motion), between {when(S, i, j)}, did {desc[X.iid]} and {desc[Y.iid]} end up closer together, farther apart, or about the same distance apart?"
                opts, ans = cat_options(cls, [c for c in ("closer together", "farther apart", "about the same distance apart") if c != cls], rng)
                out.append(mk(S, "obj_pair_distance_change", q, opts, ans, (i, j), [X.iid, Y.iid],
                              dict(dist_i_m=round(d0, 3), dist_j_m=round(d1, 3), delta_m=round(delta, 3), tau_m=round(tau, 3), cls=cls, needs_camera=False, objs=[desc[X.iid], desc[Y.iid]])))
                cls_cnt[cls] += 1
                break
        if sum(cls_cnt.values()) >= MAX_PER_FAMILY_CLIP * 2: break
    return out


def fam_nearest(S, desc, rng, rej):
    out = []
    ids = [it for it in S["insts"] if it.iid in desc]
    if len(ids) < 3: return out
    tries = 0
    for Z in ids:
        cands = [it for it in ids if it.iid != Z.iid]
        for a in range(len(cands)):
            for b in range(a + 1, len(cands)):
                X, Y = cands[a], cands[b]
                if X.kind == "static" and Y.kind == "static" and Z.kind == "static": continue   # 全静态三元组不出(与 4D 无关)
                fr = np.where(X.vis & Y.vis & Z.vis)[0]
                if len(fr) == 0: continue
                t = int(rng.choice(fr)); tries += 1
                px, py, pz = X.pos(t), Y.pos(t), Z.pos(t)
                if any(p is None for p in (px, py, pz)): continue
                if not names_ok(S, desc, [X.iid, Y.iid, Z.iid], (t,)): rej["name_swap"] += 1; continue
                zs = [cam_z(S["w2c"], p, t) for p in (px, py, pz)]
                nz = sum(noise_m(z) for z in zs)
                if not np.isfinite(nz): rej["too_far"] += 1; continue
                if not all(it.consistent(t, z) for it, z in zip((X, Y, Z), zs)): rej["kp_dense_mismatch"] += 1; continue
                dx, dy = float(np.linalg.norm(px - pz)), float(np.linalg.norm(py - pz))
                near, far = (X, Y) if dx < dy else (Y, X); dn, df = min(dx, dy), max(dx, dy)
                if dn < NEAREST_MIN_M: rej["nearest_attached"] += 1; continue
                if df < 1.5 * dn or df - dn < max(1.0, NOISE_K * nz): rej["nearest_margin"] += 1; continue
                # 2D 像面邻近(基座零样本靠它 74%):记录 2D 最近是否与 3D 一致,平衡器按 agree/disagree 50/50 → 2D 线索失效
                def uv(P):
                    E = S["w2c"][t]; xc = E[:3, :3] @ P + E[:3, 3]
                    if xc[2] <= 1e-6: return None
                    Kt = S["K"][t] if S["K"].ndim == 3 else S["K"]; u = Kt @ xc; return u[:2] / xc[2]
                ux, uy, uz = uv(px), uv(py), uv(pz)
                if ux is None or uy is None or uz is None: rej["behind_cam"] += 1; continue
                d2x, d2y = float(np.linalg.norm(ux - uz)), float(np.linalg.norm(uy - uz))
                agree = (d2x < d2y) == (dx < dy)
                if agree and max(d2x, d2y) > 1.3 * min(d2x, d2y): rej["nearest_2d_easy"] += 1; continue   # 2D 与 3D 一致且 2D 差得明显 → RGB 可答,不出
                q = f"At frame {t} of {S['T']} (t = {_fmt_s(S['times'][t])} s), which object is closer to {desc[Z.iid]} in 3D space: {desc[X.iid]} or {desc[Y.iid]}?"
                opts, ans = cat_options(desc[near.iid], [desc[far.iid]], rng)
                out.append(mk(S, "obj_nearest_to", q, opts, ans, (t, t), [X.iid, Y.iid, Z.iid],
                              dict(dist_m={desc[X.iid]: round(dx, 3), desc[Y.iid]: round(dy, 3)}, ratio=round(df / max(dn, 1e-6), 2),
                                   img_dist_px={desc[X.iid]: round(d2x, 1), desc[Y.iid]: round(d2y, 1)}, agree2d=bool(agree), cls=("2d_agree" if agree else "2d_disagree") + "|" + "", needs_camera=False)))
                if len(out) >= MAX_PER_FAMILY_CLIP: return out
    return out


def cam_obj_dist(S, it, t_cam, t_obj):
    """相机光心(t_cam)到物体(t_obj)的最近点距离(MLLM-4D obj-cam abs distance 口径);无 dense 时回退质心距离。"""
    C = cam_center(S["w2c"], t_cam)
    if it.dense is not None:
        m = it.dense_vis[:, t_obj] & np.isfinite(it.dense[:, t_obj]).all(1)
        if m.sum() >= DENSE_MIN_PTS:
            return float(np.linalg.norm(it.dense[m, t_obj] - C, axis=1).min())
    p = it.pos(t_obj)
    return None if p is None else float(np.linalg.norm(p - C))


def fam_camera_nearest(S, desc, rng, rej):
    """相对深度(相机-物体版,与 MLLM-4D 参考一致):在第 t 帧,X 和 Y 谁离相机更近。"""
    out = []
    ids = [it for it in S["insts"] if it.iid in desc]
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            X, Y = ids[a], ids[b]
            if X.kind == "static" and Y.kind == "static" and len(out) >= 1: continue   # 静-静对最多 1 题
            fr = np.where(X.vis & Y.vis)[0]
            if len(fr) == 0: continue
            t = int(rng.choice(fr))
            if not names_ok(S, desc, [X.iid, Y.iid], (t,)): rej["name_swap"] += 1; continue
            dx, dy = cam_obj_dist(S, X, t, t), cam_obj_dist(S, Y, t, t)
            if dx is None or dy is None: continue
            nz = noise_m(dx) + noise_m(dy)
            if not np.isfinite(nz): rej["too_far"] += 1; continue
            px, py = X.pos(t), Y.pos(t)
            if px is None or py is None or not (X.consistent(t, cam_z(S["w2c"], px, t)) and Y.consistent(t, cam_z(S["w2c"], py, t))): rej["kp_dense_mismatch"] += 1; continue
            near, far = (X, Y) if dx < dy else (Y, X); dn, df = min(dx, dy), max(dx, dy)
            if df < 1.5 * dn or df - dn < max(1.0, NOISE_K * nz): rej["camnear_margin"] += 1; continue
            q = f"At frame {t} of {S['T']} (t = {_fmt_s(S['times'][t])} s), which object is closer to the camera (the observer filming): {desc[X.iid]} or {desc[Y.iid]}?"
            opts, ans = cat_options(desc[near.iid], [desc[far.iid]], rng)
            out.append(mk(S, "obj_camera_nearest", q, opts, ans, (t, t), [X.iid, Y.iid],
                          dict(dist_m={desc[X.iid]: round(dx, 3), desc[Y.iid]: round(dy, 3)}, ratio=round(df / max(dn, 1e-6), 2), needs_camera=True, winner=desc[near.iid])))
            if len(out) >= MAX_PER_FAMILY_CLIP: return out
    return out


def fam_start_time(S, desc, rng, rej):
    out = []
    by_id = {}
    reprs = pickle.load(open(os.path.join(S["sdir"], "final_repr.pkl"), "rb"))
    for inst in reprs: by_id[int(inst["instance_id"])] = inst
    for it in S["insts"]:
        if it.kind == "static" or it.iid not in desc: continue
        md = _motion_duration_s(by_id[it.iid], S["times"], S["K"], S["w2c"])
        if md is None: rej["no_motion_run"] += 1; continue
        dur, fa, fb = md
        if not names_ok(S, desc, [it.iid], (fa, fb)): rej["name_swap"] += 1; continue
        t0 = float(S["times"][fa])
        vis_before = int(it.vis[:fa].sum())
        if t0 < 0.5 or vis_before < 3: rej["start_trivial"] += 1; continue     # 一开始就在动/之前看不见 → 不成题
        q = f"In the world frame (ignoring camera motion), at approximately what time (in seconds from the start of the clip) does {desc[it.iid]} start moving?"
        r = numeric_options(t0, "s", rng)
        if r is None: continue
        opts, ans, rank = r
        out.append(mk(S, "obj_start_time", q, opts, ans, (0, S["T"] - 1), [it.iid],
                      dict(value_s=round(t0, 3), start_frame=int(fa), move_frames=[int(fa), int(fb)], answer_rank=rank, needs_camera=False, obj=desc[it.iid], end_s=round(float(S["times"][fb]), 3), fps=round(float(S["fps"]), 3))))
        if len(out) >= MAX_PER_FAMILY_CLIP: break
    return out


def fam_part_motion(S, desc, rng, rej):
    out = []
    for it in S["insts"]:
        if it.kind != "nonrigid" or it.P is None or it.pid is None or it.iid not in desc: continue
        for (i, j) in frame_pairs(it.vis, S["T"], rng, n=3):
            m = it.V[:, i] & it.V[:, j]
            if not names_ok(S, desc, [it.iid], (i, j)): rej["name_swap"] += 1; continue
            dt = float(S["times"][j] - S["times"][i])
            if dt < PART_MIN_DT: rej["part_dt"] += 1; continue
            parts = sorted(set(it.pid[m].tolist()))
            if len(parts) < 3: rej["parts_lt3"] += 1; break
            cen_d = it.P[m, j] - it.P[m, i]; cen = cen_d.mean(0)
            rel = {}
            for p in parts:
                mm = m & (it.pid == p); rel[p] = (it.P[mm, j] - it.P[mm, i]).mean(0) - cen
            mags = {p: float(np.linalg.norm(v)) for p, v in rel.items()}
            order = sorted(mags, key=mags.get, reverse=True); win, second = order[0], order[1]
            z = cam_z(S["w2c"], it.c_kp[i], i); nz = noise_m(z)
            if not np.isfinite(nz): rej["too_far"] += 1; continue
            if mags[win] < max(0.25, NOISE_K * nz * 1.4) or mags[win] < 1.5 * mags[second]: rej["part_margin"] += 1; continue
            if mags[win] / dt > PART_REL_SPEED_CAP: rej["part_speed_cap"] += 1; continue
            pw = it.P[m & (it.pid == win), i].mean(0)
            fz = _frozen_disp_px(S["K"], S["w2c"], pw, pw + rel[win], i)
            if fz is None or fz < FROZEN_PX: rej["frozen_px"] += 1; continue
            shown = [win] + [p for p in order[1:]][:3]
            names = {p: f"part {p % 16 + 1}" for p in shown}          # 与 token 块 [part k] 编号一致(桶 = part_id%16+1)
            q = f"Between {when(S, i, j)}, which motion part of {desc[it.iid]} moved the most relative to the object's own center (part numbering as in the trajectory tokens)?"
            opts, ans = cat_options(names[win], [names[p] for p in shown[1:]], rng)
            out.append(mk(S, "part_motion_max", q, opts, ans, (i, j), [it.iid],
                          dict(rel_disp_m={names[p]: round(mags[p], 3) for p in shown}, obj_dist_m=round(z, 2), noise_m=nz, frozen_px=round(fz, 1), needs_camera=False, winner=names[win], obj=desc[it.iid])))
            break
        if len(out) >= MAX_PER_FAMILY_CLIP: break
    return out


def fam_camera(S, desc, rng, rej):
    """相机族(MLLM-4D 8.2 原式;GT 位姿无深度噪声)。needs_camera=True → token 不可答,单独落盘。"""
    out = []; T = S["T"]; w2c = S["w2c"]
    C = np.stack([cam_center(w2c, t) for t in range(T)])
    for (i, j) in frame_pairs(np.ones(T, bool), T, rng, n=3):
        d = float(np.linalg.norm(C[j] - C[i]))
        if d >= 0.5:
            q = f"Approximately how far (in meters) did the camera move between {when(S, i, j)}?"
            r = numeric_options(d, "m", rng)
            if r is not None:
                opts, ans, rank = r
                out.append(mk(S, "camera_abs_distance", q, opts, ans, (i, j), [], dict(value_m=round(d, 3), answer_rank=rank, needs_camera=True,
                                                                                         cam_i=[round(float(x), 2) for x in C[i]], cam_j=[round(float(x), 2) for x in C[j]])))
            dl = w2c[i][:3, :3] @ (C[j] - C[i])            # 相机 i 坐标系:+x 右 +y 下 +z 前
            ax = np.abs(dl); k = int(np.argmax(ax))
            if k != 1 and ax[k] >= 2 * max(ax[(k + 1) % 3], ax[(k + 2) % 3]) and ax[k] >= 0.5:
                cls = ("right" if dl[0] > 0 else "left") if k == 0 else ("forward" if dl[2] > 0 else "backward")
                q = f"Between {when(S, i, j)}, what was the primary translation of the camera relative to its own orientation at frame {i}?"
                opts, ans = cat_options(cls, [c for c in ("forward", "backward", "left", "right") if c != cls], rng)
                out.append(mk(S, "camera_relative_direction", q, opts, ans, (i, j), [], dict(local_disp=[round(float(x), 3) for x in dl], cls=cls, needs_camera=True)))
        break
    for it in S["insts"]:
        if it.kind == "static" or it.iid not in desc or it.dense is None: continue
        for (i, j) in frame_pairs(it.vis_dense if it.vis_dense is not None else it.vis, T, rng, n=2):
            def mind(t_cam, t_obj):
                m = it.dense_vis[:, t_obj] & np.isfinite(it.dense[:, t_obj]).all(1)
                return float(np.linalg.norm(it.dense[m, t_obj] - C[t_cam], axis=1).min()) if m.sum() >= DENSE_MIN_PTS else None
            di, dj = mind(i, i), mind(j, j)
            if di is None or dj is None: continue
            nz = noise_m(di) + noise_m(dj)
            if not np.isfinite(nz): rej["too_far"] += 1; continue
            tau = max(1.0, NOISE_K * nz); delta = dj - di
            cls = "farther" if delta >= tau else "closer" if delta <= -tau else ("almost not moving" if abs(delta) <= 0.3 * tau else None)
            if cls is None: rej["cam_pair_ambiguous"] += 1; continue
            # Object-Camera Relative Direction(8.2):物体质心在相机 i 坐标系里的横向变化 Δx(+x 右)
            pi_, pj_ = it.pos(i), it.pos(j)
            if pi_ is not None and pj_ is not None:
                Ei = w2c[i]; xi = Ei[:3, :3] @ pi_ + Ei[:3, 3]; xj = Ei[:3, :3] @ pj_ + Ei[:3, 3]; dxl = float(xj[0] - xi[0])
                nzl = noise_m(float(xi[2])) + noise_m(float(xj[2])); taul = max(0.5, NOISE_K * nzl)
                if np.isfinite(nzl):
                    clsl = "right" if dxl >= taul else "left" if dxl <= -taul else ("almost not moving" if abs(dxl) <= 0.3 * taul else None)
                    if clsl is not None and (clsl == "almost not moving" or (_frozen_disp_px(S["K"], w2c, pi_, pj_, i) or 0) >= FROZEN_PX):
                        ql = f"Between {when(S, i, j)}, is {desc[it.iid]} moving to the left or to the right relative to the camera's orientation at frame {i}?"
                        optsl, ansl = cat_options(clsl, [c for c in ("left", "right", "almost not moving") if c != clsl], rng)
                        out.append(mk(S, "obj_camera_relative_direction", ql, optsl, ansl, (i, j), [it.iid], dict(dx_cam_m=round(dxl, 3), tau_m=round(taul, 3), cls=clsl, needs_camera=True, obj=desc[it.iid],
                                                                                                                       x_cam_i=round(float(xi[0]), 2), x_cam_j=round(float(xj[0]), 2))))
            q = f"Between {when(S, i, j)}, is the distance between {desc[it.iid]} and the camera (the observer filming) getting closer or farther?"
            opts, ans = cat_options(cls, [c for c in ("closer", "farther", "almost not moving") if c != cls], rng)
            out.append(mk(S, "obj_camera_relative_distance", q, opts, ans, (i, j), [it.iid], dict(dist_i_m=round(di, 3), dist_j_m=round(dj, 3), cls=cls, needs_camera=True, obj=desc[it.iid])))
            dj2 = mind(i, j)
            if dj2 is not None and np.isfinite(noise_m(dj2)):
                q = f"What is the approximate distance (in meters) between the camera (the observer filming) at frame {i} and the nearest point of {desc[it.iid]} at frame {j} of {T}?"
                r = numeric_options(dj2, "m", rng)
                if r is not None:
                    opts, ans, rank = r
                    out.append(mk(S, "obj_camera_abs_distance", q, opts, ans, (i, j), [it.iid], dict(value_m=round(dj2, 3), answer_rank=rank, needs_camera=True, obj=desc[it.iid],
                                                                                                             cam_i=[round(float(x), 2) for x in C[i]])))
            break
    return out


FAMILIES = (fam_obj_displacement, fam_compare, fam_pair_distance, fam_camera_nearest, fam_start_time, fam_part_motion)
FAMILIES_OBJ_NEAREST = (fam_nearest,)     # 物体相对版"谁更近"(基座 rgb 零样本 71%,非 token 独有),默认不出


def gen_scene(args):
    sd, tok_path, with_camera, with_obj_nearest = args
    rej = Counter()
    try:
        S, why = load_scene(sd, tok_path)
        if S is None: return [], [], Counter({why: 1}), 0
        S["sdir"] = sd; S["_desc_cache"] = {}; S["_with_obj_nearest"] = with_obj_nearest
        rng = _seed(S["vid"], "metric")
        desc = descriptors(S, 0); S["_desc_cache"][0] = desc
        items = []
        for fam in FAMILIES:
            items += fam(S, desc, rng, rej)
        if with_camera:                              # 2026-09-13:相机进 token 后,相机族并入主集
            items += fam_camera(S, desc, rng, rej)
        if S.get("_with_obj_nearest"):
            for fam in FAMILIES_OBJ_NEAREST: items += fam(S, desc, rng, rej)
        cam = []
        if len(items) > MAX_PER_CLIP:
            rng.shuffle(items); items = items[:MAX_PER_CLIP]
        return items, cam, rej, len(desc)
    except Exception as e:  # 单场景失败只记账
        return [], [], Counter({f"exc:{type(e).__name__}": 1}), 0


NUMERIC_KEY = {"obj_abs_displacement": "value_m", "obj_speed": "value_mps", "obj_start_time": "value_s", "camera_abs_distance": "value_m", "obj_camera_abs_distance": "value_m"}


def redistractor_prior(items, rng):
    """数值题干扰项重采样(2026-09-13 定案):按该族真值的经验分布采样(+10% 抖动),与真值/彼此 ≥1.3×;时间题吸附到本 clip 的帧网格。
    原"围绕真值的等比阶梯"有两处盲泄漏(阶梯绝对位置随真值移动 + 时间题真值在 1/fps 网格而干扰项不在),A-rgb 实测 blind 61%→25%。
    答案名次由随机打乱保证均匀。"""
    import random as _r
    pool = defaultdict(list)
    for d in items:
        st = d["event_subtype"]
        if st in NUMERIC_KEY: pool[st].append(float(d["gt_payload"][NUMERIC_KEY[st]]))
    r = _r.Random(int(rng.randint(1 << 30))); n_ok = 0
    for d in items:
        st = d["event_subtype"]
        if st not in NUMERIC_KEY or len(pool[st]) < 50: continue
        v = float(d["gt_payload"][NUMERIC_KEY[st]]); unit = "m/s" if st == "obj_speed" else "m"; picks = []
        fps = float(d["gt_payload"].get("fps", 0)) if st == "obj_start_time" else 0
        for _ in range(400):
            c = float(r.choice(pool[st])) * r.uniform(0.9, 1.1)
            if fps > 0: c = round(c * fps) / fps
            if c <= 0: continue
            if all(abs(np.log(c / x)) >= np.log(1.3) for x in [v] + picks) and fmt_num(c, unit) != fmt_num(v, unit) and all(fmt_num(c, unit) != fmt_num(q, unit) for q in picks):
                picks.append(c)
            if len(picks) == 3: break
        if len(picks) < 3: continue
        vals = [v] + picks; order = list(range(4)); r.shuffle(order)
        d["options"] = {LETTERS[k]: fmt_num(vals[order[k]], unit) for k in range(4)}
        d["answer"] = LETTERS[order.index(0)]; d["gt_payload"]["answer_rank"] = int(sorted(vals).index(v)) + 1; d["gt_payload"]["distractor"] = "prior"; n_ok += 1
    return n_ok


def balance_categorical(items, rng):
    """类别题:每个子类内各答案文本等量(全局);数值题不动(名次已均匀)。"""
    by = defaultdict(list)
    for d in items: by[d["event_subtype"]].append(d)
    out = []
    for st, L in by.items():
        if "answer_rank" in L[0]["gt_payload"]:
            out += L; continue
        cls = defaultdict(list)
        for d in L:
            key = d["gt_payload"].get("cls") or d["answer"]      # 2-way/部件题:按答案字母等量(消位置偏置)
            if st == "obj_nearest_to":
                key = ("agree" if d["gt_payload"].get("agree2d") else "disagree") + "|" + d["answer"]   # 2D 线索 × 字母 四格等量
            cls[key].append(d)
        if len(cls) <= 1:
            out += L; continue
        k = min(len(v) for v in cls.values())
        for v in cls.values():
            idx = rng.permutation(len(v))[:k]; out += [v[i] for i in sorted(idx)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-root", default="data/scenes")
    ap.add_argument("--tokens", default="data/tokens_v4td")
    ap.add_argument("--out", default="outputs/metric_qa")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-camera", action="store_true", help="不生成相机族")
    ap.add_argument("--with-obj-nearest", action="store_true", help="附带物体相对版 obj_nearest_to(默认不出)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    scenes = sorted(glob.glob(os.path.join(a.batch_root, "scene_*", "final_repr.pkl")))
    jobs = []
    for f in scenes:
        sd = os.path.dirname(f); tp = os.path.join(a.tokens, os.path.basename(sd) + ".pkl")
        if os.path.exists(tp): jobs.append((sd, tp, not a.no_camera, a.with_obj_nearest))
    if a.limit: jobs = jobs[:a.limit]
    print(f"[metric] scenes={len(jobs)} workers={a.workers}", flush=True)
    items, cams, rej, n_desc = [], [], Counter(), 0
    with Pool(a.workers) as pool:
        for k, (it, cm, rj, nd) in enumerate(pool.imap(gen_scene, jobs, chunksize=16)):
            items += it; cams += cm; rej.update(rj); n_desc += nd
            if (k + 1) % 2000 == 0: print(f"  {k+1}/{len(jobs)} items={len(items)} cam={len(cams)}", flush=True)
    rng = np.random.RandomState(0)
    raw_counts = Counter(d["event_subtype"] for d in items)
    raw_cls = {st: dict(Counter(d["gt_payload"].get("cls") or d["answer"] for d in items if d["event_subtype"] == st)) for st in raw_counts}
    items = balance_categorical(items, rng); cams = balance_categorical(cams, rng)
    n_re = redistractor_prior(items, rng) + redistractor_prior(cams, rng)
    print(f"[metric] 数值题干扰项按先验重采样:{n_re} 题", flush=True)
    for name, L in (("metric_qa", items), ("metric_qa_camera", cams)):
        with open(os.path.join(a.out, name + ".jsonl"), "w") as fh:
            for d in L: fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    def dist(L):
        r = {}
        for st in sorted({d["event_subtype"] for d in L}):
            sub = [d for d in L if d["event_subtype"] == st]
            if "answer_rank" in sub[0]["gt_payload"]:
                r[st] = dict(n=len(sub), rank=dict(Counter(d["gt_payload"]["answer_rank"] for d in sub)))
            else:
                r[st] = dict(n=len(sub), cls=dict(Counter(d["gt_payload"].get("cls") or d["answer"] for d in sub)))
        return r
    rep = dict(n_scenes=len(jobs), n_groundable_desc=n_desc, rejections=dict(rej.most_common()), raw_counts=dict(raw_counts), raw_cls=raw_cls,
               metric_qa=dist(items), metric_qa_camera=dist(cams), n_metric=len(items), n_camera=len(cams),
               n_scenes_with_qa=len({d["video_id"] for d in items}))
    json.dump(rep, open(os.path.join(a.out, "_gen_report.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(rep, ensure_ascii=False, indent=1), flush=True)


if __name__ == "__main__":
    main()
