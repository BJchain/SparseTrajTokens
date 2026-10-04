"""B 集(trajectory-sensitive QA)候选生成器 v1(2026-09-14)。

定位:与 A 集(MLLM-4D 风格通用 4D QA)分开报数,只出"RGB 8 帧难答、轨迹 token 能答"的题。
本文件只产 **候选 + GT 值 + 载荷**,不配选项;网格/干扰项/配平/三分在 trajqa_b_build.py(先过 Gate 0:
token 侧逐题复算,src/tools/trajqa_b_gate0.py)。

族(wave 1;路径长度/曲率(深度噪声灌水不可证伪)、最近时刻(内部极小 <1%)推迟到 wave 2):
  disp_grid                物体在 [i,j] 的世界系位移(米;固定 log 网格 4 选 1)              尺度相关
  speed_grid               同窗口平均速率(m/s);题面不给秒 → 必须读 tsec                     尺度相关+绝对时间
  interval_disp            同一物体两个相邻区间谁位移大(2 选 1)                              尺度无关
  interval_speed           同一物体两个相邻区间谁更快(2 选 1;区间时长不等,与位移答案 50% 相反) 尺度无关+时间
  motion_duration_binary   复用 time_qa_gen 的 T-d 二分(整段 clip 协议;build 里按 token 侧复算硬过滤)
  (撤:start_time_grid / motion_duration_mcq —— 2,500 场景实测 token 节点折线对"逐帧速度门"标签的复现率 22% / 43%;
   绝对时间改由 speed_grid 与 interval_speed 承担:位移÷Δtsec,位置量稳健)
  cam_disp_grid            相机在 [i,j] 移动了几米(相机伪实例 token;GT 无深度噪声)          尺度相关
  motion_source            屏幕上的位移是物体动了还是相机动了(2 选 1)                         相机 token+世界系

踩坑清单 → 守卫:
  · 题面不写秒(旧 obj_speed 题面带 t → 速率≡位移、不考 tsec)             → when() 只写帧号
  · 窗口泄漏(帧与 token 按题目窗口裁 → 答案区间喂给模型;T-d rgb 95%)     → frame_range 一律整段 [0,T-1]
  · 深度噪声(静止树被拟出 1.2 m 轨迹)                                        → 复用 v5 防线:noise(z) 分层、≥max(0.5,3·noise)、冻结相机 ≥10 px、径向核验、速度上限
  · 答案要能从 token 复现                                                     → disp_facts 的 kp/dense 一致门 + Gate 0 逐题复算(含归一化尺度)
  · 类别先验(person 1.27 vs car 4.61 m/s)                                    → 载荷带 cls,build 按 cls 分层配平
  · "长窗口→大位移"文字捷径                                                   → 载荷带 gap,build 按 gap 档分层配平;区间比较题配平"长区间是否答案"
  · 速率≡位移 同题重复                                                        → 同 (物体,窗口) 只出 disp 或 speed 之一(md5 奇偶)
  · 指代歧义                                                                  → descriptors/names_ok(label 唯一,或 left/right 跨涉及帧一致)
  · 帧数与抽帧 jpg 数不一致                                                   → 不等的场景整体不出
  · 可复现                                                                    → md5 种子、imap 有序、id 不含选项

用法:PYTHONPATH=src/training python -m g1.trajqa_b_gen --batch-root data/scenes \
        --tokens data/tokens_v4td --out outputs/trajqa_b_candidates [--per-clip-mult 6] [--workers 32] [--scene-list f] [--limit N]
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
from g1.metric_qa_gen import (load_scene, descriptors, names_ok, disp_facts, noise_m, cam_z, cam_center,  # noqa: E402
                              frame_pairs, HUMAN, FROZEN_PX, MIN_DISP, NOISE_K, MIN_GAP_FRAMES, MIN_DT_S)
from g1.time_qa_gen import _frozen_disp_px, _motion_duration_s, _seed, _fps_bin, gen_scene as time_gen_scene  # noqa: E402

GEN_VERSION = "trajqa_b_v1"
HT = "trajqa_b"
FRAMES_ROOT = os.environ.get("G1_STEREO_FRAMES", "data/stereo4d/frames")
VEHICLE = ("car", "truck", "bus", "van", "vehicle", "motorcycle", "bicycle", "bike", "boat", "train", "tram", "scooter",
           "taxi", "suv", "jeep", "cart", "ship", "kayak", "canoe", "tractor", "trailer", "wheelchair", "stroller")
MAX_PER_FAMILY_CLIP = 2
MAX_PER_CLIP = 18
# 2026-09-16 扩量:--per-clip-mult M 把"每 clip 采样密度"整体放大 M 倍(窗口候选数与每实例每族上限同乘);
# 只影响训练数据量,门与配平一律不动。附带 WIN_IOU_MAX 去冗余门:同一(实例,族)内窗口重叠 > 该阈值的不再出题。
MULT = 1
WIN_IOU_MAX = 0.7


def win_ok(seen, key, a, b):
    """同 (实例,族) 内窗口去冗余:与已接受窗口的 IoU > WIN_IOU_MAX 就拒。seen 为 dict(key -> [(a,b)])。"""
    for (x, y) in seen.get(key, ()):
        inter = max(0, min(b, y) - max(a, x)); union = max(b, y) - min(a, x)
        if union > 0 and inter / union > WIN_IOU_MAX:
            return False
    seen.setdefault(key, []).append((a, b))
    return True
CMP_RATIO = 1.4          # 区间比较:位移/速率之比 ≥ 1.4(RGB 相对估计的分辨率附近)
CMP_DT_RATIO = 1.5       # 速率比较:两区间时长比 ≥ 1.5(否则速率答案恒等于位移答案)
IV_DISP_DT_MAX = 1.25    # 位移比较:两区间时长比 ≤ 1.25(近等长窗口 → "选长区间"无从下手;否则"长窗口位移大"占 85%+,配平后只剩 3%)
CAM_MIN_DISP = 0.5       # 相机位移题下限(米)
SRC_MIN_PX = 40.0        # motion_source:屏幕位移 ≥ 40 px 才算"看得见在动"
SRC_CAM_STILL_M, SRC_CAM_STILL_DEG = 0.10, 1.0   # "物体动"候选:相机几乎不动
SRC_CAM_MOVE_M, SRC_CAM_MOVE_DEG = 0.5, 5.0      # "相机动"候选:相机明显动
START_MIN_S = 0.5


ANIMAL = ("dog", "cat", "horse", "bird", "cow", "sheep", "goat", "duck", "pigeon", "deer", "elephant", "monkey", "fish", "animal",
          "pet", "puppy", "kitten", "seagull", "swan", "goose", "chicken", "pig", "camel", "donkey", "squirrel")
MOVABLE = ("human", "vehicle", "animal")   # motion_source 的"相机动"侧只用可动类别(建筑/柱子一看题面就知道不会动 = 文字捷径)


def cls_of(label):
    l = label.lower()
    if any(h in l for h in HUMAN): return "human"
    if any(v in l for v in VEHICLE): return "vehicle"
    if any(a in l for a in ANIMAL): return "animal"
    return "other"


def when(i, j, T):
    return f"frame {i} and frame {j} of {T}"


def parity(*keys):
    return int(hashlib.md5("|".join(str(k) for k in keys).encode()).hexdigest(), 16) & 1


def mk(S, fam, q, iids, payload):
    h = hashlib.md5((S["vid"] + fam + q).encode()).hexdigest()[:8]
    payload = dict(payload); payload.setdefault("fps_bin", S["fps_bin"]); payload.setdefault("T", S["T"])
    return dict(id=f"{S['vid']}_{fam}_{h}", video_id=S["vid"], hallucination_type=HT, event_subtype=fam, question=q,
                frame_range=[0, S["T"] - 1], involved_inst_ids=[int(x) for x in iids], gt_payload=payload)


def proj_uv(K, w2c, X, t):
    E = w2c[t]; xc = E[:3, :3] @ np.asarray(X, float) + E[:3, 3]
    if xc[2] <= 1e-6: return None
    Kt = K[t] if K.ndim == 3 else K; uv = Kt @ xc; return uv[:2] / xc[2]


def on_image(K, uv):
    if uv is None: return False
    Kt = K[0] if K.ndim == 3 else K; W, H = 2 * Kt[0, 2], 2 * Kt[1, 2]
    return 0 <= uv[0] < W and 0 <= uv[1] < H


def cam_motion(w2c, i, j):
    Ci, Cj = cam_center(w2c, i), cam_center(w2c, j)
    Rrel = w2c[j][:3, :3] @ w2c[i][:3, :3].T
    ang = float(np.degrees(np.arccos(np.clip((np.trace(Rrel) - 1) / 2, -1, 1))))
    return float(np.linalg.norm(Cj - Ci)), ang


def gated(S, it, i, j, rej):
    """位移事实 + 全部门(v5 防线);过门返回 facts,否则 None 并记账。"""
    f, why = disp_facts(S, it, i, j)
    if f is None: rej[why] += 1; return None
    if f["d"] < max(MIN_DISP, NOISE_K * f["noise"]): rej["disp_below_gate"] += 1; return None
    if f["fz"] is None or f["fz"] < FROZEN_PX: rej["frozen_px"] += 1; return None
    # 标签取 **keypoint 质心**(token 编码的正是 keypoint;dense 质心只作一致性门,disp_facts 已卡 |dk−d| ≤ max(0.3, 0.3d))
    if it.c_kp is not None:
        f["dk"] = float(np.linalg.norm(it.c_kp[j] - it.c_kp[i])); f["pki"], f["pkj"] = it.c_kp[i], it.c_kp[j]
    else:
        f["dk"] = f["d"]; f["pki"], f["pkj"] = f["pi"], f["pj"]
    return f


# ------------------------------------------------------------------ 族
def fam_disp_speed(S, desc, rng, rej):
    out = []; cnt = Counter()
    for it in S["insts"]:
        if it.kind == "static" or it.iid not in desc: continue
        for (i, j) in frame_pairs(it.vis, S["T"], rng, n=8 * MULT):
            f = gated(S, it, i, j, rej)
            if f is None: continue
            if not names_ok(S, desc, [it.iid], (i, j)): rej["name_swap"] += 1; continue
            dt = float(S["times"][j] - S["times"][i])
            base = dict(obj=desc[it.iid], cls=cls_of(it.label), kind=it.kind, gap=int(j - i), i=int(i), j=int(j), dt_s=round(dt, 3),
                        obj_dist_m=round(f["z"], 2), noise_m=f["noise"], frozen_px=round(f["fz"], 1), disp_m=round(f["dk"], 3), dense_disp_m=round(f["d"], 3),
                        pos_i=[round(float(x), 3) for x in f["pki"]], pos_j=[round(float(x), 3) for x in f["pkj"]])
            use_speed = parity(S["vid"], it.iid, i, j, "ds") == 1 and dt >= MIN_DT_S
            if use_speed:
                if cnt[("speed_grid", it.iid)] >= (MAX_PER_FAMILY_CLIP + 2) * MULT: continue
                if not win_ok(S["_win"], ("speed_grid", it.iid), i, j): rej["win_dup"] += 1; continue
                q = (f"In the world frame (ignoring camera motion), what was the average speed (in meters per second) of "
                     f"{desc[it.iid]} between {when(i, j, S['T'])}?")
                out.append(mk(S, "speed_grid", q, [it.iid], dict(base, value=round(f["dk"] / dt, 4), unit="m/s")))
                cnt[("speed_grid", it.iid)] += 1
            else:
                if cnt[("disp_grid", it.iid)] >= (MAX_PER_FAMILY_CLIP + 2) * MULT: continue
                if not win_ok(S["_win"], ("disp_grid", it.iid), i, j): rej["win_dup"] += 1; continue
                q = (f"In the world frame (ignoring camera motion), approximately how far (in meters) did {desc[it.iid]} move "
                     f"between {when(i, j, S['T'])}?")
                out.append(mk(S, "disp_grid", q, [it.iid], dict(base, value=round(f["dk"], 4), unit="m")))
                cnt[("disp_grid", it.iid)] += 1
    return out


def triples(vis, rng, n=6):
    idx = np.where(vis)[0]
    out = set()
    if len(idx) < 2 * MIN_GAP_FRAMES + 1: return []
    for _ in range(n * 8):
        j = int(rng.choice(idx))
        left = idx[idx <= j - MIN_GAP_FRAMES]; right = idx[idx >= j + MIN_GAP_FRAMES]
        if len(left) == 0 or len(right) == 0: continue
        out.add((int(rng.choice(left)), j, int(rng.choice(right))))
        if len(out) >= n: break
    return sorted(out)


def triples_equal(vis, rng, n=8):
    """等长双窗口:(i,k) 为可见帧对且 k−i ≥ 2·MIN_GAP,j = 中点。"""
    idx = np.where(vis)[0]; out = set()
    if len(idx) < 2 * MIN_GAP_FRAMES + 1: return []
    for _ in range(n * 8):
        i, k = sorted(rng.choice(idx, 2, replace=False))
        if k - i < 2 * MIN_GAP_FRAMES: continue
        out.add((int(i), int((i + k) // 2), int(k)))
        if len(out) >= n: break
    return sorted(out)


def fam_interval(S, desc, rng, rej):
    out = []; cnt = Counter()
    for it in S["insts"]:
        if it.kind == "static" or it.iid not in desc: continue
        for (i, j, k) in triples_equal(it.vis, rng, n=8 * MULT):     # ---- 位移比较(等长窗口)
            f1 = gated(S, it, i, j, rej)
            if f1 is None: continue
            f2 = gated(S, it, j, k, rej)
            if f2 is None: continue
            if not names_ok(S, desc, [it.iid], (i, j, k)): rej["name_swap"] += 1; continue
            if cnt[("interval_disp", it.iid)] >= MAX_PER_FAMILY_CLIP * MULT: break
            if not win_ok(S["_win"], ("interval_disp", it.iid), i, k): rej["win_dup"] += 1; continue
            d1, d2 = f1["dk"], f2["dk"]; dt1, dt2 = float(S["times"][j] - S["times"][i]), float(S["times"][k] - S["times"][j])
            gate = max(MIN_DISP, NOISE_K * max(f1["noise"], f2["noise"]))
            if max(d1, d2) < CMP_RATIO * min(d1, d2) or abs(d1 - d2) < gate: rej["iv_disp_margin"] += 1; continue
            disp_ans = 1 if d1 >= d2 else 2
            base = dict(obj=desc[it.iid], cls=cls_of(it.label), kind=it.kind, i=int(i), j=int(j), k=int(k), d1=round(d1, 3), d2=round(d2, 3),
                        dt1=round(dt1, 3), dt2=round(dt2, 3), v1=round(d1 / max(dt1, 1e-6), 3), v2=round(d2 / max(dt2, 1e-6), 3), noise_m=max(f1["noise"], f2["noise"]),
                        intervals=[f"between frame {i} and frame {j}", f"between frame {j} and frame {k}"])
            q = (f"In the world frame (ignoring camera motion), during which interval did {desc[it.iid]} move a longer distance: "
                 f"between frame {i} and frame {j}, or between frame {j} and frame {k} (of {S['T']} frames)?")
            out.append(mk(S, "interval_disp", q, [it.iid], dict(base, answer_interval=disp_ans, longer_window_is_answer=bool((dt1 > dt2) == (disp_ans == 1)))))
            cnt[("interval_disp", it.iid)] += 1
        for (i, j, k) in triples(it.vis, rng, n=24 * MULT):          # ---- 速率比较(不等长窗口,须读时长)
            f1 = gated(S, it, i, j, rej)
            if f1 is None: continue
            f2 = gated(S, it, j, k, rej)
            if f2 is None: continue
            if not names_ok(S, desc, [it.iid], (i, j, k)): rej["name_swap"] += 1; continue
            d1, d2 = f1["dk"], f2["dk"]; dt1, dt2 = float(S["times"][j] - S["times"][i]), float(S["times"][k] - S["times"][j])
            gate = max(MIN_DISP, NOISE_K * max(f1["noise"], f2["noise"]))
            v1, v2 = d1 / max(dt1, 1e-6), d2 / max(dt2, 1e-6)
            base = dict(obj=desc[it.iid], cls=cls_of(it.label), kind=it.kind, i=int(i), j=int(j), k=int(k), d1=round(d1, 3), d2=round(d2, 3),
                        dt1=round(dt1, 3), dt2=round(dt2, 3), v1=round(v1, 3), v2=round(v2, 3), noise_m=max(f1["noise"], f2["noise"]),
                        intervals=[f"between frame {i} and frame {j}", f"between frame {j} and frame {k}"])
            disp_ans = 1 if d1 >= d2 else 2
            if True:
                if cnt[("interval_speed", it.iid)] >= MAX_PER_FAMILY_CLIP * MULT: break
                if not win_ok(S["_win"], ("interval_speed", it.iid), i, k): rej["win_dup"] += 1; continue
                if max(dt1, dt2) / max(min(dt1, dt2), 1e-6) < CMP_DT_RATIO: rej["iv_dt_equal"] += 1; continue
                if max(v1, v2) < CMP_RATIO * min(v1, v2): rej["iv_speed_margin"] += 1; continue
                ans = 1 if v1 >= v2 else 2
                q = (f"In the world frame (ignoring camera motion), during which interval was {desc[it.iid]} moving faster on average: "
                     f"between frame {i} and frame {j}, or between frame {j} and frame {k} (of {S['T']} frames)?")
                out.append(mk(S, "interval_speed", q, [it.iid], dict(base, answer_interval=ans, longer_window_is_answer=bool((dt1 > dt2) == (ans == 1)),
                                                                        speed_eq_disp=bool(ans == disp_ans))))
                cnt[("interval_speed", it.iid)] += 1
    return out


def fam_start_time(S, desc, rng, rej):
    out = []
    by_id = {int(inst["instance_id"]): inst for inst in S["reprs"]}
    for it in S["insts"]:
        if it.kind == "static" or it.iid not in desc or it.iid not in by_id: continue
        md = _motion_duration_s(by_id[it.iid], S["times"], S["K"], S["w2c"])
        if md is None: rej["no_motion_run"] += 1; continue
        dur, fa, fb = md
        if not names_ok(S, desc, [it.iid], (fa, fb)): rej["name_swap"] += 1; continue
        t0 = float(S["times"][fa])
        if t0 < START_MIN_S or int(it.vis[:fa].sum()) < 3: rej["start_trivial"] += 1; continue
        q = (f"In the world frame (ignoring camera motion), at approximately what time (in seconds from the start of the clip) "
             f"does {desc[it.iid]} start moving?")
        out.append(mk(S, "start_time_grid", q, [it.iid], dict(value=round(t0, 4), unit="s", start_frame=int(fa), end_frame=int(fb), dur_s=round(dur, 3),
                                                              clip_s=round(float(S["times"][-1]), 3), fps=round(S["fps"], 3), obj=desc[it.iid], cls=cls_of(it.label), kind=it.kind)))
        if len(out) >= MAX_PER_FAMILY_CLIP: break
    return out


def fam_camera(S, desc, rng, rej):
    out = []
    if not getattr(S["sc"], "has_camera_object", False): rej["no_camera_token"] += 1; return out
    T = S["T"]; w2c = S["w2c"]
    C = np.stack([cam_center(w2c, t) for t in range(T)])
    for (i, j) in frame_pairs(np.ones(T, bool), T, rng, n=4 * MULT):
        d = float(np.linalg.norm(C[j] - C[i]))
        if d < CAM_MIN_DISP: rej["cam_below_gate"] += 1; continue
        if not win_ok(S["_win"], ("cam", -1), i, j): rej["win_dup"] += 1; continue
        q = f"Approximately how far (in meters) did the camera move between {when(i, j, T)}?"
        out.append(mk(S, "cam_disp_grid", q, [], dict(value=round(d, 4), unit="m", gap=int(j - i), i=int(i), j=int(j), dt_s=round(float(S["times"][j] - S["times"][i]), 3),
                                                     cam_i=[round(float(x), 3) for x in C[i]], cam_j=[round(float(x), 3) for x in C[j]], cls="camera")))
        if len(out) >= MAX_PER_FAMILY_CLIP * MULT: break
    return out


def fam_motion_source(S, desc, rng, rej):
    out = []; K, w2c = S["K"], S["w2c"]; got = Counter()
    for it in S["insts"]:
        if it.iid not in desc: continue
        if it.kind == "static":
            if got["camera"] >= MULT: continue
            if cls_of(it.label) not in MOVABLE: rej["src_static_unmovable"] += 1; continue
            X = it.c_kp[0]
            for (i, j) in frame_pairs(np.ones(S["T"], bool), S["T"], rng, n=3 * MULT):
                cd, ca = cam_motion(w2c, i, j)
                if cd < SRC_CAM_MOVE_M and ca < SRC_CAM_MOVE_DEG: rej["src_cam_still"] += 1; continue
                if not win_ok(S["_win"], ("src", it.iid), i, j): rej["win_dup"] += 1; continue
                ui, uj = proj_uv(K, w2c, X, i), proj_uv(K, w2c, X, j)
                if not (on_image(K, ui) and on_image(K, uj)): rej["src_offimage"] += 1; continue
                px = float(np.linalg.norm(uj - ui))
                if px < SRC_MIN_PX: rej["src_px_small"] += 1; continue
                if not names_ok(S, desc, [it.iid], (i, j)): rej["name_swap"] += 1; continue
                q = (f"Between {when(i, j, S['T'])}, {desc[it.iid]} appears to shift position on screen. Is that mainly because "
                     f"{desc[it.iid]} itself moved in the world, or because the camera moved?")
                out.append(mk(S, "motion_source", q, [it.iid], dict(obj=desc[it.iid], cls=cls_of(it.label), kind="static", source="camera", i=int(i), j=int(j), gap=int(j - i),
                                                                      screen_px=round(px, 1), cam_disp_m=round(cd, 3), cam_rot_deg=round(ca, 2), obj_disp_m=0.0,
                                                                      choices=[f"{desc[it.iid]} moved in the world", "the camera moved"])))
                got["camera"] += 1; break
        else:
            if got["object"] >= MULT: continue
            for (i, j) in frame_pairs(it.vis, S["T"], rng, n=3 * MULT):
                cd, ca = cam_motion(w2c, i, j)
                if cd > SRC_CAM_STILL_M or ca > SRC_CAM_STILL_DEG: rej["src_cam_moving"] += 1; continue
                if not win_ok(S["_win"], ("src", it.iid), i, j): rej["win_dup"] += 1; continue
                f = gated(S, it, i, j, rej)
                if f is None: continue
                ui, uj = proj_uv(K, w2c, f["pi"], i), proj_uv(K, w2c, f["pj"], j)
                if not (on_image(K, ui) and on_image(K, uj)): rej["src_offimage"] += 1; continue
                px = float(np.linalg.norm(uj - ui))
                if px < SRC_MIN_PX: rej["src_px_small"] += 1; continue
                if not names_ok(S, desc, [it.iid], (i, j)): rej["name_swap"] += 1; continue
                q = (f"Between {when(i, j, S['T'])}, {desc[it.iid]} appears to shift position on screen. Is that mainly because "
                     f"{desc[it.iid]} itself moved in the world, or because the camera moved?")
                out.append(mk(S, "motion_source", q, [it.iid], dict(obj=desc[it.iid], cls=cls_of(it.label), kind=it.kind, source="object", i=int(i), j=int(j), gap=int(j - i),
                                                                      screen_px=round(px, 1), cam_disp_m=round(cd, 3), cam_rot_deg=round(ca, 2), obj_disp_m=round(f["dk"], 3),
                                                                      choices=[f"{desc[it.iid]} moved in the world", "the camera moved"])))
                got["object"] += 1; break
    return out


def fam_time(S, sd, tok_path, rej):
    """复用 time_qa_gen(T-d 二分 / T-a MCQ),整段 clip 协议;载荷补 cls。"""
    td, ta, _ = time_gen_scene(sd, tok_path)
    out = []
    for d in (td or []) + (ta or []):
        d = dict(d); d["frame_range"] = [0, S["T"] - 1]; d["hallucination_type"] = HT
        gp = dict(d["gt_payload"]); gp["cls"] = cls_of(gp.get("label", "")); gp["T"] = S["T"]; gp["obj"] = "the " + gp.get("label", "object")
        gp["motion_frames"] = list(d.get("frame_range_motion", d.get("frame_range")))
        d["gt_payload"] = gp
        h = hashlib.md5((S["vid"] + d["event_subtype"] + d["question"] + json.dumps(d["options"], sort_keys=True)).encode()).hexdigest()[:8]
        d["id"] = f"{S['vid']}_{d['event_subtype']}_{h}"
        out.append(d)
    return out


FAMILIES = (fam_disp_speed, fam_interval, fam_camera, fam_motion_source)   # start_time 撤:Gate 0 一致率 22%(见 build 注)


def gen_scene(args):
    sd, tok_path = args[:2]
    if len(args) > 2:
        global MULT
        MULT = int(args[2])
    rej = Counter()
    try:
        S, why = load_scene(sd, tok_path)
        if S is None: return [], Counter({why: 1})
        clip = os.path.basename(sd)[len("scene_"):]
        n_jpg = len(glob.glob(os.path.join(FRAMES_ROOT, clip, "*.jpg")))
        if n_jpg != S["T"]: return [], Counter({"frames_T_mismatch": 1})
        S["sdir"] = sd; S["_desc_cache"] = {}; S["fps_bin"] = _fps_bin(S["fps"]); S["_win"] = {}
        S["reprs"] = pickle.load(open(os.path.join(sd, "final_repr.pkl"), "rb"))
        rng = _seed(S["vid"], "trajqa_b")
        desc = descriptors(S, 0); S["_desc_cache"][0] = desc
        items = []
        for fam in FAMILIES:
            items += fam(S, desc, rng, rej)
        # 时间族在 time_qa_gen 里自带运动区间 frame_range;这里先记下再改整段
        tmp = time_gen_scene(sd, tok_path)
        tit = []
        for d in (tmp[0] or []):   # 只留 T-d 二分;T-a MCQ 的 token 侧复算一致率 43%,撤
            d = dict(d); gp = dict(d["gt_payload"]); gp["motion_frames"] = list(d["frame_range"]); gp["cls"] = cls_of(gp.get("label", ""))
            gp["T"] = S["T"]; gp["obj"] = "the " + gp.get("label", "object"); d["gt_payload"] = gp
            d["frame_range"] = [0, S["T"] - 1]; d["hallucination_type"] = HT
            h = hashlib.md5((S["vid"] + d["event_subtype"] + d["question"] + json.dumps(d["options"], sort_keys=True)).encode()).hexdigest()[:8]
            d["id"] = f"{S['vid']}_{d['event_subtype']}_{h}"; tit.append(d)
        items += tit
        if len(items) > MAX_PER_CLIP * MULT:
            rng.shuffle(items); items = items[:MAX_PER_CLIP * MULT]
        for d in items:
            d["src"] = dict(gen=GEN_VERSION, tokens=os.path.basename(os.path.dirname(tok_path)))
        return items, rej
    except Exception as e:
        return [], Counter({f"exc:{type(e).__name__}:{str(e)[:60]}": 1})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-root", default="data/scenes")
    ap.add_argument("--tokens", default="data/tokens_v4td")
    ap.add_argument("--out", default="outputs/trajqa_b_candidates")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--scene-list", default="", help="只跑这些 scene 目录(每行一个)")
    ap.add_argument("--per-clip-mult", type=int, default=1, help="每 clip 采样密度倍数(2026-09-16 扩量;门与配平不变,附带窗口去冗余)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.scene_list:
        dirs = [l.strip() for l in open(a.scene_list) if l.strip()]
    else:
        dirs = sorted(os.path.dirname(f) for f in glob.glob(os.path.join(a.batch_root, "scene_*", "final_repr.pkl")))
    jobs = [(sd, os.path.join(a.tokens, os.path.basename(sd) + ".pkl"), a.per_clip_mult) for sd in dirs]
    jobs = [j for j in jobs if os.path.exists(j[1])]
    if a.limit: jobs = jobs[:a.limit]
    print(f"[trajqa_b] scenes={len(jobs)} workers={a.workers} per_clip_mult={a.per_clip_mult}", flush=True)
    items, rej = [], Counter()
    with Pool(a.workers) as pool:
        for k, (it, rj) in enumerate(pool.imap(gen_scene, jobs, chunksize=16)):
            items += it; rej.update(rj)
            if (k + 1) % 2000 == 0: print(f"  {k+1}/{len(jobs)} items={len(items)}", flush=True)
    with open(os.path.join(a.out, "candidates.jsonl"), "w") as fh:
        for d in items: fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    fams = Counter(d["event_subtype"] for d in items)
    def qs(v):
        v = np.asarray(v, float); return {p: round(float(np.percentile(v, p)), 3) for p in (5, 25, 50, 75, 95)} if len(v) else {}
    per = {}
    for f in sorted(fams):
        L = [d for d in items if d["event_subtype"] == f]; g = L[0]["gt_payload"]
        r = dict(n=len(L), n_scenes=len({d["video_id"] for d in L}), cls=dict(Counter(d["gt_payload"].get("cls") for d in L)),
                 fps=dict(Counter(d["gt_payload"].get("fps_bin") for d in L)))
        if "value" in g: r["value"] = qs([d["gt_payload"]["value"] for d in L])
        if "answer_interval" in g:
            r["answer_interval"] = dict(Counter(d["gt_payload"]["answer_interval"] for d in L)); r["longer_window_is_answer"] = dict(Counter(d["gt_payload"]["longer_window_is_answer"] for d in L))
            if "speed_eq_disp" in g: r["speed_eq_disp"] = dict(Counter(d["gt_payload"]["speed_eq_disp"] for d in L))
        if "source" in g: r["source"] = dict(Counter(d["gt_payload"]["source"] for d in L))
        per[f] = r
    rep = dict(gen=GEN_VERSION, tokens=a.tokens, per_clip_mult=a.per_clip_mult, win_iou_max=WIN_IOU_MAX, n_scenes=len(jobs), n_items=len(items), n_scenes_with_qa=len({d["video_id"] for d in items}),
               families=per, rejections=dict(rej.most_common()))
    json.dump(rep, open(os.path.join(a.out, "_gen_report.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(rep, ensure_ascii=False, indent=1), flush=True)


if __name__ == "__main__":
    main()
