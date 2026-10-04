"""T8 轨迹描述生成器(训练方案 v1 §1.1 的 T8;v1.1 增补的 1a 预热数据源)。

规则化、**相机无关**(红线:不用 left/right 等图像系方位词;世界系位移/速率/时序词)、逐位确定
(随机性仅来自 md5(scene,iid) 种子)。两种产物:
  t8_train.jsonl  开放式描述(1a/1b SFT 用):Q="describe how the X moves…" A=模板句
  t8_probe.jsonl  四选一 MCQ(1a **硬门**评测用):正确描述 + 3 个物理量/元动作扰动干扰项,
                  traj_only vs blind 的差距即门指标(格式与既有 QA jsonl 一致,eval.py 直接可跑)
元动作词表(全部世界系可算):stationary / moves(净位移+均速)/ accelerates / decelerates /
slows to a stop / reverses direction / approaches X / moves away from X / faster than X。
数据源:v4 token pkl(路由/label/归一化,与模型看到的完全同源)+ final_repr(米制轨迹)+
stereo4d_meta.json(fps)。label 在场景内不唯一的物体跳过 per-object 样本(防接地歧义污染 1a)。

用法(stage0_4 下):
  PYTHONPATH=src/training python -m g1.t8_gen --batch-root data/scenes \
      --tokens data/tokens_v4td --out outputs/captions
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import pickle
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(_HERE, "..", ".."), os.path.join(_HERE, "..", "..", "tokenization")):
    p = os.path.abspath(p)
    if p not in sys.path:
        sys.path.insert(0, p)
import events as ev            # noqa: E402  (src/tokenization)
import schema                  # noqa: F401,E402  (SceneTokens unpickle)

# ---- 阈值(全部显式;米制 / m/s) ------------------------------------------
V_STATIC = 0.05        # 均速低于此视为"几乎不动"的从句;与运动门 0.05m 同源量级
ACC_RATIO = 1.5        # 首/末 1/3 均速比超此 → accelerates/decelerates
STOP_RATIO = 0.30      # 末 1/4 均速 < 全程均速×此 → slows to a stop
APPROACH_ABS = 0.30    # 对间距变化的绝对下限(m)
APPROACH_REL = 0.15    # 相对下限(占初始距离)
FASTER_RATIO = 1.30    # 均速比超此 → faster than
MIN_COMMON = 5         # 对间关系所需公共可见帧数
LETTERS = ("A", "B", "C", "D")


def _seed(*keys) -> np.random.RandomState:
    h = hashlib.md5("|".join(str(k) for k in keys).encode()).hexdigest()
    return np.random.RandomState(int(h[:8], 16))


def _centroid_traj(inst):
    """实例质心轨迹(米制 W_track):逐帧对可见行取中位。返回 (T,3), vis(T,)。"""
    if "trajectories" in inst:
        P = np.asarray(inst["trajectories"], float)
        V = np.asarray(inst["visibility"], bool)
    elif "se3_R" in inst:
        ref = np.asarray(inst["ref_positions"], float)
        keep = np.abs(ref).sum(1) > 0
        if not keep.any():
            return None, None
        R = np.asarray(inst["se3_R"], float); t = np.asarray(inst["se3_t"], float)
        P = (np.einsum("tij,mj->tmi", R, ref[keep]) + t[:, None, :]).transpose(1, 0, 2)
        V = np.broadcast_to(np.asarray(inst["se3_ok"], bool)[None, :], P.shape[:2]).copy()
    else:
        return None, None
    T = P.shape[1]
    c = np.full((T, 3), np.nan); vis = np.zeros(T, bool)
    for t_ in range(T):
        m = V[:, t_]
        if m.any():
            c[t_] = np.median(P[m, t_], axis=0); vis[t_] = True
    return c, vis


def _speed_profile(c, vis, fps):
    """相邻可见帧间的速率 (m/s),对齐到区间左端帧;3 点中位平滑。"""
    f = np.nonzero(vis)[0]
    if len(f) < 2:
        return f, np.zeros(0)
    d = np.linalg.norm(c[f[1:]] - c[f[:-1]], axis=1)
    dt = (f[1:] - f[:-1]) / fps
    v = d / np.maximum(dt, 1e-9)
    if len(v) >= 3:
        v = np.array([np.median(v[max(0, i - 1):i + 2]) for i in range(len(v))])
    return f, v


def _fmt(x, nd=1):
    return f"{x:.{nd}f}".rstrip("0").rstrip(".")


def _facts(inst, fps):
    """单实例运动事实。返回 dict 或 None(无足够观测)。"""
    c, vis = _centroid_traj(inst)
    if c is None or vis.sum() < 3:
        return None
    f = np.nonzero(vis)[0]
    net = float(np.linalg.norm(c[f[-1]] - c[f[0]]))
    steps = np.linalg.norm(c[f[1:]] - c[f[:-1]], axis=1)
    path = float(steps.sum())
    dur = (f[-1] - f[0]) / fps
    fr, v = _speed_profile(c, vis, fps)
    mean_v = path / max(dur, 1e-9)
    bd = float(np.linalg.norm(np.nanmax(c[vis], 0) - np.nanmin(c[vis], 0)))
    rev = ev.detect_reversals(f, c[f], bd) if len(f) >= 3 else []
    third = max(1, len(v) // 3)
    v0, v1 = float(np.mean(v[:third])), float(np.mean(v[-third:]))
    quart = max(1, len(v) // 4)
    stop = (len(v) >= 4 and float(np.mean(v[-quart:])) < STOP_RATIO * mean_v and mean_v > V_STATIC)
    acc = "accelerates" if (min(v0, v1) > 1e-9 and v1 / max(v0, 1e-9) >= ACC_RATIO) else \
          ("decelerates" if (v0 / max(v1, 1e-9) >= ACC_RATIO and not stop) else None)
    return dict(c=c, vis=vis, f0=int(f[0]), f1=int(f[-1]), net=net, path=path,
                mean_v=mean_v, rev=[int(r) for r in rev], stop=stop, acc=acc)


def _pair_facts(fa, fb, label_b):
    """对间关系(approach/recede);无公共窗口或变化不显著返回 None。"""
    common = fa["vis"] & fb["vis"]
    f = np.nonzero(common)[0]
    if len(f) < MIN_COMMON:
        return None
    d0 = float(np.linalg.norm(fa["c"][f[0]] - fb["c"][f[0]]))
    d1 = float(np.linalg.norm(fa["c"][f[-1]] - fb["c"][f[-1]]))
    if abs(d1 - d0) < max(APPROACH_ABS, APPROACH_REL * max(d0, 1e-9)):
        return None
    return ("approaches" if d1 < d0 else "moves away from", label_b)


def _describe(facts, label, rel=None, speed_cmp=None):
    """事实 → 一句描述(相机无关)。"""
    parts = [f"the {label} moves about {_fmt(facts['net'])} m in the world frame "
             f"at an average speed of {_fmt(facts['mean_v'], 2)} m/s"]
    if facts["acc"]:
        parts.append(f"it {facts['acc']} over the clip")
    if facts["rev"]:
        parts.append(f"it reverses direction around frame {facts['rev'][0]}")
    if facts["stop"]:
        parts.append("it slows to a stop by the end")
    if rel:
        parts.append(f"it {rel[0]} the {rel[1]}")
    if speed_cmp:
        parts.append(f"it moves faster than the {speed_cmp}")
    s = "; ".join(parts)
    return s[0].upper() + s[1:] + "."


def _distractors(facts, label, rel, speed_cmp, rng):
    """3 个物理量/元动作扰动干扰项(与正确项结构同、事实错)。"""
    out = []
    out.append(f"The {label} stays in roughly the same place in the world frame throughout the video.")
    k = rng.choice([3.0, 0.3])
    d2 = (f"The {label} moves about {_fmt(facts['net'] * k)} m in the world frame "
          f"at an average speed of {_fmt(facts['mean_v'] * k, 2)} m/s")
    d2 += "; it keeps moving without reversing." if facts["rev"] else \
          f"; it reverses direction around frame {int((facts['f0'] + facts['f1']) // 2)}."
    out.append(d2)
    if rel:
        flip = "moves away from" if rel[0] == "approaches" else "approaches"
        out.append(f"The {label} moves about {_fmt(facts['net'])} m and {flip} the {rel[1]}.")
    elif speed_cmp:
        out.append(f"The {label} moves more slowly than the {speed_cmp} in the world frame.")
    elif facts["stop"]:
        out.append(f"The {label} moves at a roughly constant speed and never slows down.")
    else:
        out.append(f"The {label} slows to a complete stop early in the video and stays still afterwards.")
    return out


def gen_scene(scene_dir, token_pkl):
    sc = pickle.load(open(token_pkl, "rb"))
    reprs = pickle.load(open(os.path.join(scene_dir, "final_repr.pkl"), "rb"))
    by_id = {int(x["instance_id"]): x for x in reprs}
    meta_p = os.path.join(scene_dir, "stereo4d_meta.json")
    if not os.path.exists(meta_p):
        return [], [], "no_meta"
    fps = float(json.load(open(meta_p)).get("fps_effective") or 0)
    if fps <= 0:
        return [], [], "no_fps"
    vid = os.path.basename(scene_dir)

    dyn = [o for o in sc.objects if o.kind in ("nonrigid", "rigid")]
    static_lbls = [str(o.label) for o in sc.objects if o.kind == "static"]
    labels_all = [str(o.label) for o in sc.objects]
    facts = {}
    for o in dyn:
        fx = _facts(by_id[o.instance_id], fps)
        if fx is not None:
            facts[o.instance_id] = fx

    train, probe = [], []
    # ---- 场景级样本(动/静清单;label 计数表达,天然无歧义)
    if facts or static_lbls:
        def _cnt(lbls):
            u = {}
            for x in lbls:
                u[x] = u.get(x, 0) + 1
            return ", ".join((f"{v} {k}s" if v > 1 else f"the {k}") for k, v in u.items())
        mv = [str(o.label) for o in dyn if o.instance_id in facts]
        ans = []
        if mv:
            ans.append(f"Moving in the world frame: {_cnt(mv)}.")
        if static_lbls:
            ans.append(f"Staying in place: {_cnt(static_lbls)}.")
        train.append(dict(video_id=vid, hallucination_type="t8_desc", event_subtype="scene_inventory",
                          question="In the world frame (ignoring camera motion), which objects move "
                                   "during the video and which stay in place?",
                          answer=" ".join(ans), options=None))
    # ---- 逐对象样本(label 场景内唯一才出,防接地歧义)
    for o in dyn:
        if o.instance_id not in facts:
            continue
        lbl = str(o.label)
        if labels_all.count(lbl) != 1:
            continue
        fx = facts[o.instance_id]
        rel = None
        for o2 in sc.objects:
            if o2.instance_id == o.instance_id or labels_all.count(str(o2.label)) != 1:
                continue
            f2 = facts.get(o2.instance_id)
            if f2 is None:
                c2, v2 = _centroid_traj(by_id[o2.instance_id])
                if c2 is None or v2.sum() < MIN_COMMON:
                    continue
                f2 = dict(c=c2, vis=v2)
            rel = _pair_facts(fx, f2, str(o2.label))
            if rel:
                break
        speed_cmp = None
        for o2 in dyn:
            f2 = facts.get(o2.instance_id)
            if (f2 and o2.instance_id != o.instance_id and labels_all.count(str(o2.label)) == 1
                    and fx["mean_v"] >= FASTER_RATIO * max(f2["mean_v"], 1e-9)):
                speed_cmp = str(o2.label)
                break
        desc = _describe(fx, lbl, rel, speed_cmp)
        train.append(dict(video_id=vid, hallucination_type="t8_desc", event_subtype="object_motion",
                          question=f"In the world frame (ignoring camera motion), describe how "
                                   f"the {lbl} moves during the video.",
                          answer=desc, options=None))
        rng = _seed(vid, o.instance_id)
        opts = [desc] + _distractors(fx, lbl, rel, speed_cmp, rng)
        order = rng.permutation(4)
        options = {LETTERS[i]: opts[order[i]] for i in range(4)}
        answer = LETTERS[int(np.nonzero(order == 0)[0][0])]
        probe.append(dict(video_id=vid, hallucination_type="t8_probe", event_subtype="object_motion",
                          question=f"Which statement correctly describes the motion of the {lbl} "
                                   f"in the world frame (ignoring camera motion)?",
                          options=options, answer=answer,
                          involved_inst_ids=[int(o.instance_id)]))
    return train, probe, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-root", default="data/scenes")
    ap.add_argument("--tokens", default="data/tokens_v4td")
    ap.add_argument("--out", default="outputs/captions")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    scenes = sorted(glob.glob(os.path.join(args.batch_root, "scene_*", "final_repr.pkl")))
    n_train = n_probe = 0
    skip = {}
    ft = open(os.path.join(args.out, "t8_train.jsonl"), "w")
    fp = open(os.path.join(args.out, "t8_probe.jsonl"), "w")
    for f in scenes:
        sd = os.path.dirname(f)
        tp = os.path.join(args.tokens, os.path.basename(sd) + ".pkl")
        if not os.path.exists(tp):
            skip["no_tokens"] = skip.get("no_tokens", 0) + 1
            continue
        train, probe, err = gen_scene(sd, tp)
        if err:
            skip[err] = skip.get(err, 0) + 1
            continue
        for i, d in enumerate(train):
            d["id"] = f"{d['video_id']}_t8_{i}"
            ft.write(json.dumps(d) + "\n"); n_train += 1
        for i, d in enumerate(probe):
            d["id"] = f"{d['video_id']}_t8p_{i}"
            fp.write(json.dumps(d) + "\n"); n_probe += 1
    ft.close(); fp.close()
    print(f"scenes={len(scenes)} train={n_train} probe={n_probe} skip={skip}")


if __name__ == "__main__":
    main()
