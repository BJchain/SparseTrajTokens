"""「会问秒」的 QA 生成器(消融②的时间通道评价集;2026-09-04)。

需求见 memory project_time_qa_requirements。**操作性判据**:帧序列不变、只改 clip 的 fps,
正确答案必须随之改变 —— 即答案卡在【绝对时间】上,不能由(帧号 + 归一化 u + SCENE 头 T)推出。

两个模板(都满足判据、都是判别不计算、都过 world-frame 问法防视频通道旁路):
  T-d(主)  "the {label} 的运动是否持续超过 {X} 秒?" 二分。**X 逐题变化**、且每个 X 桶内
            yes/no 各半(X 与答案去相关);运动时长 = 冻结相机可观测的运动区间(复用 v5 深度
            噪声防线,避免深度噪声把静止段算进时长)。
  T-a(副)  "the {label} 持续移动了大约几秒?" 4 选项为秒。**干扰项 = 错误 fps 档下的读数**
            (真 fps 的秒 vs 把同样帧数按 5/10/30fps 读出的秒),选项间距 >=2x,盲/无时间臂无从选。

分层(硬要求 1):按 fps 档取样,~5 档 <= 全集 1/3;答案类别经生成器内平衡。
反捷径:label 场景内唯一才出(接地无歧义);答案二分/多分平衡;帧号不进题面(只出秒)。

用法(stage0_4 下,g1train env,纯 CPU):
  python -m g1.time_qa_gen --batch-root data/scenes --tokens data/tokens_v4td \
      --out outputs/time_qa
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
for p in (os.path.join(_HERE, "..", ".."),
          os.path.join(_HERE, "..", "..", "tokenization")):
    p = os.path.abspath(p)
    if p not in sys.path:
        sys.path.insert(0, p)
import schema  # noqa: F401,E402  (SceneTokens unpickle)
from g1.t8_gen import _centroid_traj  # noqa: E402  (复用质心轨迹提取)

LETTERS = ("A", "B", "C", "D")
FPS_BINS = [(0, 7.5, "~5"), (7.5, 12.5, "~10"), (12.5, 22.5, "~15"),
            (22.5, 45, "~30"), (45, 1e9, "~60")]
CAP5_FRAC = 1.0 / 3.0        # ~5 档在最终集里的占比上限(硬要求 1;cap_mode="five" 用)
# 2026-09-08 阈值网格往下延:Stereo4D 片段帧数大致固定 → 帧率越高片长(秒)越短,30fps 片子
# 运动时长中位只有 0.23s,旧网格最小 0.5s 让 ~15/~30 档几乎只剩 "No"(配对平衡后归零)。
# 网格对所有档共享(不按档单独配,否则 X 的取值会泄漏帧率档),每个片段只用"可测"的那部分。
X_GRID = (0.25, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
X_GRID_LEGACY = (0.5, 1.0, 1.5, 2.0, 3.0)   # 506 批(消融②)用的旧网格 + 固定 0.25s 余量
MIN_X_FRAMES = 2             # 阈值至少跨 2 个帧间隔才可测(5fps 下 0.25/0.35 自动不出)
X_REL_MARGIN = 0.2           # 边界余量 = max(1 帧间隔, 20%·X),按帧率自适应
FROZEN_PX = 10.0            # 冻结相机可观测阈(与 v5 深度噪声防线同源)
V_MOVE = 0.05              # 逐帧质心速度(m/s)高于此算"在动"(运动区间界定)
MIN_MOVE_FRAMES = 3        # 运动区间至少这么多可见帧才算数
MIN_DUR_S = 0.15          # 运动时长下限(秒);太短无法可靠判秒


def _seed(*keys) -> np.random.RandomState:
    h = hashlib.md5("|".join(str(k) for k in keys).encode()).hexdigest()
    return np.random.RandomState(int(h[:8], 16))


def _fps_bin(fps):
    for lo, hi, name in FPS_BINS:
        if lo <= fps < hi:
            return name
    return FPS_BINS[-1][2]


def _load_camera(scene_dir):
    """gt_stereo4d.npz 的世界系相机位置 (T,3);拿不到返回 None。"""
    for name in ("gt_stereo4d.npz", "gt_movi.npz"):
        p = os.path.join(scene_dir, name)
        if os.path.exists(p):
            try:
                d = np.load(p, allow_pickle=True)
            except Exception:
                return None, None
            if "extrs_w2c" not in d.files or "intrinsics_K" not in d.files:
                return None, None
            w2c = np.asarray(d["extrs_w2c"], float)          # (T,3,4) or (T,4,4)
            K = np.asarray(d["intrinsics_K"], float)
            return K, w2c
    return None, None


def _frozen_disp_px(K, w2c, X_a, X_b, t_ref):
    """冻结第 t_ref 帧相机,投影世界点 X_a/X_b 的像素位移(深度噪声防线,与 v5 同判据)。"""
    if K is None or w2c is None:
        return None
    E = w2c[t_ref]
    R, tv = E[:3, :3], E[:3, 3]
    Kf = K[t_ref] if K.ndim == 3 else K

    def proj(X):
        xc = R @ np.asarray(X, float) + tv
        z = xc[2] if abs(xc[2]) > 1e-9 else 1e-9
        uv = Kf @ xc
        return uv[:2] / z
    return float(np.linalg.norm(proj(X_b) - proj(X_a)))


def _motion_duration_s(inst, times_s, K, w2c):
    """实例的【可观测运动持续时间】(秒)。
    步骤:质心轨迹 → 逐相邻可见帧的世界速度 (m/s) → 冻结相机确证该帧对可观测(挡深度噪声)
         → 速度 > V_MOVE 且可观测的帧标为"运动帧" → 最长连续运动区间的真实时长(用 times_s)。
    返回 (dur_s, f_start, f_end) 或 None。"""
    c, vis = _centroid_traj(inst)
    if c is None:
        return None
    T = len(times_s)
    if c.shape[0] != T:
        return None
    moving = np.zeros(T, bool)
    for t in range(T - 1):
        if not (vis[t] and vis[t + 1]):
            continue
        dt = times_s[t + 1] - times_s[t]
        if dt <= 1e-6:
            continue
        v = float(np.linalg.norm(c[t + 1] - c[t])) / dt
        if v <= V_MOVE:
            continue
        fz = _frozen_disp_px(K, w2c, c[t], c[t + 1], t)   # 冻结相机可观测?
        if fz is not None and fz < FROZEN_PX:
            continue                                       # 深度噪声,不算运动
        moving[t] = moving[t + 1] = True
    # 最长连续运动区间
    best = (0, -1, -1)
    t = 0
    while t < T:
        if moving[t]:
            a = t
            while t + 1 < T and moving[t + 1]:
                t += 1
            span = t - a + 1
            if span > best[0]:
                best = (span, a, t)
        t += 1
    if best[0] < MIN_MOVE_FRAMES:
        return None
    _, a, b = best
    dur = float(times_s[b] - times_s[a])
    if dur < MIN_DUR_S:
        return None
    return dur, a, b


def _unique_label_insts(sc, reprs):
    """场景内 label 唯一的动态实例列表(接地无歧义;与主 QA line 同纪律)。返回 [(label, inst)]。"""
    labels = [str(o.label) for o in sc.objects]
    by_id = {int(x["instance_id"]): x for x in reprs}
    out = []
    for o in sc.objects:
        if o.kind not in ("nonrigid", "rigid"):
            continue
        lbl = str(o.label)
        if labels.count(lbl) != 1:
            continue
        inst = by_id.get(o.instance_id)
        if inst is not None:
            out.append((lbl, inst))
    return out


def _fmt_s(x):
    """秒数写进题面/选项的格式:最多 2 位小数、去尾零(0.25→"0.25",1.0→"1",0.5→"0.5")。
    09-09 修:原来只留 1 位小数,阈值网格延到 0.25/0.35/0.75 后题面会写成 0.2/0.3/0.8,与 gt 阈值不一致;
    T-a 的等比选项在短时长下也会撞字面(0.15/0.225→0.2)。"""
    return f"{x:.2f}".rstrip("0").rstrip(".")


def gen_scene(scene_dir, token_pkl, x_grid=X_GRID, legacy_x=False):
    """返回 (td_items, ta_items, fps_bin) 或 (None,None,reason)。"""
    sc = pickle.load(open(token_pkl, "rb"))
    times = getattr(sc, "frame_times_s", None)
    fps = getattr(sc, "fps_effective", None)
    if times is None or len(times) < 2 or not fps:
        return None, None, "no_time_axis"
    times = np.asarray(times, float)
    reprs = pickle.load(open(os.path.join(scene_dir, "final_repr.pkl"), "rb"))
    K, w2c = _load_camera(scene_dir)
    vid = os.path.basename(scene_dir)
    fbin = _fps_bin(float(fps))

    td, ta = [], []
    for lbl, inst in _unique_label_insts(sc, reprs):
        md = _motion_duration_s(inst, times, K, w2c)
        if md is None:
            continue
        dur, fa, fb = md
        rng = _seed(vid, inst["instance_id"], "time")

        # ---- T-d:二分"运动是否超过 X 秒";X 逐题选,使答案在该 X 桶内可平衡 ----
        # X 从 {0.5,1,1.5,2,3,4,5} 里挑一个"离 dur 有把握"的(|dur-X|>=0.25s 避免边界含糊);
        # 该题答案 = dur>X。X 的具体值由 rng 决定 → 同一 X 会既有 yes 又有 no,后续平衡器保证桶内均衡。
        # 每个"有把握"的 X 各出一题(|dur-X|>=0.25s 避边界含糊)—— 产量最大化,平衡放全局做
        for X in x_grid:                       # >=4s 长运动太稀(全 No,盲可猜),不进网格
            if not legacy_x and X < MIN_X_FRAMES / float(fps):
                continue                        # 帧粒度守卫:阈值不到 2 帧,判不了
            margin = 0.25 if legacy_x else max(1.0 / float(fps), X_REL_MARGIN * X)
            if abs(dur - X) < margin:           # 边界含糊不出题
                continue
            ans_yes = dur > X
            q = (f"In the world frame (ignoring camera motion), does the {lbl} keep moving "
                 f"for more than {_fmt_s(X)} seconds in total?")
            opts = {"A": "Yes", "B": "No"} if _seed(vid, inst["instance_id"], X).rand() < 0.5                 else {"A": "No", "B": "Yes"}
            ans = [k for k, v in opts.items() if v == ("Yes" if ans_yes else "No")][0]
            td.append(dict(video_id=vid, hallucination_type="time_reasoning",
                           event_subtype="motion_duration_binary", question=q,
                           options=opts, answer=ans, frame_range=[int(fa), int(fb)],
                           involved_inst_ids=[int(inst["instance_id"])],
                           gt_payload={"duration_s": round(dur, 3), "threshold_s": X,
                                       "fps": round(float(fps), 2), "fps_bin": fbin,
                                       "label": lbl}))

        # ---- T-a:"移动了大约几秒",干扰项 = 把同样帧数按错误帧率读出的秒 ----
        # 09-09 审计实测旧方案(从固定 fps 列表挑)正确答案 54.5% 落在第 3 大、从不在最小,盲选"第 3 大"
        # 就 54.5%;模拟证明固定列表 + ≥1.5× 间距下名次只能落 2/3。改成:先均匀抽目标名次 r∈{1..4},
        # 再放 r-1 个更小、4-r 个更大的读数,相邻恒 1.5×(等比 → 间距本身不带线索)。答案名次由构造均匀。
        n_move_frames = fb - fa                          # 运动跨的帧间隔数
        true_fps = float(fps)
        r = int(rng.randint(1, 5))                       # 目标名次(1=最小 … 4=最大)
        secs_sorted = [dur * (1.5 ** (j - (r - 1))) for j in range(4)]   # 第 r 个恰为真值
        secs = [dur] + [x for j, x in enumerate(secs_sorted) if j != r - 1]
        if len({_fmt_s(x) for x in secs}) < 4:            # 1 位小数下短时长会打成同一个数(0.15/0.225→0.2)
            continue                                      # 选项撞字面就不出这题
        order = rng.permutation(4)
        opts = {LETTERS[i]: f"About {_fmt_s(secs[order[i]])} seconds" for i in range(4)}
        ans = LETTERS[int(np.nonzero(order == 0)[0][0])]
        q = (f"In the world frame (ignoring camera motion), for about how many seconds "
             f"does the {lbl} keep moving?")
        ta.append(dict(video_id=vid, hallucination_type="time_reasoning",
                       event_subtype="motion_duration_mcq", question=q,
                       options=opts, answer=ans, frame_range=[int(fa), int(fb)],
                       involved_inst_ids=[int(inst["instance_id"])],
                       gt_payload={"duration_s": round(dur, 3), "n_move_frames": int(n_move_frames),
                                   "fps": round(true_fps, 2), "fps_bin": fbin, "label": lbl,
                                   "answer_rank": r,
                                   "distractor_secs": [round(x, 3) for x in secs[1:]],
                                   "implied_fps": [round(n_move_frames / x, 2) for x in secs[1:]]}))
    return td, ta, fbin


def _waterfill(counts, cap):
    """各档可用量 counts[b] → 每档保留量 k_b(<= counts[b] 且 <= cap·总量)使总量最大。"""
    if cap >= 1.0:
        return dict(counts)
    lo, hi = 0.0, float(sum(counts.values()))
    for _ in range(64):
        S = (lo + hi) / 2
        if sum(min(c, cap * S) for c in counts.values()) >= S:
            lo = S
        else:
            hi = S
    k = {b: int(min(c, int(cap * lo + 1e-9))) for b, c in counts.items()}
    while k and sum(k.values()) and max(k.values()) > cap * sum(k.values()) + 1e-9:
        b = max(k, key=k.get)               # 取整误差修正
        k[b] -= 1
    return k


def _cap_groups(groups, cap_mode, cap, rng):
    """groups: fps档 -> 单元列表(T-a 是题,T-d 是 (yes,no) 对)。按档封顶后返回同结构。
    cap_mode="all":每档占比 <= cap(水位填充);"five":只压 ~5 档到 <= cap(旧硬要求 1)。"""
    counts = {b: len(v) for b, v in groups.items()}
    if cap_mode == "all":
        k = _waterfill(counts, cap)
    elif cap_mode == "five":
        other = sum(c for b, c in counts.items() if b != "~5")
        k = dict(counts)
        if "~5" in k:
            k["~5"] = min(k["~5"], int(other / (1 - cap) * cap) if other else k["~5"])
    else:
        raise ValueError(cap_mode)
    out = {}
    for b, v in groups.items():
        out[b] = v if k[b] >= len(v) else sorted(rng.sample(v, k[b]), key=lambda u: (u if isinstance(u, int) else u[0]))
    return out


def _balance(items, rng, cap_mode="all", cap=0.5, strict_cap=1.0 / 3.0):
    """T-d 三约束一次性满足(2026-09-08 重写,顺序无关):
      A) 每个 threshold 桶内 yes/no 各半(X↔答案去相关,盲读 X 文本无优势)
      B) 帧率档封顶:cap_mode="all" 每档 <= cap;"five" 只压 ~5 档(旧硬要求 1)
      C) 全局 yes/no 各半
    做法:按 **(threshold, fps_bin) 单元格**把 Yes/No **配对**保留(A、C 由构造保证);封顶时
    **整对地丢**(丢一对 = 同格里一 Yes 一 No),A、C 不被破坏。
    另外在结果上再按 strict_cap 取一个更严的子集,标 eval_strict=True(报数用它,训练用全集)。
    🔴 旧实现是"A → 压 ~5 → 全局补平"三步串行,后两步会拆掉前面的保证:实测 42k 上
    ~5 档冲到 44.5%(要求 <=33.3%)、五个 threshold 桶全部失衡(X=0.5 上 No:Yes=186:72),
    根因是 ~5 档天然偏 "No",压它就制造全局失衡,补平时又反过来砍掉非 ~5 的一侧。
    T-a(MCQ)只做 B(+严格子集)。"""
    from collections import defaultdict
    if not items:
        return items
    is_binary = items[0]["event_subtype"] == "motion_duration_binary"

    groups = defaultdict(list)                     # fps档 -> 单元
    if is_binary:
        # 配对单元 = (threshold, fps_bin, label):同标签内 Yes/No 成对 → 标签先验也被抹平。
        # 09-09 审计实测:只按 (X,档) 配对时"只看标签猜答案"MAP 67%(man/woman 65% Yes、bicycle 41%),
        # 细到标签后恒 50%,代价题量 7,920→5,516。
        cells = defaultdict(lambda: {"Yes": [], "No": []})
        for i, d in enumerate(items):
            gp = d["gt_payload"]
            cells[(gp["threshold_s"], gp["fps_bin"], gp.get("label", ""))][d["options"][d["answer"]]].append(i)
        for (X, fb, _lbl), yn in cells.items():
            k = min(len(yn["Yes"]), len(yn["No"]))
            if k:
                groups[fb] += list(zip(rng.sample(yn["Yes"], k), rng.sample(yn["No"], k)))
    else:
        for i, d in enumerate(items):
            groups[d["gt_payload"]["fps_bin"]].append(i)

    kept = _cap_groups(groups, cap_mode, cap, rng)
    strict = _cap_groups(kept, cap_mode, strict_cap, rng)
    strict_units = {u for v in strict.values() for u in v}
    keep_idx = {}
    for b, v in kept.items():
        for u in v:
            flag = u in strict_units
            for i in ((u,) if isinstance(u, int) else u):
                keep_idx[i] = flag
    out = []
    for i in sorted(keep_idx):
        d = items[i]
        d["eval_strict"] = bool(keep_idx[i])
        out.append(d)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-root", default="data/scenes")
    ap.add_argument("--tokens", default="data/tokens_v4td")
    ap.add_argument("--out", default="outputs/time_qa")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dump-raw", action="store_true",
                    help="同时导出平衡前的候选池(_raw_td/_raw_ta.jsonl),用于离线试不同的 fps 封顶策略")
    ap.add_argument("--x-grid", default=",".join(str(x) for x in X_GRID),
                    help="T-d 阈值网格(秒,逗号分隔);所有帧率档共享")
    ap.add_argument("--legacy-x", action="store_true",
                    help="复现 506 批:旧网格 (0.5..3.0) + 固定 0.25s 余量 + 不做帧粒度守卫")
    ap.add_argument("--cap-mode", choices=["all", "five"], default="all",
                    help="帧率档封顶:all=每档 <= --cap;five=只压 ~5 档(旧硬要求 1)")
    ap.add_argument("--cap", type=float, default=0.5, help="全集的每档占比上限(默认 1/2)")
    ap.add_argument("--strict-cap", type=float, default=1.0 / 3.0,
                    help="在全集内再标一个更严的子集 eval_strict=True(默认每档 <= 1/3),报数用")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    import random
    rng = random.Random(args.seed)

    x_grid = X_GRID_LEGACY if args.legacy_x else tuple(float(x) for x in args.x_grid.split(","))
    scenes = sorted(glob.glob(os.path.join(args.batch_root, "scene_*", "final_repr.pkl")))
    td_all, ta_all = [], []
    skip = {}
    binc = {}
    for f in scenes:
        sd = os.path.dirname(f)
        tp = os.path.join(args.tokens, os.path.basename(sd) + ".pkl")
        if not os.path.exists(tp):
            skip["no_tokens"] = skip.get("no_tokens", 0) + 1
            continue
        td, ta, info = gen_scene(sd, tp, x_grid=x_grid, legacy_x=args.legacy_x)
        if td is None:
            skip[info] = skip.get(info, 0) + 1
            continue
        binc[info] = binc.get(info, 0) + 1
        td_all += td
        ta_all += ta

    if args.dump_raw:                            # 平衡前的候选池:换封顶策略可离线重算,不必重跑生成
        for nm, it in (("_raw_td", td_all), ("_raw_ta", ta_all)):
            with open(os.path.join(args.out, nm + ".jsonl"), "w") as fh:
                for d in it:
                    fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    td_all = _balance(td_all, rng, args.cap_mode, args.cap, args.strict_cap)
    ta_all = _balance(ta_all, rng, args.cap_mode, args.cap, args.strict_cap)
    for name, items in (("time_qa_binary", td_all), ("time_qa_mcq", ta_all)):
        for i, d in enumerate(items):
            d["id"] = f"{d['video_id']}_{d['event_subtype']}_{i}"
        with open(os.path.join(args.out, name + ".jsonl"), "w") as fh:
            for d in items:
                fh.write(json.dumps(d, ensure_ascii=False) + "\n")

    def _dist(items):
        from collections import Counter
        fb = Counter(d["gt_payload"]["fps_bin"] for d in items)
        ans = Counter(d["options"][d["answer"]] if len(d["options"]) == 2 else "mcq" for d in items)
        return dict(fb), dict(ans)
    def _leak(items):
        """X 文本对帧率档的泄漏:看到 X 猜档的 MAP 准确率 vs 先验(只猜最大档)。"""
        from collections import Counter
        if not items:
            return {}
        n = len(items)
        joint = Counter((d["gt_payload"]["threshold_s"], d["gt_payload"]["fps_bin"]) for d in items)
        byX = {}
        for (X, fb), c in joint.items():
            byX[X] = max(byX.get(X, 0), c)
        prior = max(Counter(d["gt_payload"]["fps_bin"] for d in items).values()) / n
        return dict(map_acc_from_X=round(sum(byX.values()) / n, 4), prior_max_bin=round(prior, 4))
    td_strict = [d for d in td_all if d.get("eval_strict")]
    ta_strict = [d for d in ta_all if d.get("eval_strict")]
    rep = dict(n_scenes_used=sum(binc.values()), skip=skip, scene_fps_bins=binc,
               config=dict(x_grid=list(x_grid), legacy_x=args.legacy_x, cap_mode=args.cap_mode,
                           cap=args.cap, strict_cap=args.strict_cap),
               T_d=dict(n=len(td_all), fps=_dist(td_all)[0], answer=_dist(td_all)[1],
                        X_leak=_leak(td_all),
                        strict=dict(n=len(td_strict), fps=_dist(td_strict)[0], answer=_dist(td_strict)[1],
                                    X_leak=_leak(td_strict))),
               T_a=dict(n=len(ta_all), fps=_dist(ta_all)[0],
                        strict=dict(n=len(ta_strict), fps=_dist(ta_strict)[0])))
    json.dump(rep, open(os.path.join(args.out, "_gen_report.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(rep, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
