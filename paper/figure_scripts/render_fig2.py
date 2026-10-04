"""图 2(Pipeline 总览)渲染 v2(09-25 少文字改版)—— 按 fig2_blueprint.md。输出 fig2_pipeline.pdf(字体嵌入)+ fig2_pipeline.png(300 dpi)。
所有真实素材只读;图上每个数字 / 引文在运行时从源文件重新核对(不一致就报错退出),包括漏斗的逐 clip 归类(约 40 秒)。
用法:<stereo4d-env python> render_fig2.py
"""
import colorsys
import json
import os
import pickle
import re
import subprocess
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import (Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, PathPatch,  # noqa: E402
                                Polygon, Rectangle)
from matplotlib.path import Path  # noqa: E402
from matplotlib.text import Text  # noqa: E402

R = str(__import__("pathlib").Path(__file__).resolve().parents[2])          # repo root(本文件在 paper/figure_scripts/)
S = R
M = R + "/data"
CLIP = "pwBorgoDJh0_141875209"
V = "scene_" + CLIP
SD = f"{M}/scenes/{V}"
FR = f"{S}/data/stereo4d/frames/{CLIP}"
OUT = R + "/paper/figure_assets/figures"          # 合成图输出目录
G1PY = os.environ.get("G1_PY", sys.executable)   # g1/data.py 需要 torch
for p in (S, S + "/src/tokenization", os.path.dirname(os.path.abspath(__file__))):
    if p not in sys.path:
        sys.path.insert(0, p)
import SCHEMA  # noqa: E402,F401  (VideoInput / Masklet 反序列化)
from render_figs import match_masks, obj_points, project  # noqa: E402

QID = V + "_disp_grid_daf506b2"
VAL = M + "/trajqa_b/final/val.jsonl"
OURS = S + "/results/eval/selection_val/s2wsd_pB_s2_cos1ep_s0_final3507_traj.json.items.jsonl"
RGB = S + "/results/eval/selection_val/rgbsw_pB_rgbsw_9002_s0_final_rgb.json.items.jsonl"
CAPTION = ("The bicycle moves about 13.5 m in the world frame at an average speed of 4.58 m/s; "
           "it moves away from the shopping bag; it moves faster than the shopping bag.")
INSTR = "In the world frame (ignoring camera motion), describe how the bicycle moves during the video."
QUESTION = ("In the world frame (ignoring camera motion), approximately how far (in meters) did the backpack "
            "move between frame 4 and frame 14 of 34?")
PERSON_T6 = 3          # ⑥ 的示例人物(图 1 里背包的主人)
TSTAR = 9              # 其参考帧 t*(scene_manifest.json 核对)
CROP6 = (40, 372, 180, 512)   # ⑥ 裁剪框 x0, y0, x1, y1(512 像素坐标)

# 漏斗(剔除原因按"运动物体在哪一步丢失"归类;check_funnel() 逐 clip 重算并核对)
FUNNEL = dict(total=108015, released=89454, vlm=29915, nomask=9207, tracks=5725, motion=2444, fail=11,
              kept=42152, videos=4079)


# ---------------------------------------------------------------- 源数据核对
def check_funnel():
    sk = f"{M}/curation/skipped_static"           # 由 data/curation/skipped_static_records.tar.gz 解压得到
    c = dict(vlm=0, nomask=0, tracks=0, motion=0)
    for n in sorted(x for x in os.listdir(sk) if x.startswith("scene_")):
        d = f"{sk}/{n}"
        if not os.path.exists(d + "/gt_dense_stats.json"):
            pre = json.load(open(d + "/stage0_pre.json"))
            assert not any(i.get("motion") == "dynamic" for i in pre["instances"]), n
            c["vlm"] += 1
            continue
        st = json.load(open(d + "/gt_dense_stats.json"))
        if st.get("skip_reason") == "no_masklets":
            c["nomask"] += 1
            continue
        ins = list((st.get("instances") or {}).values())
        assert not any(v.get("emitted") and v.get("final_motion") == "dynamic" for v in ins), n
        dyn = [v for v in ins if v.get("qwen_motion") == "dynamic"]
        if not dyn:
            c["nomask"] += 1                                  # VLM 说在动的物体一个掩码都没拿到
        elif int(st.get("n_flip_dyn2static") or 0) > 0 and any(v.get("final_motion") == "static" for v in dyn):
            c["motion"] += 1                                  # 运动门把它们降成了静态
        else:
            assert all(not v.get("emitted") for v in dyn), n
            c["tracks"] += 1                                  # 自有轨迹不足 min_tracks(8)
    fail = sum(1 for f in ("gave_up.txt", "_rej.txt") for ln in open(f"{M}/curation/{f}") if ln.strip())
    ok = [e.name for e in os.scandir(f"{M}/scenes") if e.name.startswith("scene_")]
    n_ok = sum(os.path.exists(f"{M}/scenes/{n}/final_repr.pkl") for n in ok)
    videos = len({n[len("scene_"):].rsplit("_", 1)[0] for n in ok})
    got = dict(c, fail=fail, kept=n_ok + 1, videos=videos)          # +1:gt npz 损坏被移出的那段(已交付)
    for k, v in got.items():
        assert FUNNEL[k] == v, ("漏斗数不符", k, FUNNEL[k], v)
    assert FUNNEL["released"] == FUNNEL["kept"] + sum(c.values()) + fail
    return ok


def check_keypoints(ok):
    """全量 42,151 段的关键点均值(非刚体每部件 1 点、刚体 4 点、静止物体 ≤ 4 个代表点)。"""
    tot = []
    for n in ok:
        fr = pickle.load(open(f"{M}/scenes/{n}/final_repr.pkl", "rb"))
        tot.append(sum(len(o.get("keypoint_ids") or []) for o in fr)
                   + sum(len(np.asarray(o["ref_positions"])) for o in fr if "keypoint_ids" not in o and "ref_positions" in o))
    mean = float(np.mean(tot))
    assert 44.5 <= mean < 45.5, mean                        # 图上写 "~45"
    return mean


def check_code_constants():
    src = open(S + "/src/stage4/gt_dense_frontend.py").read()
    for k, v in (("CAP", "800"), ("MIN_TRACKS", "8"), ("MIN_FRAC", "0.5"), ("AMBIG_MARGIN", "0.15")):
        assert re.search(rf"^{k} = {re.escape(v)}\b", src, re.M), k
    run = open(S + "/src/run_davis_batch_v3.py").read()
    assert re.search(r'"--min-area-frac", type=float, default=0\.003', run)
    assert re.search(r'"--n-candidates", type=int, default=3', run)


def check_sources():
    # ② VLM 清单
    pre = json.load(open(f"{M}/scenes/{V}/stage0_pre.json"))["instances"]
    by = {i["concept"]: i for i in pre}
    assert len(pre) == 12
    assert (by["person"]["object_type"], by["person"]["motion"], by["person"]["count"]) == ("non-rigid", "dynamic", 12)
    assert (by["bicycle"]["object_type"], by["bicycle"]["motion"]) == ("rigid", "dynamic")
    assert (by["storefront"]["object_type"], by["storefront"]["motion"]) == ("rigid", "static")
    # ⑥ 参考帧与关键点
    man = {d["instance_id"]: d for d in json.load(open(SD + "/scene_manifest.json"))["instances"]}
    assert man[PERSON_T6]["reference_frame_index"] == TSTAR
    fr = pickle.load(open(SD + "/final_repr.pkl", "rb"))
    o3 = next(o for o in fr if o["instance_id"] == PERSON_T6)
    assert o3["object_type"] == "non-rigid" and len(o3["keypoint_ids"]) == 4
    bike = next(o for o in fr if o["instance_id"] == 17)
    assert bike["label"] == "bicycle" and np.asarray(bike["se3_R"]).shape == (34, 3, 3) and len(bike["keypoint_ids"]) == 4
    store = next(o for o in fr if o["instance_id"] == 0)
    assert store["label"] == "storefront" and np.asarray(store["ref_positions"]).shape[0] <= 4
    # caption 与指令原文
    n_t8, cap_ok = 0, False
    for ln in open(M + "/captions/t8_train.jsonl"):
        n_t8 += 1
        if V in ln:
            d = json.loads(ln)
            if d.get("video_id") == V and d["question"] == INSTR:
                cap_ok = d["answer"] == CAPTION
    assert n_t8 == 65556 and cap_ok, (n_t8, cap_ok)
    # QA:生成报告、划分、留出视频
    rep = json.load(open(M + "/trajqa_b/candidates_m6/_gen_report.json"))
    assert rep["n_items"] == 333661
    split = {k: sum(1 for _ in open(f"{M}/trajqa_b/final/{f}.jsonl"))
             for k, f in (("train", "train"), ("val", "val"), ("test", "test"), ("heldout", "heldout_test"))}
    assert split == dict(train=56120, val=1318, test=4492, heldout=3430) and sum(split.values()) == 65360, split

    def ytids(f):
        return {json.loads(ln)["video_id"][len("scene_"):].rsplit("_", 1)[0] for ln in open(f"{M}/trajqa_b/final/{f}.jsonl")}
    ho, tr = ytids("heldout_test"), ytids("train")
    assert len(ho) == 166 and not (ho & tr)
    # 同一道题的两种回答
    q = next(json.loads(ln) for ln in open(VAL) if QID in ln)
    assert q["question"] == QUESTION and q["options"] == {"A": "1.9", "B": "1.1", "C": "1.4", "D": "2.6"} and q["answer"] == "A"
    assert f"{q['gt_payload']['value']:.2f}" == "2.09"
    ours = next(json.loads(ln) for ln in open(OURS) if QID in ln)
    rgb = next(json.loads(ln) for ln in open(RGB) if QID in ln)
    for k in ("position_frame_4_m: (-1.4, 0.9, 3.4)", "position_frame_14_m: (-1.0, 1.3, 5.5)", "displacement_m: 2.2", "<answer>A</answer>"):
        assert k in ours["gen"], k
    for k in ("position_frame_4_m: (-0.9, 1.3, 2.5)", "position_frame_14_m: (-0.8, 1.3, 3.6)", "displacement_m: 1.1", "<answer>B</answer>"):
        assert k in rgb["gen"], k
    assert ours["pred"] == "A" and rgb["pred"] == "B"
    # 训练时真实的 token 串(g1/data.py 原函数):人 → [part 1] …… 自行车 …… 相机在最后
    code = ("import sys, pickle, re; sys.path[:0] = [%r, %r]\n"
            "from g1.traj_serialize import serialize_scene_labeled, group_spans\n"
            "from g1.data import build_traj_block, TRAJ_TOKEN\n"
            "recs, labels = serialize_scene_labeled(pickle.load(open(%r, 'rb')))\n"
            "txt = build_traj_block(group_spans(recs), labels).split(chr(10))[1]\n"
            "print(re.sub('(?:%%s)+' %% re.escape(TRAJ_TOKEN), lambda m: ' <%%d> ' %% (len(m.group(0)) // len(TRAJ_TOKEN)), txt))"
            % (S + "/src/tokenization", S + "/src/training", f"{M}/tokens_v4td/{V}.pkl"))
    block = subprocess.run([G1PY, "-c", code], capture_output=True, text=True, check=True).stdout.split()
    s = " ".join(block)
    assert s.startswith("[scene] <3> [the person] <1> [part 1] <5>"), s[:80]
    assert "[the bicycle] <17>" in s and s.rstrip().endswith("[the camera] <13>"), s[-80:]
    check_code_constants()
    ok = check_funnel()
    kp_mean = check_keypoints(ok)
    return kp_mean


# ---------------------------------------------------------------- 样式与画布
TXT, SUB, EDGE, SLATE = "#1A1A1A", "#5F6368", "#B8BEC6", "#3E4C5E"
AZ, AZF, GRN, MINTF, RED = "#1F76D6", "#EAF2FB", "#2CA02C", "#EEF6EE", "#D62728"
ICE, CORAL, FLAME, CREAM, LAV, PANEL = "#DCEBF7", "#F8D9C4", "#E8702A", "#FFF7E6", "#F1ECF8", "#F4F6F8"
ORANGE, GREENO = "#FF7F0E", "#2CA02C"
FT, FB, FS = 7.0, 6.2, 6.0            # 标题 / 正文 / 注释(最小 6 pt)
LH = 0.092                            # 6 pt 行距(in)

# text.hinting=no_hinting + dpi 300:让量出来的字宽 = PDF 实际字宽(图 1 踩过的坑)
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif",
                     "font.sans-serif": ["Liberation Sans", "DejaVu Sans"], "axes.unicode_minus": False,
                     "text.hinting": "no_hinting"})
W, H = 5.5, 3.67
fig = plt.figure(figsize=(W, H), dpi=300)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
rend = fig.canvas.get_renderer()
TEXTS = []            # (artist, zone) 供最后做重叠 / 越界检查


def text(x, y, s, size=FB, color=TXT, ha="left", va="center", weight="normal", style="normal", zone=None, **kw):
    t = ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, fontweight=weight, fontstyle=style, zorder=kw.pop("zorder", 8), **kw)
    TEXTS.append((t, zone))
    return t


def width_in(t):
    return t.get_window_extent(renderer=rend).width / fig.dpi


def tw(s, size=FB, weight="normal", style="normal"):
    t = ax.text(0, -5, s, fontsize=size, fontweight=weight, fontstyle=style)
    w = width_in(t); t.remove()
    return w


def wrap(s, width, size=FS, weight="normal", style="normal"):
    words, lines, cur = s.split(" "), [], ""
    for w_ in words:
        trial = (cur + " " + w_).strip()
        if tw(trial, size, weight, style) <= width or not cur:
            cur = trial
        else:
            lines.append(cur); cur = w_
    return lines + [cur]


def para(x, y, s, width, size=FS, color=SUB, lh=LH, weight="normal", style="normal", ha="left", zone=None):
    """从 y(第一行中心)开始逐行写;返回最后一行下方的 y(= 下一段第一行中心)。"""
    for i, ln in enumerate(wrap(s, width, size, weight, style)):
        text(x, y + i * lh, ln, size=size, color=color, weight=weight, style=style, ha=ha, zone=zone)
    return y + (i + 1) * lh


def rbox(x0, y0, x1, y1, fc="white", ec=EDGE, lw=0.5, r=0.03, z=1, ls="-"):
    p = FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw,
                       zorder=z, linestyle=ls)
    ax.add_patch(p)
    return p


def arrow(p0, p1, color="#7F7F7F", lw=0.7, ms=6, cs="arc3,rad=0", z=6, ls="-", style="-|>"):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms, lw=lw, color=color, connectionstyle=cs, zorder=z,
                        linestyle=ls, shrinkA=0, shrinkB=0)
    ax.add_patch(a)
    return a


def polyarrow(pts, color="#7F7F7F", lw=0.7, ms=6, z=6, ls="-"):
    """折线箭头(只有最后一段带箭头)。"""
    xs, ys = zip(*pts[:-1])
    ax.plot(list(xs) + [pts[-2][0]], list(ys) + [pts[-2][1]], color=color, lw=lw, zorder=z, ls=ls, solid_capstyle="butt")
    arrow(pts[-2], pts[-1], color=color, lw=lw, ms=ms, z=z, ls=ls)


def img(a, x0, y0, w, h, z=3, edge=True, interp="none"):
    ax.imshow(a, extent=(x0, x0 + w, y0 + h, y0), interpolation=interp, zorder=z)
    if edge:
        ax.add_patch(Rectangle((x0, y0), w, h, fill=False, ec="#9AA0A6", lw=0.35, zorder=z + 0.1))


# ---------------------------------------------------------------- 图标(全部矢量,示意)
def frame_stack(x, y, s=0.085, z=7):
    for d in (0.03, 0.015, 0.0):
        ax.add_patch(Rectangle((x + d, y - d), s, s * 0.72, fc="white", ec="#7F7F7F", lw=0.4, zorder=z))


def snowflake(cx, cy, r=0.035, color="#2F6FB0", z=9):
    for a in (0, 60, 120):
        t = np.deg2rad(a)
        dx, dy = r * np.cos(t), r * np.sin(t)
        ax.plot([cx - dx, cx + dx], [cy - dy, cy + dy], color=color, lw=0.6, zorder=z, solid_capstyle="round")
        for sgn in (1, -1):
            ex, ey = cx + sgn * dx * 0.95, cy + sgn * dy * 0.95
            for b in (a + 150, a - 150):
                tb = np.deg2rad(b if sgn > 0 else b + 180)
                ax.plot([ex, ex + 0.35 * r * np.cos(tb)], [ey, ey + 0.35 * r * np.sin(tb)], color=color, lw=0.5, zorder=z)


def flame(cx, cy, h=0.075, color=FLAME, z=9):
    """火焰(可训练)。画布 y 轴向下:尖朝上 = y 较小。"""
    w = h * 0.64
    V_ = [(cx, cy - h / 2),
          (cx + w * 0.22, cy - h * 0.22), (cx + w / 2, cy + h * 0.02), (cx + w / 2, cy + h * 0.2),
          (cx + w / 2, cy + h * 0.42), (cx + w * 0.26, cy + h / 2), (cx, cy + h / 2),
          (cx - w * 0.26, cy + h / 2), (cx - w / 2, cy + h * 0.42), (cx - w / 2, cy + h * 0.2),
          (cx - w / 2, cy + h * 0.02), (cx - w * 0.22, cy - h * 0.22), (cx, cy - h / 2), (cx, cy - h / 2)]
    C_ = [Path.MOVETO] + [Path.CURVE4] * 12 + [Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(V_, C_), fc=color, ec="none", zorder=z))
    ax.add_patch(Ellipse((cx, cy + h * 0.22), w * 0.45, h * 0.42, fc="#FFD27A", ec="none", zorder=z + 0.1))


def shield_check(cx, cy, s=0.06, color="#4C6A92", z=8):
    pts = [(cx - s / 2, cy - s / 2), (cx + s / 2, cy - s / 2), (cx + s / 2, cy + s * 0.05), (cx, cy + s / 2),
           (cx - s / 2, cy + s * 0.05)]
    ax.add_patch(Polygon(pts, closed=True, fc="#E3EAF4", ec=color, lw=0.5, zorder=z))
    ax.plot([cx - s * 0.2, cx - s * 0.03, cx + s * 0.22], [cy - s * 0.02, cy + s * 0.16, cy - s * 0.2], color=GRN, lw=0.8,
            zorder=z + 0.1, solid_capstyle="round", solid_joinstyle="round")


def box2d(cx, cy, s=0.06, z=8):
    ax.add_patch(Rectangle((cx - s / 2, cy - s * 0.4), s, s * 0.8, fill=False, ec=AZ, lw=0.5, ls=(0, (1.5, 1)), zorder=z))


def eye_badge(cx, cy, r=0.045, z=9):
    ax.add_patch(Circle((cx, cy), r, fc=SLATE, ec="white", lw=0.6, zorder=z))
    ax.add_patch(Ellipse((cx, cy), r * 1.35, r * 0.8, fc="white", ec="none", zorder=z + 0.1))
    ax.add_patch(Circle((cx, cy), r * 0.26, fc=SLATE, ec="none", zorder=z + 0.2))


def bicycle(cx, cy, s=0.16, color="#4D4D4D", z=8):
    r = s * 0.24
    for dx in (-s * 0.3, s * 0.3):
        ax.add_patch(Circle((cx + dx, cy + s * 0.12), r, fill=False, ec=color, lw=0.6, zorder=z))
    pts = [(cx - s * 0.3, cy + s * 0.12), (cx - s * 0.05, cy - s * 0.12), (cx + s * 0.18, cy - s * 0.12),
           (cx + s * 0.3, cy + s * 0.12), (cx + 0.0, cy + s * 0.12), (cx - s * 0.05, cy - s * 0.12)]
    ax.plot(*zip(*pts), color=color, lw=0.6, zorder=z)
    ax.plot([cx + s * 0.18, cx + s * 0.22], [cy - s * 0.12, cy - s * 0.2], color=color, lw=0.6, zorder=z)


def storefront(cx, cy, s=0.16, color="#4D4D4D", z=8):
    x0, y0 = cx - s * 0.4, cy - s * 0.28
    ax.add_patch(Rectangle((x0, y0 + s * 0.14), s * 0.8, s * 0.42, fill=False, ec=color, lw=0.6, zorder=z))
    for i in range(4):
        ax.add_patch(Rectangle((x0 + i * s * 0.2, y0), s * 0.2, s * 0.14, fc=("#E8702A" if i % 2 == 0 else "white"),
                               ec=color, lw=0.4, zorder=z))
    ax.add_patch(Rectangle((cx - s * 0.08, y0 + s * 0.3), s * 0.16, s * 0.26, fill=False, ec=color, lw=0.5, zorder=z))


def xyz_axes(ox, oy, L=0.05, z=9):
    for (dx, dy), c in (((L, 0), "#D62728"), ((0, -L), "#2CA02C"), ((-L * 0.6, L * 0.5), "#1F76D6")):
        arrow((ox, oy), (ox + dx, oy + dy), color=c, lw=0.6, ms=3.5, z=z)


def cylinder(cx, cy, w=0.1, h=0.1, fc="#DCE7F5", ec="#4C6A92", z=8):
    e = h * 0.28
    ax.add_patch(Rectangle((cx - w / 2, cy - h / 2 + e / 2), w, h - e, fc=fc, ec="none", zorder=z))
    ax.plot([cx - w / 2, cx - w / 2], [cy - h / 2 + e / 2, cy + h / 2 - e / 2], color=ec, lw=0.5, zorder=z + 0.1)
    ax.plot([cx + w / 2, cx + w / 2], [cy - h / 2 + e / 2, cy + h / 2 - e / 2], color=ec, lw=0.5, zorder=z + 0.1)
    ax.add_patch(Ellipse((cx, cy + h / 2 - e / 2), w, e, fc=fc, ec=ec, lw=0.5, zorder=z + 0.05))
    ax.add_patch(Ellipse((cx, cy - h / 2 + e / 2), w, e, fc=fc, ec=ec, lw=0.5, zorder=z + 0.2))


def red_x(cx, cy, r=0.022, lw=1.0, z=9):
    ax.plot([cx - r, cx + r], [cy - r, cy + r], color=RED, lw=lw, zorder=z, solid_capstyle="round")
    ax.plot([cx - r, cx + r], [cy + r, cy - r], color=RED, lw=lw, zorder=z, solid_capstyle="round")


def check_mark(cx, cy, s=1.0, z=9):
    ax.plot([cx - 0.04 * s, cx - 0.01 * s, cx + 0.045 * s], [cy - 0.004 * s, cy + 0.028 * s, cy - 0.04 * s], color=GRN,
            lw=1.4, solid_capstyle="round", solid_joinstyle="round", zorder=z)


# ---------------------------------------------------------------- 真实数据面板(numpy RGB,原样嵌入)
from matplotlib import cm  # noqa: E402

_tab = [tuple(int(255 * c) for c in cm.tab20(i)[:3]) for i in range(20)] + [(120, 120, 120)]
COL = [_tab[i] for i in list(range(0, 20, 2)) + list(range(1, 20, 2)) + [20]]   # masklet i → 颜色(深色在前)
KPC = [(214, 39, 40), (31, 119, 180), (44, 160, 44), (148, 103, 189)]            # ⑥ 关键点(按部件)


def load_rgb(p):
    a = cv2.imread(p)
    assert a is not None, p
    return cv2.cvtColor(a, cv2.COLOR_BGR2RGB)


def mask_overlay(frame, masklets, t, alpha=0.5, thick=2):
    out = frame.astype(float).copy()
    for i, m in enumerate(masklets):
        mk = m.masks[t]
        if mk.any():
            out[mk] = out[mk] * (1 - alpha) + np.array(COL[i]) * alpha
    out = out.astype(np.uint8)
    for i, m in enumerate(masklets):
        mk = m.masks[t].astype(np.uint8)
        if mk.any():
            cs, _ = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(out, cs, -1, COL[i], thick, cv2.LINE_AA)
    return out


def partsam_crops():
    a = cv2.imread(R + "/paper/figure_assets/partsam_parts_diagnostic.png", cv2.IMREAD_UNCHANGED)
    assert a.shape[:2] == (2000, 1550), a.shape
    c = a[132:360, 1320:1384]                                    # 只取给定框(y 132–360, x 1320–1384)
    rgb = cv2.cvtColor(c[..., :3], cv2.COLOR_BGR2RGB).astype(float)
    alpha = c[..., 3:4].astype(float) / 255.0 if c.shape[2] == 4 else np.ones(rgb.shape[:2] + (1,))
    color = (rgb * alpha + 255 * (1 - alpha)).astype(np.uint8)   # 叠到白底
    fg = (np.abs(color.astype(int) - 255).sum(-1) > 40)
    # 按色相把部件分开(只用于:灰度版消去部件间亮度差 + 在部件中心撒示意点)
    hsv = cv2.cvtColor(color, cv2.COLOR_RGB2HSV).astype(np.float32)
    feat = np.stack([np.cos(hsv[..., 0] / 90 * np.pi) * hsv[..., 1], np.sin(hsv[..., 0] / 90 * np.pi) * hsv[..., 1]], -1)[fg]
    k = 5
    _, lab, _ = cv2.kmeans(feat, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5), 5, cv2.KMEANS_PP_CENTERS)
    lab_img = np.full(fg.shape, -1); lab_img[fg] = lab.ravel()
    v = hsv[..., 2] / 255.0
    gray = np.full(fg.shape, 255.0)
    dots = []
    for j in range(k):
        sel = lab_img == j
        if sel.sum() < 30:
            continue
        shade = np.clip(v[sel] / np.median(v[sel]), 0.7, 1.15)
        gray[sel] = 185 * shade
        ys, xs = np.nonzero(sel)
        cy, cx = np.median(ys), np.median(xs)
        # 取离中位点最近的部件像素(保证点落在部件上)
        i = int(np.argmin((ys - cy) ** 2 + (xs - cx) ** 2))
        dots.append((xs[i] + 0.5, ys[i] + 0.5, tuple(int(x) for x in np.median(color[sel], axis=0))))
    g3 = np.repeat(np.clip(gray, 0, 255).astype(np.uint8)[..., None], 3, -1)
    return g3, color, dots


def build_panels():
    vi = pickle.load(open(f"{M}/figure_scenes/{V}/video_input.pkl", "rb"))
    assert len(vi.masklets) == 21 and vi.masklets[0].masks.shape == (34, 512, 512) and vi.frames.shape == (34, 512, 512, 3)
    fr = pickle.load(open(SD + "/final_repr.pkl", "rb"))
    g = np.load(SD + "/gt_stereo4d.npz", allow_pickle=True)
    iids = [o["instance_id"] for o in fr]
    mm = match_masks(vi, SD, iids, 34, 512, 512)
    assert sorted(mm) == sorted(iids) and min(f for _, f in mm.values()) >= 0.5, mm
    mi_of = {iid: vi.masklets.index(m) for iid, (m, _) in mm.items()}
    P = {}
    P["frames"] = [load_rgb(f"{FR}/{t:05d}.jpg") for t in (0, 16, 33)]
    assert np.array_equal(P["frames"][1], vi.frames[16])
    # ③ 掩码
    P["mask16"] = mask_overlay(vi.frames[16], vi.masklets, 16)
    P["mask_strip"] = [mask_overlay(vi.frames[t], vi.masklets, t, thick=3) for t in (0, 16, 33)]
    # ④-B GT 轨迹点(第 16 帧可见),按实例颜色(与 ③ 同色)
    a = (vi.frames[16].astype(float) * 0.5).astype(np.uint8)
    n_pts = 0
    for iid in sorted(mi_of):
        d = np.load(f"{SD}/dense_tracks_{iid}.npz")
        assert d["uv_norm"].shape[0] <= 800
        uv = d["uv_norm"][:, 16] * 512
        for (u, v_) in uv[d["vis"][:, 16]]:
            cv2.circle(a, (int(round(u)), int(round(v_))), 3, COL[mi_of[iid]], -1, cv2.LINE_AA); n_pts += 1
    P["tracks16"] = a
    # ⑥ 关键点 ↔ 像素 ↔ 轨迹(person 3,t* = 9)
    x0, y0, x1, y1 = CROP6
    sc = 3
    f9 = load_rgb(f"{FR}/{TSTAR:05d}.jpg")[y0:y1, x0:x1]
    b = cv2.resize(f9, ((x1 - x0) * sc, (y1 - y0) * sc), interpolation=cv2.INTER_CUBIC)
    b = (b.astype(float) * 0.75).astype(np.uint8)
    d3 = np.load(f"{SD}/dense_tracks_{PERSON_T6}.npz")
    uv = d3["uv_norm"][:, TSTAR] * 512; vis = d3["vis"][:, TSTAR]
    for (u, v_) in uv[vis]:
        cv2.circle(b, (int(round((u - x0) * sc)), int(round((v_ - y0) * sc))), 2, (225, 225, 225), -1, cv2.LINE_AA)
    o3 = next(o for o in fr if o["instance_id"] == PERSON_T6)
    Pw, Vw = obj_points(o3)
    E = np.asarray(g["extrs_w2c"], float); K = np.asarray(g["intrinsics_K"], float)
    uvk, _ = project(Pw, E, K, 512, 512)
    for k in range(Pw.shape[0]):
        assert Vw[k, TSTAR]
        dd = np.linalg.norm(uv - uvk[k, TSTAR], axis=1); dd[~vis] = 1e9
        assert dd.min() < 0.05, (k, dd.min())                     # 关键点恰好落在一条 GT 轨迹上
        fs = [f for f in range(TSTAR - 3, TSTAR + 4) if 0 <= f < 34 and Vw[k, f]]
        for fa, fb in zip(fs[:-1], fs[1:]):
            if fb == fa + 1:
                pa = (int(round((uvk[k, fa, 0] - x0) * sc)), int(round((uvk[k, fa, 1] - y0) * sc)))
                pb = (int(round((uvk[k, fb, 0] - x0) * sc)), int(round((uvk[k, fb, 1] - y0) * sc)))
                cv2.line(b, pa, pb, KPC[k], 3, cv2.LINE_AA)
    for k in range(Pw.shape[0]):
        c = (int(round((uvk[k, TSTAR, 0] - x0) * sc)), int(round((uvk[k, TSTAR, 1] - y0) * sc)))
        cv2.circle(b, c, 10, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(b, c, 8, KPC[k], -1, cv2.LINE_AA)
    P["link"] = b
    P["n_link_tracks"] = int(vis.sum())
    # ⑤ 网格 / 部件 / 关键点(示意点落在真实部件中心)
    P["mesh_gray"], P["mesh_parts"], P["part_dots"] = partsam_crops()
    # ⑦ 只画轨迹的第 16 帧格
    tp = load_rgb(R + f"/paper/figure_assets/{CLIP}/fig1_teaser_key_objects_panel_traj_only.png")
    assert tp.shape[:2] == (512, 2592), tp.shape
    P["traj16"] = tp[:, 1560:2072]
    P["n_track_pts16"] = n_pts
    return P




PURP = "#7B4FA8"


def t4w_schematic(x0, y0, s):
    """示意(不是真实输出):一个物体掩码(不规则轮廓)里撒规则网格点,每点拖一小段轨迹。"""
    th = np.linspace(0, 2 * np.pi, 80)
    rr = 0.36 + 0.06 * np.sin(3 * th + 0.5) + 0.04 * np.cos(5 * th)
    cx, cy = x0 + s * 0.5, y0 + s * 0.5
    ax.add_patch(Polygon(np.c_[cx + s * rr * np.cos(th) * 0.95, cy + s * rr * np.sin(th) * 1.1], closed=True,
                         fc="#E4EEF9", ec="#7F9CC0", lw=0.6, zorder=3))
    for gx in np.linspace(cx - s * 0.2, cx + s * 0.2, 4):
        for gy in np.linspace(cy - s * 0.24, cy + s * 0.24, 4):
            tt = np.linspace(0, 1, 10)
            ax.plot(gx + s * 0.13 * tt, gy - s * 0.05 * tt + s * 0.035 * np.sin(np.pi * tt), color=AZ, lw=0.55, zorder=4)
            ax.plot([gx], [gy], marker="o", ms=1.5, color=SLATE, zorder=5)


# ================================================================ 绘制(v2:两条横带、短标签、同题对比作结论)
kp_mean = check_sources()
P = build_panels()
# v2 新出现在图上的事实也要核对
RUNS = S + "/results/training_logs"
for d, (arm, stage) in {"1a_fit_tail24k_s0": ("traj", "1a"), "pB_wsd_tok_s0_tail7k_d20": ("traj_only", "1b"),
                        "pB_s2_cos1ep_s0": ("traj", "1b")}.items():
    m_ = json.load(open(f"{RUNS}/{d}/meta.json"))
    assert (m_["arm"], m_["stage"], m_["n_frames"]) == (arm, stage, 8), d      # 预热 / 只 token / 联合;8 帧
assert f"{FUNNEL['total'] / 1000:.0f}k" == "108k" and f"{FUNNEL['kept'] / 1000:.0f}k" == "42k"
assert min(FUNNEL[k] for k in ("vlm", "nomask", "tracks", "motion")) > 0         # 每个过滤项都真的剔除了 clip
Q_SHORT = "In the world frame, how far did the backpack move between frames 4 and 14?"
for frag in ("In the world frame", "how far", "did the backpack move between frame 4 and frame 14"):
    assert frag in QUESTION, frag                                                # 缩写只删词、不改意思

TITLE, LAB = 7.0, 6.2
BA0, BA1 = 0.02, 2.00                      # A 带(数据管线)
BB0 = 2.06                                 # B 带(模型与训练)
CX = 0.24                                  # 内容左边(左侧是竖排带标题)
CARD_E = "#D5DAE1"
TINT_A, TINT_B, TAB_A, TAB_B = "#F4F7FB", "#F8F6F2", "#DDE6F1", "#ECE6DC"


def card(x0, y0, x1, y1, fc="white", ec=CARD_E, lw=0.5):
    rbox(x0, y0, x1, y1, fc, ec, lw, r=0.035, z=1)


def ctitle(x0, x1, y, s):
    return text((x0 + x1) / 2, y, s, TITLE, SLATE, ha="center", weight="bold")


def csub(x0, x1, y, s, color=SUB):
    return text((x0 + x1) / 2, y, s, LAB, color, ha="center")


# ---- 两条带的底色与竖排标题
rbox(0.02, BA0, W - 0.02, BA1, TINT_A, "none", 0, r=0.05, z=0.2)
rbox(0.02, BB0, W - 0.02, H - 0.02, TINT_B, "none", 0, r=0.05, z=0.2)
rbox(0.02, BA0, 0.17, BA1, TAB_A, "none", 0, r=0.05, z=0.3)
rbox(0.02, BB0, 0.17, H - 0.02, TAB_B, "none", 0, r=0.05, z=0.3)
text(0.095, (BA0 + BA1) / 2, "Data pipeline", 8, SLATE, ha="center", weight="bold", rotation=90)
text(0.095, (BB0 + H - 0.02) / 2, "Model & training", 8, SLATE, ha="center", weight="bold", rotation=90)

# ================================================================ A 带:数据管线
XW = [0.70, 0.84, 0.80, 1.70, 0.80]
GAPX = (W - 0.06 - CX - sum(XW)) / 4
xs_ = [CX]
for w_ in XW[:-1]:
    xs_.append(xs_[-1] + w_ + GAPX)
(S1, S2, S3, S45, S6) = [(x, x + w_) for x, w_ in zip(xs_, XW)]
TOP = 0.20
C1 = (TOP, 1.14)                            # 前三张卡的上下边
MIDY = (C1[0] + C1[1]) / 2 + 0.02

# 1 Video clips
card(S1[0], C1[0], S1[1], C1[1])
ctitle(*S1, C1[0] + 0.10, "Video clips")
TS1, OFF = 0.44, 0.05
x_ = (S1[0] + S1[1]) / 2 - (TS1 + 2 * OFF) / 2
for k, a in enumerate(P["frames"]):
    xx, yy = x_ + k * OFF, C1[0] + 0.21 + k * OFF
    ax.add_patch(Rectangle((xx - 0.008, yy - 0.008), TS1 + 0.016, TS1 + 0.016, fc="white", ec="none", zorder=3 + k))
    ax.imshow(a, extent=(xx, xx + TS1, yy + TS1, yy), interpolation="none", zorder=3.1 + k)
    ax.add_patch(Rectangle((xx, yy), TS1, TS1, fill=False, ec="#9AA0A6", lw=0.35, zorder=3.2 + k))
csub(*S1, C1[1] - 0.09, "mono / stereo")

# 2 Object inventory
card(S2[0], C1[0], S2[1], C1[1])
ctitle(*S2, C1[0] + 0.10, "Object inventory")
eye_badge(S2[0] + 0.12, C1[0] + 0.40, 0.06)
for k, s_ in enumerate(("person ×12", "bicycle", "storefront")):
    yy = C1[0] + 0.30 + k * 0.125
    ti_ = text(S2[0] + 0.23, yy, s_, LAB, TXT)
    box2d(S2[1] - 0.075, yy, 0.06)
    assert S2[0] + 0.23 + width_in(ti_) + 0.04 <= S2[1] - 0.105, ("清单项贴图标", s_)
csub(*S2, C1[1] - 0.09, "Qwen3-VL-Plus")

# 3 Instance masks
card(S3[0], C1[0], S3[1], C1[1])
ctitle(*S3, C1[0] + 0.10, "Instance masks")
TS3 = 0.58
img(P["mask16"], (S3[0] + S3[1]) / 2 - TS3 / 2, C1[0] + 0.19, TS3, TS3)
csub(*S3, C1[1] - 0.09, "SAM 3")

# 4 3D tracking(上)∥ 5 Object geometry(下)
C4 = (TOP, 1.10)
C5 = (1.17, 1.94)
card(S45[0], C4[0], S45[1], C4[1], ec="#9DBFE6", lw=0.6)
ctitle(*S45, C4[0] + 0.10, "3D tracking")
TK = 0.34
xa_ = S45[0] + 0.30
xb_ = S45[1] - 0.30 - TK
t4w_schematic(xa_, C4[0] + 0.18, TK)
ax.add_patch(Rectangle((xa_, C4[0] + 0.18), TK, TK, fill=False, ec="#9AA0A6", lw=0.35, zorder=3))
img(P["tracks16"], xb_, C4[0] + 0.18, TK, TK)
ax.add_patch(Circle(((xa_ + TK + xb_) / 2, C4[0] + 0.18 + TK / 2), 0.055, fc="white", ec=AZ, lw=0.7, zorder=5))
text((xa_ + TK + xb_) / 2, C4[0] + 0.18 + TK / 2, "or", 6.0, AZ, ha="center", weight="bold")
text(xa_ + TK / 2, C4[0] + 0.60, "Track4World", LAB, TXT, ha="center", weight="bold")
text(xb_ + TK / 2, C4[0] + 0.60, "GT 3D tracks", LAB, TXT, ha="center", weight="bold")
csub(*S45, C4[0] + 0.72, "assign tracks to masks")
csub(*S45, C4[0] + 0.81, "visibility + camera poses")

card(S45[0], C5[0], S45[1], C5[1], ec="#A9D3A9", lw=0.6)
ctitle(*S45, C5[0] + 0.10, "Object geometry")


def pill(x, y, s, fc):
    w_ = tw(s, 6.0, "bold") + 0.08
    rbox(x, y - 0.045, x + w_, y + 0.045, fc, "none", 0, r=0.045, z=4)
    text(x + 0.04, y, s, 6.0, "white", weight="bold")
    return x + w_


g3, colr, dots = P["mesh_gray"], P["mesh_parts"], P["part_dots"]
THH = 0.34
THW = THH * g3.shape[1] / g3.shape[0]
yN = C5[0] + 0.24
pill(S45[0] + 0.06, yN, "non-rigid", "#2E7D32")
tx0 = S45[0] + 0.12
for j, (a_, lab) in enumerate(((g3, "mesh"), (colr, "parts"), (g3, "keypoints"))):
    xx = tx0 + j * 0.32
    img(a_, xx, yN + 0.07, THW, THH, edge=False)
    if lab == "keypoints":
        for (u, v, cc) in dots:
            ax.plot([xx + u / g3.shape[1] * THW], [yN + 0.07 + v / g3.shape[0] * THH], marker="o", ms=1.8,
                    mfc=np.array(cc) / 255, mec="white", mew=0.3, zorder=6)
    text(xx + THW / 2, yN + 0.07 + THH + 0.065, lab, 6.0, TXT, ha="center")
    if j < 2:
        arrow((xx + THW + 0.045, yN + 0.07 + THH / 2), (xx + 0.28, yN + 0.07 + THH / 2), color="#6B7C6B", lw=0.6, ms=4)
xr = S45[0] + 1.02
yR = C5[0] + 0.28
pill(xr, yR, "rigid", "#558B2F")
tr_ = text(xr, yR + 0.11, "4 points + SE(3)", LAB, TXT)
pill(xr, yR + 0.27, "static", "#7C8B7C")
text(xr, yR + 0.38, "≤ 4 points", LAB, TXT)
assert xr + width_in(tr_) <= S45[1] - 0.04 and yR + 0.38 + 0.05 <= C5[1] - 0.04, "几何卡片贴边"

# 6 Sparse 4D object tracks(输出)
C6 = (0.46, 1.64)
card(S6[0], C6[0], S6[1], C6[1], ec="#B9A3D6", lw=0.7)
ctitle(*S6, C6[0] + 0.10, "Sparse 4D")
ctitle(*S6, C6[0] + 0.20, "object tracks")
TS6 = 0.66
img(P["traj16"], (S6[0] + S6[1]) / 2 - TS6 / 2, C6[0] + 0.28, TS6, TS6)
csub(*S6, C6[1] - 0.09, "~45 keypoints / clip")

# 过滤(压缩成两行)
FB0 = (1.26, 1.94)
card(S1[0], FB0[0], S3[1], FB0[1])
fx, fy = S1[0] + 0.06, FB0[0] + 0.16
ax.add_patch(Polygon([(fx, fy), (fx + 0.20, fy), (fx + 0.13, fy + 0.14), (fx + 0.13, fy + 0.26), (fx + 0.07, fy + 0.26),
                      (fx + 0.07, fy + 0.14)], closed=True, fc="#DDE3EA", ec=SLATE, lw=0.5, zorder=3))
tx = fx + 0.26
text(tx, FB0[0] + 0.13, "Stereo4D GT-track branch", 6.5, SLATE, weight="bold")
text(tx, FB0[0] + 0.32, "108k clips → 42k kept", 8.0, TXT, weight="bold")
FL_ = ["filters: moving object · valid mask", "≥ 8 tracks · motion check"]
assert all(tx + tw(ln, 6.0) <= S3[1] - 0.05 for ln in FL_), FL_
for i, ln in enumerate(FL_):
    text(tx, FB0[0] + 0.48 + i * 0.09, ln, 6.0, SUB)

# A 带连线:1 → 2 → 3 → 分叉 → 4 ∥ 5 → 汇合 → 6
G = "#7F7F7F"
arrow((S1[1] + 0.005, MIDY), (S2[0] - 0.005, MIDY), color=G, lw=0.8, ms=6)
arrow((S2[1] + 0.005, MIDY), (S3[0] - 0.005, MIDY), color=G, lw=0.8, ms=6)
FX = (S3[1] + S45[0]) / 2
Y4 = (C4[0] + C4[1]) / 2
Y5 = (C5[0] + C5[1]) / 2
ax.plot([S3[1], FX, FX], [MIDY, MIDY, Y5], color=G, lw=0.8, zorder=5)
arrow((FX, Y4), (S45[0] - 0.005, Y4), color=G, lw=0.8, ms=6)
ax.plot([FX, FX], [Y4, MIDY], color=G, lw=0.8, zorder=5)
arrow((FX, Y5), (S45[0] - 0.005, Y5), color=G, lw=0.8, ms=6)
MX = (S45[1] + S6[0]) / 2
Y6 = (C6[0] + C6[1]) / 2
ax.plot([S45[1], MX, MX, S45[1]], [Y4, Y4, Y5, Y5], color=G, lw=0.8, zorder=5)
arrow((MX, Y6), (S6[0] - 0.005, Y6), color=G, lw=0.8, ms=6)

# ================================================================ B 带:模型与训练
LANE_T, LANE_F, LANE_Q = BB0 + 0.30, BB0 + 0.72, BB0 + 1.04
TKX = (CX, CX + 0.78)                       # Trajectory tokenization 方块
EBX = (1.40, 2.10)                          # projector / vision encoder 方块
MBX = (2.22, 2.42)                          # 模型竖条
MBY = (BB0 + 0.16, BB0 + 1.18)
CPX = (2.60, W - 0.06)                      # 同题对比
# 轨迹通道
rbox(TKX[0], LANE_T - 0.17, TKX[1], LANE_T + 0.17, "#F3EEF9", PURP, 0.8, r=0.035, z=3)
text((TKX[0] + TKX[1]) / 2, LANE_T - 0.075, "Trajectory", 6.6, TXT, ha="center", weight="bold")
text((TKX[0] + TKX[1]) / 2, LANE_T + 0.025, "tokenization", 6.6, TXT, ha="center", weight="bold")
text((TKX[0] + TKX[1]) / 2, LANE_T + 0.115, "(Fig. 3)", 6.0, SUB, ha="center")
sq0 = TKX[1] + 0.09
NSQ = 4
for i in range(NSQ):
    ax.add_patch(Rectangle((sq0 + i * 0.058, LANE_T - 0.023), 0.046, 0.046, fc=PURP, ec="none", zorder=4))
arrow((TKX[1] + 0.01, LANE_T), (sq0 - 0.015, LANE_T), color=PURP, lw=0.8, ms=5)
arrow((sq0 + NSQ * 0.058 + 0.005, LANE_T), (EBX[0] - 0.01, LANE_T), color=PURP, lw=0.8, ms=5)
rbox(EBX[0], LANE_T - 0.10, EBX[1], LANE_T + 0.10, CORAL, FLAME, 0.6, r=0.03, z=3)
text((EBX[0] + EBX[1]) / 2 + 0.05, LANE_T, "projector", LAB, TXT, ha="center")
flame(EBX[0] + 0.08, LANE_T, 0.075)
arrow((EBX[1] + 0.005, LANE_T), (MBX[0] - 0.005, LANE_T), color=PURP, lw=0.9, ms=5)
# 帧通道
frame_stack(CX + 0.01, LANE_F + 0.02, 0.09)
text(CX + 0.17, LANE_F, "8 frames", LAB, TXT)
arrow((CX + 0.17 + tw("8 frames", LAB) + 0.04, LANE_F), (EBX[0] - 0.01, LANE_F), color=G, lw=0.7, ms=5)
rbox(EBX[0], LANE_F - 0.10, EBX[1], LANE_F + 0.10, ICE, "#5B8DB8", 0.6, r=0.03, z=3)
text((EBX[0] + EBX[1]) / 2 + 0.05, LANE_F, "vision encoder", 6.0, TXT, ha="center")
snowflake(EBX[0] + 0.065, LANE_F, 0.028)
arrow((EBX[1] + 0.005, LANE_F), (MBX[0] - 0.005, LANE_F), color=G, lw=0.7, ms=5)
# 问题通道
for i in range(3):
    ax.plot([CX + 0.02, CX + (0.11 if i < 2 else 0.07)], [LANE_Q - 0.03 + i * 0.03] * 2, color=SUB, lw=0.7, zorder=5)
text(CX + 0.17, LANE_Q, "question", LAB, TXT)
arrow((CX + 0.17 + tw("question", LAB) + 0.04, LANE_Q), (MBX[0] - 0.005, LANE_Q), color=G, lw=0.7, ms=5)
# 模型竖条 + LoRA 小标签
rbox(MBX[0], MBY[0], MBX[1], MBY[1], SLATE, "none", 0, r=0.03, z=3)
text((MBX[0] + MBX[1]) / 2, (MBY[0] + MBY[1]) / 2 + 0.05, "Qwen3-VL-8B", TITLE, "white", ha="center", weight="bold", rotation=90)
rbox(MBX[0] - 0.05, MBY[0] - 0.055, MBX[1] + 0.05, MBY[0] + 0.05, CORAL, FLAME, 0.5, r=0.02, z=4)
flame(MBX[0] - 0.015, MBY[0] - 0.003, 0.045, z=10)
text((MBX[0] + MBX[1]) / 2 + 0.025, MBY[0] - 0.003, "LoRA", 6.0, TXT, ha="center", weight="bold")

# 同题对比(全图结论)
YQ0 = BB0 + 0.14
q_lines = wrap(Q_SHORT, CPX[1] - CPX[0] - 0.14, 6.5)
assert len(q_lines) <= 2, q_lines
qh = 0.07 + len(q_lines) * 0.105 + 0.11
rbox(CPX[0], YQ0, CPX[1], YQ0 + qh, "#F3F4F6", "#C9CDD3", 0.5, r=0.035, z=2)
for i, ln in enumerate(q_lines):
    text(CPX[0] + 0.07, YQ0 + 0.085 + i * 0.105, ln, 6.5, TXT)
text(CPX[1] - 0.07, YQ0 + 0.085 + len(q_lines) * 0.105, "ground truth 2.09 m", 6.5, GRN, ha="right", weight="bold")
YP0 = YQ0 + qh + 0.09
PBH, PBG = 0.34, 0.07                       # 两个预测框上下叠放,各占满整宽
PB1 = (YP0, YP0 + PBH)
PB2 = (YP0 + PBH + PBG, YP0 + 2 * PBH + PBG)
YP1 = PB2[1]
arrow((MBX[1] + 0.005, (YP0 + YP1) / 2), (CPX[0] - 0.01, (YP0 + YP1) / 2), color=SLATE, lw=0.9, ms=6)
rbox(CPX[0], PB1[0], CPX[1], PB1[1], "#EEEEEE", "#BDBDBD", 0.5, r=0.04, z=2)
rbox(CPX[0], PB2[0], CPX[1], PB2[1], AZF, AZ, 0.8, r=0.04, z=2)
for (y0b, y1b), lab, val, col, ok in ((PB1, "Frames only", "1.1 m", RED, False), (PB2, "Frames + trajectory tokens", "2.2 m", GRN, True)):
    yc = (y0b + y1b) / 2
    frame_stack(CPX[0] + 0.08, yc + 0.02, 0.08)
    tx_ = CPX[0] + 0.23
    if ok:
        for i, cc in enumerate((AZ, ORANGE, GREENO)):
            ax.add_patch(Rectangle((CPX[0] + 0.21 + i * 0.034, yc - 0.013), 0.027, 0.027, fc=cc, ec="none", zorder=7))
        tx_ = CPX[0] + 0.34
    text(tx_, yc, lab, 7.0, TXT, weight="bold")
    vt = text(CPX[1] - 0.30, yc, val, 11, col, ha="right", weight="bold")
    if ok:
        check_mark(CPX[1] - 0.16, yc, 1.25)
    else:
        red_x(CPX[1] - 0.16, yc, 0.034, 1.6)
    assert tx_ + tw(lab, 7.0, "bold") + 0.1 < CPX[1] - 0.30 - width_in(vt), "对比框文字相撞"

# 监督 + 三段训练(紧凑一条)
YS1 = LANE_Q + 0.25
STEPS = [("Projector", "warm-up"), ("Token-only", "fine-tuning"), ("Joint", "fine-tuning")]
sw = (MBX[1] - CX - 2 * 0.06) / 3
sx = [CX + i * (sw + 0.06) for i in range(3)]
SH = 0.24
for i, (l1, l2) in enumerate(STEPS):
    x0, x1 = sx[i], sx[i] + sw
    pts = [(x0, YS1), (x1 - 0.05, YS1), (x1, YS1 + SH / 2), (x1 - 0.05, YS1 + SH), (x0, YS1 + SH)] + (
        [(x0 + 0.05, YS1 + SH / 2)] if i else [])
    ax.add_patch(Polygon(pts, closed=True, fc="#E9EDF2", ec="#8792A2", lw=0.5, zorder=3))
    text((x0 + x1) / 2 + (0.02 if i else 0), YS1 + SH / 2 - 0.045, l1, 6.2, TXT, ha="center", weight="bold")
    text((x0 + x1) / 2 + (0.02 if i else 0), YS1 + SH / 2 + 0.05, l2, 6.2, TXT, ha="center")
YSUP = YS1 - 0.075
text((sx[0] + sx[0] + sw) / 2, YSUP, "motion captions", 6.0, SUB, ha="center", style="italic")
ax.plot([sx[1] + 0.05, sx[1] + 0.05, sx[2] + sw - 0.06, sx[2] + sw - 0.06], [YSUP + 0.035, YSUP + 0.025, YSUP + 0.025, YSUP + 0.035],
        color="#9AA0A6", lw=0.5, zorder=4)
text((sx[1] + sx[2] + sw) / 2, YSUP - 0.005, "trajectory QA", 6.0, SUB, ha="center", style="italic",
     bbox=dict(boxstyle="square,pad=0.08", fc=TINT_B, ec="none"))
BOT_B = max(YS1 + SH, YP1) + 0.06

# 回程粗箭头:A 带输出 → 两带分界 → B 带轨迹通道起点
XR6 = (S6[0] + S6[1]) / 2
YBND = (BA1 + BB0) / 2
XIN = (TKX[0] + TKX[1]) / 2
ax.plot([XR6, XR6, XIN], [C6[1] + 0.005, YBND, YBND], color=PURP, lw=1.6, zorder=6, solid_joinstyle="round")
arrow((XIN, YBND), (XIN, LANE_T - 0.175), color=PURP, lw=1.6, ms=7)

# ================================================================ 高度、重叠 / 越界检查、输出
H_NEED = BOT_B + 0.02
fig.canvas.draw()
bbs, oob = [], []
for t in fig.findobj(Text):
    if not t.get_visible() or not t.get_text().strip():
        continue
    e = t.get_window_extent(renderer=rend)
    if e.width < 1:
        continue
    x0, x1 = e.x0 / fig.dpi, e.x1 / fig.dpi
    y0_, y1_ = H - e.y1 / fig.dpi, H - e.y0 / fig.dpi
    if not (x0 >= -0.001 and x1 <= W + 0.001 and y0_ >= -0.001 and y1_ <= H + 0.001):
        oob.append((t.get_text(), round(x0, 2), round(x1, 2), round(y0_, 2), round(y1_, 2)))
    bbs.append((x0 + 0.004, y0_ + 0.012, x1 - 0.004, y1_ - 0.012, t.get_text()))
bad = [(p_[4], q_[4]) for i, p_ in enumerate(bbs) for q_ in bbs[i + 1:]
       if p_[0] < q_[2] and q_[0] < p_[2] and p_[1] < q_[3] and q_[1] < p_[3]]
N_TEXT = len(bbs)
if oob or bad or abs(H_NEED - H) >= 0.03:
    fig.savefig(os.path.join(os.environ.get("TMPDIR", "/tmp"), "fig2_debug.png"), dpi=200)
    raise AssertionError(dict(H_NEED=round(H_NEED, 3), H=H, out_of_canvas=oob, overlaps=bad[:12]))
fig.savefig(os.path.join(OUT, "fig2_pipeline.pdf"))
fig.savefig(os.path.join(OUT, "fig2_pipeline.png"), dpi=300)
print("wrote", os.path.join(OUT, "fig2_pipeline.pdf"), "and .png;", f"H={H} in, text items={N_TEXT}, keypoints/clip={kp_mean:.2f}")
