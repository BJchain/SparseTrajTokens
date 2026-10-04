"""图 1(Teaser)渲染 v2 —— 按 fig1_blueprint.md(09-25 用户意见版)。输出 fig1_teaser.pdf(字体嵌入)+ fig1_teaser.png(300 dpi)。
所有真实素材只读;帧图原样嵌入;题目 / 回答 / token 串结构在运行时从源文件核对(不一致就报错退出)。
token 条按用户意见只画示意数目 + "(…)":核对画出的标签是真实序列的有序子序列、画出的方块不多于真实数目。
用法:<stereo4d-env python> render_fig1.py
"""
import colorsys
import json
import os
import re
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402
from matplotlib.text import Text  # noqa: E402
from PIL import Image  # noqa: E402

R = str(__import__("pathlib").Path(__file__).resolve().parents[2])          # repo root(本文件在 paper/figure_scripts/)
S = R
CLIP = "pwBorgoDJh0_141875209"
OUT = R + "/paper/figure_assets/figures"          # 合成图输出目录
FIGDIR = R + "/paper/figure_assets"
TILES = [FIGDIR + f"/{CLIP}/fig1_teaser_key_objects_f{t:02d}.png" for t in (0, 8, 12, 16, 24)]
TOK = S + f"/data/tokens_v4td/scene_{CLIP}.pkl"
QID = f"scene_{CLIP}_speed_grid_314351d3"
VAL = S + "/data/trajqa_b/final/val.jsonl"
OURS = S + "/results/eval/selection_val/s2wsd_pB_s2_cos1ep_s0_final3507_traj.json.items.jsonl"
RGB = S + "/results/eval/selection_val/rgbsw_pB_rgbsw_9002_s0_final_rgb.json.items.jsonl"
G1PY = os.environ.get("G1_PY", sys.executable)      # g1/data.py 需要 torch

BLUE, ORANGE, GREEN = "#1F76D6", "#FF7F0E", "#2CA02C"
GRAY, DARK, TXT, SUB, RED = "#969696", "#4D4D4D", "#1A1A1A", "#5F6368", "#D62728"
FB, FS = 6.2, 6.0

# text.hinting=no_hinting + dpi 300:量出来的字宽 = PDF 实际字宽
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif",
                     "font.sans-serif": ["Liberation Sans", "DejaVu Sans"], "axes.unicode_minus": False,
                     "text.hinting": "no_hinting"})

# token 条(示意):("L", 标签, 颜色) / ("S", 方块数, 颜色) / ("M", "(…)", 颜色) = 该组还有更多 / ("G", "…") = 省略若干物体
STRIP = [
    [("L", "[scene]", DARK), ("S", 2, DARK), ("M", "(…)", DARK), ("L", "[the person]", ORANGE), ("S", 1, ORANGE),
     ("L", "[part 1]", ORANGE), ("S", 2, ORANGE), ("M", "(…)", ORANGE), ("L", "[part 2]", ORANGE), ("S", 2, ORANGE),
     ("M", "(…)", ORANGE), ("G", "…", SUB)],
    [("L", "[the shopping bag]", GREEN), ("S", 1, GREEN), ("G", "…", SUB),
     ("L", "[the bicycle]", BLUE), ("S", 2, BLUE), ("M", "(…)", BLUE), ("G", "…", SUB),
     ("L", "[the camera]", DARK), ("S", 2, DARK), ("M", "(…)", DARK)],
]


# ---------------------------------------------------------------- 源数据核对
def real_block_items():
    code = ("import sys, pickle, re; sys.path[:0] = [%r, %r]\n"
            "from g1.traj_serialize import serialize_scene_labeled, group_spans\n"
            "from g1.data import build_traj_block, TRAJ_TOKEN\n"
            "recs, labels = serialize_scene_labeled(pickle.load(open(%r, 'rb')))\n"
            "txt = build_traj_block(group_spans(recs), labels).split(chr(10))[1]\n"
            "print(re.sub('(?:%%s)+' %% re.escape(TRAJ_TOKEN), lambda m: ' <%%d> ' %% (len(m.group(0)) // len(TRAJ_TOKEN)), txt))"
            % (S + "/src/tokenization", S + "/src/training", TOK))
    out = subprocess.run([G1PY, "-c", code], capture_output=True, text=True, check=True).stdout
    items = []
    for m in re.finditer(r"<(\d+)>|(\[[^\]]+\])", out):
        items.append(("S", int(m.group(1))) if m.group(1) else ("L", m.group(2)))
    return items


def check_strip(real, strip):
    """画出的标签必须按顺序出现在真实序列里;标签后画出的方块数 ≤ 真实数目,少画时必须紧跟 "(…)"。"""
    shown = [it for line in strip for it in line]
    i = 0
    for j, it in enumerate(shown):
        if it[0] != "L":
            continue
        while i < len(real) and real[i] != ("L", it[1]):
            i += 1
        assert i < len(real), ("标签不在真实序列里(或顺序不对)", it[1])
        i += 1
        nxt = shown[j + 1] if j + 1 < len(shown) else None
        if nxt and nxt[0] == "S":
            assert real[i][0] == "S" and nxt[1] <= real[i][1], ("方块多于真实数目", it[1], nxt[1], real[i])
            if nxt[1] < real[i][1]:
                assert j + 2 < len(shown) and shown[j + 2][0] == "M", ("少画了方块却没有 (…)", it[1])
            else:
                assert not (j + 2 < len(shown) and shown[j + 2][0] == "M"), ("数目已全画却写了 (…)", it[1])
    return True


def check_sources():
    q = next(json.loads(ln) for ln in open(VAL) if QID in ln)
    assert q["options"] == {"A": "1.7", "B": "4.3", "C": "3.2", "D": "2.3"} and q["answer"] == "B"
    ours = next(json.loads(ln) for ln in open(OURS) if QID in ln)
    rgb = next(json.loads(ln) for ln in open(RGB) if QID in ln)
    for k in ("position_frame_12_m: (2.7, 2.2, 6.4)", "position_frame_16_m: (3.8, 2.8, 9.3)",
              "displacement_m: 3.2", "time_s: 0.80", "speed_mps: 3.95", "<answer>B</answer>"):
        assert k in ours["gen"], k
    for k in ("displacement_m: 1.0", "time_s: 0.80", "speed_mps: 1.29", "<answer>A</answer>"):
        assert k in rgb["gen"], k
    assert ours["pred"] == "B" and rgb["pred"] == "A" and "speed_mps: 4.13" in ours["evidence_gt"]
    real = real_block_items()
    assert sum(n for k, n in real if k == "S") == 457
    check_strip(real, STRIP)
    for p in TILES:
        a = np.asarray(Image.open(p))
        assert a.shape[:2] == (512, 512), (p, a.shape)
    return q["question"]


# ---------------------------------------------------------------- 画布与工具
W, H = 5.5, 2.35
fig = plt.figure(figsize=(W, H), dpi=300)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
rend = fig.canvas.get_renderer()


def text(x, y, s, size=FB, color=TXT, ha="left", va="center", weight="normal", **kw):
    return ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, fontweight=weight, zorder=kw.pop("zorder", 8), **kw)


def width_in(t):
    return t.get_window_extent(renderer=rend).width / fig.dpi


def tw(s, size=FB, weight="normal"):
    t = ax.text(0, -5, s, fontsize=size, fontweight=weight)
    w = width_in(t); t.remove()
    return w


def wrap(s, width, size):
    words, lines, cur = s.split(), [], ""
    for w_ in words:
        trial = (cur + " " + w_).strip()
        if tw(trial, size) <= width or not cur:
            cur = trial
        else:
            lines.append(cur); cur = w_
    return lines + [cur]


def rainbow(x):
    return colorsys.hsv_to_rgb(0.78 * (1.0 - float(x)), 1.0, 1.0)


def rbox(x0, y0, x1, y1, fc, ec, lw=0.5, r=0.04, z=1):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, zorder=z))


def frame_icon(x, y):
    for d in (0.03, 0.015, 0.0):
        ax.add_patch(Rectangle((x + d, y - 0.035 - d), 0.085, 0.06, fc="white", ec="#7F7F7F", lw=0.4, zorder=6))


def token_icon(x, y):
    for i, c in enumerate((BLUE, ORANGE, GREEN)):
        ax.add_patch(Rectangle((x + i * 0.035, y - 0.015), 0.028, 0.028, fc=c, ec="none", zorder=6))


# ================================================================ 绘制
question = check_sources()

# ---- Zone 1:五帧通栏(约占画布一半)
X0, GAP = 0.02, 0.05
TS = (W - 2 * X0 - 4 * GAP) / 5
TY = 0.02
tile_x = [X0 + k * (TS + GAP) for k in range(5)]
for k, (p, t) in enumerate(zip(TILES, (0, 8, 12, 16, 24))):
    im = np.asarray(Image.open(p).convert("RGB"))
    ax.imshow(im, extent=(tile_x[k], tile_x[k] + TS, TY + TS, TY), interpolation="none", zorder=1)
    ax.add_patch(Rectangle((tile_x[k], TY), TS, TS, fill=False, ec="#BDBDBD", lw=0.4, zorder=2))
    text(tile_x[k] + TS / 2, TY + TS + 0.065, f"t = {t}", FS, SUB, ha="center")
YB = TY + TS + 0.17                                   # 下半段顶

# ---- Zone 3 / 4 的几何(先定,左列据此对齐)
MX0, MX1 = 3.21, 3.40
BX0, BX1 = 3.48, W - 0.02
# ---- Zone 2a:问题框
QX0, QX1 = X0, MX0 - 0.09
qlines = wrap(question, QX1 - QX0 - 0.12, FB)
assert len(qlines) <= 2, qlines
QH = 0.05 + 0.105 * (len(qlines) + 1) + 0.025
rbox(QX0, YB, QX1, YB + QH, "#F3F4F6", "#C9CDD3")
for i, ln in enumerate(qlines):
    text(QX0 + 0.06, YB + 0.075 + 0.105 * i, ln, FB)
yo = YB + 0.075 + 0.105 * len(qlines)
text(QX0 + 0.06, yo, "A. 1.7   B. 4.3   C. 3.2   D. 2.3", FB, weight="bold")
text(QX1 - 0.06, yo, "ground truth: 4.13 m/s", FS, SUB, ha="right")

# ---- Zone 2b:轨迹 token 条(示意数目 + "(…)")
YT = YB + QH + 0.12
text(QX0, YT, "trajectory tokens", 6.4, SUB, weight="bold")
SQ, PIT = 0.05, 0.062
LY0, LP = YT + 0.14, 0.13
strip_end = []
for li, items in enumerate(STRIP):
    x, yc = QX0, LY0 + li * LP
    for kind, a, col in items:
        if kind == "S":
            for _ in range(a):
                ax.add_patch(Rectangle((x, yc - SQ / 2), SQ, SQ, fc=col, ec="none", zorder=3)); x += PIT
            x += 0.01
        else:
            wt = "bold" if kind == "L" and a.startswith("[the") else "normal"
            t_ = text(x, yc, a, FB, col if kind != "G" else SUB, weight=wt)
            x += width_in(t_) + (0.07 if kind == "M" else 0.03)          # "(…)" 之后留出组间距
    strip_end.append(x)
    assert x <= QX1 + 0.01, ("token 条溢出", li, x)
# ---- 彩虹色条(轨迹颜色 = 时间)
YC = LY0 + len(STRIP) * LP + 0.06
cb_x0 = QX0 + tw("start", FS) + 0.05
grad = np.array([[rainbow(v) for v in np.linspace(0, 1, 256)]])
ax.imshow(grad, extent=(cb_x0, cb_x0 + 1.0, YC + 0.022, YC - 0.022), interpolation="bilinear", zorder=1)
text(cb_x0 - 0.04, YC, "start", FS, SUB, ha="right")
text(cb_x0 + 1.04, YC, "end", FS, SUB)
LEFT_BOT = YC + 0.05

# ---- Zone 4:两个回答
BH1 = 0.50
b1 = (YB, YB + BH1)
b2 = (YB + BH1 + 0.07, YB + 2 * BH1 + 0.07)
MY0, MY1 = YB, b2[1]
rbox(BX0, b1[0], BX1, b1[1], "#EEEEEE", "#BDBDBD")
hy = b1[0] + 0.10
frame_icon(BX0 + 0.06, hy + 0.02)
text(BX0 + 0.21, hy, "Qwen3-VL-8B", 6.8, weight="bold")
text(BX0 + 0.06, hy + 0.165, "moved 1.0 m in 0.80 s", FB)
c1 = text(BX0 + 0.06, hy + 0.28, "→ 1.29 m/s → A", FB)
r_ = 0.03
cx = BX0 + 0.06 + width_in(c1) + 0.07 + r_
ax.plot([cx - r_, cx + r_], [hy + 0.28 - r_, hy + 0.28 + r_], color=RED, lw=1.3, solid_capstyle="round", zorder=6)
ax.plot([cx - r_, cx + r_], [hy + 0.28 + r_, hy + 0.28 - r_], color=RED, lw=1.3, solid_capstyle="round", zorder=6)

rbox(BX0, b2[0], BX1, b2[1], "#E8F1FB", BLUE, lw=0.7)
hy = b2[0] + 0.10
frame_icon(BX0 + 0.06, hy + 0.02)
token_icon(BX0 + 0.19, hy)
tt = text(BX0 + 0.32, hy, "Frames + trajectory tokens", 6.8, weight="bold")
assert BX0 + 0.32 + width_in(tt) <= BX1 - 0.04, "标题超出气泡"
text(BX0 + 0.06, hy + 0.165, "(2.7, 2.2, 6.4) m → (3.8, 2.8, 9.3) m", FB)
c2 = text(BX0 + 0.06, hy + 0.28, "3.2 m in 0.80 s → 3.95 m/s → B", FB)
cx = BX0 + 0.06 + width_in(c2) + 0.07 + 0.045
assert cx + 0.05 <= BX1 - 0.04
ax.plot([cx - 0.045, cx - 0.012, cx + 0.05], [hy + 0.28 - 0.005, hy + 0.28 + 0.03, hy + 0.28 - 0.045], color=GREEN, lw=1.5,
        solid_capstyle="round", solid_joinstyle="round", zorder=6)

# ---- Zone 3:MLLM 竖条
ax.add_patch(FancyBboxPatch((MX0, MY0), MX1 - MX0, MY1 - MY0, boxstyle="round,pad=0,rounding_size=0.03",
                            fc="#3E4C5E", ec="none", zorder=2))
text((MX0 + MX1) / 2, (MY0 + MY1) / 2, "MLLM", 7.5, "white", ha="center", weight="bold", rotation=90)

# ---- 连线
GXA = (tile_x[2] + TS + tile_x[3]) / 2                 # 第 3、4 帧之间的缝
assert MX0 < GXA < MX1, (MX0, GXA, MX1)
ax.add_patch(FancyArrowPatch((GXA, TY + TS + 0.01), (GXA, MY0 - 0.005), arrowstyle="-|>", mutation_scale=6, lw=0.7,
                             color="#7F7F7F", zorder=3, shrinkA=0, shrinkB=0))
ax.add_patch(FancyArrowPatch((QX1 + 0.005, YB + QH / 2), (MX0 - 0.005, YB + QH / 2), arrowstyle="-|>", mutation_scale=6,
                             lw=0.7, color="#7F7F7F", zorder=3, shrinkA=0, shrinkB=0))
ya = (b2[0] + b2[1]) / 2
ax.add_patch(FancyArrowPatch((max(strip_end) + 0.03, LY0 + LP / 2), (MX0 - 0.005, ya), arrowstyle="-|>", mutation_scale=8,
                             lw=1.6, color=BLUE, zorder=3, connectionstyle="arc3,rad=0.15", shrinkA=0, shrinkB=0))

# ================================================================ 高度、重叠 / 越界检查、输出
H_NEED = max(LEFT_BOT, MY1) + 0.04
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
if oob or bad or abs(H_NEED - H) >= 0.03:
    raise AssertionError(dict(H_NEED=round(H_NEED, 3), H=H, out_of_canvas=oob, overlaps=bad[:12]))
fig.savefig(os.path.join(OUT, "fig1_teaser.pdf"))
fig.savefig(os.path.join(OUT, "fig1_teaser.png"), dpi=300)
print("wrote", os.path.join(OUT, "fig1_teaser.pdf"), "and .png;", f"H={H} in, tile={TS:.3f} in, frames band={TY + TS + 0.12:.2f} in")
