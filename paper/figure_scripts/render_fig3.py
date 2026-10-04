"""图 3(Tokenization 详细步骤)渲染 v2 —— 按 fig3_blueprint.md(09-25 按用户意见改版:两行 6 步、少字)。
输出 fig3_tokenization.pdf(字体嵌入)+ fig3_tokenization.png(300 dpi)。
3D 轨迹框、三张小图的散点 / 曲线 / 误差管 / 节点、token 序列与示例 token 都从真实文件读出现场画;
图上每个数字在运行时核对(不一致就报错退出)。v1 渲染脚本备份在会话 scratchpad(v1_render_fig3.py)。
用法:<stereo4d-env python> render_fig3.py
"""
import json
import os
import pickle
import re
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle  # noqa: E402
from matplotlib.text import Text  # noqa: E402
from scipy.interpolate import splev  # noqa: E402

R = str(__import__("pathlib").Path(__file__).resolve().parents[2])          # repo root(本文件在 paper/figure_scripts/)
S = R
M = R + "/data"
CLIP = "pwBorgoDJh0_141875209"
V = "scene_" + CLIP
SD = f"{M}/scenes/{V}"
TOKP = f"{M}/tokens_v4td/{V}.pkl"
OUT = R + "/paper/figure_assets/figures"          # 合成图输出目录
G1PY = os.environ.get("G1_PY", sys.executable)   # g1/data.py 需要 torch
DBG = os.path.join(os.environ.get("TMPDIR", "/tmp"), "fig3_debug.png")
for p in (S, S + "/src/tokenization", S + "/src/training"):
    if p not in sys.path:
        sys.path.insert(0, p)
import SCHEMA  # noqa: E402,F401
import curve_fit_v3 as cf3  # noqa: E402
import tokenizer as tkz  # noqa: E402
from g1 import traj_serialize as ts  # noqa: E402

IID, KPID, OBJ_IDX, REV_IDX = 13, 645, 4, 139     # 示例:person 实例 13 的关键点 645;序列下标 139 = 第 3 帧折返节点
BIKE, STORE = 17, 0
PLOT_DIM = 2                                        # 小图纵轴 = z 分量(折返尖角最清楚)


def check_sources():
    sc = pickle.load(open(TOKP, "rb"))
    fr = pickle.load(open(SD + "/final_repr.pkl", "rb"))
    assert sc.T == 34
    c, s = np.asarray(sc.norm_center, float), float(sc.norm_scale)
    assert [f"{v:.2f}" for v in c] == ["1.46", "2.12", "10.06"] and f"{s:.2f}" == "10.23", (c, s)
    o = sc.objects[OBJ_IDX]
    assert (o.instance_id, o.label, o.kind) == (IID, "person", "nonrigid")
    kt = next(k for k in o.keypoint_tracks if k.keypoint_id == KPID)
    assert kt.part_id == 1
    segs = kt.pl_segments
    runs = [(int(round(sg.u_start * 33)), int(round(sg.u_end * 33))) for sg in segs]
    assert runs == [(0, 7), (9, 12), (14, 19), (21, 21), (23, 23), (25, 27), (29, 33)], runs
    assert len(kt.occlusion_gaps) == 6
    assert np.allclose(segs[0].fnodes, [0, 1.5, 3, 3, 13 / 3, 17 / 3, 7], atol=1e-6), segs[0].fnodes
    assert segs[0].knot_types == ["endpoint", "smooth", "reversal", "reversal", "smooth", "smooth", "endpoint"]
    assert [len(sg.fnodes) for sg in segs] == [7, 2, 4, 1, 1, 2, 4]            # 共 21 个节点
    kts = [k for sg in segs for k in sg.knot_types]
    assert kts.count("reversal") == 2 and "impact" not in kts                    # 只有第 3 帧一个事件
    # "one token per spline coefficient":每段节点数 = 各样条片的系数个数之和(单点段 = 1 个节点)
    for sg in segs:
        ncoef = sum(np.asarray(c_).shape[1] for (_, c_, _, _, _) in sg.tcks)
        assert len(sg.fnodes) == (ncoef if sg.tcks else 1), (sg.fnodes, ncoef)
    degs = [k_ for sg in segs for (_, _, k_, _, _) in sg.tcks]
    assert degs == [2, 3, 1, 3, 1, 3], degs                                      # 第 0–3 帧只有 4 帧 → 二次;2 节点段 = 直线
    inst = next(x for x in fr if x["instance_id"] == IID)
    assert list(inst["keypoint_ids"])[1] == KPID
    traj_n = (np.asarray(inst["trajectories"], float) - c) / s
    vis = np.asarray(inst["visibility"], bool)
    bd = tkz._cloud_diag(traj_n[vis])
    eps = tkz.DEFAULT_ALPHA * bd
    assert abs(eps - 0.0159) < 0.0005, eps
    rec = tkz.reconstruct_keypoint(kt, 34)
    err = np.linalg.norm(rec[vis[1]] - traj_n[1][vis[1]], axis=1)
    assert np.nanmax(err) <= eps, (np.nanmax(err), eps)                          # 误差管确实罩住了数据
    for sg in segs:                                                              # 每段内逐帧可见(缺口只在段与段之间)
        assert vis[1, int(round(sg.u_start * 33)):int(round(sg.u_end * 33)) + 1].all()
    # token 序列与示例 token
    recs = ts.serialize_scene(sc)
    assert len(recs) == 457
    r = recs[REV_IDX]
    assert ts.ROLES[r.role] == "PL_VERT" and ts.KNOTS[r.knot] == "reversal" and (r.inst, r.part, r.kp) == (5, 2, 6)
    assert r.kind == 0 and r.otype == 0                     # 节点 token 的 kind / otype 通道为 PAD(物体头 token 携带)→ 图上画 "–"
    assert ts.KINDS[0] == "PAD" and ts.OTYPES[0] == "PAD"
    assert [f"{v:.2f}" for v in r.vec3] == ["0.61", "0.02", "0.48"] and f"{r.scalar:.2f}" == "0.09" and f"{r.tsec:.2f}" == "0.60"
    assert r.has_vec3 and r.has_scalar and r.has_tsec
    blk = [x for x in recs if x.group == OBJ_IDX and x.part == r.part and x.kp == r.kp]
    roles = [ts.ROLES[x.role] for x in blk]
    assert len(blk) == 28 and roles.count("VIS_SUMMARY") == 1 and roles.count("OCC_GAP") == 6 and roles.count("PL_VERT") == 21, roles
    assert tuple(ts.KNOTS[1:]) == ("smooth", "reversal", "impact", "endpoint")
    # tokenizer / 拟合常量
    assert tkz.DEFAULT_ALPHA == 0.02 and cf3.DEFAULT_M == 12 and tkz.CAMERA_ABS_BUDGET_M == 0.10 and tkz.MOTION_THRESH == 0.05
    assert tkz.RIGID_PARAM == "centroid" and tkz.DEFAULT_RIGID_MODE == "joint" and tkz.DEFAULT_CURVE == "bspline"
    src = open(S + "/src/tokenization/tokenizer.py").read()
    assert "scale = float(rad.max())" in src                                     # 共享中心 + 最大半径 → 单位球
    tok_names = [n for n in ("norm_center", "norm_scale") if hasattr(sc, n)]
    assert tok_names == ["norm_center", "norm_scale"]                            # 中心 / 尺度随 token 一起存 → 米制可还原
    # 投影器:结构 + 参数量(逐层算)
    enc = open(S + "/src/training/g1/traj_encoder.py").read()
    for pat in (r"nn\.Linear\(self\.n_channels \* d, hidden_size\)", r"nn\.Linear\(hidden_size, hidden_size \* 2\)",
                r"nn\.Linear\(hidden_size \* 2, hidden_size\)", r"self\.norm_in = nn\.LayerNorm\(hidden_size\)",
                r"self\.norm_out = nn\.LayerNorm\(hidden_size\)", r"h = h \+ self\.mlp\(h\)", r"torch\.cat\(chans, dim=-1\)"):
        assert re.search(pat, enc), pat
    assert len(re.findall(r"= nn\.Embedding\(", enc)) == 7 and len(re.findall(r"self\.proj_\w+ = nn\.Linear\(1|self\.proj_vec3 = nn\.Linear\(3", enc)) >= 2
    d, h = 456, 4096
    n = ((len(ts.ROLES) + len(ts.KNOTS) + len(ts.KINDS) + len(ts.OTYPES) + ts.N_INST_BUCKET + ts.N_PART_BUCKET + ts.N_KP_BUCKET + 3) * d
         + (3 * d + d) + 2 * (d + d) + (10 * d * h + h) + 2 * (2 * h) + (h * 2 * h + 2 * h) + (2 * h * h + h))
    assert n == 85_930_656, n
    meta = json.load(open(S + "/results/training_logs/pB_s2_cos1ep_s0/meta.json"))
    assert (meta["fusion"], meta["ch_dim"], meta["use_tsec"], meta["n_frames"], meta["max_side"]) == ("concat", 456, True, 8, 384)
    # 真实序列结构(训练代码 build_traj_block 原函数,g1train 子进程)
    code = ("import sys, pickle, re; sys.path[:0] = [%r, %r]\n"
            "from g1.traj_serialize import serialize_scene_labeled, group_spans\n"
            "from g1.data import build_traj_block, TRAJ_TOKEN\n"
            "recs, labels = serialize_scene_labeled(pickle.load(open(%r, 'rb')))\n"
            "txt = build_traj_block(group_spans(recs), labels).split(chr(10))[1]\n"
            "print(re.sub('(?:%%s)+' %% re.escape(TRAJ_TOKEN), lambda m: ' <%%d> ' %% (len(m.group(0)) // len(TRAJ_TOKEN)), txt))"
            % (S + "/src/tokenization", S + "/src/training", TOKP))
    blk_s = " ".join(subprocess.run([G1PY, "-c", code], capture_output=True, text=True, check=True).stdout.split())
    assert blk_s.startswith("[scene] <3> [the person] <1> [part 1] <5> [part 2] <5>"), blk_s[:80]
    assert "[the person] <1> [part 1] <7> [part 2] <28> [part 4]" in blk_s               # 示例人物(实例 13)那一组
    assert blk_s.endswith("[the camera] <13>")
    return dict(sc=sc, fr=fr, kt=kt, traj_n=traj_n, vis=vis, eps=eps, recs=recs, blk=blk)


# ---------------------------------------------------------------- 样式与画布
TXT, SUB, SLATE = "#1A1A1A", "#5F6368", "#3E4C5E"
PURP, PURP_L, AZ, GREY, DARK = "#7B4FA8", "#B7A0D8", "#1F76D6", "#969696", "#4D4D4D"
C_END, C_SMOOTH, C_REV = SLATE, PURP, "#E4572E"
TUBE, GAPC, CELL_D, CELL_C = "#E7DDF3", "#E3E3E3", "#EAF2FB", "#FCE9DD"
CARD_F, CARD_E, ARR = "#F7F8FA", "#CDD3DB", "#6B7079"
FT, FS = 7.0, 6.0                 # 只用两种字号:卡片标题 7 pt 粗体 / 其余 6 pt
LH = 0.092                        # 6 pt 行距(in)

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif",
                     "font.sans-serif": ["Liberation Sans", "DejaVu Sans"], "axes.unicode_minus": False,
                     "text.hinting": "no_hinting"})
W, H = 5.5, 2.70
fig = plt.figure(figsize=(W, H), dpi=300)
ax = fig.add_axes([0, 0, 1, 1], zorder=0)
ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
rend = fig.canvas.get_renderer()
CARDS = []                        # (x0, y0, x1, y1):每段文字必须完整落在某一张卡片里


def text(x, y, s, size=FS, color=TXT, ha="left", va="center", weight="normal", style="normal", **kw):
    return ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, fontweight=weight, fontstyle=style,
                   zorder=kw.pop("zorder", 8), **kw)


def width_in(t):
    return t.get_window_extent(renderer=rend).width / fig.dpi


def tw(s, size=FS, weight="normal", style="normal"):
    t = ax.text(0, -5, s, fontsize=size, fontweight=weight, fontstyle=style)
    w = width_in(t); t.remove()
    return w


def rbox(x0, y0, x1, y1, fc="white", ec=CARD_E, lw=0.5, r=0.03, z=1, ls="-"):
    p = FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw,
                       zorder=z, linestyle=ls)
    ax.add_patch(p)
    return p


def arrow(p0, p1, color=ARR, lw=0.7, ms=6, z=6, ls="-"):
    a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms, lw=lw, color=color, zorder=z,
                        linestyle=ls, shrinkA=0, shrinkB=0)
    ax.add_patch(a)
    return a


def axes_at(x0, y0, w, h):
    """在英寸坐标 (左上角 x0,y0,宽 w,高 h) 放一个数据坐标轴。"""
    a = fig.add_axes([x0 / W, 1 - (y0 + h) / H, w / W, h / H], zorder=2)
    a.patch.set_alpha(0)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        a.spines[sp].set_color("#9AA0A6"); a.spines[sp].set_linewidth(0.5)
    a.tick_params(length=2, width=0.5, colors="#9AA0A6", labelleft=False, labelbottom=False)
    return a


def to_in(a, x, y):
    """数据坐标 (轴 a) → 画布英寸坐标。"""
    X, Y = a.transData.transform((x, y))
    return X / fig.dpi, H - Y / fig.dpi


def card(xr, y0, y1):
    rbox(xr[0], y0, xr[1], y1, CARD_F, CARD_E, 0.6, r=0.04, z=0.5)
    CARDS.append((xr[0], y0, xr[1], y1))


def ctitle(xr, y0, lines):
    for i, s_ in enumerate(lines):
        t = text((xr[0] + xr[1]) / 2, y0 + 0.10 + i * 0.10, s_, FT, SLATE, ha="center", weight="bold")
        assert width_in(t) <= xr[1] - xr[0] - 0.08, ("标题溢出", s_)


def legend_row(x0, y, items, x_max, gap_mk=0.025, gap_item=0.075):
    """一行图例:items = [(画标记的函数 | None, 文字, 颜色)];返回右端 x,并断言不越过 x_max。"""
    x = x0
    for i, (mk, lab, col) in enumerate(items):
        if mk is not None:
            x = mk(x, y) + gap_mk
        t = text(x, y, lab, FS, col)
        x += width_in(t) + (gap_item if i < len(items) - 1 else 0)
    assert x <= x_max, ("图例溢出", [it[1] for it in items], round(x, 3), round(x_max, 3))
    return x


def mk_tri(x, y):
    ax.plot([x + 0.025], [y], marker="v", ms=4.2, color=C_REV, mec="white", mew=0.3, zorder=9)
    return x + 0.05


def mk_diamond(x, y):
    ax.plot([x + 0.025], [y], marker="D", ms=2.8, mfc="white", mec=C_REV, mew=0.6, zorder=9)
    return x + 0.05


def mk_swatch(fc):
    def f(x, y):
        ax.add_patch(Rectangle((x, y - 0.03), 0.06, 0.06, fc=fc, ec="none", zorder=9))
        return x + 0.06
    return f


# ================================================================ 数据
Dd = check_sources()
sc, fr, kt, traj_n, vis, eps, recs, blk = (Dd[k] for k in ("sc", "fr", "kt", "traj_n", "vis", "eps", "recs", "blk"))
C0, S0 = np.asarray(sc.norm_center, float), float(sc.norm_scale)
Pk, Vk = traj_n[1], vis[1]                          # 关键点 645(归一化世界系)
segs = kt.pl_segments
pieces = []                                         # (ff, zz):每个样条片的稠密曲线
for sg in segs:
    for (t_, c_, k_, f0, f1) in sg.tcks:
        uu = np.linspace(0, 1, 90)
        pieces.append((f0 + uu * (f1 - f0), np.asarray(splev(uu, (t_, c_, k_)))[PLOT_DIM]))
nodes = [(float(f), float(p[PLOT_DIM]), k) for sg in segs for f, p, k in zip(sg.fnodes, sg.nodes, sg.knot_types)]
assert len(nodes) == 21
single = [float(sg.fnodes[0]) for sg in segs if len(sg.fnodes) == 1]
two = [sg for sg in segs if len(sg.fnodes) == 2]
assert single == [21.0, 23.0] and [(t.fnodes[0], t.fnodes[1]) for t in two] == [(9.0, 12.0), (25.0, 27.0)]
corner = [(f, z) for f, z, k in nodes if k == "reversal"]
assert len(corner) == 2 and all(abs(f - 3) < 1e-9 for f, _ in corner)
ZREV = float(np.mean([z for _, z in corner]))
gaps = [(a_ * 33, b_ * 33) for a_, b_ in kt.occlusion_gaps]
fV = np.where(Vk)[0]
zV = Pk[Vk, PLOT_DIM]
zall = np.concatenate([zV] + [p_[1] for p_ in pieces])
YL = (zall.min() - eps - 0.02, zall.max() + eps + 0.02)

# ================================================================ 版面
M0, GAP = 0.04, 0.12                                # 外边距 / 卡片间距(放箭头)
CW1 = 1.26                                          # 第一张卡宽;其余三张等宽
CWn = (W - 2 * M0 - 3 * GAP - CW1) / 3
xs_ = [M0]
for w_ in (CW1, CWn, CWn):
    xs_.append(xs_[-1] + w_ + GAP)
X1, X2, X3, X4 = [(x, x + w_) for x, w_ in zip(xs_, (CW1, CWn, CWn, CWn))]
R1 = (0.03, None)
PY0, PH = R1[0] + 0.30, 0.60                        # 三张小图:顶 / 高
YTK = PY0 + PH + 0.068                              # 刻度文字行
YL1 = PY0 + PH + 0.17                               # 图例 / 提示两行
YL2 = YL1 + LH
R1 = (R1[0], YL2 + 0.068)
R2 = (R1[1] + 0.17, None)
PXS = [(xr[0] + 0.11, xr[1] - 0.08) for xr in (X2, X3, X4)]

for xr, tl in ((X1, ["Sparse 4D", "object tracks"]), (X2, ["Split by", "visibility + events"]),
               (X3, ["Cubic B-spline /", "line / point"]), (X4, ["On-curve nodes", "→ tokens"])):
    card(xr, *R1)
    ctitle(xr, R1[0], tl)

# ---------------------------------------------------------------- 卡 1:稀疏 4D 物体轨迹(真实世界系轨迹,斜俯视)+ 归一化


def obj_world(iid):
    return next(x for x in fr if x["instance_id"] == iid)


def ortho(Pw, az=np.deg2rad(-25), el=np.deg2rad(58)):
    """世界系(x 右、y 下、z 前)→ 斜俯视正交投影:返回 (水平, 竖直向上)。"""
    x, up, z = Pw[..., 0], -Pw[..., 1], Pw[..., 2]
    x1 = x * np.cos(az) + z * np.sin(az)
    z1 = -x * np.sin(az) + z * np.cos(az)
    return np.stack([x1, up * np.cos(el) + z1 * np.sin(el)], -1)


o13, o17, o0 = obj_world(IID), obj_world(BIKE), obj_world(STORE)
T13 = (np.asarray(o13["trajectories"], float) - C0) / S0
V13 = np.asarray(o13["visibility"], bool)
assert V13[1, 3]
Rb, tb = np.asarray(o17["se3_R"], float), np.asarray(o17["se3_t"], float)
refb = np.asarray(o17["ref_positions"], float)
okb = np.asarray(o17["se3_ok"], bool)
assert refb.shape == (4, 3) and okb[16]
bike16 = ((Rb[16] @ refb.T).T + tb[16] - C0) / S0
bike_c = ((np.einsum("tij,j->ti", Rb, refb.mean(0)) + tb) - C0) / S0
store = (np.asarray(o0["ref_positions"], float) - C0) / S0
assert store.shape[0] <= 4
allp = np.concatenate([T13[V13], bike_c[okb], bike16, store])
lo3, hi3 = allp.min(0), allp.max(0)
cube = np.array([[x, y, z] for x in (lo3[0], hi3[0]) for y in (lo3[1], hi3[1]) for z in (lo3[2], hi3[2])])
BX0, BX1, BY0, BY1 = X1[0] + 0.08, X1[1] - 0.08, PY0 - 0.02, YL1 - 0.10
cq = ortho(cube)
q_lo, q_hi = cq.min(0), cq.max(0)
k3 = min((BX1 - BX0) / (q_hi[0] - q_lo[0]), (BY1 - BY0) / (q_hi[1] - q_lo[1]))
ox = BX0 + (BX1 - BX0 - k3 * (q_hi[0] - q_lo[0])) / 2
oy = BY0 + (BY1 - BY0 - k3 * (q_hi[1] - q_lo[1])) / 2


def to_box(Pw):
    q = ortho(Pw)
    return ox + k3 * (q[..., 0] - q_lo[0]), oy + k3 * (q_hi[1] - q[..., 1])


for i in range(8):
    for j in range(i + 1, 8):
        if np.sum(np.abs(cube[i] - cube[j]) > 1e-9) == 1:
            xs, ys = to_box(cube[[i, j]])
            ax.plot(xs, ys, color="#C9CDD3", lw=0.4, zorder=2)
xs, ys = to_box(store)
ax.plot(xs, ys, "o", ms=2.2, color=GREY, zorder=4)
xs, ys = to_box(bike_c[okb])
ax.plot(xs, ys, color=AZ, lw=0.9, zorder=4)
xs, ys = to_box(bike16)
ax.plot(xs, ys, "o", ms=2.2, color=AZ, mec="white", mew=0.3, zorder=5)
for m in range(T13.shape[0]):
    xs, ys = to_box(T13[m])
    for f in range(33):
        if V13[m, f] and V13[m, f + 1]:
            ax.plot(xs[f:f + 2], ys[f:f + 2], color=PURP, lw=(1.3 if m == 1 else 0.5), alpha=(1.0 if m == 1 else 0.5),
                    zorder=5, solid_capstyle="round")
pxs, pys = to_box(T13[V13])                         # 人物轨迹在框里的位置 → 右侧框线外标 "person"(与序列里的 [the person] 同色)
i_r = int(np.argmax(pxs))
yp_ = float(pys[i_r])
xe_ = [pxs[i_r]]
for i in range(8):
    for j in range(i + 1, 8):
        if np.sum(np.abs(cube[i] - cube[j]) > 1e-9) == 1:
            (xa_, xb_), (ya_, yb_) = to_box(cube[[i, j]])
            if min(ya_, yb_) <= yp_ <= max(ya_, yb_) and abs(yb_ - ya_) > 1e-6:
                xc_ = xa_ + (yp_ - ya_) / (yb_ - ya_) * (xb_ - xa_)
                if xc_ > pxs[i_r] - 0.02:
                    xe_.append(xc_)
t_ = text(max(xe_) + 0.035, yp_, "person", FS, PURP)
assert max(xe_) + 0.035 + width_in(t_) <= X1[1] - 0.04, "person 标签出卡"
xr_, yr_ = to_box(T13[1, 3])
ax.plot([xr_], [yr_], "o", ms=2.6, color=C_REV, mec="white", mew=0.3, zorder=6)     # 第 3 帧折返(贯穿全图的示例点)
# 归一化:单位球小图标 + 两行短语(不写具体数值)
SR = 0.068
scx, scy = X1[0] + 0.06 + SR, (YL1 + YL2) / 2
ax.add_patch(Circle((scx, scy), SR, fc="#F1ECF8", ec=PURP, lw=0.6, zorder=3))
ax.add_patch(Ellipse((scx, scy), 2 * SR, 0.7 * SR, fill=False, ec=PURP, lw=0.4, ls=(0, (2, 1.5)), zorder=3))
ax.plot([scx], [scy], marker="+", ms=3.6, mew=0.6, color=SLATE, zorder=4)
for (dx, dy), cc in (((-0.03, -0.016), PURP), ((0.024, 0.024), AZ), ((0.03, -0.024), GREY)):
    ax.plot([scx + dx], [scy + dy], "o", ms=1.6, color=cc, zorder=4)
for y_, s_, col in ((YL1, "shared center + scale", TXT), (YL2, "metric recovery retained", SUB)):
    t_ = text(scx + SR + 0.05, y_, s_, FS, col)
    assert scx + SR + 0.05 + width_in(t_) <= X1[1] - 0.04, ("卡 1 文字溢出", s_)

# ---------------------------------------------------------------- 卡 2–4:同一关键点的三张同坐标小图
AXS = []
for x0, x1 in PXS:
    a = axes_at(x0, PY0, x1 - x0, PH)
    a.set_xlim(-1.5, 34.5); a.set_ylim(*YL)
    a.set_xticks([0, 33]); a.set_yticks([])
    AXS.append(a)
a2, a3, a4 = AXS
# 刻度文字与轴名只在第一张图上写一次(三张图同坐标)
for fx, s_ in ((0, "0"), (33, "33")):
    text(to_in(a2, fx, YL[0])[0], YTK, s_, FS, SUB, ha="center")
text(to_in(a2, 16.5, YL[0])[0], YTK, "frame", FS, SUB, ha="center")
text(PXS[0][0] - 0.045, PY0 + PH / 2, "z", FS, SUB, ha="center")

# 卡 2:可见帧(每个可见段连成一串)+ 遮挡缺口 + 事件切分
for g0, g1 in gaps:
    a2.axvspan(g0 + 0.3, g1 - 0.3, color=GAPC, lw=0, zorder=0)
for sg in segs:
    ff = np.arange(int(round(sg.u_start * 33)), int(round(sg.u_end * 33)) + 1)
    a2.plot(ff, Pk[ff, PLOT_DIM], "-", color=PURP_L, lw=0.6, zorder=2)
    a2.plot(ff, Pk[ff, PLOT_DIM], "o", ms=2.0, color=PURP, mec="none", alpha=0.75, zorder=3)
a2.axvline(3, color=C_REV, lw=0.8, ls=(0, (2, 1.4)), zorder=2)
a2.plot([3], [YL[1]], marker="v", ms=4.6, color=C_REV, mec="white", mew=0.3, zorder=5, clip_on=False)
xl0, xl_max = X2[0] + 0.06, X2[1] - 0.04
legend_row(xl0, YL1, [(mk_tri, "reversal", TXT), (mk_diamond, "impact", TXT)], xl_max)
legend_row(xl0, YL2, [(mk_swatch(GAPC), "never fit across gaps", TXT)], xl_max)

# 卡 3:误差管 + 拟合曲线 / 直线 / 单点 + C⁰ 角点
a3.plot(fV, zV, "o", ms=1.3, color="#BDBDBD", mec="none", zorder=3)
for ff, zz in pieces:
    a3.fill_between(ff, zz - eps, zz + eps, color=TUBE, lw=0, zorder=1)
    a3.plot(ff, zz, color=PURP, lw=1.0, zorder=4)
for f in single:
    z_ = next(z for ff_, z, k in nodes if abs(ff_ - f) < 1e-9)
    a3.add_patch(Rectangle((f - 0.45, z_ - eps), 0.9, 2 * eps, fc=TUBE, lw=0, zorder=1))
    a3.plot([f], [z_], "o", ms=2.1, color=PURP, zorder=4)
a3.plot([3], [ZREV], "o", ms=3.2, mfc="white", mec=C_REV, mew=0.7, zorder=6)
z_line = float(np.mean([p_[PLOT_DIM] for p_ in two[0].nodes]))
z_pt = next(z for ff_, z, k in nodes if abs(ff_ - 21) < 1e-9)
pc = pieces[-1]                                     # 第 29–33 帧那段三次样条
assert abs(pc[0][0] - 29) < 1e-9 and abs(pc[0][-1] - 33) < 1e-9
LAB3 = [(10.5, z_line + 0.075, "line"), (21.5, z_pt + 0.07, "point"), (31.6, float(pc[1].max()) + 0.055, "curve")]
for fx, zy, s_ in LAB3:
    a3.text(fx, zy, s_, fontsize=FS, color=SUB, ha="center", va="center", zorder=7)
legend_row(X3[0] + 0.06, YL1, [(mk_swatch(TUBE), "max error ≤ 0.02 × bbox diag", TXT)], X3[1] - 0.04)

# 卡 4:节点(按 knot type 上色),折返节点加圈
for ff, zz in pieces:
    a4.plot(ff, zz, color=PURP, lw=0.55, alpha=0.5, zorder=2)
KC = {"endpoint": C_END, "smooth": C_SMOOTH, "reversal": C_REV}
for f, z, k in nodes:
    a4.plot([f], [z], "o", ms=3.2, color=KC[k], mec="white", mew=0.35, zorder=5)
a4.plot([3], [ZREV], "o", ms=7.5, mfc="none", mec=C_REV, mew=0.7, zorder=6)


def mk_dot(col):
    def f(x, y):
        ax.plot([x + 0.018], [y], "o", ms=2.8, color=col, mec="white", mew=0.3, zorder=9)
        return x + 0.036
    return f


xl0 = X4[0] + 0.06
legend_row(xl0, YL1, [(mk_dot(C_END), "endpoint", TXT), (mk_dot(C_SMOOTH), "smooth", TXT), (mk_dot(C_REV), "reversal", TXT)],
           X4[1] - 0.04, gap_mk=0.012, gap_item=0.05)
legend_row(xl0, YL2, [(None, "one token per spline coefficient", TXT)], X4[1] - 0.04)

# ================================================================ 下行:序列化 → 投影
SQ, PIT = 0.036, 0.046
rev_pos = [x_ is recs[REV_IDX] for x_ in blk].index(True)          # 示例 token 在这 28 个里的位置
blk_kind = [ts.ROLES[x_.role] for x_ in blk]
assert blk_kind[rev_pos] == "PL_VERT" and rev_pos == 1 + 6 + 2
SEQ = [
    [("L", "[scene]", DARK, "normal"), ("S", 3, DARK), ("D", "…", SUB, "normal"), ("L", "[the person]", PURP, "bold"),
     ("S", 1, PURP), ("L", "[part 1]", PURP, "normal"), ("S", 7, PURP_L)],
    [("L", "[part 2]", PURP, "normal"), ("K", 28, PURP), ("D", "…", SUB, "normal"), ("L", "[the camera]", DARK, "bold"),
     ("S", 13, DARK)],
]
YQ = [R2[0] + 0.28, R2[0] + 0.43]
XQ0 = M0 + 0.10


def seq_line(items, x, yc, draw=True):
    """画一行序列(draw=False 只量宽度);返回 (右端 x, 示例 token 方块左边 x)。"""
    xrev = None
    for it in items:
        kind = it[0]
        if kind in ("L", "D"):
            w_ = tw(it[1], FS, it[3])
            if draw:
                text(x, yc, it[1], FS, it[2], weight=it[3])
            x += w_ + 0.03
        elif kind == "S":
            for _ in range(it[1]):
                if draw:
                    ax.add_patch(Rectangle((x, yc - SQ / 2), SQ, SQ, fc=it[2], ec="none", zorder=4))
                x += PIT
            x += 0.025
        else:                                          # 示例关键点的 28 个 token:可见性摘要 / 遮挡缺口 / 节点
            for j in range(it[1]):
                r_ = blk_kind[j]
                if draw:
                    if r_ == "VIS_SUMMARY":
                        ax.add_patch(Rectangle((x, yc - SQ / 2), SQ, SQ, fc="white", ec=it[2], lw=0.5, hatch="//////", zorder=4))
                    elif r_ == "OCC_GAP":
                        ax.add_patch(Rectangle((x, yc - SQ / 2), SQ, SQ, fc="#BDBDBD", ec="none", zorder=4))
                    else:
                        ax.add_patch(Rectangle((x, yc - SQ / 2), SQ, SQ, fc=it[2], ec="none", zorder=4))
                if j == rev_pos:
                    xrev = x
                x += PIT
            x += 0.025
    return x, xrev


# 字段表(一个 token = 离散字段一行 + 连续字段一行;数值 = 第 3 帧折返节点那个 token)
DISC = [("role", "node"), ("knot", "reversal"), ("kind", "–"), ("otype", "–"), ("inst", "5"), ("part", "2"), ("kp", "6")]
CONT = [("xyz", "(0.61, 0.02, 0.48)"), ("u", "0.09"), ("t", "0.60 s")]
CPAD, CGAP, CHH = 0.06, 0.02, 0.20
GL_W = max(tw("discrete", FS, style="italic"), tw("continuous", FS, style="italic")) + 0.07
seq_w = max(seq_line(items, XQ0, 0, draw=False)[0] for items in SEQ) - XQ0


def row_w(cells):
    return sum(max(tw(a_, FS), tw(b_, FS, "bold")) + CPAD for a_, b_ in cells) + CGAP * (len(cells) - 1)


tab_w = GL_W + max(row_w(DISC), row_w(CONT))
X5 = (M0, M0 + 0.10 + max(seq_w, tab_w) + 0.10)
X6 = (X5[1] + GAP, W - M0)
YT = YQ[1] + 0.19                                   # 字段表第一行顶
YT2 = YT + CHH + 0.06
R2 = (R2[0], None)

# 投影器(结构真实;模块尺寸为示意)
CELL_G = "#E9EDF2"
MODS = [("pair", ("embeddings", "linear projections")), ("concat", CELL_G, TXT), ("Linear → 4096", SLATE, "white"),
        ("LayerNorm", CELL_G, TXT), ("residual MLP", SLATE, "white"), ("LayerNorm", CELL_G, TXT)]
MH, MG = 0.095, 0.033
PCX = (X6[0] + X6[1]) / 2
MW = min(1.7, X6[1] - X6[0] - 0.30)
YM0 = R2[0] + 0.21
YM_END = YM0 + len(MODS) * MH + (len(MODS) - 1) * MG
YSLOT = YM_END + 0.07
YOUT = YSLOT + 0.045 + 0.07
R2 = (R2[0], max(YT2 + CHH, YOUT + 0.04) + 0.07)

card(X5, *R2)
ctitle(X5, R2[0], ["Serialize"])
xr_rev = None
ends = []
for items, yc in zip(SEQ, YQ):
    xe, xr = seq_line(items, XQ0, yc)
    ends.append(xe)
    xr_rev = xr if xr is not None else xr_rev
assert max(ends) - 0.025 <= X5[1] - 0.08, ("序列溢出", ends)
ax.add_patch(Rectangle((xr_rev - 0.012, YQ[1] - SQ / 2 - 0.012), SQ + 0.024, SQ + 0.024, fill=False, ec=C_REV, lw=0.8, zorder=5))
# 字段表 + 放大锥(从描边方块到表格)
TX0 = (X5[0] + X5[1]) / 2 - tab_w / 2                # 字段表在卡片里水平居中
cell_x0 = TX0 + GL_W
cx1 = {}
for yt, cells, fc, ec, lab in ((YT, DISC, CELL_D, "#8FB3DA", "discrete"), (YT2, CONT, CELL_C, "#E3A98B", "continuous")):
    text(TX0, yt + CHH / 2, lab, FS, SUB, style="italic")
    x = cell_x0
    for name, val in cells:
        w_ = max(tw(name, FS), tw(val, FS, "bold")) + CPAD
        rbox(x, yt, x + w_, yt + CHH, fc, ec, 0.5, r=0.02, z=3)
        text(x + w_ / 2, yt + 0.058, name, FS, TXT, ha="center")
        text(x + w_ / 2, yt + 0.142, val, FS, (SUB if val == "–" else C_REV), ha="center", weight=("normal" if val == "–" else "bold"))
        x += w_ + CGAP
    cx1[lab] = x - CGAP
assert max(cx1.values()) <= X5[1] - 0.08, ("字段表溢出", cx1)
ybot_sq = YQ[1] + SQ / 2 + 0.012
ax.add_patch(Polygon([(xr_rev - 0.012, ybot_sq), (xr_rev + SQ + 0.012, ybot_sq), (cx1["discrete"], YT - 0.012), (cell_x0, YT - 0.012)],
                     closed=True, fc=C_REV, ec="none", alpha=0.08, zorder=2))
for xa_, xb_ in ((xr_rev - 0.012, cell_x0), (xr_rev + SQ + 0.012, cx1["discrete"])):
    ax.plot([xa_, xb_], [ybot_sq, YT - 0.012], color=C_REV, lw=0.4, alpha=0.7, zorder=2)

card(X6, *R2)
ctitle(X6, R2[0], ["Project to 4096-d embeddings"])
ym = YM0
MOD_C = []
for i, md in enumerate(MODS):
    if md[0] == "pair":
        half = (MW - 0.04) / 2
        for j, (s_, fc, ec) in enumerate(((md[1][0], CELL_D, "#8FB3DA"), (md[1][1], CELL_C, "#E3A98B"))):
            xa_ = PCX - MW / 2 + j * (half + 0.04)
            rbox(xa_, ym, xa_ + half, ym + MH, fc, ec, 0.5, r=0.02, z=3)
            t_ = text(xa_ + half / 2, ym + MH / 2, s_, FS, TXT, ha="center")
            assert width_in(t_) <= half - 0.03, ("模块文字溢出", s_)
    else:
        s_, fc, tc = md
        rbox(PCX - MW / 2, ym, PCX + MW / 2, ym + MH, fc, ("none" if fc == SLATE else "#8792A2"), 0.5, r=0.02, z=3)
        text(PCX, ym + MH / 2, s_, FS, tc, ha="center")
    MOD_C.append(ym + MH / 2)
    if i < len(MODS) - 1:
        arrow((PCX, ym + MH), (PCX, ym + MH + MG), color=SLATE, lw=0.6, ms=3.5)
    ym += MH + MG
assert abs(ym - MG - YM_END) < 1e-9
arrow((PCX, YM_END), (PCX, YSLOT - 0.005), color=PURP, lw=1.0, ms=5)
slots = ["t", "t", "p", "p", "p", "t", "p", "p", "t", "t"]
sx0 = PCX - len(slots) * 0.055 / 2
for i, s_ in enumerate(slots):
    ax.add_patch(Rectangle((sx0 + i * 0.055, YSLOT), 0.045, 0.045, fc=(PURP if s_ == "p" else "#D0D0D0"), ec="none", zorder=4))
t_ = text(PCX, YOUT, "fill <traj> slots in Qwen input embeddings", FS, TXT, ha="center")
assert width_in(t_) <= X6[1] - X6[0] - 0.08, "投影器结尾一句溢出"

# ================================================================ 箭头
YA = PY0 + PH / 2
for (xa_, xb_) in ((X1[1], X2[0]), (X2[1], X3[0]), (X3[1], X4[0])):
    arrow((xa_ + 0.018, YA), (xb_ - 0.018, YA), color=ARR, lw=1.3, ms=8)
arrow((X5[1] + 0.018, MOD_C[0]), (X6[0] - 0.018, MOD_C[0]), color=ARR, lw=1.3, ms=8)
X4C, YMID, XIN = (X4[0] + X4[1]) / 2, (R1[1] + R2[0]) / 2, M0 + 0.32     # 第 4 步 → 两行之间的通道 → 序列化
ax.plot([X4C, X4C, XIN], [R1[1], YMID, YMID], color=PURP, lw=1.6, zorder=6, solid_joinstyle="round")
arrow((XIN, YMID), (XIN, R2[0] + 0.004), color=PURP, lw=1.6, ms=7)

# ================================================================ 画布高度、重叠 / 越界 / 出卡片检查、输出
H_NEED = R2[1] + 0.03
fig.canvas.draw()
bbs, oob, outcard = [], [], []
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
    if not any(cx0 + 0.02 <= x0 and x1 <= cx1_ - 0.02 and cy0 + 0.01 <= y0_ + 0.012 and y1_ - 0.012 <= cy1 - 0.01
               for cx0, cy0, cx1_, cy1 in CARDS):
        outcard.append((t.get_text(), round(x0, 2), round(x1, 2), round(y0_, 2), round(y1_, 2)))
    bbs.append((x0 + 0.004, y0_ + 0.012, x1 - 0.004, y1_ - 0.012, t.get_text()))
bad = [(p_[4], q_[4]) for i, p_ in enumerate(bbs) for q_ in bbs[i + 1:]
       if p_[0] < q_[2] and q_[0] < p_[2] and p_[1] < q_[3] and q_[1] < p_[3]]
if oob or bad or outcard or abs(H_NEED - H) >= 0.03:
    fig.savefig(DBG, dpi=200)
    raise AssertionError(dict(H_NEED=round(H_NEED, 3), H=H, out_of_canvas=oob, out_of_card=outcard, overlaps=bad[:12]))
fig.savefig(os.path.join(OUT, "fig3_tokenization.pdf"))
fig.savefig(os.path.join(OUT, "fig3_tokenization.png"), dpi=300)
print("wrote", os.path.join(OUT, "fig3_tokenization.pdf"), "and .png;", f"H={H} in, texts={len(bbs)}, eps={eps:.4f},",
      "cards:", [tuple(round(v, 2) for v in c_) for c_ in CARDS])
