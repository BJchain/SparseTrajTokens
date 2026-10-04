"""图 4b(附录追加例子:baseline 错、我们对;09-26 老师要求,版式同 fig4 系列)渲染 —— 由 render_fig4.py 改出。
原图 4 v2 说明:图 4 v2(附录定性图,每个 example 一张)渲染 —— 按 fig4_blueprint.md(09-25 用户意见:拆成 5 张附录图,
上 = 3 帧 + caption(同 v1),中 = 题目栏,下 = 左右并排的两个模型原样输出)。
输出 fig4_<scene>.pdf(字体嵌入)+ fig4_<scene>.png(300 dpi);没有留出集题目的 example 跳过并打印原因。
运行时核对:帧图 = 所标 clip 的所标帧(叠加层以外与原始帧逐像素一致)、caption 与 T8 原文逐字一致、时间轴 fps 一致、
上色物体的轮廓颜色确实出现在帧图里、裁剪框包住全部叠加层;题目 / 选项 / 正确答案 / 真值取自留出集原文件,
问到的物体 = 帧上上色的那个实例;两个模型的输出取自逐题预测原文件,<answer> 里的字母 = 记录的预测,
两个模型的评测协议相同(整段 clip 均匀采 8 帧:B 集每题的 frame_range 都是整段)。任何一项不符就报错退出。v1 脚本备份在会话 scratchpad(v1_render_fig4.py)。
用法:<stereo4d-env python> render_fig4.py
"""
import colorsys
import json
import os
import re
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle  # noqa: E402
from matplotlib.text import Text  # noqa: E402
from PIL import Image  # noqa: E402

R = str(__import__("pathlib").Path(__file__).resolve().parents[2])          # repo root(本文件在 paper/figure_scripts/)
S = R
M = R + "/data"
FIG = R + "/paper/figure_assets"
FRAW = S + "/data/stereo4d/frames"
QA = M + "/trajqa_b/final"
PREDS = R + "/baselines/preds"
OUT = R + "/paper/figure_assets/figures"          # 合成图输出目录
DBG = os.path.join(os.environ.get("TMPDIR", "/tmp"), "fig4_debug")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # render_figs.py
from render_figs import PAL  # noqa: E402  (渲染帧时用的调色板,BGR)

# expect = (只看帧模型的字母, 我们的字母):写进蓝图的结论,运行时核对
EX = [
    dict(key="distant_animal", clip="N6EH3Lk0eRo_255522189", crop=True, name="Distant animal",
         qid="scene_N6EH3Lk0eRo_255522189_speed_grid_f27c768c", split="test", expect=("B", "C")),
    dict(key="crossing_truck", clip="WWdorMyLg3M_426693360", crop=False, name="Crossing truck",
         qid="scene_WWdorMyLg3M_426693360_disp_grid_1b56523e", split="test", expect=("C", "D")),
    dict(key="camera_motion", clip="7uTgv66Q29U_36169503", crop=False, name="Camera motion",
         qid="scene_7uTgv66Q29U_36169503_cam_disp_grid_84fb6571", split="test", expect=("D", "C")),
    dict(key="carried_object", clip="uDWH1bRUvYQ_217984651", crop=False, name="Carried object",
         qid="scene_uDWH1bRUvYQ_217984651_disp_grid_fddedbbd", split="heldout_test", expect=("B", "A")),
    dict(key="approaching_person", clip="q-B2xu8B3Js_29963297", crop=False, name="Approaching person",
         qid="scene_q-B2xu8B3Js_29963297_speed_grid_c8681538", split="heldout_test", expect=("D", "A")),
]
MODELS = [("rgb", "Ours_RGB-arm", "Qwen3-VL-8B · frames only"), ("ours", "Ours_TrajRGB-final", "Ours · frames + trajectory tokens")]


def hex_of(bgr):
    b, g, r = bgr
    return f"#{r:02X}{g:02X}{b:02X}"


def overlay_mask(ren, raw):
    """叠加层 = 轮廓 / 轨迹线(与原帧差 > 60)膨胀后把轮廓围住的内部(半透明填色)一并填满。"""
    from scipy.ndimage import binary_fill_holes
    diff = np.abs(ren.astype(int) - raw.astype(int)).sum(-1) > 60
    m = cv2.morphologyEx(diff.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return binary_fill_holes(cv2.dilate(m, np.ones((5, 5), np.uint8)) > 0)


def parse_output(raw):
    """模型原样输出 → 行列表;核对结构 <evidence> … </evidence> <answer>X</answer>,返回 (行, 字母)。"""
    lines = raw.split("\n")
    assert lines[0] == "<evidence>" and lines[-2] == "</evidence>", raw
    m = re.fullmatch(r"<answer>([A-D])</answer>", lines[-1])
    assert m, raw
    return lines, m.group(1)


# ---------------------------------------------------------------- 源数据核对
def check_sources():
    summ = {r["clip"]: r for r in json.load(open(FIG + "/render_summary_extra.json"))}
    t8 = {}
    want = {"scene_" + e["clip"] for e in EX}
    for ln in open(M + "/captions/t8_train.jsonl"):
        if "scene_" in ln:
            d = json.loads(ln)
            if d["video_id"] in want:
                t8.setdefault(d["video_id"], []).append(d["answer"])
    for e in EX:                                                        # caption = 该 clip 的 T8 原文(清单句在前)
        a_ = t8["scene_" + e["clip"]]
        e["caps"] = [x for x in a_ if x.startswith("Moving in the world frame")] + [x for x in a_ if not x.startswith("Moving in the world frame")]
        assert len(e["caps"]) >= 1 and e["caps"][0].startswith("Moving in the world frame")
        e["prefix"] = "fig4b_" + e["clip"]
    qrec = {}
    for sp in ("test", "heldout_test", "val", "train"):
        for ln in open(f"{QA}/{sp}.jsonl"):
            d = json.loads(ln)
            if d["video_id"] in want:
                d["_split"] = sp
                qrec.setdefault(d["video_id"], {})[d["id"]] = d
    preds = {}
    for key, fn, _ in MODELS:
        meta = json.load(open(f"{PREDS}/{fn}.meta.json"))
        assert meta["protocol"].startswith("OURS (not Q1/Q2): 8 frames sampled uniformly INSIDE the question's asked interval"), fn
        assert meta["base_model"] == "Qwen3-VL-8B-Instruct" and "NOT zero-shot" in meta["note"], fn
        preds[key] = {json.loads(ln)["id"]: json.loads(ln) for ln in open(f"{PREDS}/{fn}.jsonl")}
    assert "pB_rgbsw_9002_s0/final" in json.load(open(f"{PREDS}/Ours_RGB-arm.meta.json"))["checkpoint"]
    assert json.load(open(f"{PREDS}/Ours_TrajRGB-final.meta.json"))["checkpoint"] == "runs_sft110k/pB_s2_cos1ep_s0/final"
    for e in EX:
        v = "scene_" + e["clip"]
        s = summ[e["clip"]]
        assert s["prefix"] == e["prefix"]
        e["frames"] = tuple(s["frames"])
        for c in e["caps"]:                                             # caption 逐字
            assert c in t8[v], ("caption 与 T8 原文不符", e["prefix"], c[:60])
        g = np.load(f"{M}/scenes/{v}/gt_stereo4d.npz", allow_pickle=True)   # 时间轴 fps 一致
        ts = np.asarray(g["timestamps_us"], float) / 1e6
        fps_meta = float(json.load(open(f"{M}/scenes/{v}/stereo4d_meta.json"))["fps_effective"])
        assert abs((len(ts) - 1) / (ts[-1] - ts[0]) / fps_meta - 1) < 0.1, (e["prefix"], "fps 不一致")
        tiles, masks = [], []
        for t in e["frames"]:                                           # 帧图 = 原始帧 + 叠加层
            ren = np.asarray(Image.open(f"{FIG}/{e['clip']}/{e['prefix']}_f{t:02d}.png").convert("RGB"))
            raw = cv2.cvtColor(cv2.imread(f"{FRAW}/{e['clip']}/{t:05d}.jpg"), cv2.COLOR_BGR2RGB)
            assert ren.shape == raw.shape == (512, 512, 3)
            om = overlay_mask(ren, raw)
            assert om.sum() > 150 and np.abs(ren.astype(int) - raw.astype(int)).sum(-1)[~om].mean() < 3.0, (e["prefix"], t)
            tiles.append(ren); masks.append(om)
        e["tiles"] = tiles
        colors, iid_of = {}, {}
        for i, o in enumerate(s["objects"]):                            # objects 顺序 = 调色板顺序
            colors.setdefault("the " + o["label"], hex_of(PAL[i % len(PAL)]))
            iid_of.setdefault("the " + o["label"], o["iid"])
        e["colors"] = {}
        for nm in re.findall(r"Moving in the world frame: (.*?)\. Staying", e["caps"][0])[0].split(", "):
            if nm.startswith("the "):
                col = colors[nm]
                rgb = np.array([int(col[k:k + 2], 16) for k in (1, 3, 5)])
                n = sum(int((np.abs(t_.astype(int) - rgb).max(-1) <= 10).sum()) for t_ in tiles)
                assert n >= 40, (e["prefix"], nm, col, n)
                e["colors"][nm] = col
        if e["crop"]:                                                   # 裁剪框:三帧叠加层并集外扩后取正方形
            ys, xs = np.nonzero(np.any(np.stack(masks), 0))
            x0, x1 = np.percentile(xs, [0.5, 99.5]); y0, y1 = np.percentile(ys, [0.5, 99.5])
            side = min(512, 1.18 * max(x1 - x0, y1 - y0) + 16)
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            bx0 = int(np.clip(round(cx - side / 2), 0, 512 - side)); by0 = int(np.clip(round(cy - side / 2), 0, 512 - side))
            box = (bx0, by0, bx0 + int(side), by0 + int(side))
            assert box[0] <= x0 and box[1] <= y0 and box[2] >= x1 and box[3] >= y1, (e["prefix"], box)
            e["box"] = box
        else:
            e["box"] = (0, 0, 512, 512)
        # 题目 / 两个模型的输出
        if e["qid"] is None:
            assert v not in qrec, (e["key"], "蓝图说它没有 TrajQA-B 题目,但找到了", sorted(qrec[v])[:3])
            continue
        q = qrec[v][e["qid"]]
        assert q["_split"] == e["split"] and q["_split"] in ("test", "heldout_test")            # 只用留出集
        assert q["frame_range"] == [0, q["gt_payload"]["T"] - 1]     # B 集每题 frame_range = 整段 → 两个模型都看整段均匀采的 8 帧
        assert {d["_split"] for d in qrec[v].values()} <= {"test", "heldout_test"}              # 整个 clip 不在训练集
        gp = q["gt_payload"]
        assert e["frames"] == (gp["i"], (gp["i"] + gp["j"]) // 2, gp["j"]), (e["key"], e["frames"])          # 帧 = 题目区间
        if q["event_subtype"] == "cam_disp_grid":
            obj = None
            assert q["involved_inst_ids"] == [] and "the camera move" in q["question"]
        else:
            obj = re.search(r"(?:did|of) (the [a-z /]+?) (?:move|between)", q["question"]).group(1)
            assert obj in e["colors"] and q["involved_inst_ids"] == [iid_of[obj]], (e["key"], obj, q["involved_inst_ids"])
        e["q"], e["obj"] = q, obj
        e["out"] = {}
        for key, _, _ in MODELS:
            p = preds[key][e["qid"]]
            lines, letter = parse_output(p["raw"])
            assert letter == p["pred"], (e["key"], key)
            e["out"][key] = (lines, letter)
        assert (e["out"]["rgb"][1], e["out"]["ours"][1]) == e["expect"], (e["key"], e["out"]["rgb"][1], e["out"]["ours"][1])
        unit = q["gt_payload"]["unit"]
        assert unit in ("m", "m/s") and (unit == "m/s") == (q["event_subtype"] == "speed_grid")
        e["keyline"] = "speed_mps:" if unit == "m/s" else ("camera_displacement_m:" if obj is None else "displacement_m:")
        for key in ("rgb", "ours"):                                   # 决定答案的那一行存在且唯一
            assert sum(ln.startswith(e["keyline"]) for ln in e["out"][key][0]) == 1
    return EX


# ---------------------------------------------------------------- 样式与排版工具
TXT, SUB, SLATE = "#1A1A1A", "#5F6368", "#3E4C5E"
GRN, RED, TAG_E, TAG_A = "#2CA02C", "#D62728", "#A33B2B", "#2E7D32"
FN, FH, FC, FM, FL = 8.0, 7.0, 6.5, 6.3, 6.0          # 场景名 / 栏标题 / 正文 / 模型输出(等宽)/ 帧号与色条
LHC, LHM = 0.100, 0.098
MONO = "Liberation Mono"
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif",
                     "font.sans-serif": ["Liberation Sans", "DejaVu Sans"], "axes.unicode_minus": False,
                     "text.hinting": "no_hinting"})
W = 5.5
C = {}                                                   # 当前画布:fig / ax / rend / H / 文字所属的框


def new_canvas(H):
    fig = plt.figure(figsize=(W, H), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    C.update(fig=fig, ax=ax, rend=fig.canvas.get_renderer(), H=H, boxes=[])


def text(x, y, s, size=FC, color=TXT, ha="left", va="center", weight="normal", **kw):
    return C["ax"].text(x, y, s, fontsize=size, color=color, ha=ha, va=va, fontweight=weight, zorder=8, **kw)


def width_in(t):
    return t.get_window_extent(renderer=C["rend"]).width / C["fig"].dpi


def tw(s, size=FC, weight="normal", family=None):
    t = C["ax"].text(0, -5, s, fontsize=size, fontweight=weight, **({"fontfamily": family} if family else {}))
    w = width_in(t)
    t.remove()
    return w


def styled_words(sentence, colors, first_only=False):
    """句子 → [(词, 颜色, 粗体, 紧贴前词)]。colors: {物体名: 颜色};物体名上色加粗(first_only=只认句首那个)。
    紧贴前词 = 原文里它前面没有空格(如名字后面的逗号 / 句号),排版时不加空格、也不在它前面折行。"""
    spans = []
    for nm, col in sorted(colors.items(), key=lambda kv: -len(kv[0])):
        for m in re.finditer(re.escape(nm), sentence, flags=re.I):
            if first_only and m.start() != 0:
                continue
            if not any(a < m.end() and m.start() < b for a, b, _ in spans):
                spans.append((m.start(), m.end(), col))
    spans.sort()
    segs, pos = [], 0
    for a, b, col in spans:
        segs += [(sentence[pos:a], TXT, False), (sentence[a:b], col, True)]
        pos = b
    segs.append((sentence[pos:], TXT, False))
    out, prev_space = [], True
    for seg, col, bold in segs:
        for m in re.finditer(r"\S+|\s+", seg):
            piece = m.group(0)
            if piece.isspace():
                prev_space = True
                continue
            out.append((piece, col, bold, not prev_space and bool(out)))
            prev_space = False
    rebuilt = "".join(("" if g else " ") + w for w, _, _, g in out).strip()
    assert rebuilt == sentence, (rebuilt, sentence)                 # 排版前后文字逐字一致
    return out


def layout_rich(words, width, size=FC):
    """按宽度折行(紧贴前词的 token 与前词同组,不在组内折行),返回 ([[token, ...], ...], 空格宽)。"""
    sp = tw("a a", size) - 2 * tw("a", size)
    groups = []
    for tk in words:
        if tk[3] and groups:
            groups[-1].append(tk)
        else:
            groups.append([tk])
    lines, cur, cw = [], [], 0.0
    for g in groups:
        gw = sum(tw(w_, size, "bold" if b else "normal") for w_, _, b, _ in g)
        add = gw if not cur else sp + gw
        if cur and cw + add > width:
            lines.append(cur); cur, cw = list(g), gw
        else:
            cur += g; cw += add
    return lines + ([cur] if cur else []), sp


def layout_balanced(words, width, size=FC):
    """折行后若最后一行不到行宽的 1/4(孤字),在行数不变的前提下把行宽收到最小,让各行长度接近。"""
    lines, sp = layout_rich(words, width, size)
    last = sum(tw(w_, size, "bold" if b else "normal") for w_, _, b, _ in lines[-1])
    if len(lines) < 2 or last >= 0.25 * width:
        return lines, sp
    lo, hi = width / len(lines), width
    for _ in range(30):
        mid = (lo + hi) / 2
        if len(layout_rich(words, mid, size)[0]) == len(lines):
            hi = mid
        else:
            lo = mid
    return layout_rich(words, hi, size)


def draw_rich(x, y, lines, sp, size=FC, lh=LHC):
    for i, ln in enumerate(lines):
        xx = x
        runs = []                                        # 同样式且相连的词合成一段
        for w_, c, b, glue in ln:
            if runs and runs[-1][1] == c and runs[-1][2] == b:
                runs[-1][0] += ("" if glue else " ") + w_
            else:
                runs.append([w_, c, b, glue])
        for k, (s_, c, b, glue) in enumerate(runs):
            if k and not glue:
                xx += sp
            t = text(xx, y + i * lh, s_, size, c, weight="bold" if b else "normal")
            xx += width_in(t)
    return y + len(lines) * lh


def rbox(x0, y0, x1, y1, fc, ec, lw=0.5, r=0.04, z=1, holds_text=True):
    C["ax"].add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle=f"round,pad=0,rounding_size={r}",
                                     fc=fc, ec=ec, lw=lw, zorder=z))
    if holds_text:
        C["boxes"].append((x0, y0, x1, y1))


def mark(cx, cy, ok, r=0.042):
    """✓ / ✗ 圆标(矢量画,不依赖字体)。"""
    ax = C["ax"]
    ax.add_patch(Circle((cx, cy), r, fc=(GRN if ok else RED), ec="none", zorder=9))
    if ok:
        ax.plot([cx - 0.52 * r, cx - 0.12 * r, cx + 0.55 * r], [cy + 0.02 * r, cy + 0.42 * r, cy - 0.45 * r], color="white",
                lw=1.0, zorder=10, solid_capstyle="round", solid_joinstyle="round")
    else:
        for s_ in (1, -1):
            ax.plot([cx - 0.42 * r, cx + 0.42 * r], [cy - 0.42 * r * s_, cy + 0.42 * r * s_], color="white", lw=1.0, zorder=10,
                    solid_capstyle="round")


def rainbow(x):
    return colorsys.hsv_to_rgb(0.78 * (1.0 - float(x)), 1.0, 1.0)


def fmt_gt(v, unit):
    return f"{v:.2f} {unit}"


# ================================================================ 一张图
TS, TG, X0 = 1.00, 0.04, 0.02
CBX0, CBX1 = X0 + 3 * TS + 2 * TG + 0.08, W - 0.02
PAD = 0.07
QX0, QX1 = 0.02, W - 0.02
PX = [(0.02, (W - 0.08) / 2 + 0.02), ((W - 0.08) / 2 + 0.08, W - 0.02)]


def draw_figure(e, H):
    new_canvas(H)
    ax = C["ax"]
    y = 0.02
    # ---- 上段:三帧 + 帧号 + 小色条;右侧 caption 框(与 v1 一行相同)
    bx0, by0, bx1, by1 = e["box"]
    for k, (t, a) in enumerate(zip(e["frames"], e["tiles"])):
        x = X0 + k * (TS + TG)
        ax.imshow(a[by0:by1, bx0:bx1], extent=(x, x + TS, y + TS, y), interpolation="none", zorder=2)
        ax.add_patch(Rectangle((x, y), TS, TS, fill=False, ec="#9AA0A6", lw=0.35, zorder=3))
        text(x + TS / 2, y + TS + 0.07, f"t = {t}", FL, SUB, ha="center")
    cb_w, cb_h = 0.9, 0.04
    cb_x0, cb_y = X0 + (3 * TS + 2 * TG - cb_w) / 2, y + TS + 0.17
    grad = np.array([[rainbow(v) for v in np.linspace(0, 1, 256)]])
    ax.imshow(grad, extent=(cb_x0, cb_x0 + cb_w, cb_y + cb_h, cb_y), interpolation="bilinear", zorder=2)
    text(cb_x0 - 0.04, cb_y + cb_h / 2, "start", FL, SUB, ha="right")
    text(cb_x0 + cb_w + 0.04, cb_y + cb_h / 2, "end", FL, SUB, ha="left")
    C["boxes"].append((0.0, y + TS, CBX0 - 0.04, cb_y + cb_h + 0.06))           # 帧号 + 色条所在的条带
    inner = CBX1 - CBX0 - 2 * PAD
    blocks = [layout_rich(styled_words(c, e["colors"], first_only=(j > 0)), inner) for j, c in enumerate(e["caps"])]
    n_lines = sum(len(b[0]) for b in blocks)
    box_h = 0.075 + 0.13 + n_lines * LHC + (len(blocks) - 1) * 0.035 + 0.05
    rbox(CBX0, y, CBX1, y + box_h, "#F6F7F9", "#D5D9DF")
    text(CBX0 + PAD, y + 0.085, e["name"], FN, TXT, weight="bold")
    yy = y + 0.085 + 0.14
    for lines, sp in blocks:
        yy = draw_rich(CBX0 + PAD, yy, lines, sp) + 0.035
    y += max(cb_y + cb_h + 0.04 - y, box_h) + 0.10
    # ---- 中段:题目栏
    q = e["q"]
    qin = QX1 - QX0 - 2 * 0.08
    qlines, qsp = layout_balanced(styled_words(q["question"], ({e["obj"]: e["colors"][e["obj"]]} if e["obj"] else {})), qin)
    bar_h = 0.09 + 0.14 + len(qlines) * LHC + 0.035 + LHC + 0.02
    rbox(QX0, y, QX1, y + bar_h, "#EEF0F3", "#D5D9DF")
    text(QX0 + 0.08, y + 0.09, "Question", FH, SLATE, weight="bold")
    text(QX1 - 0.08, y + 0.09, f"Right answer: {q['answer']}", FH, TAG_A, ha="right", weight="bold")
    yq = draw_rich(QX0 + 0.08, y + 0.09 + 0.14, qlines, qsp)
    yo = yq + 0.035
    xo = QX0 + 0.08
    for L in sorted(q["options"]):
        good = L == q["answer"]
        t_ = text(xo, yo, f"{L}. {q['options'][L]}", FC, (TAG_A if good else TXT), weight=("bold" if good else "normal"))
        xo += width_in(t_) + 0.22
    t_ = text(QX1 - 0.08, yo, f"ground truth: {fmt_gt(q['gt_payload']['value'], q['gt_payload']['unit'])}", FC, SUB, ha="right")
    assert xo - 0.22 + 0.2 < QX1 - 0.08 - width_in(t_), "选项行与真值相撞"
    y += bar_h + 0.10
    # ---- 下段:两个模型的原样输出(左右并排,等高)
    n_out = max(len(e["out"][k][0]) for k in ("rgb", "ours"))
    ph = 0.09 + 0.14 + n_out * LHM + 0.05
    for (key, _, head), (px0, px1) in zip(MODELS, PX):
        ours = key == "ours"
        rbox(px0, y, px1, y + ph, ("#EEF5FC" if ours else "#F4F4F4"), ("#1F76D6" if ours else "#BDBDBD"), lw=(0.8 if ours else 0.5))
        text(px0 + 0.08, y + 0.09, head, FH, TXT, weight="bold")
        lines, letter = e["out"][key]
        for i, ln in enumerate(lines):
            yl = y + 0.09 + 0.14 + i * LHM
            xl = px0 + 0.10
            m = re.fullmatch(r"(<answer>)([A-Z])(</answer>)", ln)
            if ln in ("<evidence>", "</evidence>"):
                t_ = text(xl, yl, ln, FM, TAG_E, fontfamily=MONO)
            elif m:
                xx = xl
                for s_, col, wt in ((m.group(1), TAG_A, "normal"), (m.group(2), TXT, "bold"), (m.group(3), TAG_A, "normal")):
                    t_ = text(xx, yl, s_, FM, col, weight=wt, fontfamily=MONO)
                    xx += width_in(t_)
                mark(xx + 0.09, yl, letter == q["answer"])
            else:
                t_ = text(xl, yl, ln, FM, TXT, weight=("bold" if ln.startswith(e["keyline"]) else "normal"), fontfamily=MONO)
                assert xl + width_in(t_) <= px1 - 0.06, ("输出行溢出", ln)
    return y + ph + 0.02


def finish(e):
    fig, H, rend = C["fig"], C["H"], C["rend"]
    fig.canvas.draw()
    bbs, oob, outbox = [], [], []
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        ex_ = t.get_window_extent(renderer=rend)
        if ex_.width < 1:
            continue
        x0, x1 = ex_.x0 / fig.dpi, ex_.x1 / fig.dpi
        y0_, y1_ = H - ex_.y1 / fig.dpi, H - ex_.y0 / fig.dpi
        if not (x0 >= -0.001 and x1 <= W + 0.001 and y0_ >= -0.001 and y1_ <= H + 0.001):
            oob.append((t.get_text(), round(x0, 2), round(x1, 2), round(y0_, 2), round(y1_, 2)))
        if not any(bx0 + 0.02 <= x0 and x1 <= bx1 - 0.02 and by0 + 0.005 <= y0_ + 0.012 and y1_ - 0.012 <= by1 - 0.005
                   for bx0, by0, bx1, by1 in C["boxes"]):
            outbox.append((t.get_text(), round(x0, 2), round(x1, 2), round(y0_, 2), round(y1_, 2)))
        bbs.append((x0 + 0.004, y0_ + 0.012, x1 - 0.004, y1_ - 0.012, t.get_text()))
    bad = [(p_[4], q_[4]) for i, p_ in enumerate(bbs) for q_ in bbs[i + 1:]
           if p_[0] < q_[2] and q_[0] < p_[2] and p_[1] < q_[3] and q_[1] < p_[3]]
    if oob or bad or outbox:
        os.makedirs(DBG, exist_ok=True)
        fig.savefig(f"{DBG}/debug_{e['key']}.png", dpi=150)
        raise AssertionError(dict(fig=e["key"], out_of_canvas=oob, out_of_box=outbox, overlaps=bad[:12]))
    base = os.path.join(OUT, f"fig4b_{e['key']}")
    fig.savefig(base + ".pdf")
    fig.savefig(base + ".png", dpi=300)
    plt.close(fig)
    return len(bbs)


# ================================================================ 主流程:每个 example 先量高度,再按实际高度重画一次
for e in check_sources():
    if e["qid"] is None:
        print(f"skip fig4_{e['key']}: 该 clip 在 TrajQA-B 里没有题目(没有两个模型的输出),见蓝图口径 3")
        continue
    h_need = draw_figure(e, 9.0)
    plt.close(C["fig"])
    h = round(h_need + 0.005, 3)
    assert h <= 8.5, (e["key"], "超过一页", h)
    assert abs(draw_figure(e, h) - h) < 0.01
    n = finish(e)
    print(f"wrote fig4b_{e['key']}.pdf/.png  H={h:.2f} in  texts={n}  q={e['qid']}  "
          f"frames-only {e['out']['rgb'][1]} / ours {e['out']['ours'][1]} / key {e['q']['answer']}")
