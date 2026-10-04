#!/usr/bin/env python3
"""把 RESULTS 里的数字渲染成 baseline表格.png 那种学术表格 (PNG)。

    python tools/make_table.py --out figure/trajqa_b_baseline.png

数字**全部从 preds/*.jsonl + qa/answers.jsonl 现算**,不解析 markdown,保证与 score.py 同源。
本机没有中文字体(fc-list :lang=zh 为空),且参考表本身就是全英文,所以表内一律用英文。
"""
import argparse, collections, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import LETTERS, NUMERIC3

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACK4 = ["motion_duration_binary", "motion_source", "interval_disp", "interval_speed"]

# (Q2 pred 基名, 显示名, 权重GiB, 组)。Q1 的文件名是 Q1_<slug>.jsonl。
# 用户 2026-09-24:表里去掉 Cfg 列 → 每个模型只出一行,有 Q1 用 Q1,否则用 Q2
# (于是 Qwen3-VL-32B 只留 Q1 那行;7–8B 档只有 Q2)。口径细节在 RESULTS.md。
# 用户 2026-09-25:闭源模型单列 "Proprietary Models" 组,放最前(同参考表 baseline表格.png);
# API 按预算只跑 heldout_test / test 各 500 题的子集(RESULTS §0.993),按列照常填、不加标注;Fable 行先占位。
MODELS = [
    ("GPT-6-Astra",         "GPT-6 Astra",          "-",    "proprietary"),
    ("Claude-Fable-5.1",    "Claude Fable 5.1",     "-",    "proprietary"),
    ("qwen25vl7b",          "Qwen2.5-VL-7B",        "15.4", "open8b"),
    ("Qwen3-VL-8B",         "Qwen3-VL-8B",          "16.3", "open8b"),
    ("LLaVA-NeXT-Video-7B", "LLaVA-NeXT-Video-7B",  "13.2", "open8b"),
    ("InternVideo2.5-8B",   "InternVideo2.5-8B",    "15.0", "open8b"),
    ("InternVL2.5-8B",      "InternVL2.5-8B",       "15.0", "open8b"),
    ("InternVL3.5-8B",      "InternVL3.5-8B",       "15.9", "open8b"),
    ("Qwen2.5-VL-32B",      "Qwen2.5-VL-32B",       "63.6", "open32b"),
    ("Qwen3-VL-32B",        "Qwen3-VL-32B",         "62.1", "open32b"),
    ("InternVL2.5-38B",     "InternVL2.5-38B",      "71.5", "open38b"),
    ("InternVL3.5-38B",     "InternVL3.5-38B",      "71.5", "open38b"),
    ("VG-LLM-8B",           "VG-LLM (Qwen2.5-VL-7B)",   "17.2", "3d"),
    ("VLM-3R-7B",           "VLM-3R (LLaVA-Video-7B)",  "17.4", "3d"),
    ("MLLM-4D-SFT-8B",      "MLLM-4D-SFT (Qwen3-VL-8B)", "16.3", "ft4d"),
    # 2026-09-25(用户指示):我方两臂不再用写死的锚点,改为与 baseline 同一套 compute() 从 preds/Ours_*.jsonl 现算(两个 split 全部 7,922 题)。
    # 口径 = OURS(见 preds/Ours_*.meta.json):帧从题目问的区间内均匀采 8 帧 + 结构化 evidence→answer 输出;微调过,非零样本,不参与加粗。
    ("Ours_TrajRGB-final",  "Ours: Traj+RGB (FT)",          "17.9", "ours"),   # 2026-09-25 用户指示:只展示联合臂;RGB 臂的 preds 仍在 preds/Ours_RGB-arm*.jsonl
]
GROUP_TITLE = {
    "proprietary": "Proprietary Models",
    "open8b":  "Open-Source Base Models, 7-8B  (zero-shot)",
    "open32b": "Open-Source Base Models, 32B  (zero-shot)",
    "open38b": "Open-Source Base Models, 38B  (zero-shot)",
    "3d":      "3D Spatial Reasoning Models  (zero-shot)",
    "ft4d":    "4D Spatio-Temporal SFT Model  (NOT zero-shot: trained on Stereo4D QA; 85% of eval clips in its SFT data)",
    "ours":    "Ours  (FINE-TUNED on TrajQA-B train, not zero-shot)",
}
NOT_ZERO_SHOT = {"ft4d", "ours"}  # 不参与"零样本最优"加粗
SUBSET_OK = {"proprietary"}       # 这些组按 pred 里已有的题直接算,不当"还在跑"
# 2026-09-25:MLLM-4D 的 SFT 数据 MLLM4D-2M 与 TrajQA-B 同源(Stereo4D),按 tools/overlap_mllm4d.py 的分档
# (logs/overlap_mllm4d.json)在主行下再拆两行:评测 clip 在其 SFT 训练集里 / YouTube 视频都没出现过。
# 2026-09-25(用户指示):图里 MLLM-4D 只展示主行,不再画"IN / NOT in its SFT data"两行子集(分档数字仍在 RESULTS.md §0.992 / logs/overlap_mllm4d.json)。
SUBROWS = {}
TIER_FILE = os.path.join(PKG, "logs", "overlap_mllm4d.json")
# 2026-09-25 前这里是写死的锚点(RGB 47.8 / 旧联合 73.8,只有 heldout);现已改为从 preds 现算(见 MODELS 末尾两行),留空以兼容 export_table_csv。
OURS = []


def load_preds(p):
    out = {}
    if not p or not os.path.exists(p):
        return out
    for line in open(p):
        line = line.strip()
        if line:
            d = json.loads(line)
            out[d["id"]] = d
    return out


def compute(slug, gold, pkg, keep=None, full=True):
    """keep:只统计这些题 id(子集行用);None = 全部。full=False:子集跑法,不要求 7,922 题。"""
    pred = load_preds(os.path.join(pkg, "preds", f"{slug}.jsonl"))
    blind = load_preds(os.path.join(pkg, "preds", f"{slug}_blind.jsonl"))
    if not pred:
        return None
    if full and len(pred) < 7922:   # 还在跑:残缺数字不进表,免得误导
        return {"partial": len(pred)}

    def tally(P):
        r = collections.defaultdict(lambda: [0, 0])
        for qid, d in P.items():
            g = gold.get(qid)
            if g is None or (keep is not None and qid not in keep):
                continue
            ok = int((d.get("pred") or "?").strip().upper()[:1] == g["answer"])
            fam, sp = g["family"], g["split"]
            r[f"fam:{fam}"][0] += ok; r[f"fam:{fam}"][1] += 1
            if fam in NUMERIC3:
                for k in ("NUM3", f"NUM3:{sp}"):
                    r[k][0] += ok; r[k][1] += 1
            else:
                r["BACK4"][0] += ok; r["BACK4"][1] += 1
        return r

    R, B = tally(pred), tally(blind)
    a = lambda d, k: (100.0 * d[k][0] / d[k][1]) if d[k][1] else float("nan")
    return dict(
        n=sum(v[1] for k, v in R.items() if k.startswith("fam:")), n_num3=R["NUM3"][1],
        num3=a(R, "NUM3"), blind=a(B, "NUM3"),
        held=a(R, "NUM3:heldout_test"), test=a(R, "NUM3:test"),
        cam=a(R, "fam:cam_disp_grid"), disp=a(R, "fam:disp_grid"),
        speed=a(R, "fam:speed_grid"), back4=a(R, "BACK4"),
    )


def f(x, nd=1, plus=False):
    if x is None or x != x:
        return "--"
    return f"{x:+.{nd}f}" if plus else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", default=PKG)
    ap.add_argument("--out", default="figure/trajqa_b_baseline.png")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--notes-in-figure", action="store_true", help="把脚注画回图里(2026-09-25 起默认只保留表)")
    a = ap.parse_args()

    gold = {d["id"]: d for d in map(json.loads,
            open(os.path.join(a.pkg, "qa", "answers.jsonl")))}

    rows = []
    for slug, disp, w, grp in MODELS:
        base = next((b for b in (f"Q1_{slug}", slug)
                     if os.path.exists(os.path.join(a.pkg, "preds", f"{b}.jsonl"))), None)
        rows.append((base or slug, disp, w, grp,
                     compute(base, gold, a.pkg, full=grp not in SUBSET_OK) if base else None))

    # 子集行(MLLM-4D:按其 SFT 训练集是否含该 clip 拆分)
    subrows = {}
    if any(slug.replace("Q1_", "", 1) in SUBROWS for slug, *_ in rows) and os.path.exists(TIER_FILE):
        tier = json.load(open(TIER_FILE))["tier"]
        qclip = {}
        for q in map(json.loads, open(os.path.join(a.pkg, "qa", "questions.jsonl"))):
            v = q["video_id"]
            qclip[q["id"]] = v[len("scene_"):] if v.startswith("scene_") else v
        for slug, disp, w, grp, st in rows:
            key = slug.replace("Q1_", "", 1)
            if key in SUBROWS and st and "partial" not in st:
                subrows[slug] = []
                for t, label in SUBROWS[key]:
                    keep = {qid for qid, c in qclip.items() if tier.get(c) == t}
                    subrows[slug].append((label, compute(slug, gold, a.pkg, keep=keep)))

    # ---- 表格骨架 ----
    COLS = ["Models", "Wt.", "Numeric-3", "Heldout", "Test",
            "Cam.Disp", "Obj.Disp", "Speed", "Non-num."]
    # 列 x 位置(0-1 归一)
    X = [0.008, 0.262, 0.366, 0.474, 0.566, 0.672, 0.775, 0.868, 0.956]
    ALIGN = ["left"] + ["center"] * 8

    body = []
    last_grp = None
    for slug, disp, w, grp, st in rows:
        if grp != last_grp:
            body.append(("GROUP", GROUP_TITLE[grp]))
            last_grp = grp
        body.append(("ROW", (slug, disp, w, st, grp)))
        for label, sst in subrows.get(slug, []):
            body.append(("SUB", (label, sst)))
    for nm, w, sg, bl in OURS:                    # 兼容旧锚点(现为空)
        body.append(("OURS", (nm, w, sg, bl)))

    n_lines = len(body)
    fig_h = 1.55 + 0.236 * n_lines
    fig, ax = plt.subplots(figsize=(11.6, fig_h), dpi=a.dpi)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    fig.patch.set_facecolor("white")

    top = 0.965
    rh = (top - 0.075) / (n_lines + 3.2)          # 行高
    FS = 9.3

    def line(y, lw=0.8, x0=0.0, x1=1.0, c="black"):
        ax.plot([x0, x1], [y, y], lw=lw, color=c, solid_capstyle="butt")

    # ---------- 表头 ----------
    y = top
    line(y, 1.5)                                    # toprule
    y -= rh * 0.85
    ax.text(0.50, y + rh * 0.16, "TrajQA-B  (heldout_test + test, 7,922 Q; zero-shot unless noted)",
            ha="center", va="center", fontsize=FS + 1.0, style="italic")
    y -= rh * 0.82
    # 分组表头
    ax.text(X[2], y, "main, n=7,500", ha="center", va="center", fontsize=FS - 0.9)
    ax.text((X[3] + X[4]) / 2, y, "By split", ha="center", va="center", fontsize=FS - 0.9)
    ax.text((X[5] + X[7]) / 2, y, "By family", ha="center", va="center", fontsize=FS - 0.9)
    # 组下划线
    for x0, x1 in ((X[2] - 0.048, X[2] + 0.048), (X[3] - 0.046, X[4] + 0.044),
                   (X[5] - 0.048, X[7] + 0.044)):
        line(y - rh * 0.36, 0.6, x0, x1)
    y -= rh * 0.92
    for c, x, al in zip(COLS, X, ALIGN):
        ax.text(x, y, c, ha=al, va="center", fontsize=FS, fontweight="bold")
    y -= rh * 0.42
    line(y, 0.9)                                    # midrule

    # ---------- 找每列最优(只在零样本开源+3D 里比) ----------
    zs = [st for slug, d, w, g, st in rows if st and "partial" not in st and g not in NOT_ZERO_SHOT]
    best = {}
    for key in ("num3", "held", "test", "cam", "disp", "speed"):
        vals = [s[key] for s in zs if s and s[key] == s[key]]
        if vals:
            best[key] = max(vals)

    # ---------- 表体 ----------
    for kind, payload in body:
        y -= rh
        if kind == "GROUP":
            ax.add_patch(Rectangle((0, y - rh * 0.40), 1, rh * 0.86,
                                   facecolor="#eaf0f7", edgecolor="none", zorder=0))
            ax.text(0.5, y + rh * 0.03, payload, ha="center", va="center",
                    fontsize=FS - 0.1, style="italic", zorder=1)
            continue
        if kind == "OURS":
            nm, w, sg, bl = payload
            cells = [nm, w, f(sg), f(sg), "--", "--", "--", "--", "--"]
            for c, x, al in zip(cells, X, ALIGN):
                ax.text(x, y, c, ha=al, va="center", fontsize=FS, color="#333333")
            continue
        if kind == "SUB":                            # 子集行:灰色斜体,n 写进标签,不加粗
            label, st = payload
            cells = [f"{label}", f"n={st['n_num3']:,}", f(st["num3"]), f(st["held"]), f(st["test"]),
                     f(st["cam"]), f(st["disp"]), f(st["speed"]), f(st["back4"])]
            for c, x, al in zip(cells, X, ALIGN):
                ax.text(x, y, c, ha=al, va="center", fontsize=FS - 0.8, color="#555555", style="italic")
            continue
        slug, disp, w, st, grp = payload
        if st and "partial" in st:
            ax.text(X[0], y, disp, ha="left", va="center", fontsize=FS)
            ax.text(X[1], y, w, ha="center", va="center", fontsize=FS)
            ax.text((X[2]+X[8])/2, y, f"... running ({st['partial']}/7,922) ...",
                    ha="center", va="center", fontsize=FS, color="#888888", style="italic")
            continue
        if st is None:                               # 还没有 pred(占位行,如 Claude Fable 5.1)
            cells = [disp, w] + ["--"] * 7
            for c, x, al in zip(cells, X, ALIGN):
                ax.text(x, y, c, ha=al, va="center", fontsize=FS,
                        color="#999999" if c == "--" else "black")
            continue
        cells = [disp, w, f(st["num3"]),
                 f(st["held"]), f(st["test"]), f(st["cam"]), f(st["disp"]),
                 f(st["speed"]), f(st["back4"])]
        keys = [None, None, "num3", "held", "test", "cam", "disp", "speed", None]
        for c, x, al, k in zip(cells, X, ALIGN, keys):
            bold = k and k in best and st.get(k) == best[k] and grp not in NOT_ZERO_SHOT
            ax.text(x, y, c, ha=al, va="center", fontsize=FS,
                    fontweight="bold" if bold else "normal")

    y -= rh * 0.55
    line(y, 1.5)                                    # bottomrule

    # ---------- 脚注:2026-09-25 用户指示,图里只保留表;脚注文字保留在 NOTES 里,由 export_table_csv.py --notes 写成 md 备用 ----------
    if a.notes_in_figure:
        ax.text(0.0, y - rh * 0.62, NOTES, ha="left", va="top", fontsize=FS - 1.9,
                color="#333333", linespacing=1.5)
    else:                                            # 只保留表:把画布下沿收到 bottomrule 下面一点,不留脚注的空白
        ax.set_ylim(y - rh * 0.9, 1)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=a.dpi, bbox_inches="tight", facecolor="white", pad_inches=0.16)
    got = sum(1 for r in rows if r[4] and "partial" not in r[4])
    print(f"✅ {a.out}  ({got}/{len(rows)} 个模型有数据)")
    for slug, d, w, g, st in rows:
        if st is None:
            print(f"   n/a  {d}")


NOTES_ITEMS = [   # 每项一个逻辑段落;图里按行画(--notes-in-figure),md 里按段落写(export_table_csv.py --notes)
    ("Main metric = Numeric-3 combined (cam_disp_grid + disp_grid + speed_grid), 4-way MCQ, chance = 25.0.  All numbers are plain accuracy with frames given."),
    ("Frames are sampled uniformly over the WHOLE clip and NEVER cropped to the asked interval; greedy, max_new_tokens=4, first A/B/C/D taken."),
    ("Non-num. = mean over the four small families (n=76-146), reported not interpreted."),
    ("Base models are zero-shot; our arms are fine-tuned on TrajQA-B train (videos disjoint from both splits), so the comparison is\n"
     "protocol-mismatched (same caveat as MLLM4D-Bench Table 1). Ours: same 7,922 Q on both splits; 8 frames sampled INSIDE the asked interval\n"
     "+ trajectory tokens from the 4D scene representation; structured evidence->answer output, greedy. Recipe = projector warm-up -> token-only\n"
     "stage -> joint RGB+traj stage (cosine, 1 epoch). Reference (not shown): the same fine-tuning WITHOUT tokens (RGB-only arm) scores 48.3\n"
     "(heldout 47.8 / test 48.8; preds/Ours_RGB-arm.jsonl)."),
    ("MLLM-4D-SFT: official usage (transformers 4.57, 32 frames via the official Qwen3-VL video processor with timestamps, flash-attn 2;\n"
     "max_new_tokens=16 + official answer parser, every output was a single letter). Its SFT set MLLM4D-2M is built from Stereo4D like\n"
     "TrajQA-B: 2,635 of the 3,103 eval clips are in it (same clip id). Sub-rows split its questions by that (n = Numeric-3 count); \"NOT in\"\n"
     "= the clip's YouTube video never appears in MLLM4D-2M (264 Numeric-3 questions whose video appears only via other clips are in neither).\n"
     "Its 25.0 is a shortcut, not chance guessing: in MLLM4D-2M's 856k numeric QAs the answer is the 2nd/3rd-smallest option 99.4% of the time;\n"
     "on TrajQA-B it picks the 3rd-smallest option in 92.6% of numeric questions, and TrajQA-B's answer positions are balanced (25% each)."),
    ("MLLM-4D-SFT split by whether the eval clip is in its SFT set (Numeric-3): clips IN its SFT data (n=6,332) 24.9 / 24.5 / 25.2 / 25.4 / 24.4 / 24.7 / 68.0;\n"
     "clips NOT in (video never in MLLM4D-2M; n=904) 26.1 / 24.1 / 29.3 / 24.7 / 28.7 / 25.6 / 75.7 (columns as in the table); see RESULTS.md and logs/overlap_mllm4d.json."),
]
NOTES = "\n".join(NOTES_ITEMS)


if __name__ == "__main__":
    sys.exit(main())
