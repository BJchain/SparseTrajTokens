#!/usr/bin/env python3
"""消融表(2026-09-25,用户指示):我方四臂各一行,画成与 trajqa_b_baseline.png 同风格的学术表格 PNG(只含表格)。

    python tools/make_ablation.py --out figure/Ablation.png [--md figure/Ablation.md] [--notes figure/Ablation_notes.md]

数字与主表同源:用 make_table.compute() 从 preds/Ours_*.jsonl + qa/answers.jsonl 现算(两个 split 全部 7,922 题)。
四臂 = 输入模态消融:
  (1) Question only        定稿联合模型,不给帧、不给轨迹 token(题面地板)          preds/Ours_TrajRGB-final_blind.jsonl
  (2) RGB frames only      同一微调流程但训练/推理都不给轨迹 token 的 RGB 臂(SPEC 17)  preds/Ours_RGB-arm.jsonl
  (3) RGB + donor tokens   定稿联合模型,帧 + **别的场景**的轨迹 token(格式不变)     preds/Ours_TrajRGB-final_donor.jsonl
  (4) RGB + Traj (full)    定稿联合模型,帧 + 轨迹 token                             preds/Ours_TrajRGB-final.jsonl
备用(写进 md/notes,不进图):只给轨迹 token 不给帧,preds/Ours_TrajRGB-final_trajonly.jsonl。
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_table import PKG, compute, f

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ARMS = [   # (pred slug, 显示名, 模型, 给帧?, 给轨迹 token?)   —— 2026-09-25 用户指示:第 3 行改为 donor 对照,token-only 行移到 notes
    ("Ours_TrajRGB-final_blind",    "Question only",                     "final (joint)", False, False),
    ("Ours_RGB-arm",                "RGB frames only",                   "RGB-only FT",   True,  False),
    ("Ours_TrajRGB-final_donor",    "RGB + donor tokens (control)",      "final (joint)", True,  "donor"),
    ("Ours_TrajRGB-final",          "RGB + trajectory tokens (full)",    "final (joint)", True,  True),
]
CONTROL = ("Ours_TrajRGB-final_trajonly", "Trajectory tokens only (no frames)", "final (joint)", False, True)

NOTES_ITEMS = [
    "Ablation on TrajQA-B (heldout_test + test, 7,922 Q). Main metric = Numeric-3 combined (cam_disp_grid + disp_grid + speed_grid, n = 7,500), "
    "4-way MCQ, chance = 25.0; Heldout / Test = Numeric-3 per split; By family = per-family accuracy over both splits; Non-num. = mean over the four small families.",
    "All four rows are fine-tuned on TrajQA-B train (videos disjoint from both splits). Rows 1, 3, 4 are the SAME final model (projector warm-up -> "
    "token-only stage -> joint RGB+traj stage, cosine 1 epoch) with inputs changed at inference; row 2 is a separately fine-tuned RGB-only arm "
    "(same data and budget family, no trajectory tokens at training or inference; SPEC sec 17). Frames: 8, sampled inside the asked interval; "
    "structured evidence->answer output, greedy.",
    "Row 3 (donor control) = the final model given frames + the trajectory tokens of ANOTHER scene (format, token count, timestamps unchanged): "
    "Numeric-3 {donor_num3} (heldout {donor_held} / test {donor_test}), i.e. at the question-only floor, so the gain of row 4 comes from the CONTENT of the tokens, "
    "not from their format. RGB-only arm's own question-only floor: {rgbblind_num3} (heldout {rgbblind_held} / test {rgbblind_test}).",
    "Not in the figure: trajectory tokens only (final model, no frames) = Numeric-3 {tok_num3} (heldout {tok_held} / test {tok_test}; cam/obj/speed {tok_cam} / {tok_disp} / {tok_speed}); "
    "frames add {frame_gain:+.1f} on top of tokens (row 4 - tokens only); tokens add {token_gain:+.1f} on top of frames (row 4 - row 2). TrajQA-B is built so that the asked "
    "quantities are not recoverable from RGB alone, so these gains characterise the benchmark, not fusion in general.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", default=PKG)
    ap.add_argument("--out", default=os.path.join(PKG, "figure", "Ablation.png"))
    ap.add_argument("--md", default=None); ap.add_argument("--notes", default=None); ap.add_argument("--csv", default=None)
    ap.add_argument("--dpi", type=int, default=200)
    a = ap.parse_args()
    gold = {d["id"]: d for d in map(json.loads, open(os.path.join(a.pkg, "qa", "answers.jsonl")))}
    rows = [(disp, model, fr, tk, compute(slug, gold, a.pkg)) for slug, disp, model, fr, tk in ARMS]
    for disp, model, fr, tk, st in rows:
        assert st and "partial" not in st, f"{disp}: preds 不全"
    tokonly = compute(CONTROL[0], gold, a.pkg)
    donor = rows[2][4]
    rgbblind = compute("Ours_RGB-arm_blind", gold, a.pkg)

    COLS = ["Arm (inputs at inference)", "Model", "Frames", "Traj tok.", "Numeric-3", "Heldout", "Test", "Cam.Disp", "Obj.Disp", "Speed", "Non-num."]
    X = [0.008, 0.262, 0.352, 0.416, 0.494, 0.578, 0.650, 0.736, 0.822, 0.900, 0.972]
    ALIGN = ["left"] + ["center"] * 10
    n_lines = len(rows)
    fig, ax = plt.subplots(figsize=(13.0, 1.05 + 0.30 * n_lines), dpi=a.dpi)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off"); fig.patch.set_facecolor("white")
    top = 0.96; rh = (top - 0.06) / (n_lines + 3.0); FS = 9.3
    def line(y, lw=0.8, x0=0.0, x1=1.0):
        ax.plot([x0, x1], [y, y], lw=lw, color="black", solid_capstyle="butt")
    y = top; line(y, 1.5); y -= rh * 0.85
    ax.text(0.50, y + rh * 0.16, "Ablation on TrajQA-B  (heldout_test + test, 7,922 Q; all arms fine-tuned)",
            ha="center", va="center", fontsize=FS + 1.0, style="italic")
    y -= rh * 0.82
    ax.text((X[2] + X[3]) / 2, y, "Inputs", ha="center", va="center", fontsize=FS - 0.9)
    ax.text(X[4], y, "main, n=7,500", ha="center", va="center", fontsize=FS - 0.9)
    ax.text((X[5] + X[6]) / 2, y, "By split", ha="center", va="center", fontsize=FS - 0.9)
    ax.text((X[7] + X[9]) / 2, y, "By family", ha="center", va="center", fontsize=FS - 0.9)
    for x0, x1 in ((X[2] - 0.030, X[3] + 0.030), (X[4] - 0.038, X[4] + 0.038), (X[5] - 0.034, X[6] + 0.032), (X[7] - 0.036, X[9] + 0.034)):
        line(y - rh * 0.36, 0.6, x0, x1)
    y -= rh * 0.92
    for c, x, al in zip(COLS, X, ALIGN):
        ax.text(x, y, c, ha=al, va="center", fontsize=FS, fontweight="bold")
    y -= rh * 0.42; line(y, 0.9)
    keys = ["num3", "held", "test", "cam", "disp", "speed", "back4"]
    for i, (disp, model, fr, tk, st) in enumerate(rows):
        y -= rh
        if i == len(rows) - 1:
            ax.add_patch(Rectangle((0, y - rh * 0.45), 1, rh * 0.9, facecolor="#eaf0f7", edgecolor="none", zorder=0))
        cells = [disp, model, "Y" if fr else "-", ("donor" if tk == "donor" else ("Y" if tk else "-"))] + [f(st[k]) for k in keys]
        for c, x, al in zip(cells, X, ALIGN):           # 消融表不加粗"最优"(全模型并非每列最高),只用底色标出全模型行
            ax.text(x, y, c, ha=al, va="center", fontsize=FS, zorder=1)
    y -= rh * 0.55; line(y, 1.5)
    ax.set_ylim(y - rh * 0.9, 1)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=a.dpi, bbox_inches="tight", facecolor="white", pad_inches=0.16)
    print(f"✅ {a.out}")

    full, rgb, tok = rows[3][4], rows[1][4], tokonly
    fill = dict(donor_num3=f(donor["num3"]), donor_held=f(donor["held"]), donor_test=f(donor["test"]),
                tok_num3=f(tokonly["num3"]), tok_held=f(tokonly["held"]), tok_test=f(tokonly["test"]), tok_cam=f(tokonly["cam"]), tok_disp=f(tokonly["disp"]), tok_speed=f(tokonly["speed"]),
                rgbblind_num3=f(rgbblind["num3"]), rgbblind_held=f(rgbblind["held"]), rgbblind_test=f(rgbblind["test"]),
                frame_gain=full["num3"] - tok["num3"], token_gain=full["num3"] - rgb["num3"])
    if a.md:
        md = ["| Arm (inputs at inference) | Model | Frames | Traj tok. | **Numeric-3** | Heldout | Test | Cam.Disp | Obj.Disp | Speed | Non-num. |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for disp, model, fr, tk, st in rows:
            md.append(f"| {disp} | {model} | {'Y' if fr else '-'} | {'donor' if tk == 'donor' else ('Y' if tk else '-')} | **{f(st['num3'])}** | " + " | ".join(f(st[k]) for k in keys[1:]) + " |")
        md.append(f"| *(not in figure) {CONTROL[1]}* | {CONTROL[2]} | - | Y | *{f(tokonly['num3'])}* | " + " | ".join(f"*{f(tokonly[k])}*" for k in keys[1:]) + " |")
        open(a.md, "w").write("\n".join(md) + "\n"); print(f"✅ {a.md}")
    if a.csv:
        import csv
        with open(a.csv, "w", newline="") as fo:
            wr = csv.writer(fo); wr.writerow(["arm", "model", "frames", "traj_tokens", "pred_file", "n_numeric3"] + ["numeric3", "heldout", "test", "cam_disp", "obj_disp", "speed", "non_num"])
            for (slug, disp, model, fr, tk), (*_, st) in zip(ARMS, rows):
                wr.writerow([disp, model, int(fr), ("donor" if tk == "donor" else int(bool(tk))), f"preds/{slug}.jsonl", st["n_num3"]] + [f"{st[k]:.4f}" for k in keys])
            wr.writerow([CONTROL[1], CONTROL[2], 0, 1, f"preds/{CONTROL[0]}.jsonl", tokonly["n_num3"]] + [f"{tokonly[k]:.4f}" for k in keys])
        print(f"✅ {a.csv}")
    if a.notes:
        open(a.notes, "w").write("# Notes for figure/Ablation.png (kept out of the figure)\n\n"
                                 + "\n\n".join("- " + it.format(**fill) for it in NOTES_ITEMS) + "\n")
        print(f"✅ {a.notes}")


if __name__ == "__main__":
    sys.exit(main())
