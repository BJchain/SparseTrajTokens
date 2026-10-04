"""阶段一(1b,token-only)WSD 步数寻优的汇总表(2026-09-22)。
用法:python src/tools/wsd1b_report.py [run_tag ...]      默认 pB_wsd_tok_s0 以及它的 decay 尾巴(pB_wsd_tok_s0_tail*)
两类数据按步数对齐:
  内联验证(训练中每 500 步,teacher forcing,全量 val):L_val = L_legacy;digΔ = 数字 token NLL 的 donor gap
  生成式评测(逐 1k 存档,口径同 SPEC 15/18/19 节,只在 val 上):Acc = 数值三族合并;Δdonor;E_obj / E_cam = 位置误差中位(米)
旧基线(SPEC 15 节,旧 1a + 6k cosine):Acc 71.1 / Δdonor +43.2 / E_obj 0.32 / E_cam 0.10 / val loss 0.148
选点规则见 <run>/PREREG.md:相差 < 2.6 pt 视为打平,取步数最少;Δdonor 低 >5 pt 或 E_obj 高 >25% 否决。
"""
from __future__ import annotations
import glob, json, os, re, sys


def load_eval():
    ev = {}
    p = os.environ.get("EVAL_CURVE", "results/eval/selection_val/1b_wsd.jsonl")
    if os.path.exists(p):
        for line in open(p):
            d = json.loads(line); ev[d["tag"]] = d
    return ev


def table(tag, ev):
    run = os.path.join(os.environ.get("RUNS_DIR", "results/training_logs"), tag)
    vc, tl = {}, {}
    for fn, dst in (("val_curve.jsonl", vc), ("train_log.jsonl", tl)):
        p = os.path.join(run, fn)
        if os.path.exists(p):
            for line in open(p):
                d = json.loads(line); dst[d["step"]] = d
    steps = set(vc)
    for t in ev:
        m = re.match(re.escape(tag) + r"_(?:step|final)(\d+)_val$", t)
        if m:
            steps.add(int(m.group(1)))
    f = lambda x, n=1: "  -  " if x is None else f"{x:.{n}f}"
    rows = []
    for s in sorted(steps):
        a = ev.get(f"{tag}_step{s}_val") or ev.get(f"{tag}_final{s}_val")
        b = ev.get(f"{tag}_step{s}_valdonor") or ev.get(f"{tag}_final{s}_valdonor")
        acc = a["acc_numeric_pooled"][1] * 100 if a and a.get("acc_numeric_pooled") else None
        dac = b["acc_numeric_pooled"][1] * 100 if b and b.get("acc_numeric_pooled") else None
        fam = "/".join(f(a["acc"][k][1] * 100) for k in ("cam_disp_grid", "disp_grid", "speed_grid")) if a else "  -  "
        win = [tl[k]["loss"] for k in tl if s - 500 < k <= s]
        v = vc.get(s)
        rows.append(dict(step=s, epoch=f"{s / 3507:.2f}", trainL=f(sum(win) / len(win) if win else None, 4),
                         L_val=f(v["clean"]["L_legacy"] if v else None, 4),
                         digΔ=f(v.get("gap", {}).get("dig_nll/all") if v else None, 3),
                         Acc=f(acc), donor=f(dac), Δdonor=f(None if acc is None or dac is None else acc - dac),
                         cam_disp_speed=fam, E_obj=f(a.get("pos_err_med_obj") if a else None, 2),
                         E_cam=f(a.get("pos_err_med_cam") if a else None, 2)))
    print(f"\n=== {tag}")
    if not rows:
        print("(还没有数据)"); return
    cols = list(rows[0].keys()); w = {k: max(len(k), max(len(str(r[k])) for r in rows)) for k in cols}
    print("  ".join(k.rjust(w[k]) for k in cols))
    for r in rows:
        print("  ".join(str(r[k]).rjust(w[k]) for k in cols))


def main():
    tags = sys.argv[1:] or (["pB_wsd_tok_s0"] + sorted(os.path.basename(p) for p in glob.glob(os.path.join(os.environ.get("RUNS_DIR", "results/training_logs"), "pB_wsd_tok_s0_tail*"))))
    ev = load_eval()
    for t in tags:
        table(t, ev)


if __name__ == "__main__":
    main()
