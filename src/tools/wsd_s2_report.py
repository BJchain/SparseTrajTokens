"""阶段二(1b 联合 RGB+traj)WSD 步数寻优的汇总表(2026-09-22)。
用法:python src/tools/wsd_s2_report.py [run_tag ...]      默认 pB_wsd_s2_s0 及其尾巴(pB_wsd_s2_s0_tail*)
列(口径同 SPEC 第 16 节;全部 val,Acc = 数值三族合并):
  trainL / L_val / digΔ   训练 loss(500 步窗口)/ 内联验证 L_legacy(带帧,teacher forcing)/ 内联数字 token NLL 的 donor gap
  联合 = g1.eval --mode traj;donor = eval_permute --corrupt donor --mode traj(带帧口径);Δdonor = 联合 − donor
  只token = g1.eval --mode traj_only;保持度 = 只token − 阶段一起点(81.4)
  E_obj / E_cam = 联合口径下 evidence 位置误差中位(米);val_loss = meta.json 里训练结束时的旧式 [val](尾巴用)
旧阶段二(SPEC 16,旧阶段一起 3,000 步 cosine):联合 74.0 / 只 token 73.7 / Δdonor +45.8 / 保持度 +2.6 / E_obj 0.30 / E_cam 0.10 / val loss 0.1415
"""
from __future__ import annotations
import glob, json, os, re, sys

START = 81.4          # 阶段一起点(pB_wsd_tok_s0_tail7k_d20,val 数值三族)
STEPS_PER_EPOCH = 3507


def load_eval():
    ev = {}
    p = os.environ.get("EVAL_CURVE", "results/eval/selection_val/1b_s2_wsd.jsonl")
    if os.path.exists(p):
        for line in open(p):
            d = json.loads(line); ev[d["tag"]] = d
    return ev


def acc(d):
    return d["acc_numeric_pooled"][1] * 100 if d and d.get("acc_numeric_pooled") else None


def table(tag, ev):
    run = os.path.join(os.environ.get("RUNS_DIR", "results/training_logs"), tag)
    vc, tl = {}, {}
    for fn, dst in (("val_curve.jsonl", vc), ("train_log.jsonl", tl)):
        p = os.path.join(run, fn)
        if os.path.exists(p):
            for line in open(p):
                d = json.loads(line); dst[d["step"]] = d
    meta = json.load(open(os.path.join(run, "meta.json"))) if os.path.exists(os.path.join(run, "meta.json")) else {}
    steps = set(vc)
    for t in ev:
        m = re.match(re.escape(tag) + r"_(?:step|final)(\d+)_traj$", t)
        if m:
            steps.add(int(m.group(1)))
    f = lambda x, n=1: "  -  " if x is None else f"{x:.{n}f}"
    rows = []
    for s in sorted(steps):
        g = lambda kind: ev.get(f"{tag}_step{s}_{kind}") or ev.get(f"{tag}_final{s}_{kind}")
        a, b, c = g("traj"), g("donor"), g("trajonly")
        ja, da, ta = acc(a), acc(b), acc(c)
        win = [tl[k]["loss"] for k in tl if s - 500 < k <= s]
        v = vc.get(s)
        is_final = meta.get("opt_steps_done") == s
        rows.append(dict(step=s, epoch=f"{s / STEPS_PER_EPOCH:.2f}", trainL=f(sum(win) / len(win) if win else None, 4),
                         L_val=f(v["clean"]["L_legacy"] if v else (meta.get("val_loss") if is_final else None), 4),
                         digΔ=f(v.get("gap", {}).get("dig_nll/all") if v else None, 3),
                         联合=f(ja), donor=f(da), Δdonor=f(None if ja is None or da is None else ja - da),
                         只token=f(ta), 保持度=f(None if ta is None else ta - START),
                         cam_disp_speed=("/".join(f(a["acc"][k][1] * 100) for k in ("cam_disp_grid", "disp_grid", "speed_grid")) if a else "  -  "),
                         E_obj=f(a.get("pos_err_med_obj") if a else None, 2), E_cam=f(a.get("pos_err_med_cam") if a else None, 2)))
    print(f"\n=== {tag}" + (f"(训练结束 {meta.get('opt_steps_done')} 步,val loss {meta.get('val_loss')})" if meta else ""))
    if not rows:
        print("(还没有数据)"); return
    cols = list(rows[0].keys()); w = {k: max(len(k), max(len(str(r[k])) for r in rows)) for k in cols}
    print("  ".join(k.rjust(w[k]) for k in cols))
    for r in rows:
        print("  ".join(str(r[k]).rjust(w[k]) for k in cols))


def main():
    tags = sys.argv[1:] or (["pB_wsd_s2_s0"] + sorted(os.path.basename(p) for p in glob.glob(os.path.join(os.environ.get("RUNS_DIR", "results/training_logs"), "pB_wsd_s2_s0_tail*"))))
    ev = load_eval()
    for t in tags:
        table(t, ev)


if __name__ == "__main__":
    main()
