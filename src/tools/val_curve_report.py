"""把 g1.train --val-every 写的 val_curve.jsonl 打成一张表(2026-09-20,1a 拟合步数诊断用)。
用法:python src/tools/val_curve_report.py results/training_logs/1a_fit_const_s0 [更多 run 目录...] [--csv out.csv]
列(OM = object_motion,T8 里唯一依赖 token 内容的题型;c/d = clean / donor;gap>0 才说明在读内容):
  L_leg     全验证集 loss(批 2 口径,可与历史 meta.json 的 val_loss 直接比;旧 1a 1,000 步 = 0.147)
  OM_L c/d/gap    object_motion 逐 token loss:正确轨迹 / 换成别的场景的轨迹 / 差
  dig c/d   object_motion 答案里数字 token 的 NLL
  disp/speed/frame  三类数字的首位命中率 clean/donor(teacher forcing;第一个数字的首位与自由生成等价)
  em_disp   位移整个数字全对率 clean/donor
  b16       投影器权重舍入到 bf16 后 OM loss 的变化(1b 载入时就是这么舍入;≈0 说明 fp32 训出来的东西 bf16 接得住)
  dθ/1k     相邻两次验证之间投影器参数的 RMS 位移(×1e-4);plateau 时它仍大 = 在噪声里打转,它也塌 = 真停了
"""
from __future__ import annotations
import argparse, json, os


def load(run):
    p = os.path.join(run, "val_curve.jsonl")
    recs = {}
    if os.path.exists(p):
        for line in open(p):
            line = line.strip()
            if line:
                d = json.loads(line); recs[d["step"]] = d          # 续训重复的步取最后一次
    return [recs[k] for k in sorted(recs)]


def row(d):
    c, dn, g = d["clean"], d.get("donor", {}), d.get("gap", {})
    b = d.get("clean_bf16cast", {}); pm = d.get("param", {})
    f = lambda x, n=3: "  -  " if x is None else f"{x:.{n}f}"
    pair = lambda k, n=3: f"{f(c.get(k), n)}/{f(dn.get(k), n)}"
    b16 = None if "L_tok/object_motion" not in b else b["L_tok/object_motion"] - c["L_tok/object_motion"]
    return dict(step=d["step"], epoch=round(d["step"] / 4033, 2), L_leg=f(c["L_legacy"], 4),   # epoch 按 T8 全量 4,033 步折算
                OM_L=f"{f(c.get('L_tok/object_motion'), 4)}/{f(dn.get('L_tok/object_motion'), 4)}",
                OM_gap=f(g.get("L_tok/object_motion"), 4), inv_L=f(c.get("L_tok/scene_inventory"), 4),
                dig=pair("dig_nll/object_motion"), dig_gap=f(g.get("dig_nll/object_motion"), 3),
                disp=pair("num_disp/first_acc"), speed=pair("num_speed/first_acc"), frame=pair("num_frame/first_acc"),
                em_disp=pair("num_disp/em"), b16=f(b16, 4),
                dth=("  -  " if not pm else f"{pm['d_rms_since_prev'] * 1e4:.2f}"))


def trend(recs, n):
    """最近 n 个整千步验证点的线性趋势(每 1k 步的斜率 ± 标准误,t 值)。比"最好值有没有刷新"稳:单点噪声大时看斜率是否显著 ≠ 0。"""
    import numpy as np
    R = [d for d in recs if d["step"] % 1000 == 0 and d["step"] > 0][-n:]
    if len(R) < 4:
        return
    x = np.array([d["step"] / 1000 for d in R], float)
    series = {"OM_clean_loss": [d["clean"].get("L_tok/object_motion") for d in R],
              "OM_donor_gap": [d.get("gap", {}).get("L_tok/object_motion") for d in R],
              "digit_NLL_gap": [d.get("gap", {}).get("dig_nll/object_motion") for d in R],
              "disp_first_acc_gap": [d.get("gap", {}).get("num_disp/first_acc") for d in R],
              "L_legacy": [d["clean"].get("L_legacy") for d in R]}
    print(f"  趋势(最近 {len(R)} 个点,{int(x[0])}k–{int(x[-1])}k;斜率 = 每 1k 步的变化):")
    for k, y in series.items():
        if any(v is None for v in y):
            continue
        y = np.array(y, float); A = np.vstack([x, np.ones_like(x)]).T
        coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
        dof = len(x) - 2; s2 = float(((y - A @ coef) ** 2).sum()) / max(dof, 1)
        se = (s2 / float(((x - x.mean()) ** 2).sum())) ** 0.5
        t = coef[0] / se if se > 0 else float("nan")
        print(f"    {k:<20} 斜率 {coef[0]:+.5f} ± {se:.5f}  (t={t:+.1f}{'，显著' if abs(t) > 2.3 else '，不显著'})   均值 {y.mean():.4f}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--csv", default="")
    ap.add_argument("--trend", type=int, default=0, help=">0:再给最近 N 个整千步点的线性趋势")
    a = ap.parse_args(); allrows = []
    for run in a.runs:
        recs = load(run)
        print(f"\n=== {run}({len(recs)} 个验证点)")
        if not recs:
            continue
        rows = [row(d) for d in recs]; cols = list(rows[0].keys())
        w = {k: max(len(k), max(len(str(r[k])) for r in rows)) for k in cols}
        print("  ".join(k.rjust(w[k]) for k in cols))
        for r in rows:
            print("  ".join(str(r[k]).rjust(w[k]) for k in cols)); allrows.append(dict(run=os.path.basename(run.rstrip("/")), **r))
        if a.trend:
            trend(recs, a.trend)
    if a.csv and allrows:
        import csv
        with open(a.csv, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(allrows[0].keys())); wr.writeheader(); wr.writerows(allrows)


if __name__ == "__main__":
    main()
