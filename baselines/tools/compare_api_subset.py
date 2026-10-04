#!/usr/bin/env python3
"""API 模型(子集)评分 + 与表里其它模型在**同一批题**上的配对比较(McNemar)。

  python3 tools/compare_api_subset.py [--pred preds/Q1_GPT-6-Astra.jsonl] [--per-split 500] [--seed 0]

子集用 run_api_native.subset() 按 (seed, per_split) 重建,先核对 pred 恰好覆盖这批题;
其它模型只在这批题上现算(同时列出它们的全集数字,看子集是否有代表性)。
McNemar 与 paired_mllm4d.py 同一实现(连续性校正正态近似,双侧)。
"""
import argparse, collections, json, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import LETTERS, NUMERIC3, load_split
from paired_mllm4d import mcnemar
from run_api_native import subset
from make_table import MODELS

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACK4 = ("motion_duration_binary", "motion_source", "interval_disp", "interval_speed")


def load(p):
    return {d["id"]: d for d in map(json.loads, open(p)) if d}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=os.path.join(PKG, "preds", "Q1_GPT-6-Astra.jsonl"))
    ap.add_argument("--name", default=None, help="默认取 make_table.MODELS 里的显示名")
    ap.add_argument("--per-split", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json-out", default=None, help="默认 logs/compare_api_subset_<slug>.json")
    ap.add_argument("--allow-missing", action="store_true", help="子集没跑满时先评已完成部分(会标出缺题)")
    a = ap.parse_args()
    slug = os.path.basename(a.pred).replace(".jsonl", "").replace("Q1_", "", 1)
    a.name = a.name or next((d for s, d, *_ in MODELS if s == slug), slug)
    a.json_out = a.json_out or os.path.join(PKG, "logs", f"compare_api_subset_{slug}.json")

    gold = {d["id"]: d for d in map(json.loads, open(os.path.join(PKG, "qa", "answers.jsonl")))}
    ids = [d["id"] for d in subset(load_split(PKG, "heldout_test,test"), a.per_split, a.seed)]
    A = load(a.pred)
    miss, extra = [i for i in ids if i not in A], [i for i in A if i not in set(ids)]
    print(f"子集 {len(ids)} 题(per_split={a.per_split}, seed={a.seed});pred {len(A)} 条,"
          f"缺 {len(miss)},子集外 {len(extra)}")
    if extra or (miss and not a.allow_missing):
        print("❌ pred 与子集不一致,先补跑/核对再评(缺题时可加 --allow-missing 先评已完成部分)"); return 1
    if miss:
        print(f"⚠️ 只评已完成的 {len(ids) - len(miss)} 题,缺:{miss}")
        ids = [i for i in ids if i in A]

    ok = lambda P, i: (P.get(i, {}).get("pred") or "?")[:1] == gold[i]["answer"]
    fam = lambda i: gold[i]["family"]
    groups = [("数值三族", lambda i: fam(i) in NUMERIC3),
              ("  heldout_test", lambda i: fam(i) in NUMERIC3 and gold[i]["split"] == "heldout_test"),
              ("  test", lambda i: fam(i) in NUMERIC3 and gold[i]["split"] == "test"),
              ("Cam.Disp", lambda i: fam(i) == "cam_disp_grid"),
              ("Obj.Disp", lambda i: fam(i) == "disp_grid"),
              ("Speed", lambda i: fam(i) == "speed_grid"),
              ("Non-num.(后四族)", lambda i: fam(i) in BACK4)]
    out = {"n": len(ids), "missing": miss, "per_split": a.per_split, "seed": a.seed, "groups": {}, "vs": {}}
    print(f"\n{a.name} 在子集上:")
    for name, f in groups:
        S = [i for i in ids if f(i)]
        c = sum(ok(A, i) for i in S); p = c / len(S)
        se = 100 * math.sqrt(p * (1 - p) / len(S))
        out["groups"][name.strip()] = {"acc": round(100 * p, 2), "se": round(se, 2), "n": len(S)}
        print(f"  {name:18s} {100*p:5.1f} ±{se:.1f}  (n={len(S)})")

    num = [i for i in ids if fam(i) in NUMERIC3]
    let = collections.Counter(A[i]["pred"] for i in num)
    gt = collections.Counter(gold[i]["answer"] for i in num)
    hit = {L: 100 * sum(ok(A, i) for i in num if A[i]["pred"] == L) / max(let[L], 1) for L in LETTERS}
    print(f"  数值三族预测字母 {dict(sorted(let.items()))} / 真值 {dict(sorted(gt.items()))};各字母命中率 "
          + " ".join(f"{L}={hit[L]:.1f}" for L in LETTERS))
    out["letters"], out["gt_letters"] = dict(let), dict(gt)

    print(f"\n同一批 {len(num)} 道数值三族题上的配对比较(A = {a.name}):")
    print(f"  {'模型':34s} {'子集':>6s} {'全集':>6s} {'差(A-它)':>9s}  A对它错/A错它对   p")
    for slug, disp, _, grp in MODELS:
        q1 = os.path.join(PKG, "preds", f"Q1_{slug}.jsonl")
        p = q1 if os.path.exists(q1) else os.path.join(PKG, "preds", f"{slug}.jsonl")
        if not os.path.exists(p) or os.path.abspath(p) == os.path.abspath(a.pred):
            continue                                # 没有 pred / 就是被比的模型自己
        B = load(p)
        S = [i for i in num if i in B]              # 共同题(另一个 API 模型也可能只跑了子集的一部分)
        if not S:
            continue
        ca, cb = sum(ok(A, i) for i in S), sum(ok(B, i) for i in S)
        full = [i for i in B if i in gold and gold[i]["family"] in NUMERIC3]
        cfull = 100 * sum(ok(B, i) for i in full) / len(full)
        b = sum(ok(A, i) and not ok(B, i) for i in S)
        c = sum(ok(B, i) and not ok(A, i) for i in S)
        z, pv = mcnemar(b, c)
        tag = ("Q1" if p == q1 else "Q2") + (",非零样本" if grp == "ft4d" else "")
        fcell = f"{cfull:6.1f}" if len(full) >= 7500 else f"{'—':>6s}"   # 只跑了子集的模型没有全集数字
        print(f"  {disp + ' [' + tag + ']':34s} {100*cb/len(S):6.1f} {fcell} {100*(ca-cb)/len(S):+9.1f}  "
              f"{b:5d} / {c:<5d}  {pv:.2g}" + ("" if len(S) == len(num) else f"  (共同 n={len(S)})"))
        out["vs"][disp] = {"cfg": tag, "n": len(S), "acc_subset": round(100 * cb / len(S), 2),
                           "acc_full": round(cfull, 2) if len(full) >= 7500 else None,
                           "b": b, "c": c, "p": pv}
    json.dump(out, open(a.json_out, "w"), ensure_ascii=False, indent=1)
    print(f"\n-> {a.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
