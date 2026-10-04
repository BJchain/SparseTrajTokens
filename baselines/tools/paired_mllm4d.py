#!/usr/bin/env python3
"""MLLM-4D-SFT 与其它模型在同一批数值三族题上的配对比较(McNemar),并按重叠档拆分。

  python3 tools/paired_mllm4d.py [--a preds/Q1_MLLM-4D-SFT-8B.jsonl] [--b preds/Qwen3-VL-8B.jsonl ...]
分档来自 logs/overlap_mllm4d.json(tools/overlap_mllm4d.py)。
"""
import argparse, collections, json, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import NUMERIC3, LETTERS

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(p):
    return {d["id"]: d for d in map(json.loads, open(p))}


def mcnemar(b, c):
    """b = A 对 B 错,c = A 错 B 对;连续性校正的正态近似,双侧 p。"""
    if b + c == 0:
        return 0.0, 1.0
    z = (abs(b - c) - 1) / math.sqrt(b + c)
    return z, math.erfc(z / math.sqrt(2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default=os.path.join(PKG, "preds", "Q1_MLLM-4D-SFT-8B.jsonl"))
    ap.add_argument("--b", nargs="*", default=[os.path.join(PKG, "preds", f) for f in
                    ("Qwen3-VL-8B.jsonl", "Q1_Qwen3-VL-32B.jsonl", "Q1_InternVL2.5-38B.jsonl",
                     "Q1_VLM-3R-7B.jsonl", "Q1_VG-LLM-8B.jsonl")])
    a = ap.parse_args()
    gold = {d["id"]: d for d in map(json.loads, open(os.path.join(PKG, "qa", "answers.jsonl")))}
    q = {d["id"]: d for d in map(json.loads, open(os.path.join(PKG, "qa", "questions.jsonl")))}
    tier = json.load(open(os.path.join(PKG, "logs", "overlap_mllm4d.json")))["tier"]
    clip = lambda v: v[len("scene_"):] if v.startswith("scene_") else v
    A = load(a.a)
    ids = [i for i in A if i in gold and gold[i]["family"] in NUMERIC3]
    ok = lambda P, i: (P.get(i, {}).get("pred") or "?")[:1] == gold[i]["answer"]

    # 字母分布 / 各字母命中率
    let = collections.Counter(A[i]["pred"] for i in ids)
    gt = collections.Counter(gold[i]["answer"] for i in ids)
    hit = {L: sum(ok(A, i) for i in ids if A[i]["pred"] == L) / max(let[L], 1) for L in LETTERS}
    print(f"{os.path.basename(a.a)} 数值三族 {len(ids)} 题:预测字母 {dict(let)} / 真值 {dict(gt)};"
          f"各字母命中率 " + " ".join(f"{L}={100*hit[L]:.1f}" for L in LETTERS))
    raw = collections.Counter(("单字母" if len(A[i]["raw"]) == 1 else "其它") for i in A)
    print(f"全部 {len(A)} 条原始输出形态 {dict(raw)};取字母规则 {dict(collections.Counter(A[i].get('rule') for i in A))}")

    for pb in a.b:
        if not os.path.exists(pb):
            continue
        B = load(pb)
        print(f"\n== vs {os.path.basename(pb)} ==")
        for t in ("ALL", "seen_clip", "seen_video", "unseen"):
            S = [i for i in ids if i in B and (t == "ALL" or tier[clip(q[i]["video_id"])] == t)]
            if not S:
                continue
            ca, cb = sum(ok(A, i) for i in S), sum(ok(B, i) for i in S)
            b = sum(ok(A, i) and not ok(B, i) for i in S)
            c = sum(ok(B, i) and not ok(A, i) for i in S)
            z, p = mcnemar(b, c)
            print(f"  {t:10s} n={len(S):5d}  MLLM-4D {100*ca/len(S):5.1f}  对方 {100*cb/len(S):5.1f}  "
                  f"差 {100*(ca-cb)/len(S):+5.1f}  (A对B错 {b} / A错B对 {c}, z={z:.2f}, p={p:.3g})")


if __name__ == "__main__":
    main()
