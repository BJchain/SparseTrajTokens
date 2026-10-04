#!/usr/bin/env python3
"""数值三族:把每题 4 个选项按数值排序,看模型选中的位置与真值位置差几格(rank error)。
纯随机时 P(err=0)=25%;按本集真值位置分布算随机基线的 P(|err|≤1)。|err|≤1 明显高于随机 = 有粗略尺度感。
  python3 tools/rank_error.py preds/Q1_MLLM-4D-SFT-8B.jsonl preds/Qwen3-VL-8B.jsonl ...
"""
import collections, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import NUMERIC3, LETTERS
PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
q = {d["id"]: d for d in map(json.loads, open(os.path.join(PKG, "qa", "questions.jsonl")))}
g = {d["id"]: d for d in map(json.loads, open(os.path.join(PKG, "qa", "answers.jsonl")))}
ids = [i for i in g if g[i]["family"] in NUMERIC3 and g[i]["split"] in ("heldout_test", "test")]


def rank(i, L):
    opts = sorted(LETTERS, key=lambda k: float(q[i]["options"][k]))
    return opts.index(L)


gtr = collections.Counter(rank(i, g[i]["answer"]) for i in ids)
rand_le1 = sum(gtr[r] * sum(abs(r - c) <= 1 for c in range(4)) / 4 for r in range(4)) / len(ids)
print(f"真值在(按数值排序的)4 个位置上的分布 {dict(sorted(gtr.items()))};随机基线 P(|err|≤1)={100*rand_le1:.1f}%")
for p in sys.argv[1:]:
    P = {d["id"]: d for d in map(json.loads, open(p))}
    err = collections.Counter(); pick = collections.Counter(); n = 0
    for i in ids:
        L = (P.get(i, {}).get("pred") or "?")[:1]
        if L not in LETTERS:
            continue
        n += 1
        e = rank(i, L) - rank(i, g[i]["answer"]); err[e] += 1; pick[rank(i, L)] += 1
    le1 = sum(v for k, v in err.items() if abs(k) <= 1) / n
    print(f"{os.path.basename(p):32s} n={n}  err=0 {100*err[0]/n:4.1f}%  |err|≤1 {100*le1:4.1f}%  "
          f"偏大(err>0) {100*sum(v for k,v in err.items() if k>0)/n:4.1f}% / 偏小 {100*sum(v for k,v in err.items() if k<0)/n:4.1f}%  "
          f"选中位置(小→大) {[round(100*pick[r]/n,1) for r in range(4)]}")
