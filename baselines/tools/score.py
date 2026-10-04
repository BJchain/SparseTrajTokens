#!/usr/bin/env python3
"""TrajQA-B 评分。

    python tools/score.py --pred preds/qwen25vl7b_heldout.jsonl [--blind preds/..._blind.jsonl]

pred jsonl 每行: {"id":..., "pred":"A"}  (多余字段忽略;pred 不在 ABCD 记为错)
输出: 总体 / 逐 split / 逐族 + **数值三族合并**(论文主指标) + 字母分布 + 可选"净增益"。

为什么要盲测(--blind):B 集所有数字都要减掉**该模型自己的**盲测地板才可比 ——
不同模型的选项先验不同,共用一个 25% 地板会系统性偏袒先验强的模型(见 TRAJQA_B_SPEC §17.6)。
"""
import argparse, collections, json, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import LETTERS, NUMERIC3

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_preds(p):
    out = {}
    for line in open(p):
        line = line.strip()
        if line:
            d = json.loads(line)
            out[d["id"]] = (d.get("pred") or "?").strip().upper()[:1]
    return out


def tally(preds, gold):
    rows = collections.defaultdict(lambda: [0, 0])          # key -> [correct, n]
    letters = collections.Counter()
    for qid, p in preds.items():
        g = gold.get(qid)
        if g is None:
            continue
        ok = int(p == g["answer"])
        letters[p if p in LETTERS else "?"] += 1
        for k in ("ALL", f"split:{g['split']}", f"fam:{g['family']}",
                  f"{g['split']}/{g['family']}"):
            rows[k][0] += ok; rows[k][1] += 1
        if g["family"] in NUMERIC3:
            for k in ("NUM3", f"NUM3:{g['split']}"):
                rows[k][0] += ok; rows[k][1] += 1
    return rows, letters


def acc(r):
    return 100.0 * r[0] / r[1] if r[1] else float("nan")


def se(r):
    a = r[0] / r[1] if r[1] else 0
    return 100.0 * math.sqrt(max(a * (1 - a), 1e-9) / r[1]) if r[1] else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--blind", default=None, help="同模型 --no-frames 跑出的预测,用于算净增益")
    ap.add_argument("--pkg", default=PKG)
    a = ap.parse_args()

    gold = {d["id"]: d for d in map(json.loads, open(os.path.join(a.pkg, "qa", "answers.jsonl")))}
    preds = load_preds(a.pred)
    rows, letters = tally(preds, gold)
    miss = len(preds) - rows["ALL"][1]
    print(f"# {a.pred}\n评分 {rows['ALL'][1]} 题" + (f"(有 {miss} 条 id 对不上,已忽略)" if miss else ""))

    print(f"\n{'键':34s} {'n':>6s} {'Acc%':>7s} {'±SE':>5s}")
    def show(k, label=None):
        if rows[k][1]:
            print(f"{(label or k):34s} {rows[k][1]:6d} {acc(rows[k]):7.1f} {se(rows[k]):5.2f}")
    show("ALL", "总体")
    print("-" * 56)
    for sp in ("heldout_test", "test", "val", "train"):
        show(f"split:{sp}", f"  split={sp}")
    print("-" * 56)
    print("数值三族合并(cam_disp/disp/speed = 论文主指标)")
    show("NUM3", "  全部 split")
    for sp in ("heldout_test", "test", "val", "train"):
        show(f"NUM3:{sp}", f"    {sp}")
    print("-" * 56)
    for f in sorted({k.split(":", 1)[1] for k in rows if k.startswith("fam:")}):
        show(f"fam:{f}", f"  {f}")
    print(f"\n预测字母分布: {dict(letters)}")

    if a.blind:
        brows, _ = tally(load_preds(a.blind), gold)
        print(f"\n净增益(减该模型自己的盲测地板;盲测 = 同 prompt 不给帧)")
        print(f"{'键':34s} {'看帧':>7s} {'盲测':>7s} {'净增':>7s}")
        for k, lab in [("NUM3", "数值三族合并")] + \
                      [(f"NUM3:{s}", f"  {s}") for s in ("heldout_test", "test", "val")] + \
                      [(f"fam:{f}", f"  {f}") for f in NUMERIC3]:
            if rows[k][1] and brows[k][1]:
                print(f"{lab:34s} {acc(rows[k]):7.1f} {acc(brows[k]):7.1f} {acc(rows[k])-acc(brows[k]):+7.1f}")


if __name__ == "__main__":
    main()
