#!/usr/bin/env python3
"""MLLM4D-2M(MLLM-4D-SFT 的训练集)数值题的"正确选项在 4 个选项里按数值排第几"统计(论文附录:99.4% 是第二或第三小)。
复现 logs/mllm4d_2m_answer_position.log(原统计在评测服务器上临时算的,这里补成脚本)。
  python3 tools/mllm4d_answer_position.py [--ds data/external/MLLM-4D-Datasets/MLLM4D-2M.json] > logs/mllm4d_2m_answer_position.log
"""
import argparse, collections, json, os, re

BASELINE_ROOT = os.environ.get("BASELINE_ROOT", os.path.expanduser("~/baseline_workspace"))
FAMS = ("Camera_Absolute_Distance", "Object_Absolute_Distance", "Object_Camera_Absolute_Distance")
OPT = re.compile(r"^([A-D])\.\s*([-+]?\d+(?:\.\d+)?)\s*$", re.M)
ANS = re.compile(r"<answer>\s*([A-D])\s*</answer>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default=os.path.join(os.environ.get("MLLM4D_DATASETS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "external", "MLLM-4D-Datasets")), "MLLM4D-2M.json"))
    a = ap.parse_args()
    cnt = {f: collections.Counter() for f in FAMS}; bad = 0
    letters = collections.defaultdict(collections.Counter)          # 全部题型的答案字母分布(选项字母本身是均衡的)
    for d in json.load(open(a.ds)):
        f = d.get("question_type")
        conv = {c["from"]: c["value"] for c in d["conversations"]}
        m = ANS.search(conv.get("assistant", ""))
        if m:
            letters[f][m.group(1)] += 1
        if f not in cnt:
            continue
        opts = dict(OPT.findall(conv.get("user", ""))); m = ANS.search(conv.get("assistant", ""))
        if len(opts) != 4 or not m or m.group(1) not in opts:
            bad += 1; continue
        order = sorted(opts, key=lambda k: float(opts[k]))
        cnt[f][order.index(m.group(1))] += 1
    print(f"无法解析答案 {bad}")
    tot = collections.Counter()
    for f in FAMS:
        n = sum(cnt[f].values()); tot.update(cnt[f])
        print(f"{f:34s} n={n}  真值在 4 个选项中按数值的位置(小→大): " + " ".join(f"{100 * cnt[f][i] / n:.1f}%" for i in range(4)))
    n = sum(tot.values())
    print(f"三类数值题合计 n={n}: " + " ".join(f"{100 * tot[i] / n:.1f}%" for i in range(4)))
    print(f"各题型答案字母分布: {dict((k, dict(v)) for k, v in letters.items())}")


if __name__ == "__main__":
    main()
