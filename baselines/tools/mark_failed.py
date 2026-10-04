#!/usr/bin/env python3
"""把"未跑 / 降级 / 失败"如实记进 RESULTS.md §4,不编数字。

    python tools/mark_failed.py --model InternVL2.5-8B --situation 失败 \\
        --reason "transformers 5.17 与 remote code 不兼容,见 logs/run_xxx.log"
"""
import argparse, os, sys

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--situation", required=True, help="未跑 / 降级 / 失败")
    ap.add_argument("--reason", required=True)
    ap.add_argument("--results", default=os.path.join(PKG, "RESULTS.md"))
    a = ap.parse_args()

    row = f"| {a.model} | {a.situation} | {a.reason} |"
    t = open(a.results).read()
    head = "## 4. 未跑 / 降级 / 失败记录"
    i = t.index(head)
    j = t.find("\n### ", i + 1)
    if j == -1:
        j = t.find("\n## ", i + 1)
    if j == -1:
        j = len(t)
    block = t[i:j].split("\n")
    # 同一模型已有记录就替换
    hit = next((n for n, l in enumerate(block)
                if l.startswith("|") and l.split("|")[1].strip() == a.model), None)
    if hit is not None:
        block[hit] = row
    else:
        last = max((n for n, l in enumerate(block) if l.startswith("|")), default=None)
        block.insert((last + 1) if last is not None else len(block), row)
    open(a.results, "w").write(t[:i] + "\n".join(block) + t[j:])
    print(f"✅ §4 已记录: {row}")


if __name__ == "__main__":
    sys.exit(main())
