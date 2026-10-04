#!/usr/bin/env python3
"""阳性对照:本机用官方 evaluation/main.py 跑出的 VLM4D 结果 vs 官方仓库自带的 outputs_demo(作者机器上的输出)。
口径:response 字母对应的选项文本 == answer(与官方 demo 同算法;官方 evaluation.py 用 Qwen3-VL 当裁判,
response 已是单字母时二者等价)。
  python3 tools/compare_official_vlm4d.py <ours.json> [<ours2.json> ...]
"""
import json, math, os, sys
BASELINE_ROOT = os.environ.get("BASELINE_ROOT", os.path.expanduser("~/baseline_workspace"))  # 基模权重/代码/数据所在工作目录
DEMO = f"{BASELINE_ROOT}/scratchpad/MLLM-4D/evaluation/outputs_demo/real_mc/MLLM-4D-SFT-1.0.json"


def key(x):
    return next((k for k, v in x["choices"].items() if v == x["answer"]), None)


def acc(d):
    ok = sum(1 for x in d if x["response"] and x["response"] == key(x))
    return ok, len(d)


demo = {x["id"]: x for x in json.load(open(DEMO))}
c, n = acc(demo.values())
print(f"官方 demo(作者机器,采样解码):{100*c/n:.2f}% ({c}/{n}),空回答 {sum(1 for x in demo.values() if not x['response'])}")
for p in sys.argv[1:]:
    d = json.load(open(p))
    c2, n2 = acc(d)
    se = 100 * math.sqrt((c2 / n2) * (1 - c2 / n2) / n2)
    agree = sum(1 for x in d if x["id"] in demo and x["response"] == demo[x["id"]]["response"])
    both = [x for x in d if x["id"] in demo]
    b = sum(1 for x in both if x["response"] == key(x) and demo[x["id"]]["response"] != key(demo[x["id"]]))
    cc = sum(1 for x in both if x["response"] != key(x) and demo[x["id"]]["response"] == key(demo[x["id"]]))
    print(f"{p}:\n   本机 {100*c2/n2:.2f}% ±{se:.1f} ({c2}/{n2}),空回答 {sum(1 for x in d if not x['response'])};"
          f"与官方 demo 逐题同答 {100*agree/len(both):.1f}%;本机对 demo 错 {b} / 本机错 demo 对 {cc}")
