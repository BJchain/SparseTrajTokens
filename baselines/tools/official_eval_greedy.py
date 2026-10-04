#!/usr/bin/env python3
"""原样运行 MLLM-4D 官方 evaluation/main.py,唯一改动:把 checkpoint 的 generation_config 改成 greedy
(官方默认采样 T=0.7/top_p 0.8/top_k 20;TrajQA-B 协议是 greedy)。用于阳性对照:greedy 会不会把成绩拉低。
  cd $BASELINE_ROOT/code/MLLM-4D/evaluation && python .../official_eval_greedy.py <main.py 的参数...>
"""
import os, runpy, sys
from transformers import Qwen3VLForConditionalGeneration

_orig = Qwen3VLForConditionalGeneration.from_pretrained


def _greedy(*a, **k):
    m = _orig(*a, **k)
    g = m.generation_config
    g.do_sample, g.temperature, g.top_p, g.top_k = False, None, None, None
    print(f"[greedy patch] generation_config: do_sample={g.do_sample}", flush=True)
    return m


Qwen3VLForConditionalGeneration.from_pretrained = staticmethod(_greedy)
sys.path.insert(0, os.getcwd())          # python main.py 会把脚本目录放进 sys.path,run_path 不会
sys.argv = ["main.py"] + sys.argv[1:]
runpy.run_path("main.py", run_name="__main__")
