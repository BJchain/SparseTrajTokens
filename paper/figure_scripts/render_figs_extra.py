"""附录追加例子(09-26,老师:多找 baseline 不行、我们成功的例子)的帧图渲染 —— 复用 render_figs.render(),
不动已有 6 个 clip 的输出与 render_summary.json;结果写 <clip>/fig4b_<key>_fXX.png 与 render_summary_extra.json。
选帧 = 题目问到的区间:第 i 帧、中间帧、第 j 帧(轨迹画到当前帧为止)。
用法:<sam3d-objects 或 stereo4d python> paper/figure_scripts/render_figs_extra.py <clip:i:j> ...
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_figs import OUT, render  # noqa: E402

summ_p = os.path.join(OUT, "render_summary_extra.json")
summary = {r["clip"]: r for r in json.load(open(summ_p))} if os.path.exists(summ_p) else {}
for arg in sys.argv[1:]:
    clip, i, j = arg.split(":")
    i, j = int(i), int(j)
    sel = [i, (i + j) // 2, j]
    r = render(f"fig4b_{clip}", clip, sel)
    summary[clip] = r
    print(f"[{clip}] frames={sel} masklets={r['n_masklets_rerun']} moving={r['n_moving']} with_mask={r['n_moving_with_mask']} "
          f"objects={[(o.get('iid'), o.get('label'), o.get('mask_matched')) for o in r['objects'] if 'iid' in o]}", flush=True)
json.dump(list(summary.values()), open(summ_p, "w"), ensure_ascii=False, indent=1)
