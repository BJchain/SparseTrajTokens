#!/bin/bash
# 场景成品(data/scenes/scene_*/final_repr.pkl)→ 轨迹 token → Stage 1 运动描述(T8 caption)。
#   tokens_v4td:训练 / QA 用的 token(三次 B 样条节点 + 事件切分 + 刚体联合拟合 + 相机伪物体 --camera)= 发布的 data/tokens_v4td
#   tokens_nocam:同一 tokenizer 但不加相机伪物体,只用来生成 caption —— caption 生成器按 token 的 kind 判断"在动的物体",
#                 相机伪物体的 kind 是 rigid,混进去会被当成运动物体写进描述。
#   发布的 data/captions/t8_train.jsonl 是 2026-09-09 用当时的无相机 token(v4t,刚体参数化修复之前)生成的;
#   caption 只用到物体的 kind / label 与 final_repr 的关键点轨迹,两版在这些字段上一致。逐字节复现请以发布文件为准。
# 用法:bash scripts/data/build_tokens_and_captions.sh [OUT=outputs]
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.."
PY=${PY:-python}; O=${1:-outputs}
$PY src/tokenization/build_tokens.py --root data/scenes --out $O/tokens_v4td --camera
$PY src/tokenization/build_tokens.py --root data/scenes --out $O/tokens_nocam
PYTHONPATH=src/training $PY -m g1.t8_gen --batch-root data/scenes --tokens $O/tokens_nocam --out $O/captions
