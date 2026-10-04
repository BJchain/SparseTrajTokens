#!/bin/bash
# TrajQA-B 从 token 到最终四个划分的完整配方(2026-09-14 → 09-17 实际执行的命令,路径换成本仓库布局)。
#   ① 两个候选池:mult=1(val/test 来源)与 mult=6(train / heldout 来源,333,661 条候选)
#   ② Gate 0:按模型看到的 token 逐题复算答案,不一致即丢
#   ③ build_m1:默认参数构建 train/val/test(val 1,318 / test 4,492 原样进最终集;train 只作 mixmatch 的族占比参照)
#   ④ heldout_test:视频哈希 u∈[0.25,0.30) 波段,单独按 test 口径构建(3,430 题 / 166 视频)
#   ⑤ train:mult=6 池,先整体剔除 heldout 波段再配平,放宽每视频封顶,然后按 build_m1 的族占比重采样(56,120 题)
#   ⑥ 结构校验 + 文字捷径审计
# 用法:bash scripts/data/build_trajqa_b.sh [OUT=outputs/trajqa_b]   (CPU 即可;约 1–2 小时 / 40 进程)
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.."          # repo root
PY=${PY:-python}; W=${WORKERS:-40}; TOK=${TOKENS:-data/tokens_v4td}; O=${1:-outputs/trajqa_b}
export PYTHONPATH=src/training
mkdir -p $O/candidates_m1 $O/candidates_m6 $O/build_m1 $O/final $O/_tmp_train_raw
for m in 1 6; do C=$O/candidates_m$m
  $PY -m g1.trajqa_b_gen --workers $W --batch-root data/scenes --tokens $TOK --per-clip-mult $m --out $C
  $PY src/tools/trajqa_b_gate0.py $C/candidates.jsonl $TOK $C/gate0.jsonl $W
done
$PY -m g1.trajqa_b_build --cand $O/candidates_m1/candidates.jsonl --gate0 $O/candidates_m1/gate0.jsonl --out $O/build_m1
$PY -m g1.trajqa_b_build --cand $O/candidates_m6/candidates.jsonl --gate0 $O/candidates_m6/gate0.jsonl --out $O/final \
    --band 0.25,0.30 --band-name heldout_test --cap-vf 30 --cap-v 120 --fam-cap 3000
$PY -m g1.trajqa_b_build --cand $O/candidates_m6/candidates.jsonl --gate0 $O/candidates_m6/gate0.jsonl --out $O/_tmp_train_raw \
    --splits train --copy-from $O/build_m1 --cap-vf 80 --cap-v 300 --fam-cap 60000 --exclude-band 0.25,0.30
$PY src/tools/trajqa_b_mixmatch.py $O/_tmp_train_raw $O/final --ref $O/build_m1
cp $O/build_m1/val.jsonl $O/build_m1/test.jsonl $O/final/
$PY src/tools/trajqa_b_validate.py $O/final --gate0 $O/candidates_m6/gate0.jsonl
$PY src/tools/trajqa_b_audit.py $O/final $O/final/_audit.json
wc -l $O/final/*.jsonl
