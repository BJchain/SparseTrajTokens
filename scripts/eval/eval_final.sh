#!/bin/bash
# 定稿模型在留出集上的四种输入口径(只推理;与论文表 2/表 3 同口径):
#   traj      帧 + 轨迹 token(主结果)          traj_only 只给 token
#   donor     帧 + 换成别的场景的 token(对照)   blind     题面盲测(无帧无 token)
# 用法:bash scripts/eval/eval_final.sh <heldout|dev> [ckpt=checkpoints/stage3_final] [GPUS="0 1 2 3"]
# 结果:results/eval/final/final_{hld,dev}_<mode>.json(.items.jsonl) + 汇总 final_{heldout,dev}.jsonl(数值三族 Acc / 证据误差)
# RGB 对照臂:MODES="rgb blind" CKPT=checkpoints/rgb_only_arm TAG=cmp bash scripts/eval/eval_final.sh heldout
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.."          # repo root
SPLIT=${1:?heldout|dev}; CK=${2:-${CKPT:-checkpoints/stage3_final}}
PY=${PY:-python}; read -r -a G <<< "${GPUS:-0 1 2 3}"; MODES=(${MODES:-traj donor traj_only blind})
export QWEN3VL_PATH=${QWEN3VL_PATH:-checkpoints/Qwen3-VL-8B-Instruct} G1_STEREO_TOKENS=${G1_STEREO_TOKENS:-data/tokens_v4td} \
       G1_STEREO_FRAMES=${G1_STEREO_FRAMES:-data/stereo4d/frames} PYTHONPATH=src/training \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTORCH_ALLOC_CONF=expandable_segments:True
case $SPLIT in heldout) QA=data/trajqa_b/final/heldout_test.jsonl; S=hld; CURVE_NAME=heldout;;
               dev)     QA=data/trajqa_b/final/test.jsonl;         S=dev; CURVE_NAME=dev;; *) echo "split?"; exit 1;; esac
OUTD=${OUTD:-results/eval/final}; mkdir -p "$OUTD"; TAG=${TAG:-final}; CURVE=$OUTD/${TAG}_${CURVE_NAME}.jsonl
[ -f "$CK/traj_encoder.pt" ] || [ "${MODES[*]}" = "rgb blind" ] || { echo "ckpt 不完整:$CK"; exit 1; }
run(){ local g=$1 m=$2 t=${TAG}_${S}_$2; local M=(-m g1.eval --mode $m)
  [ "$m" = donor ] && M=(-m g1.eval_permute --corrupt donor --mode traj)
  [ "$m" = traj_only ] && t=${TAG}_${S}_trajonly
  CUDA_VISIBLE_DEVICES=$g $PY "${M[@]}" --structured --ckpt "$CK" --qa "$QA" --on all --out "$OUTD/$t.json" > "$OUTD/$t.log" 2>&1
  $PY src/tools/evidence_metrics.py "$OUTD/$t.json.items.jsonl" --json "$CURVE" --tag "$t" > /dev/null
  echo "$t acc_overall=$($PY -c "import json;print(json.load(open('$OUTD/$t.json'))['acc_overall'])")"; }
i=0; for m in "${MODES[@]}"; do run "${G[$((i % ${#G[@]}))]}" "$m" & i=$((i + 1)); done; wait
echo "done -> $CURVE"
