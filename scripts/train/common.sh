#!/bin/bash
# 三个训练阶段共用的环境(被 stage*.sh source)。全部可用环境变量覆盖。
set -euo pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.."          # repo root
PY=${PY:-python}                                                       # 训练环境(environment/g1train.pip.txt)
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
RUNS_DIR=${RUNS_DIR:-runs}; mkdir -p "$RUNS_DIR"
export QWEN3VL_PATH=${QWEN3VL_PATH:-checkpoints/Qwen3-VL-8B-Instruct}   # Qwen/Qwen3-VL-8B-Instruct 本地目录
export G1_STEREO_TOKENS=${G1_STEREO_TOKENS:-data/tokens_v4td}
export G1_STEREO_FRAMES=${G1_STEREO_FRAMES:-data/stereo4d/frames}
export PYTHONPATH=src/training OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTORCH_ALLOC_CONF=expandable_segments:True
NGPU=$(echo "$GPUS" | tr ',' '\n' | grep -c .)
# 全局批固定 16 = 每卡微批 × 卡数 × 梯度累积
accum_for(){ local mb=$1; local acc=$(( 16 / (mb * NGPU) )); [ $((acc * mb * NGPU)) -eq 16 ] || { echo "全局批 16 不能被 每卡批 $mb × 卡数 $NGPU 整除" >&2; exit 1; }; echo $acc; }
launch(){ CUDA_VISIBLE_DEVICES=$GPUS $PY -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=$NGPU -m g1.train "$@"; }
