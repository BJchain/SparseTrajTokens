#!/bin/bash
# 通用训练启动(支持多卡 DDP):bash src/tools/train_launch.sh <gpus 如 0,1,2,3> <run_tag> [g1.train 其余参数...]
# 全局批固定 GLOBAL_BATCH(默认 16):每卡微批 BATCH(默认 2),grad_accum = GLOBAL_BATCH/(BATCH×卡数)。多卡走 torch.distributed.run。
# 环境变量:TOKENS(默认 data/tokens_v4td)、BATCH、GLOBAL_BATCH、PY(默认 python)、RUNS_DIR(默认 runs)。
# 输出 $RUNS_DIR/<tag>,日志 $RUNS_DIR/<tag>.log。
set -u; cd "$(dirname "$(readlink -f "$0")")/../.." || exit 1     # repo root
GPUS=$1; TAG=$2; shift 2
PY=${PY:-python}; RUNS_DIR=${RUNS_DIR:-runs}; mkdir -p "$RUNS_DIR"
N=$(echo "$GPUS" | tr ',' '\n' | grep -c .); BATCH=${BATCH:-2}; GB=${GLOBAL_BATCH:-16}; ACC=$(( GB / (BATCH * N) ))
[ $((ACC * BATCH * N)) -eq "$GB" ] || { echo "全局批 $GB 不能被 每卡批 $BATCH × 卡数 $N 整除"; exit 1; }
OUT=$RUNS_DIR/$TAG; LOG=$RUNS_DIR/$TAG.log
[ -d "$OUT" ] && { echo "已存在 $OUT"; exit 1; }
if [ "$N" -gt 1 ]; then LAUNCH="$PY -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=$N -m g1.train"; else LAUNCH="$PY -m g1.train"; fi
setsid nohup env CUDA_VISIBLE_DEVICES=$GPUS OMP_NUM_THREADS=4 G1_STEREO_TOKENS=${TOKENS:-data/tokens_v4td} PYTHONPATH=src/training \
  $LAUNCH "$@" --batch "$BATCH" --grad-accum "$ACC" --out "$OUT" > "$LOG" 2>&1 < /dev/null &
disown; echo "started $TAG on gpus=$GPUS (N=$N, batch=$BATCH, accum=$ACC, global=$GB) $(date +%T) log=$LOG"
