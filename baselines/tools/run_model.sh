#!/bin/bash
# 跑一个模型的完整流程:看帧臂 -> 盲测臂 -> 写 RESULTS.md。
#   用法: ./tools/run_model.sh <slug> <hf_repo> <family> <weights_gib> [batch] [extra run_hf args...]
# 两臂都断点续跑(--out 已有 id 自动跳过),中断了重跑同一条命令即可。
# 口径:8 帧 @384 plain greedy max_new=4,与 README §3 一致;batch 只影响速度不影响 greedy 结果。
cd "$(dirname "$0")/.." || exit 1
export HF_HUB_OFFLINE=1
unset all_proxy ALL_PROXY

SLUG="$1"; REPO="$2"; FAM="$3"; W="$4"; BATCH="${5:-32}"; shift 5 2>/dev/null || shift 4
EXTRA=("$@")
PRED="preds/${SLUG}.jsonl"
BLIND="preds/${SLUG}_blind.jsonl"
LOG="logs/run_${SLUG}.log"

echo "################ $SLUG ($REPO) $(date +%F_%H:%M:%S) ################" | tee -a "$LOG"

# ---------- 看帧臂 ----------
T0=$(date +%s)
"${PYBIN:-python}" tools/run_hf.py --model "$REPO" --family "$FAM" --split heldout_test,test \
    --batch "$BATCH" --out "$PRED" "${EXTRA[@]}" >> "$LOG" 2>&1
RC1=$?
WALL=$(( $(date +%s) - T0 ))
echo "[看帧臂] exit=$RC1 wall=${WALL}s 条数=$(wc -l < "$PRED" 2>/dev/null)" | tee -a "$LOG"

# ---------- 盲测臂(NO_BLIND=1 时跳过)----------
# 用户 2026-09-21 决定:只比常规使用下基模的推理能力,不需要盲测地板。
if [ "${NO_BLIND:-0}" = "1" ]; then
  echo "[盲测臂] 按要求跳过" | tee -a "$LOG"
  BLIND=""
else
  "${PYBIN:-python}" tools/run_hf.py --model "$REPO" --family "$FAM" --split heldout_test,test \
      --batch "$BATCH" --no-frames --out "$BLIND" "${EXTRA[@]}" >> "$LOG" 2>&1
  RC2=$?
  echo "[盲测臂] exit=$RC2 条数=$(wc -l < "$BLIND" 2>/dev/null)" | tee -a "$LOG"
fi

# ---------- 出表 ----------
CFG="8帧@384 plain b=${BATCH} ${ENGINE_TAG:-transformers}"
[ ${#EXTRA[@]} -gt 0 ] && CFG="$CFG ${EXTRA[*]}"
BLIND_ARG=()
[ -n "$BLIND" ] && BLIND_ARG=(--blind "$BLIND")
python tools/report.py --model "$SLUG" --weights "$W" --config "$CFG" \
    --pred "$PRED" "${BLIND_ARG[@]}" --wall "$WALL" --engine "${ENGINE_TAG:-transformers 5.17.0}" 2>&1 | tee -a "$LOG"

echo "################ $SLUG DONE $(date +%F_%H:%M:%S) ################" | tee -a "$LOG"
