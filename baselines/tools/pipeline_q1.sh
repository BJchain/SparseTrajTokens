#!/bin/bash
# Q1「能力上限」流水线(用户 2026-09-22 决定):
#   - 已跑完/在跑的模型维持 Q2 原样,不动
#   - Qwen3-VL-32B:Q2 跑完后**保留权重**,再按官方原生配置跑一遍 Q1(桥接模型,可做 Q1/Q2 配对对比)
#   - 从 Qwen2.5-VL-32B 起,直接只跑 Q1
#
# Q1 = 整段 clip 全帧 + 源分辨率 512 不降采样 + 各家族官方视频通路 + 时间信息给。
# 不跑盲测臂。
BASELINE_ROOT=${BASELINE_ROOT:-$HOME/baseline_workspace}   # 放模型权重 models/、各基模代码 code/、独立环境 *_env/ 的工作目录
cd "$(dirname "$0")/.." || exit 1
export HF_HUB_OFFLINE=1
unset all_proxy ALL_PROXY
MR=${BASELINE_ROOT}/models
PLOG=logs/pipeline_q1.log

run_q1 () {
  SLUG="$1"; DIR="$2"; FAM="$3"; MODE="$4"; W="$5"; BATCH="${6:-2}"
  PRED="preds/Q1_${SLUG}.jsonl"
  LOG="logs/run_Q1_${SLUG}.log"
  echo "=== [$(date +%F_%H:%M:%S)] Q1 开跑 $SLUG (family=$FAM mode=$MODE b=$BATCH) ===" >> "$PLOG"
  T0=$(date +%s)
  python tools/run_native.py --model "$MR/$DIR" --family "$FAM" --mode "$MODE" \
      --split heldout_test,test --batch "$BATCH" --out "$PRED" >> "$LOG" 2>&1
  WALL=$(( $(date +%s) - T0 ))
  N=$(wc -l < "$PRED" 2>/dev/null || echo 0)
  echo "=== [$(date +%F_%H:%M:%S)] Q1 $SLUG 结束 $N/7922 墙钟=${WALL}s ===" >> "$PLOG"
  if [ "$N" -ge 7922 ]; then
    python tools/report.py --model "${SLUG} [Q1原生]" --weights "$W" \
        --config "Q1 全帧@512 ${FAM}/${MODE} 给时间 b=${BATCH}" \
        --pred "$PRED" --wall "$WALL" --engine "transformers 5.17.0" >> "$PLOG" 2>&1
    echo "=== [$(date +%F_%H:%M:%S)] Q1 $SLUG 已入表 ===" >> "$PLOG"
  else
    python tools/mark_failed.py --model "${SLUG} [Q1原生]" --situation "不完整" \
        --reason "Q1 只产出 $N/7922 条,见 $LOG" >> "$PLOG" 2>&1
  fi
}

# ---- 1) Qwen3-VL-32B:等它的 Q2 跑完(权重已硬链接到 _keep,不怕被删)----
echo "=== [$(date +%F_%H:%M:%S)] 等 Qwen3-VL-32B 的 Q2 收尾 ===" >> "$PLOG"
while [ "$(wc -l < preds/Qwen3-VL-32B.jsonl 2>/dev/null || echo 0)" -lt 7922 ]; do sleep 30; done
while pgrep -f "run_hf.p[y] .*qwen3vl-32b" > /dev/null; do sleep 30; done
echo "=== [$(date +%F_%H:%M:%S)] Q2 完成,开始 Q1 ===" >> "$PLOG"
run_q1 "Qwen3-VL-32B" "qwen3vl-32b_keep" "qwen" "video" "62.1" 2

# ---- 2) Qwen2.5-VL-32B:只跑 Q1 ----
echo "=== [$(date +%F_%H:%M:%S)] 等 Qwen2.5-VL-32B 下载完 ===" >> "$PLOG"
while ! grep -q "完成 fail=" "logs/ms_qwen25vl-32b.log" 2>/dev/null; do sleep 60; done
if grep -q "完成 fail=0" "logs/ms_qwen25vl-32b.log" 2>/dev/null; then
  # 腾盘:Qwen3-VL-32B 的 Q1 已跑完,权重可以删了
  rm -rf "$MR/qwen3vl-32b_keep" "$MR/qwen3vl-32b"
  echo "=== [$(date +%F_%H:%M:%S)] 已删 Qwen3-VL-32B 权重,剩 $(df -h ${BASELINE_ROOT}|tail -1|awk '{print $4}') ===" >> "$PLOG"
  run_q1 "Qwen2.5-VL-32B" "qwen25vl-32b" "qwen" "video" "63.6" 2
  N=$(wc -l < preds/Q1_Qwen2.5-VL-32B.jsonl 2>/dev/null || echo 0)
  [ "$N" -ge 7922 ] && rm -rf "$MR/qwen25vl-32b" && \
    echo "=== [$(date +%F_%H:%M:%S)] 已删 Qwen2.5-VL-32B 权重 ===" >> "$PLOG"
else
  python tools/mark_failed.py --model "Qwen2.5-VL-32B [Q1原生]" --situation "未跑" \
      --reason "ModelScope 下载失败,见 logs/ms_qwen25vl-32b.log" >> "$PLOG" 2>&1
fi
echo "=== [$(date +%F_%H:%M:%S)] Q1 PIPELINE DONE ===" >> "$PLOG"
