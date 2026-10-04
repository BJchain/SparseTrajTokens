#!/bin/bash
# 38B 档,Q1 原生口径(用户 2026-09-22 决定)。
#
# 关键约束:
#  1) InternVL 家族必须用 tf4_env(transformers 4.51.3)。系统的 5.17 会在
#     configuration_internvl_chat.py 上报 'InternVLChatConfig' object has no attribute 'llm_config'。
#  2) 若权重里带 sentencepiece 的 tokenizer.model,要先跑 fix_sp_tokenizer.py
#     修掉那个含 NUL 的 piece(sentencepiece>=0.2.0 拒载;Python 3.12 没有 0.1.99 轮子)。
#     InternVL2.5-38B / 3.5-38B 的 LLM 是 Qwen 系,可能走 tokenizer.json 不需要修 —— 脚本自动判断。
#  3) Q1 帧数取 **32**,这正是 InternVL 官方 load_video 的默认 num_segments=32。
#     本数据集 clip 只有 20~38 帧,32 段≈全帧,同时把 InternViT-6B 的前向峰值压住
#     (71.5 GiB 权重 + 38 帧全过视觉塔有 OOM 风险,单卡 95.6 GiB 不宽裕)。
#  4) batch=1。GPU_BUDGET:38B b=1 = 76.4 GiB,32帧 b=4 = 89.4 GiB;单卡留足余量。
#  5) 跑满 7,922 才删权重,腾给下一个。
BASELINE_ROOT=${BASELINE_ROOT:-$HOME/baseline_workspace}   # 放模型权重 models/、各基模代码 code/、独立环境 *_env/ 的工作目录
cd "$(dirname "$0")/.." || exit 1
export HF_HUB_OFFLINE=1
unset all_proxy ALL_PROXY
MR=${BASELINE_ROOT}/models
TF4=${BASELINE_ROOT}/tf4_env/bin/python
PLOG=logs/pipeline_38b.log

run_q1_internvl () {
  SLUG="$1"; DIR="$2"; W="$3"
  PRED="preds/Q1_${SLUG}.jsonl"
  LOG="logs/run_Q1_${SLUG}.log"

  # --- tokenizer 体检 ---
  if [ -f "$MR/$DIR/tokenizer.model" ]; then
    echo "=== [$(date +%F_%H:%M:%S)] $SLUG 带 sentencepiece tokenizer,跑 NUL piece 修复 ===" >> "$PLOG"
    python tools/fix_sp_tokenizer.py "$MR/$DIR" >> "$PLOG" 2>&1
  else
    echo "=== [$(date +%F_%H:%M:%S)] $SLUG 无 tokenizer.model(走 fast tokenizer),无需修复 ===" >> "$PLOG"
  fi

  echo "=== [$(date +%F_%H:%M:%S)] Q1 开跑 $SLUG (internvl, 32 帧@448 官方口径, b=1, tf4_env) ===" >> "$PLOG"
  T0=$(date +%s)
  "$TF4" tools/run_native.py --model "$MR/$DIR" --family internvl \
      --split heldout_test,test --n-frames 32 --batch 1 --out "$PRED" >> "$LOG" 2>&1
  WALL=$(( $(date +%s) - T0 ))
  N=$(wc -l < "$PRED" 2>/dev/null || echo 0)
  echo "=== [$(date +%F_%H:%M:%S)] Q1 $SLUG 结束 $N/7922 墙钟=${WALL}s ===" >> "$PLOG"

  if [ "$N" -ge 7922 ]; then
    python tools/report.py --model "${SLUG} [Q1原生]" --weights "$W" \
        --config "Q1 32帧@448 internvl(官方num_segments=32,每帧1tile) 给时间 b=1" \
        --pred "$PRED" --wall "$WALL" --engine "transformers 4.51.3 (tf4_env)" >> "$PLOG" 2>&1
    rm -rf "$MR/$DIR"
    echo "=== [$(date +%F_%H:%M:%S)] $SLUG 已入表并删权重,剩 $(df -h ${BASELINE_ROOT}|tail -1|awk '{print $4}') ===" >> "$PLOG"
  else
    python tools/mark_failed.py --model "${SLUG} [Q1原生]" --situation "不完整" \
        --reason "Q1 只产出 $N/7922 条,见 $LOG;权重保留可续跑" >> "$PLOG" 2>&1
  fi
}

for SPEC in "InternVL2.5-38B:internvl2_5-38b:71.5" "InternVL3.5-38B:internvl3_5-38b:71.5"; do
  IFS=: read -r SLUG DIR W <<< "$SPEC"
  echo "=== [$(date +%F_%H:%M:%S)] 等 $SLUG 下载完 ===" >> "$PLOG"
  while ! grep -q "完成 fail=" "logs/ms_${DIR}.log" 2>/dev/null; do sleep 120; done
  if grep -q "完成 fail=0" "logs/ms_${DIR}.log" 2>/dev/null; then
    run_q1_internvl "$SLUG" "$DIR" "$W"
  else
    echo "!!! [$(date +%F_%H:%M:%S)] $SLUG 下载失败" >> "$PLOG"
    python tools/mark_failed.py --model "${SLUG} [Q1原生]" --situation "未跑" \
        --reason "ModelScope 下载失败,见 logs/ms_${DIR}.log" >> "$PLOG" 2>&1
  fi
done
echo "=== [$(date +%F_%H:%M:%S)] 38B Q1 PIPELINE DONE ===" >> "$PLOG"
