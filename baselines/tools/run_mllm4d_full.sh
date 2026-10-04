#!/bin/bash
# MLLM-4D-SFT 单模型全链路(2026-09-25,用户:用 SFT 后的 MLLM-4D 做 Q1,确保官方用法、GPU 正确):
#   CUDA 自检 → Q1 推理 7,922(tools/run_mllm4d.py,断点续跑)→ 跑满才入表(report.py)→ 重画表格图
#   用法: setsid nohup ./tools/run_mllm4d_full.sh > logs/pipeline_mllm4d.log 2>&1 &
# 正式跑之前的核对见 logs/verify_mllm4d.log(tools/verify_mllm4d.py):权重 750/750 逐张量一致、
# flash 内核确实被调用、帧数组喂法与官方"视频文件路径"喂法逐比特一致、冒烟 21/21 单字母。
BASELINE_ROOT=${BASELINE_ROOT:-$HOME/baseline_workspace}   # 放模型权重 models/、各基模代码 code/、独立环境 *_env/ 的工作目录
set -o pipefail
cd "$(dirname "$0")/.." || exit 1
export HF_HUB_OFFLINE=1
unset all_proxy ALL_PROXY
PY=${BASELINE_ROOT}/mllm4d_env/bin/python
SLUG=MLLM-4D-SFT-8B
MODEL=${BASELINE_ROOT}/models/MLLM-4D/MLLM-4D-SFT-1.0
W=16.3
ENGINE="transformers 4.57.0 + torch 2.8.0+cu128 + flash-attn 2.8.3 (mllm4d_env)"
CFG="Q1 官方num_frames=32(整段clip)@512 Qwen3-VL video通道+官方时间戳 b=1 flash_attention_2 greedy max_new=16 官方取字母"
PRED=preds/Q1_${SLUG}.jsonl
LOG=logs/run_Q1_${SLUG}.log
PLOG=logs/pipeline_mllm4d.log
log(){ echo "=== [$(date +%F_%H:%M:%S)] $* ===" | tee -a "$PLOG"; }

log "$SLUG 检查 CUDA"
if ! $PY -c "
import torch
torch.zeros(8, device='cuda'); torch.cuda.synchronize()
f, t = torch.cuda.mem_get_info(0)
print(f'CUDA OK: {torch.cuda.get_device_name(0)} 空闲 {f/2**30:.1f}/{t/2**30:.1f} GiB')
" 2>&1 | tee -a "$PLOG"; then
  log "❌ $SLUG CUDA 不可用,中止"; exit 1
fi

log "$SLUG Q1 开跑:$CFG"
T0=$(date +%s)
$PY tools/run_mllm4d.py --model "$MODEL" --split heldout_test,test --out "$PRED" >> "$LOG" 2>&1
RC=$?
WALL=$(( $(date +%s) - T0 ))
N=$(wc -l < "$PRED" 2>/dev/null || echo 0)
log "$SLUG 推理结束 rc=$RC $N/7922 墙钟=${WALL}s"
grep -m1 "设备自检" "$LOG" | tee -a "$PLOG"
grep -m1 "\[attn\]" "$LOG" | tee -a "$PLOG"

if [ "$N" -lt 7922 ]; then
  log "❌ $SLUG 未跑满"
  python tools/mark_failed.py --model "${SLUG} [Q1原生]" --situation "不完整" \
      --reason "Q1 只产出 $N/7922 条(rc=$RC),见 $LOG;重跑本脚本即可续跑" 2>&1 | tee -a "$PLOG"
  exit 1
fi

(
  flock 9
  python tools/report.py --model "${SLUG} [Q1原生]" --weights "$W" --config "$CFG" \
      --pred "$PRED" --wall "$WALL" --engine "$ENGINE" \
      --note "⚠️ 非零样本:MLLM-4D 的 SFT 数据 MLLM4D-2M 与 TrajQA-B 同源(Stereo4D),评测 clip 大量出现在其训练集中,见 RESULTS §0.992"
  python tools/make_table.py --out figure/trajqa_b_baseline.png
) 9> logs/.results.lock 2>&1 | tee -a "$PLOG"
if grep -qF "| ${SLUG} [Q1原生] |" RESULTS.md; then
  log "$SLUG 已入表 ✓"
else
  log "❌ $SLUG 入表失败"; exit 1
fi
