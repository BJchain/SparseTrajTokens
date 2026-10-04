#!/bin/bash
# 3D 组单模型全链路(2026-09-24,用户:冒烟通过后正式跑,显存允许可两个同时跑):
#   CUDA 自检 → Q1 推理 7,922(tools/run_3d.py,断点续跑)→ 跑满才入表(report.py)→ 重画表格图
#   用法: setsid nohup ./tools/run_3d_full.sh VG-LLM-8B > logs/pipeline_3d_VG-LLM-8B.log 2>&1 &
#         setsid nohup ./tools/run_3d_full.sh VLM-3R-7B > logs/pipeline_3d_VLM-3R-7B.log 2>&1 &
# 冒烟(21 题,7 族各 3)实测峰值显存 VG-LLM 27.2 GiB / VLM-3R 23.3 GiB → 两个同时跑放得下。
# 硬校验同 resume_38b.sh:加载后参数 100% 在 GPU(run_3d.py 自检)、必须 7,922/7,922 才入表,
# 否则 mark_failed 记 §4、不编数字。两个进程可能先后写 RESULTS.md → 入表段加 flock。
BASELINE_ROOT=${BASELINE_ROOT:-$HOME/baseline_workspace}   # 放模型权重 models/、各基模代码 code/、独立环境 *_env/ 的工作目录
set -o pipefail
cd "$(dirname "$0")/.." || exit 1
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset all_proxy ALL_PROXY
M=${BASELINE_ROOT}/models
SLUG="$1"
case "$SLUG" in
  VG-LLM-8B)
    PY=${BASELINE_ROOT}/vgllm_env/bin/python; W=17.2; ENGINE="transformers 4.50.0 (vgllm_env)"
    ARGS=(--family vgllm --model "$M/vgllm-qa-vggt-8b")
    CFG="Q1 ≤32帧 VG-LLM官方(逐帧image@504+VGGT@518) 给时间 b=1 sdpa(视觉塔逐块,无flash-attn)" ;;
  VLM-3R-7B)
    PY=${BASELINE_ROOT}/vlm3r_env/bin/python; W=17.4; ENGINE="transformers 4.40.0 + peft 0.10.0 (vlm3r_env)"
    ARGS=(--family vlm3r --model "$M/vlm-3r-llava-qwen2-lora" --model-base "$M/LLaVA-Video-7B-Qwen2")
    CFG="Q1 ≤32帧@384 VLM-3R官方(VSiBench LoRA+CUT3R,qwen_1_5) 给时间 b=1 sdpa fp16" ;;
  *) echo "usage: run_3d_full.sh VG-LLM-8B|VLM-3R-7B"; exit 2 ;;
esac
PRED=preds/Q1_${SLUG}.jsonl
LOG=logs/run_Q1_${SLUG}.log
PLOG=logs/pipeline_3d.log
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
$PY tools/run_3d.py "${ARGS[@]}" --split heldout_test,test --out "$PRED" >> "$LOG" 2>&1
RC=$?
WALL=$(( $(date +%s) - T0 ))
N=$(wc -l < "$PRED" 2>/dev/null || echo 0)
log "$SLUG 推理结束 rc=$RC $N/7922 墙钟=${WALL}s"
grep -m1 "设备自检" "$LOG" | tee -a "$PLOG"

if [ "$N" -lt 7922 ]; then
  log "❌ $SLUG 未跑满"
  python tools/mark_failed.py --model "${SLUG} [Q1原生]" --situation "不完整" \
      --reason "Q1 只产出 $N/7922 条(rc=$RC),见 $LOG;重跑本脚本即可续跑" 2>&1 | tee -a "$PLOG"
  exit 1
fi

(
  flock 9
  python tools/report.py --model "${SLUG} [Q1原生]" --weights "$W" --config "$CFG" \
      --pred "$PRED" --wall "$WALL" --engine "$ENGINE"
  python tools/make_table.py --out figure/trajqa_b_baseline.png
) 9> logs/.results.lock 2>&1 | tee -a "$PLOG"
if grep -qF "| ${SLUG} [Q1原生] |" RESULTS.md; then
  log "$SLUG 已入表 ✓"
else
  log "❌ $SLUG 入表失败"; exit 1
fi
