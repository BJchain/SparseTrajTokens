#!/bin/bash
# 32B 全量。喂法 = base(现行逐帧 image),依据 2026-09-22 的 300 题三路 A/B:
#   base 28.2 / frameidx 27.5 / video 18.7(同一子集,n=284)
#   原生 video 通道**显著更差**(McNemar p=0.0001),不采用;frameidx 与 base 无显著差异。
# 不跑盲测臂(NO_BLIND=1)。跑满 7,922 才删权重。
BASELINE_ROOT=${BASELINE_ROOT:-$HOME/baseline_workspace}   # 放模型权重 models/、各基模代码 code/、独立环境 *_env/ 的工作目录
cd "$(dirname "$0")/.." || exit 1
export HF_HUB_OFFLINE=1
unset all_proxy ALL_PROXY
export NO_BLIND=1
MR=${BASELINE_ROOT}/models
PLOG=logs/pipeline_32b.log

run_one () {
  SLUG="$1"; DIR="$2"; W="$3"
  echo "=== [$(date +%F_%H:%M:%S)] 开跑 $SLUG 全量 (b=8, base 喂法) ===" >> "$PLOG"
  ./tools/run_model.sh "$SLUG" "$MR/$DIR" qwen "$W" 8 >> "$PLOG" 2>&1
  N=$(wc -l < "preds/${SLUG}.jsonl" 2>/dev/null || echo 0)
  echo "=== [$(date +%F_%H:%M:%S)] $SLUG 结束 条数=$N ===" >> "$PLOG"
  if [ "$N" -ge 7922 ]; then
    rm -rf "$MR/$DIR"
    echo "=== [$(date +%F_%H:%M:%S)] 已删 $DIR 腾盘,剩 $(df -h ${BASELINE_ROOT}|tail -1|awk '{print $4}') ===" >> "$PLOG"
  else
    python tools/mark_failed.py --model "$SLUG" --situation "不完整" \
        --reason "只产出 $N/7922 条,见 logs/run_${SLUG}.log;权重已保留可续跑" >> "$PLOG" 2>&1
  fi
}

# #1 已下完,直接跑
run_one "Qwen3-VL-32B" "qwen3vl-32b" "62.1"

# #2 等下完再跑
echo "=== [$(date +%F_%H:%M:%S)] 等 Qwen2.5-VL-32B 下载完 ===" >> "$PLOG"
while ! grep -q "完成 fail=" "logs/ms_qwen25vl-32b.log" 2>/dev/null; do sleep 60; done
if grep -q "完成 fail=0" "logs/ms_qwen25vl-32b.log" 2>/dev/null; then
  run_one "Qwen2.5-VL-32B" "qwen25vl-32b" "63.6"
else
  echo "!!! [$(date +%F_%H:%M:%S)] Qwen2.5-VL-32B 下载失败" >> "$PLOG"
  python tools/mark_failed.py --model "Qwen2.5-VL-32B" --situation "未跑" \
      --reason "ModelScope 下载失败,见 logs/ms_qwen25vl-32b.log" >> "$PLOG" 2>&1
fi
echo "=== [$(date +%F_%H:%M:%S)] 32B FULL PIPELINE DONE ===" >> "$PLOG"
