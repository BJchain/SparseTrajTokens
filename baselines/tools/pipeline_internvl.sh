#!/bin/bash
# 三个 InternVL 系模型的补跑(第一轮全挂了,原因与修法见 RESULTS.md §4)。
# 必须用 tf4_env(transformers 4.51.3):InternVL 自带的 configuration_internvl_chat.py
# 在 transformers 5.17 下会 AttributeError: 'InternVLChatConfig' object has no attribute 'llm_config'。
# 等 GPU 上的 Qwen3-VL 跑完再开,避免两个模型抢显存。
BASELINE_ROOT=${BASELINE_ROOT:-$HOME/baseline_workspace}   # 放模型权重 models/、各基模代码 code/、独立环境 *_env/ 的工作目录
cd "$(dirname "$0")/.." || exit 1
export HF_HUB_OFFLINE=1
unset all_proxy ALL_PROXY
export PYBIN=${BASELINE_ROOT}/tf4_env/bin/python
export ENGINE_TAG="transformers 4.51.3 (tf4_env)"
PLOG=logs/pipeline_internvl.log

echo "=== [$(date +%F_%H:%M:%S)] 等 Qwen3-VL-8B 跑完(让出 GPU)===" >> "$PLOG"
while pgrep -f "run_hf.p[y] .*qwen3vl-8b" > /dev/null; do sleep 60; done
echo "=== [$(date +%F_%H:%M:%S)] GPU 空了,开跑 InternVL 三兄弟 ===" >> "$PLOG"

# slug : 目录 : 权重GiB   (batch 用 8:InternVL 的 .chat() 是逐条的,大 batch 没意义还占显存)
for SPEC in \
  "InternVideo2.5-8B:internvideo2_5-8b:15.0" \
  "InternVL2.5-8B:internvl2_5-8b:15.0" \
  "InternVL3.5-8B:internvl3_5-8b:15.9" ; do
  IFS=: read -r SLUG DIR W <<< "$SPEC"
  echo "=== [$(date +%F_%H:%M:%S)] 开跑 $SLUG ===" >> "$PLOG"
  ./tools/run_model.sh "$SLUG" "${BASELINE_ROOT}/models/$DIR" internvl "$W" 8 >> "$PLOG" 2>&1
  echo "=== [$(date +%F_%H:%M:%S)] $SLUG 结束 条数=$(wc -l < "preds/${SLUG}.jsonl" 2>/dev/null || echo 0) ===" >> "$PLOG"
  if [ ! -s "preds/${SLUG}.jsonl" ]; then
    python tools/mark_failed.py --model "$SLUG" --situation "失败" \
      --reason "tf4_env 下仍未产出预测,见 logs/run_${SLUG}.log" >> "$PLOG" 2>&1
  fi
done
echo "=== [$(date +%F_%H:%M:%S)] INTERNVL PIPELINE DONE ===" >> "$PLOG"
