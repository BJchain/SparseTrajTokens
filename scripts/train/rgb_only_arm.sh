#!/bin/bash
# 只看帧的对照臂(论文消融 "Frames only (separate FT)",run 名 pB_rgbsw_9002_s0):同数据、同结构化目标,不给轨迹 token。
# 预算 9,002 步(2.5662 epoch)余弦退火;是 3,000 / 6,000 / 9,002 三个预算中 val 数值三族 Acc 最高者。
# 原始脚本 docs/records/original_launch_scripts/rgb_sweep.sh / final_compare.sh(选点规则)。
source "$(dirname "$(readlink -f "$0")")/common.sh"
D=data/trajqa_b/final; OUT=$RUNS_DIR/rgb_only_arm
launch --arm rgb --stage 1b --qa $D/train.jsonl --val-qa $D/val.jsonl --split-mode video --test-frac 0.1 --val-frac 0.1 \
  --seed 0 --log-every 20 --structured --save-every 1000 --epochs 2.5662 \
  --batch 2 --grad-accum "$(accum_for 2)" --out "$OUT" 2>&1 | tee "$OUT.log"
