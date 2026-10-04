#!/bin/bash
# Stage 1(论文 Stage 1 / 代码里的 "1a"):只训轨迹投影器(fp32),LLM 全冻结;数据 = 规则生成的运动描述(T8 caption)。
# 26,000 步 = 500 warmup + 恒定 LR 2e-4 到 24,000 步 + 线性退火到 0(2,000 步)。全局批 16。
# 原始实验是两段跑的(恒定段长跑 1a_fit_const_s0 → 从 24,000 步的完整训练状态续一个退火尾巴 1a_fit_tail24k_s0),
# 这里合成一条等价命令;原始脚本见 docs/records/original_launch_scripts/1a_fit_{launch,tail}.sh。
# 产物:$RUNS_DIR/stage1_projector/final/traj_encoder.pt(= checkpoints/stage1_projector)
source "$(dirname "$(readlink -f "$0")")/common.sh"
OUT=$RUNS_DIR/stage1_projector
launch --arm traj --stage 1a --fusion concat --use-tsec --qa data/captions/t8_train.jsonl --allow-open \
  --val-frac 0.02 --test-frac 0 --split-mode video --seed 0 --log-every 20 \
  --sched constant --warmup-steps 500 --decay-from 24000 --decay-steps 2000 --max-steps 26000 \
  --save-every 1000 --val-every 1000 --val-donor-subs object_motion \
  --fp32-encoder --no-grad-ckpt --answer-logits-only \
  --batch 1 --grad-accum "$(accum_for 1)" --out "$OUT" 2>&1 | tee "$OUT.log"
