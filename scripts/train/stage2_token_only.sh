#!/bin/bash
# Stage 2(代码里的 "1b token-only",run 名 pB_wsd_tok_s0_tail7k_d20):只给轨迹 token、不给帧;训练 LoRA(r128)+ 投影器。
# 8,400 步 = 180 warmup + 恒定 LR(LoRA 1e-4 / 投影器 2e-4)到 7,000 步 + 线性退火 1,400 步。数据 = TrajQA-B train,结构化目标。
# 原始实验同样是"恒定长跑 + 从 step7000 续退火尾巴"两段;原始脚本 docs/records/original_launch_scripts/1b_wsd_{launch,tails}.sh。
# 产物:$RUNS_DIR/stage2_token_only/final(= checkpoints/stage2_token_only)
source "$(dirname "$(readlink -f "$0")")/common.sh"
INIT=${INIT:-$RUNS_DIR/stage1_projector/final/traj_encoder.pt}     # 或 checkpoints/stage1_projector/traj_encoder.pt
D=data/trajqa_b/final; OUT=$RUNS_DIR/stage2_token_only
launch --arm traj_only --stage 1b --fusion concat --use-tsec --qa $D/train.jsonl --val-qa $D/val.jsonl \
  --split-mode video --test-frac 0.1 --val-frac 0.1 --seed 0 --log-every 20 --structured \
  --sched constant --warmup-steps 180 --decay-from 7000 --decay-steps 1400 --max-steps 8400 \
  --save-every 1000 --val-every 500 \
  --batch 2 --grad-accum "$(accum_for 2)" --init-encoder "$INIT" --out "$OUT" 2>&1 | tee "$OUT.log"
