#!/bin/bash
# Stage 3(定稿,run 名 pB_s2_cos1ep_s0):帧 + 轨迹 token 联合训练,每个样本以 0.2 概率整体丢帧(token 不丢)。
# 从 Stage 2 的 LoRA + 投影器继续(优化器重新初始化);1 epoch = 3,507 步,warmup 3% + 余弦退火到 0,峰值 LR 同 Stage 2。
# 原始脚本 docs/records/original_launch_scripts/1b_s2_cosine_launch.sh。
# 产物:$RUNS_DIR/stage3_joint/final(= checkpoints/stage3_final,论文报告的模型)
source "$(dirname "$(readlink -f "$0")")/common.sh"
SRC=${SRC:-$RUNS_DIR/stage2_token_only/final}                      # 或 checkpoints/stage2_token_only
D=data/trajqa_b/final; OUT=$RUNS_DIR/stage3_joint
launch --arm traj --stage 1b --fusion concat --use-tsec --rgb-dropout 0.2 --qa $D/train.jsonl --val-qa $D/val.jsonl \
  --split-mode video --test-frac 0.1 --val-frac 0.1 --seed 0 --log-every 20 --structured \
  --epochs 1.0 --warmup 0.03 --save-every 1000 --val-every 1000 \
  --batch 2 --grad-accum "$(accum_for 2)" --init-encoder "$SRC/traj_encoder.pt" --init-adapter "$SRC" --out "$OUT" 2>&1 | tee "$OUT.log"
