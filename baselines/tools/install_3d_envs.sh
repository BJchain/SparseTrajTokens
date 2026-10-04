#!/bin/bash
# 3D 组两个模型的推理环境(2026-09-24)。做法同 install_tf4.sh:venv --system-site-packages,
# 复用系统 miniconda 的 torch 2.7.0+cu128(本卡 sm_120 只有 ≥2.7 的 cu128 轮子能用),
# 只在 venv 里装各自钉死的 transformers 等纯 Python 包。
#   vgllm_env : VG-LLM  官方钉 transformers 4.50.0(其 modeling_qwen2_5_vl.py 是照 4.50 抄改的)
#   vlm3r_env : VLM-3R  官方钉 transformers 4.40.0 / peft 0.10.0(LLaVA-NeXT 代码)
# 官方还钉了 torch 2.5.1 / 2.1.1 + flash_attn —— 本卡用不了,改走 sdpa(见 HANDOFF §2.1)。
# 只装推理路径真正 import 到的包;训练/可视化用的(deepspeed、open3d、gsplat、viser…)不装。
#   用法: ./tools/install_3d_envs.sh [vgllm|vlm3r|all]
BASELINE_ROOT=${BASELINE_ROOT:-$HOME/baseline_workspace}   # 放模型权重 models/、各基模代码 code/、独立环境 *_env/ 的工作目录
unset all_proxy ALL_PROXY
PIP_ARGS="--index-url https://pypi.org/simple --timeout 120 --retries 10"
PY=/root/miniconda3/bin/python3
WHICH="${1:-all}"

mk() {  # <venv> <包...>
  local V="$1"; shift
  [ -x "$V/bin/python" ] || "$PY" -m venv --system-site-packages "$V"
  "$V/bin/python" -m pip install -q --upgrade pip $PIP_ARGS
  echo "=== [$(date +%H:%M:%S)] $V: pip install $* ==="
  "$V/bin/python" -m pip install $PIP_ARGS "$@"
  echo "=== [$(date +%H:%M:%S)] exit=$? ==="
}

if [ "$WHICH" = "vgllm" ] || [ "$WHICH" = "all" ]; then
  mk ${BASELINE_ROOT}/vgllm_env \
    "transformers==4.50.0" "accelerate==1.4.0" "huggingface_hub<1.0" safetensors \
    qwen-vl-utils einops opencv-python-headless
fi
if [ "$WHICH" = "vlm3r" ] || [ "$WHICH" = "all" ]; then
  mk ${BASELINE_ROOT}/vlm3r_env \
    "transformers==4.40.0" "tokenizers==0.19.1" "peft==0.10.0" "accelerate==0.29.1" \
    "huggingface_hub<1.0" safetensors "numpy<2" \
    einops einops-exts timm open_clip_torch roma opencv-python-headless scipy omegaconf
fi
