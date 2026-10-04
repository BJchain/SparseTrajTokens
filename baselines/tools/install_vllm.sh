#!/bin/bash
# 在独立 venv 里装 vLLM,绝不污染主环境(主环境 torch 2.7.0+cu128 / transformers 5.17.0,
# run_hf.py 依赖它)。本机是 RTX PRO 6000 Blackwell = sm_120,需要 cu128 及以上的轮子。
# aliyun 是国内源,走代理反而慢一个量级(实测 316MB 轮子卡 80 分钟),这里把代理全摘掉
BASELINE_ROOT=${BASELINE_ROOT:-$HOME/baseline_workspace}   # 放模型权重 models/、各基模代码 code/、独立环境 *_env/ 的工作目录
unset all_proxy ALL_PROXY http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
VENV=${BASELINE_ROOT}/vllm_env
$VENV/bin/python -m ensurepip --upgrade 2>/dev/null
$VENV/bin/python -m pip install -q --upgrade pip
echo "=== installing vllm ($(date +%H:%M:%S)) ==="
$VENV/bin/python -m pip install vllm=="${1:-0.29.0}"
echo "=== install exit=$? ($(date +%H:%M:%S)) ==="
echo "=== version check ==="
$VENV/bin/python -c "
import torch, vllm
print('vllm', vllm.__version__)
print('torch', torch.__version__)
print('cuda avail', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device', torch.cuda.get_device_name(0), 'sm_%d%d' % torch.cuda.get_device_capability(0))
    print('arch list', torch.cuda.get_arch_list())
"
