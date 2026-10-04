#!/bin/bash
# InternVL 家族的退路环境:系统装的是 transformers 5.17.0,而 InternVL 自带的 remote code
# (modeling_internvl_chat.py)是按 4.37–4.4x 写的,5.x 删改了不少 API,大概率 import 就崩。
# 这里建一个 --system-site-packages 的 venv,只把 transformers 降到 4.x,复用系统的 torch 2.7.0,
# 所以只需要下十几 MB,不用重下 torch。
#
# pip 源说明(实测):aliyun 直连 <50 KB/s(60s 下不完 3MB 轮子);pypi.org 走代理 230 KB/s。
# 所以保留 http_proxy,只摘掉 socks 的 all_proxy,并强制 index-url 走 pypi.org。
BASELINE_ROOT=${BASELINE_ROOT:-$HOME/baseline_workspace}   # 放模型权重 models/、各基模代码 code/、独立环境 *_env/ 的工作目录
unset all_proxy ALL_PROXY
VENV=${BASELINE_ROOT}/tf4_env
PIP_ARGS="--index-url https://pypi.org/simple --timeout 120 --retries 10"

python -m venv --system-site-packages "$VENV"
"$VENV/bin/python" -m pip install -q --upgrade pip $PIP_ARGS
echo "=== 装 transformers ${1:-4.51.3} ($(date +%H:%M:%S)) ==="
"$VENV/bin/python" -m pip install $PIP_ARGS "transformers==${1:-4.51.3}"
echo "=== exit=$? ($(date +%H:%M:%S)) ==="
"$VENV/bin/python" -c "
import transformers, torch
print('transformers', transformers.__version__)
print('torch', torch.__version__, '| cuda', torch.cuda.is_available())
from transformers import AutoModel, AutoTokenizer
print('AutoModel / AutoTokenizer 可用 ✅')
"
