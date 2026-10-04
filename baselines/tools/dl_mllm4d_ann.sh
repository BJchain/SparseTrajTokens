#!/bin/bash
# MLLM-4D 数据集的标注文件(只要 json,不要 100 GB 训练视频),用于核对 TrajQA-B clip 与 MLLM-4D 训练集是否重叠。
# 布局照官方 scripts/download_dataset_hf.py(../data/MLLM-4D-Datasets)。逐文件 Range 续传 + sha256 校验。
#   用法: setsid nohup ./tools/dl_mllm4d_ann.sh > logs/dl_mllm4d_ann.log 2>&1 &
BASELINE_ROOT=${BASELINE_ROOT:-$HOME/baseline_workspace}   # 放模型权重 models/、各基模代码 code/、独立环境 *_env/ 的工作目录
unset all_proxy ALL_PROXY
DST=${MLLM4D_DATASETS:-$(dirname "$(readlink -f "$0")")/../../data/external/MLLM-4D-Datasets}
while read -r F SZ SHA; do
  OUT="$DST/$F"; VIA=proxy
  for a in $(seq 1 200); do
    CUR=$(stat -c%s "$OUT" 2>/dev/null || echo 0); [ "$CUR" -ge "$SZ" ] && break
    if [ $VIA = mirror ]; then U="https://hf-mirror.com/datasets/flow666/MLLM-4D-Datasets/resolve/main/$F"; PX=(--noproxy '*')
    else U="https://huggingface.co/datasets/flow666/MLLM-4D-Datasets/resolve/main/$F"; PX=(${HTTPS_PROXY:+-x "$HTTPS_PROXY"}); fi
    CODE=$(curl -sL "${PX[@]}" -r "${CUR}-" -o >(cat >> "$OUT") -w "%{http_code}" --connect-timeout 60 --max-time 3600 --speed-limit 1000000 --speed-time 45 "$U")
    sleep 1
    NEW=$(stat -c%s "$OUT" 2>/dev/null || echo 0)
    if [ "$CODE" != 206 ] && ! { [ "$CODE" = 200 ] && [ "$CUR" = 0 ]; }; then truncate -s "$CUR" "$OUT"; NEW=$CUR; fi
    echo "[$(date +%T)] $F via $VIA HTTP $CODE -> $NEW/$SZ"
    [ $VIA = mirror ] && VIA=proxy || VIA=mirror
  done
  GOT=$(sha256sum "$OUT" | cut -d' ' -f1)
  [ "$GOT" = "$SHA" ] && echo "[$(date +%T)] OK sha256 $F" || echo "[$(date +%T)] ❌ sha256 不符 $F $GOT"
done <<LIST
MLLM4D-R1-30k.jsonl 52273967 86f27ee5ff12264d30683539d54f089324e92f4233c3bd3d7188579d1225681a
MLLM4D-Cold-Start-Data.json 28283018 9aedb0960dd850257d31b155901fda8afa18ce4387032a6548b91ad92a3faee1
MLLM4D-2M.json 2277188518 5a47adb7578ec132efdf4a90457a50290891e9c54c68f14e545c3f2c9f51eae4
LIST
echo "[$(date +%T)] 全部结束"
