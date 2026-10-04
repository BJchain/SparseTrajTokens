#!/bin/bash
# Stereo4D → 物体中心稀疏 4D 轨迹(<产物根>/scene_<clip>/final_repr.pkl)的整条数据管线。
# 论文的 42,152 段就是这条链在 89,454 个有官方 3D 轨迹的 clip 上跑出来的(8×L20,约 3 周;原始调度脚本见
# docs/records/original_ops_scripts/,它们只是在此基础上加了断点续跑 / 动态占卡 / 磁盘水位守卫)。
#
#   ① 取数(INGEST=1 时):从官方 mp4 tar(data/stereo4d/raw_mp4/)抽左目视频 + 下载官方 npz
#      → 自适应抽帧(20–34 帧,512×512)+ gt_stereo4d.npz,写到 data/stereo4d/frames/<clip>/。
#      发布的数据里已经带了 42,151 段的帧与 gt_stereo4d.npz,只重跑管线时不需要这一步。
#   ② Qwen3-VL-Plus 物体清单预取(DashScope API,不占 GPU;需 DASHSCOPE_API_KEY)
#   ③ GPU 流水线(动态队列,每卡一条 lane):SAM 3 掩码 → 官方轨迹归属 + 运动门 → 刚体路由 → Fast-SAM3D 网格
#      → PartSAM 部件 → 关键点两遍采样 → 关键点↔轨迹链接 → final_repr
#   ④(COMPRESS_GT=1 时)把 gt_stereo4d.npz 压成只含相机的切片(下游只读相机),省 99.9% 磁盘
#
# 用法:GPUS="0 1 2 3" bash scripts/data/run_stereo4d_pipeline.sh <clip 清单> [产物根=outputs/scenes]
#   clip 清单:每行一个 clip 名(例:data/curation/delivered_scenes.txt 去掉 scene_ 前缀);clip 名可能以 '-' 开头。
# 依赖:environment/ 下的 sam3 / sam3d-objects / PartSAM 环境(CONDA_ENVS_DIR 或 SAM3_PY/SAM3D_PY/PARTSAM_PY 指过去),
#       third_party/ 下的 sam3、sam-3d-objects(+ 补丁)、Fast-SAM3D、PartSAM 及其权重(见 third_party/README.md)。
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.."          # repo root
CLIPS=${1:?clip list}; ROOT=${2:-outputs/scenes}; GPUS=${GPUS:-0}; PY=${PY:-python}
: "${DASHSCOPE_API_KEY:?需要 DASHSCOPE_API_KEY(Qwen3-VL-Plus 物体清单)}"
mkdir -p "$ROOT"
if [ "${INGEST:-0}" = 1 ]; then
  # 取数是常驻进程(按波次取、按待跑量暂停/恢复),与 GPU 流水线并行;它把建好帧的 clip 追加进队列清单
  $PY src/tools/stereo4d_ingest.py --queue-list "$ROOT/_ingested.txt" --root "$ROOT" --wave "${WAVE:-2000}" \
      --min-free-g "${MIN_FREE_G:-60}" > "$ROOT/_ingest.log" 2>&1 &
  ING=$!
  while kill -0 $ING 2>/dev/null || [ -s "$ROOT/_ingested.txt" ]; do
    [ -s "$ROOT/_ingested.txt" ] && { sort -u "$ROOT/_ingested.txt" > "$ROOT/_batch.txt"
      S4D_EXTRA="--recon fastsam3d" bash src/tools/stereo4d_prod.sh "$GPUS" "$ROOT/_batch.txt" "$ROOT" "${CHUNK:-12}"; }
    kill -0 $ING 2>/dev/null || break; sleep 600
  done
else
  S4D_EXTRA="--recon fastsam3d" bash src/tools/stereo4d_prod.sh "$GPUS" "$CLIPS" "$ROOT" "${CHUNK:-12}"
fi
[ "${COMPRESS_GT:-0}" = 1 ] && $PY src/tools/compress_gt_npz.py --batch-root "$ROOT" --frames-root data/stereo4d/frames
echo "done -> $ROOT"
