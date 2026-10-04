#!/bin/bash
# Stereo4D 生产总入口(2026-08-13)—— 一条命令跑完整批,把这一轮验证过的编排优化全部接上。
#
#   bash src/tools/stereo4d_prod.sh "<gpu 列表>" <clip清单> <产物根> [chunk=12]
# 例:
#   bash src/tools/stereo4d_prod.sh "0 2 3 5 6 7" data/curation/clips_ready.txt data/scenes
#
# 它按顺序做三件事:
#   ① Qwen 预取(**完全不占 GPU**)。stage0-pre 是 DashScope 往返(~17s/clip),串在 GPU 流水线里
#      就是纯空转 —— 2026-08-01 那次 1k 跑起跑 49 分钟后六卡均值只有 32%、三张卡读 0%,就是
#      卡在这。48 路并发跑完全量,GPU 流水线之后 --skip-stages p 只读磁盘上的 json。
#   ② reclaim:释放上次崩溃残留的僵尸认领(必须在没有 lane 存活时做,子命令自带守卫)。
#   ③ 动态拉取队列(stereo4d_queue.sh)跑 GPU 流水线,全程挂 gpu_trace 采样。
#
# 已接上的优化(每一条都有实测,见 GPU利用率改进.md / STAGE1_FRAME_SCORER_PLAN.md):
#   * stage0 SAM3 批处理(模型每 chunk 加载一次而不是每 clip)—— 省 ~27s/clip,逐位一致
#   * Qwen 并发预取 —— 串行 519h → 48 路约 11h,且不占 GPU
#   * 动态拉取队列 + 原子认领 —— 独占机 +13.5%,共享机 +40~42%,且两条 lane 跑同一 clip
#     在结构上不可能
#   * stage1 提前放弃(--abandon-frac,默认 0.2)—— 省 11.9% 的 stage1 调用,下游 ⑦ 未检出退化
#   * stage2a 2 worker —— **1.283x**(S4D_STAGE2A_WORKERS,默认 2,显存不够会自动降级)
#
# ⚠️ stage2a 多 worker 的收益被改小过一次,别再引用旧数字:
#   2026-08-12 在**旧版** stage2a 上测到 1.62x —— 当时它每 8 个 prompt 调一次
#   empty_cache()+gc.collect(),占墙钟 52%,GPU 在那期间是空的(58.4% TDP),第二个 worker 填的
#   正是这些空档。那两句已于 2026-08-12 删除(单卡本身就快了 53%)。
#   **2026-08-13 在新版上重测(264 实例,同一张卡顺序跑):1 worker 4974s vs 2 worker 3878s
#   = 1.283x**,0 次 OOM。新版 1 worker 仍有 25.7% util=0 / 77% TDP,所以还有得填,只是少了。
#   显存:1 worker 峰值 26.7GB,**2 worker 37.4GB / 46GB —— 只剩 8.6GB 余量**。
#   共享卡上别人占掉 10GB 就会 OOM 整个 chunk,所以下面会按实时空闲显存自动降级。
set -u
cd "$(dirname "$(readlink -f "$0")")/../.." || exit 1     # repo root
GPUS=${1:?usage: stereo4d_prod.sh "<gpus>" <cliplist> <root> [chunk]}
CLIPS=${2:?}
ROOT=${3:?}
CHUNK=${4:-12}
TM=${TMPDIR:-/tmp}
NW2=${S4D_STAGE2A_WORKERS:-2}
PREFETCH_WORKERS=${S4D_QWEN_WORKERS:-48}

export STEREO4D_ROOT=$ROOT
export STEREO4D_CLIPLIST=$CLIPS
# ⚠️ 额外 skip 必须走 STEREO4D_SKIP_EXTRA,**不能塞进 STEREO4D_EXTRA** —— 后者拼在命令末尾,
# argparse 取最后一个 --skip-stages,会把队列脚本自己的 GPU/CPU 切分整个覆盖掉且不报错。
export STEREO4D_SKIP_EXTRA="p"
# 显存守卫:2 worker 实测峰值 37.4GB。任一张卡空闲不足就整体降到 1 —— 宁可慢 28%,
# 也别让一个 chunk 在 stage2a 中途 OOM(那会连累整块重排,而且 PartSAM 的失败痕迹不对称)。
if [ "$NW2" -gt 1 ]; then
  NEED=40000
  low=""
  for G in $GPUS; do
    free=$(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i "$G" \
           | awk -F', ' '{print $1-$2}')
    [ "${free:-0}" -lt "$NEED" ] && low="$low g$G(${free}M)"
  done
  if [ -n "$low" ]; then
    echo "⚠️ 显存不足以跑 $NW2 个 stage2a worker(需 ≥${NEED}M):$low —— 自动降级为 1"
    NW2=1
  fi
fi
# 纯 2a(motion-part 退场,2026-08-13 定案)自 2026-08-15 起已是**编排器的默认**,这里不必再传。
# 本入口曾漏传 --no-motion-groups(2026-08-13→08-15 期间走生产入口的批次都还在用 motion-part),
# 这也是把它改成默认值的直接原因 —— 定案了但不是默认,就迟早会有人忘记传。
# 想跑回旧的运动臂做对照:S4D_MOTION_GROUPS=1(注意要传的是 --motion-groups,不是把 flag 置空)。
# 依据见 memory: project_motion_part_retired / project_pure2a_change_spec。
ARM=""
[ "${S4D_MOTION_GROUPS:-0}" = "1" ] && ARM="--motion-groups"
export STEREO4D_EXTRA="--stage2a-workers $NW2 $ARM ${S4D_EXTRA:-}"   # S4D_EXTRA 用来传 --recon 等

echo "=== ① Qwen 预取($PREFETCH_WORKERS 路并发,不占 GPU) ==="
[ -n "${DASHSCOPE_API_KEY:-}" ] || { echo "FATAL: 没有 DASHSCOPE_API_KEY"; exit 1; }
python src/tools/qwen_prefetch.py --clips "@$CLIPS" --batch-root "$ROOT" \
       --frames-root "${STEREO4D_FRAMES:-data/stereo4d/frames}" \
       --workers "$PREFETCH_WORKERS" || { echo "预取失败 —— 中止"; exit 2; }

echo
echo "=== ② reclaim(释放僵尸认领) ==="
bash src/tools/stereo4d_queue.sh reclaim || exit 2

echo
echo "=== ③ GPU 流水线:动态队列 gpus=[$GPUS] chunk=$CHUNK stage2a-workers=$NW2 ==="
PIDS=()
for G in $GPUS; do
  bash src/tools/stereo4d_queue.sh "$G" "$CHUNK" 2 > "$TM/q_lane_g$G.log" 2>&1 &
  PIDS+=($!)
  echo "[prod] lane g$G pid ${PIDS[-1]}"
done
wait "${PIDS[@]}"

echo
nf=$(ls -d "$ROOT"/scene_*/final_repr.pkl 2>/dev/null | wc -l)
nt=$(grep -c . "$CLIPS")
ng=0; [ -f "$ROOT/_q/gave_up.txt" ] && ng=$(sort -u "$ROOT/_q/gave_up.txt" | wc -l)
echo "S4DPROD_DONE final=$nf/$nt gave_up=$ng"
grep -ql "_LOW_DISK" "$TM"/q_lane_g*.log 2>/dev/null && echo "⚠️ 有 lane 因磁盘水位停了"
