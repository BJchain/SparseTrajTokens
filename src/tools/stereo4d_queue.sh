#!/bin/bash
# Stereo4D 动态拉取队列 lane(2026-08-12)—— 替代 stereo4d_lane.sh 的静态分片 + 轮屏障。
#
#   bash src/tools/stereo4d_queue.sh <GPU id> [chunk=12] [max_inflight_cpu=2]
#   bash src/tools/stereo4d_queue.sh reclaim            # 只在【所有 lane 都停了】时跑
#
# chunk 为什么是 12(用 1k 批 897 个场景的真实实例数分布模拟出来的,别拍脑袋改):
#   两个方向相反的效应 ——
#     · chunk 越小,尾部균衡越好(各 lane 领到的活更接近)
#     · chunk 越小,每 chunk 的固定模型加载(SAM3 26s + SAM3D 51s + PartSAM 13s x2 ≈ 103s)摊得越薄
#   合起来算,最优在 12 附近,8–32 之间曲线很平:
#     chunk  4    8   12   16   24   48
#     相对静态分片 +7.2 +12.5 +13.5 +12.1 +11.3 +6.4  (%)
#   ⚠️ 直觉"chunk 越大越好"是错的:48 比 12 差 7 个点。
#
# 为什么换掉静态分片(实测依据见 GPU利用率.md / GPU利用率改进.md):
#   ① 尾部空转:静态分片下每条 lane 跑完自己那份就退出,clip 复杂度差异很大(实测单场景
#      1.9–27 min),先跑完的卡干等最慢的那条。轮间等待实测占 3.3%,尾部还要更多。
#   ② 抢跑坑:旧脚本用 `NR%n==p` 取模,一旦取模作用在"剩余清单"上,别的 lane 每 final 掉一个
#      clip 就整体位移,能领到别人正在跑的 clip,两次不同重建的产物混进同一个 scene 目录
#      (SAM3D/PartSAM 都非确定 → inst_*.npz 与 parts_*.npz 顶点对不上)。2026-08-11 的止血办法
#      是"取模必须作用在稳定全量清单上",治标。
#   这里改成【原子认领】治本:`mkdir` 在 POSIX 上是原子的(本机 XFS 24 进程抢 8 key 实测每 key
#   恰好 1 个赢家),认领成功才动那个 clip —— 两条 lane 跑同一个 clip 在结构上不可能。
#   代价:没有了"这份是我的"这种静态归属,所以断点续跑必须靠 reclaim 释放僵尸认领(见下)。
#
# 收益有多大(897 场景真实分布模拟,含每 chunk 模型加载):
#   · 独占机器 + 真实清单(按 clip 名排序 => 同视频相邻 => 成本相关): **+13.5%**
#   · 共享机器(别人的负载把某些卡压慢 1.3-2.0x,本机的常态):        **+40~42%**
#     ← 这才是大头。静态分片对"某张卡变慢"完全无法适应,整批都得等它。
#   · ⚠️ 反例:如果清单先随机打散(成本去相关)且各卡速度一致,静态取模已接近最优,
#     chunk=24 反而慢 2.1% —— 取模本身就是个很好的交错均衡器。别无条件说动态更快。
#
# 与旧脚本一致保留的:
#   * GPU 段(p,0,t,1,tt,2a)前台 + CPU 尾巴(b,3-7,纯 numpy)后台,与下一个 chunk 的 GPU 段重叠
#   * 每个 chunk 独立 sub-root,避免批量 stage 的 glob scene_* 撞上并发 chunk
#   * merge_scene 文件级搬运;done 判据 = ROOT/scene_<clip>/final_repr.pkl
#   * kill -0 按 PID 判活(不用 pgrep —— 会匹配到自己的命令行,实测翻过车)
#   * 磁盘水位守卫
#
# 环境覆盖:STEREO4D_ROOT / STEREO4D_FRAMES / STEREO4D_CLIPLIST / STEREO4D_SLIM / STEREO4D_EXTRA
#   STEREO4D_SKIP_EXTRA  额外要跳过的 stage 码(逗号,如 "p")。**必须用这个,不要往
#     STEREO4D_EXTRA 里塞 --skip-stages** —— EXTRA 是拼在命令末尾的,argparse 取最后一个,
#     会把本脚本自己的 GPU/CPU 切分整个覆盖掉(CPU 尾巴的 b,3-7 会跑进 GPU 段),且不报错。
set -u
cd "$(dirname "$(readlink -f "$0")")/../.." || exit 1     # repo root
export TMPDIR=${TMPDIR:-/tmp}
TM=$TMPDIR
ROOT=${STEREO4D_ROOT:-data/scenes}
FRAMES=${STEREO4D_FRAMES:-data/stereo4d/frames}
MAXTRY=${STEREO4D_MAXTRY:-2}
Q=$ROOT/_q
mkdir -p "$ROOT" "$Q/claim" "$Q/try"

# ---------- 全量清单(稳定,与进度无关) ----------
ALLF=$Q/all_clips.txt
build_list() {
  # ⚠️ 临时文件必须带 $$ —— 所有 lane 同时启动、都发现清单不存在、都来建。共用一个 .tmp 名字时
  # 第一个 mv 走了源文件,其余 lane 的 mv 报 "No such file",然后读到空清单直接退出(实测 6 条
  # lane 里死了 2 条)。各写各的私有 tmp + mv -f 覆盖:mv 同分区内是原子的,内容又完全一样,
  # 谁最后落地都对,读者也永远看不到半个文件。
  local tmp="$ALLF.$$.tmp"
  if [ -n "${STEREO4D_CLIPLIST:-}" ]; then
    sort "$STEREO4D_CLIPLIST" > "$tmp"
  else
    ls "$FRAMES" | while read -r c; do
      [ -f "$FRAMES/$c/gt_stereo4d.npz" ] && printf '%s\n' "$c"
    done | sort > "$tmp"
  fi
  mv -f "$tmp" "$ALLF"
}

# ---------- reclaim:释放僵尸认领 ----------
# lane 崩了/被 kill 掉时,它持有的认领会一直挂着,那些 clip 再也没人领。只有在【确认所有 lane
# 都停了】的时候跑这个(通常由 master 在起 lane 之前跑一次)。跑在有 lane 存活的时候 = 把抢跑坑
# 又请回来了。
if [ "${1:-}" = "reclaim" ]; then
  # 存活判定走 PID 文件 + kill -0,**不能用 pgrep -f**:方括号技巧只防 pgrep 匹配自己,防不住
  # 命令行里恰好含 "stereo4d_queue.sh 0" 的父 shell / 编辑器 / 别的脚本 —— 实测就是这么误判的
  # (lane 早退出了还报"有 1 条在跑")。同一个坑在 feedback_pgrep_selfmatch 里记过。
  alive=0
  for pf in "$Q"/lane_*.pid; do
    [ -f "$pf" ] || continue
    p=$(cat "$pf" 2>/dev/null)
    if [ -n "$p" ] && kill -0 "$p" 2>/dev/null; then
      alive=$((alive+1))
    else
      rm -f "$pf"                                       # 陈旧 PID 文件,顺手清掉
    fi
  done
  if [ "$alive" -gt 0 ]; then
    echo "[reclaim] 还有 $alive 条 lane 在跑 —— 拒绝执行(会造成两条 lane 领同一 clip)"; exit 2
  fi
  n=0
  for d in "$Q"/claim/*; do
    [ -d "$d" ] || continue
    c=$(basename "$d")
    [ -f "$ROOT/scene_$c/final_repr.pkl" ] && continue        # 已完成,认领留着当墓碑
    grep -qxF -e "$c" "$Q/skipped.txt" 2>/dev/null && continue   # 静态过滤器已剔除,同上
    t=0; [ -f "$Q/try/$c" ] && t=$(wc -l < "$Q/try/$c")
    [ "$t" -ge "$MAXTRY" ] && continue                        # 已放弃,别再放回队列
    rmdir "$d" 2>/dev/null && n=$((n+1))
  done
  echo "[reclaim] 释放了 $n 个僵尸认领"
  exit 0
fi

G=${1:?usage: stereo4d_queue.sh <gpu> [chunk] [max_inflight_cpu]  |  stereo4d_queue.sh reclaim}
CHUNK_SZ=${2:-12}
MAX_CPU=${3:-2}
# GPU 段跑 p,0,t,r,1,tt,2a;CPU 尾巴跑 b,3-7。STEREO4D_SKIP_EXTRA 追加到【两边】。
# ⚠️ 查 clip 名一律用 `grep -qxF -e "$c"` —— Stereo4D 的 clip 名全是 YouTube id、**以 `-` 开头**,
# 写成 `grep -qxF "$c"` 会被当成选项解析(实测 `ugrep: invalid argument -J=...`,rc=2),
# 于是"是否已剔除"的判断永远为假。2026-08-16 的 1k 跑就是这么把 387 个已剔除 clip 重跑了一遍、
# 并全部误记进 gave_up.txt(305 里 303 是假警报)。同一形态的坑:--clips 也必须用 `=` 或 @file。
SKIP_GPU="b,3,4,5,6,7"
SKIP_CPU="p,0,t,r,1,tt,2"
if [ -n "${STEREO4D_SKIP_EXTRA:-}" ]; then
  SKIP_GPU="$SKIP_GPU,$STEREO4D_SKIP_EXTRA"
  SKIP_CPU="$SKIP_CPU,$STEREO4D_SKIP_EXTRA"
fi
if [ "${STEREO4D_DRYRUN:-0}" != "1" ]; then
  eval "$(grep '^export DASHSCOPE_API_KEY' ~/.bashrc | tail -1)"
  [ -n "${DASHSCOPE_API_KEY:-}" ] || { echo "[q$G] FATAL: no DASHSCOPE_API_KEY"; exit 1; }
fi

# 宣告存活(reclaim 靠这个判断能不能安全释放认领);退出时自动清掉
printf '%s\n' "$$" > "$Q/lane_$G.pid"
trap 'rm -f "$Q/lane_$G.pid"' EXIT

[ -s "$ALLF" ] || build_list
mapfile -t CLIPS < "$ALLF"
N=${#CLIPS[@]}
[ "$N" -gt 0 ] || { echo "[q$G] 清单为空($ALLF)"; exit 1; }
# 各 lane 从不同位置起扫,降低认领碰撞(纯性能优化,正确性只靠 mkdir 原子性)
CURSOR=$(( (G * N / 8) % N ))
echo "Q$G START clips=$N chunk=$CHUNK_SZ cursor0=$CURSOR $(date +%s)"

# ---------- dry-run 接缝 ----------
# STEREO4D_DRYRUN=1 时用桩替换两次 python 调用,让【编排逻辑本身】能脱离 GPU 验证。
# 这不是可有可无的测试糖:本项目的编排 bug 出过真事故(静态分片抢跑 → 两次不同重建的产物
# 混进同一 scene 目录,SAM3D/PartSAM 非确定使得 inst_*.npz 与 parts_*.npz 顶点对不上),
# 而那种 bug 只在并发下暴露,靠读代码看不出来。桩会按 DRYRUN_FAIL_RE 让部分 clip"失败",
# 以便验证认领释放/重试/放弃三条路径。
_stub_gpu() {                          # $1=part 文件  $2=sub-root
  # 逐 clip 计时:动态队列的全部意义在于【每个 clip 成本差很多】(实测单场景 1.9–27 min),
  # 均匀成本下静态分片和动态队列没差别,测出来会得到"改了等于没改"的假结论。
  local c t
  while read -r c; do
    mkdir -p "$2/scene_$c"; printf 'x' > "$2/scene_$c/video_input.pkl"
    t=${DRYRUN_GPU_SEC:-1}
    if [ -n "${DRYRUN_HEAVY_RE:-}" ] && printf '%s' "$c" | grep -qE "$DRYRUN_HEAVY_RE"; then
      t=${DRYRUN_HEAVY_SEC:-5}
    fi
    [ "$t" != "0" ] && sleep "$t"
  done < "$1"
}
_stub_cpu() {                          # $1=part 文件  $2=sub-root
  local c
  while read -r c; do
    if [ -n "${DRYRUN_FAIL_RE:-}" ] && printf '%s' "$c" | grep -qE "$DRYRUN_FAIL_RE"; then
      continue                         # 模拟这个 clip 跑挂了:不产 final_repr
    fi
    printf 'x' > "$2/scene_$c/final_repr.pkl"
  done < "$1"
  sleep "${DRYRUN_CPU_SEC:-1}"
}

merge_scene() {                       # merge_scene <src_scene_dir> <dst_root>
  local sd=$1 dst=$2/$(basename "$1")
  if [ ! -d "$dst" ]; then mv "$sd" "$2/"; return; fi
  local f
  for f in "$sd"/*; do [ -e "$dst/$(basename "$f")" ] || mv "$f" "$dst/"; done
  rm -rf "$sd"
}

# 起手回收本 lane 上次中断残留的 sub-root(产物拿回 ROOT,skip-if-exists 续跑)
for SB in "$ROOT"/_q${G}_chunk*; do
  [ -d "$SB" ] || continue
  for sd in "$SB"/scene_*; do [ -d "$sd" ] && merge_scene "$sd" "$ROOT"; done
  rm -rf "$SB"
done

CPU_PIDS=""
chunk_i=0
empty_pass=0

while true; do
  # ---------- 认领一个 chunk ----------
  CHUNK=()
  scanned=0
  while [ ${#CHUNK[@]} -lt "$CHUNK_SZ" ] && [ "$scanned" -lt "$N" ]; do
    c=${CLIPS[$CURSOR]}
    CURSOR=$(( (CURSOR + 1) % N ))
    scanned=$((scanned + 1))
    [ -f "$ROOT/scene_$c/final_repr.pkl" ] && continue
    grep -qxF -e "$c" "$Q/skipped.txt" 2>/dev/null && continue   # 静态过滤器已剔除 = 视为完成
    [ -d "$Q/claim/$c" ] && continue
    if [ -f "$Q/try/$c" ]; then
      t=$(wc -l < "$Q/try/$c")
      [ "$t" -ge "$MAXTRY" ] && continue
    fi
    mkdir "$Q/claim/$c" 2>/dev/null || continue      # 输给别的 lane,跳过
    printf 'x\n' >> "$Q/try/$c"
    CHUNK+=("$c")
  done

  if [ ${#CHUNK[@]} -eq 0 ]; then
    # 空扫一遍。可能只是别人正在跑(之后会释放重试),先等在飞的 CPU 尾巴收尾再确认一次。
    empty_pass=$((empty_pass + 1))
    for p in $CPU_PIDS; do wait "$p" 2>/dev/null; done
    CPU_PIDS=""
    [ "$empty_pass" -ge 2 ] && break
    continue
  fi
  empty_pass=0
  chunk_i=$((chunk_i + 1))

  SB=$ROOT/_q${G}_chunk$chunk_i
  mkdir -p "$SB"
  PART=$TM/q_${G}_chunk$chunk_i.txt
  printf '%s\n' "${CHUNK[@]}" > "$PART"
  # 领取:ROOT 里已有部分产物的场景搬进 sub-root(skip-if-exists 才看得见)
  # ⚠️ 必须同时看 _skipped_static/:静态过滤器把被剔除的场景搬去了那儿,而 stage0_pre.json
  # (已经烧掉的 Qwen 调用)就在里面。只看 $ROOT/scene_$c 的后果实测过 —— 一旦某个已剔除的
  # clip 因任何原因被重新认领,这里合不进来 → _prepare_scene 建裸目录 → has_pre 为假 →
  # stage0 退回兜底 `--concepts person` → 产出一份垃圾 final_repr 且过滤器不再拦它。
  # 2026-08-17 就是这样污染了 84 个 clip(触发源是 grep 的减号 bug,已修,但这里是链条上
  # 独立的一环,单独堵死)。见 memory: project_s4d_1k_batch_outcome。
  for c in "${CHUNK[@]}"; do
    if [ -d "$ROOT/scene_$c" ]; then
      merge_scene "$ROOT/scene_$c" "$SB"
    elif [ -d "$ROOT/_skipped_static/scene_$c" ]; then
      merge_scene "$ROOT/_skipped_static/scene_$c" "$SB"
      echo "Q$G chunk$chunk_i RECLAIM_FROM_SKIPPED $c" >&2
    fi
  done

  # ---------- GPU 段(前台) ----------
  T0=$(date +%s)
  echo "Q$G chunk$chunk_i GPU_START n=${#CHUNK[@]} $T0"
  if [ "${STEREO4D_DRYRUN:-0}" = "1" ]; then
    _stub_gpu "$PART" "$SB" > "$TM/q_g${G}_chunk${chunk_i}_gpu.log" 2>&1
  else
    CUDA_VISIBLE_DEVICES=$G python src/run_stereo4d_batch_v3.py \
      --clips="@$PART" --batch-root "$SB" --frames-root "$FRAMES" \
      --skip-stages "$SKIP_GPU" ${STEREO4D_EXTRA:-} \
      > "$TM/q_g${G}_chunk${chunk_i}_gpu.log" 2>&1
  fi
  rc=$?
  T1=$(date +%s)
  echo "Q$G chunk$chunk_i GPU_END rc=$rc dur=$((T1-T0))s"

  # ---------- CPU 尾巴(后台,与下一个 chunk 的 GPU 段重叠) ----------
  (
    TC0=$(date +%s)
    if [ "${STEREO4D_DRYRUN:-0}" = "1" ]; then
      _stub_cpu "$PART" "$SB" > "$TM/q_g${G}_chunk${chunk_i}_cpu.log" 2>&1
    else
      CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
      MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 VECLIB_MAXIMUM_THREADS=8 \
      python src/run_stereo4d_batch_v3.py \
        --clips="@$PART" --batch-root "$SB" --frames-root "$FRAMES" \
        --skip-stages "$SKIP_CPU" ${STEREO4D_EXTRA:-} \
        > "$TM/q_g${G}_chunk${chunk_i}_cpu.log" 2>&1
    fi
    crc=$?
    # STEREO4D_SLIM: 0=全留(默认) | pkl=只删 video_input.pkl | 1=激进(连 inst/dense 一起删)
    # video_input.pkl 是帧+mask 的 pickle 缓存,实测占单 scene 产物的 56%(144MB/256MB),
    # 且下游只有 viz 会读它 —— 110k 规模上这一项就是 9TB,跑完即删是默认该做的事。
    if [ "${STEREO4D_SLIM:-0}" != "0" ]; then
      for sd in "$SB"/scene_*; do
        [ -f "$sd/final_repr.pkl" ] || continue
        [ -f "$sd/scene_manifest.json" ] || continue     # 见下面完成判据:没 manifest = stage1 没跑
        rm -f "$sd"/video_input.pkl 2>/dev/null
        if [ "${STEREO4D_SLIM}" = "1" ]; then
          # ⚠️ 历史遗留档位,两处是反的(2026-08-19 核算发现):它不删 parts_*.npz
          # (20.32 MiB/scene,110k 上漏 1168 GiB),却删了 QA 硬挖要读的 dense_tracks_*.npz。
          # 保留只为复现旧产物;新跑一律用 jpg 档。
          rm -f "$sd"/inst_*.npz "$sd"/inst_*.obj "$sd"/dense_tracks_*.npz 2>/dev/null
        elif [ "${STEREO4D_SLIM}" = "jpg" ]; then
          # 「最省磁盘版 - jpg 版」= 110k 落盘 396 GiB 的那一档。保留清单按【下游真实消费方】定:
          #   tokenizer 读 final_repr + gt 相机切片;QA 硬挖读 final_repr + dense_tracks + gt 相机;
          #   训练读 data/stereo4d/frames/<clip>/*.jpg。
          # 留:final_repr.pkl + 8 个小 json + dense_tracks_*.npz(合计 2.393 MiB/scene)
          # 删:video_input.pkl(只有 viz 读)/ inst_*(网格只用于采关键点,已烘进 final_repr)/
          #     parts_*(part_id 同上)/ tracks_d4rt / pass1_keypoints / stage2b* / pass2*
          rm -f "$sd"/inst_*.npz "$sd"/inst_*.obj "$sd"/parts_*.npz "$sd"/parts_motion_*.npz \
                "$sd"/tracks_d4rt.npz "$sd"/pass1_keypoints.pkl "$sd"/pass1_keypoints_coverage.json \
                "$sd"/stage2b.npz "$sd"/stage2b_early.npz "$sd"/pass2.npz 2>/dev/null
        fi
      done
      # 被静态过滤器剔除的 scene 永远不会有 final_repr,但它们的 video_input.pkl 一样占地方
      # (实测 42.4MB/clip,而且是它们目录里的全部内容),同样删掉 —— Qwen 结果在 stage0_pre.json。
      for sd in "$SB"/_skipped_static/scene_*; do
        [ -d "$sd" ] && rm -f "$sd"/video_input.pkl 2>/dev/null
      done
    fi
    for sd in "$SB"/scene_*; do [ -d "$sd" ] && merge_scene "$sd" "$ROOT"; done
    # 静态 clip 过滤器(2026-08-15)把被剔除的 scene 目录搬进 $SB/_skipped_static/、名单写
    # $SB/skipped_static_clips.json。**这两样也必须并回 ROOT**,否则:①下面的 rmdir 静默失败,
    # sub-batch 目录搁浅(与 lane 脚本那个已知坑同一形态);②剔除名单停在 sub-batch 里传不回去,
    # ROOT 级重跑幂等就断了(下次又会把这些 clip 重新拉进来跑一遍 Qwen/SAM3)。
    if [ -d "$SB/_skipped_static" ]; then
      mkdir -p "$ROOT/_skipped_static"
      for sd in "$SB"/_skipped_static/scene_*; do
        [ -d "$sd" ] && merge_scene "$sd" "$ROOT/_skipped_static"
      done
      rmdir "$SB/_skipped_static" 2>/dev/null
    fi
    # 同时把 clip 名追加到 $Q/skipped.txt —— 队列的 done 判据是 ROOT/scene_<c>/final_repr.pkl,
    # 而被剔除的 clip 永远不会有它:不记下来的话它们会被反复重新认领直到 MAXTRY,最后全进
    # gave_up.txt(110k 上就是几万次空转 + 一个没有意义的 gave_up 计数)。
    if [ -d "$ROOT/_skipped_static" ]; then
      for sd in "$ROOT"/_skipped_static/scene_*; do
        [ -d "$sd" ] || continue
        c=$(basename "$sd"); c=${c#scene_}
        grep -qxF -e "$c" "$Q/skipped.txt" 2>/dev/null || printf '%s\n' "$c" >> "$Q/skipped.txt"
      done
    fi
    if [ -f "$SB/skipped_static_clips.json" ]; then
      python3 - "$SB/skipped_static_clips.json" "$ROOT/skipped_static_clips.json" <<'PYMERGE'
import json, os, sys
src, dst = sys.argv[1], sys.argv[2]
cur = {}
if os.path.exists(dst):
    try: cur = json.load(open(dst)).get("clips", {})
    except Exception: cur = {}
try: cur.update(json.load(open(src)).get("clips", {}))
except Exception: pass
json.dump({"note": "无动态实例、已剔除;scene 目录在 _skipped_static/,搬回上一级即可恢复",
           "clips": cur}, open(dst, "w"), indent=1, ensure_ascii=False)
PYMERGE
      rm -f "$SB/skipped_static_clips.json"
    fi
    rm -f "$PART"
    rmdir "$SB" 2>/dev/null
    # ---------- 结算认领 ----------
    # 成功 -> 认领留着当墓碑(扫描时被 final_repr 判据跳过,留着便于审计)
    # 失败且未达 MAXTRY -> 释放,让任意一条 lane(不一定是本条)重试 —— 这就是"动态"的意义
    done_n=0; rel=0; gave=0
    while read -r c; do
      # ⚠️ 判据必须带 scene_manifest.json:它是 run_stage1.py【无条件】写的(哪怕实例全被
      #    静态/rigid 路由跳过也写),所以"有 manifest"= stage1 真的跑过。只看 final_repr 的话,
      #    GPU 段失败(如别人抢显存导致 SAM3D 加载 OOM)后 CPU 尾巴仍会产出一个没有任何网格的
      #    final_repr,于是这个 clip 被算作完成、认领被留作墓碑、永远不会重试 —— 一次崩溃被
      #    静默转成一条低质量数据。2026-08-19 实测:3k 跑头 36 个成品里有 6 个是这么来的。
      if { [ -f "$ROOT/scene_$c/final_repr.pkl" ] && [ -f "$ROOT/scene_$c/scene_manifest.json" ]; } \
         || grep -qxF -e "$c" "$Q/skipped.txt" 2>/dev/null; then
        done_n=$((done_n+1)); continue      # 被静态过滤器剔除的也算完成,否则会一直重试到 gave_up
      fi
      t=0; [ -f "$Q/try/$c" ] && t=$(wc -l < "$Q/try/$c")
      if [ "$t" -lt "$MAXTRY" ]; then
        rmdir "$Q/claim/$c" 2>/dev/null && rel=$((rel+1))
      else
        gave=$((gave+1)); printf '%s\n' "$c" >> "$Q/gave_up.txt"
      fi
    done < <(printf '%s\n' "${CHUNK[@]}")
    TC1=$(date +%s)
    echo "Q$G chunk$chunk_i CPU_END rc=$crc dur=$((TC1-TC0))s done=$done_n 释放重试=$rel 放弃=$gave"
  ) &
  CPU_PIDS="$CPU_PIDS $!"

  # 在飞 CPU 尾巴上限(多 lane 共享 CPU)。kill -0 按 PID 判活 —— 不用 pgrep,它会匹配自己。
  while :; do
    live=0; alive_pids=""
    for p in $CPU_PIDS; do
      if kill -0 "$p" 2>/dev/null; then live=$((live+1)); alive_pids="$alive_pids $p"; fi
    done
    CPU_PIDS=$alive_pids
    [ "$live" -lt "$MAX_CPU" ] && break
    sleep 15
  done

  gfree=$(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i "$G" \
          2>/dev/null | awk -F', ' '{print $1-$2}')
  if [ -n "${gfree:-}" ] && [ "$gfree" -lt "${STEREO4D_MIN_GPU_MB:-24000}" ]; then
    echo "Q${G}_GPU_TAKEN 空闲显存仅 ${gfree}M — 本 lane 退出(autoscale 会在卡空出来时重起)"
    for p in $CPU_PIDS; do wait "$p" 2>/dev/null; done
    exit 4
  fi

  free=$(df /data | awk 'NR==2{print int($4/1048576)}')
  if [ "$free" -lt "${STEREO4D_MIN_FREE_G:-20}" ]; then
    echo "Q${G}_LOW_DISK ${free}G — 停止领新活,等在飞的收尾"
    for p in $CPU_PIDS; do wait "$p" 2>/dev/null; done
    exit 2
  fi
done

for p in $CPU_PIDS; do wait "$p" 2>/dev/null; done
n_done=$(ls -d "$ROOT"/scene_*/final_repr.pkl 2>/dev/null | wc -l)
echo "Q${G}_DONE chunks=$chunk_i total_final=$n_done $(date +%s)"
