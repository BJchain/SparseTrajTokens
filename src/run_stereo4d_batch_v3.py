"""Stereo4D batch orchestrator: the V3 chain on Stereo4D clips (thin runner).

Stereo4D ingest is DAVIS-shaped: per-clip frame dirs already produced by
src/stage0/stereo4d_adapter.py under data/stereo4d/frames/<clip>/ (left-eye 512x512
perspective frames %05d.jpg + gt_stereo4d.npz + stereo4d_meta.json, adaptive stride
20-38 frames). No GT instance masks -> SAM 3 makes the masklets. GT intrinsics ARE
known (60deg hfov: fx=fy=443.405, cx=cy=256) -> injected via run_stage0.py --intrinsics
(read per clip from stereo4d_meta.json). Per clip:
  1. scene_<clip>/ setup: symlink gt_stereo4d.npz + copy stereo4d_meta.json from the
     frames dir (GT stays with the scene for later scoring; frames dir is never written)
  2. stage0-pre (Qwen VLM, combined 1-call default) + stage0 SAM 3 --intrinsics
     -> scene_<clip>/video_input.pkl carrying GT camera_intrinsics
  3. stages t,1..7 DELEGATED UNCHANGED to run_davis_batch_v3.py --skip-stages p,0
     (same pattern as run_tapvid3d_batch_v3.py)

Frame policy: the adapter already subsampled each clip to ~20-38 frames, so this runner
defaults to --max-frames 0 (= keep ALL frames, stride 1) — gt_stereo4d.npz index i
corresponds 1:1 to frames %05d.jpg, and cutting frames here would break that alignment.

Clip names may start with "-" (YouTube ids, e.g. -3Sx43OYGJ8_15081748): pass
--clips=@/path/list.txt (one clip per line) or the '=' form --clips=-3Sx...,... .

Example (smoke):
  python src/run_stereo4d_batch_v3.py --clips=@/tmp/clips.txt \
      --batch-root data/scenes
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]                 # stage0_4
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from src.run_davis_batch_v3 import _BASE_ENV, _run_soft, _run, PY  # noqa: E402

DEFAULT_FRAMES_ROOT = str(_ROOT / "data" / "stereo4d" / "frames")
GT_FILENAME = "gt_stereo4d.npz"
META_FILENAME = "stereo4d_meta.json"
# 512x512 60deg-hfov rectified left eye (stereo4d_adapter.intrinsics_arg fallback)
_FALLBACK_INTRINSICS = "{0:.6f},{0:.6f},256.000000,256.000000".format(
    0.5 * 512 / math.tan(math.radians(30.0)))
# SAM 3 text-concept fallback when the VLM front end is off/failed (open-vocab; Stereo4D
# is VR180 web video, mostly people-centric scenes).
_FALLBACK_CONCEPT = "person"


SKIPPED_JSON = "skipped_static_clips.json"     # 批级剔除名单(可审计 + 重跑幂等)
SKIPPED_DIR = "_skipped_static"                # 被剔除的 scene 目录搬到这里(下划线开头,
                                               # 不会被下游 glob("scene_*") 命中 = 真正剔除)


def _load_skipped(br: str) -> dict:
    """读批级剔除名单 {clip: reason}。文件不存在返回 {}。"""
    p = _ROOT / br / SKIPPED_JSON
    if not p.exists():
        return {}
    try:
        return json.load(open(p)).get("clips", {})
    except Exception:
        return {}


def _record_skipped(br: str, newly: dict) -> None:
    """把 newly {clip: reason} 并进批级名单,并把它们的 scene 目录搬进 _skipped_static/。

    为什么是"搬"而不是"删":scene 目录里有 stage0_pre.json —— 那是**已经花掉的 Qwen API 调用**,
    删了重跑就得重烧。搬进下划线开头的子目录后,`glob("scene_*")` 的下游一个都碰不到(= 用户
    要的"完全剔除"),但名单和 Qwen 结果都留着可审计、可回滚(搬回来即可)。
    """
    if not newly:
        return
    cur = _load_skipped(br)
    cur.update(newly)
    root = _ROOT / br
    (root / SKIPPED_DIR).mkdir(exist_ok=True)
    for clip in newly:
        src, dst = root / f"scene_{clip}", root / SKIPPED_DIR / f"scene_{clip}"
        if src.exists():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.move(str(src), str(dst))
    with open(root / SKIPPED_JSON, "w") as f:
        json.dump({"note": "无动态实例、已从本批剔除;scene 目录在 " + SKIPPED_DIR +
                           "/,把它搬回上一级即可恢复", "clips": cur}, f,
                  indent=1, ensure_ascii=False)


def _qwen_all_static(br: str, clip: str) -> bool:
    """Qwen 是否认为这个 clip 里【没有任何会动的东西】。

    ⚠️ 只有在**单向运动门**(gt_dense_frontend 永不把 static 升级成 dynamic)下,这个判断才等价于
    "最终不会有 dynamic 实例" —— 实测 96 场景零泄漏。若哪天把升级方向加回去,本过滤器必须同时
    重新评估,否则会误杀。
    未知一律当"有动"(保守):没有 stage0_pre.json、解析失败、或 motion 字段缺失/为 None
    (与 pipeline 的 "motion=None 视为 dynamic" 约定一致)。
    """
    p = _ROOT / br / f"scene_{clip}" / "stage0_pre.json"
    if not p.exists():
        return False
    try:
        insts = json.load(open(p)).get("instances", [])
    except Exception:
        return False
    if not insts:
        return False                            # 一个概念都没有 -> 交给下游按老路处理
    return all(i.get("motion") == "static" for i in insts)


def _parse_clips(spec: str, frames_root: str) -> list[str]:
    spec = spec.strip()
    if spec.upper() == "ALL":
        root = Path(frames_root)
        return sorted(d.name for d in root.iterdir()
                      if d.is_dir() and (d / GT_FILENAME).exists())
    if spec.startswith("@"):
        lines = Path(spec[1:]).read_text().splitlines()
        return [l.strip() for l in lines if l.strip()]
    return [s.strip() for s in spec.split(",") if s.strip()]


def _prepare_scene(clip: str, frames_root: str, br: str) -> dict | None:
    """scene_<clip>/ setup: symlink GT + copy meta from the frames dir. Returns the meta
    dict (intrinsics etc.) or None if the clip's frames/GT are missing."""
    fd = Path(frames_root) / clip
    meta_p, gt_p = fd / META_FILENAME, fd / GT_FILENAME
    if not fd.is_dir() or not any(fd.glob("*.jpg")):
        print(f"[adapter {clip}] frames dir missing/empty: {fd} — skipping", flush=True)
        return None
    sd = _ROOT / br / f"scene_{clip}"
    sd.mkdir(parents=True, exist_ok=True)
    if gt_p.exists() and not (sd / GT_FILENAME).exists():
        (sd / GT_FILENAME).symlink_to(gt_p.resolve())      # absolute: survives scene moves
    meta = {}
    if meta_p.exists():
        meta = json.loads(meta_p.read_text())
        if not (sd / META_FILENAME).exists():
            shutil.copy2(meta_p, sd / META_FILENAME)
    meta.setdefault("intrinsics", _FALLBACK_INTRINSICS)
    meta["frames_dir"] = str(fd)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True,
                    help="'ALL' | comma list | @file (one clip per line). Use the '=' "
                         "form (--clips=-3Sx...) — clip names may start with '-'.")
    ap.add_argument("--frames-root", default=DEFAULT_FRAMES_ROOT)
    ap.add_argument("--batch-root", default="data/scenes")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="0 = ALL frames (default; the adapter already subsampled to "
                         "20-38 and GT is index-aligned — do not cut here)")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--prompt-frame", type=int, default=0)
    ap.add_argument("--min-area-frac", type=float, default=0.003)
    ap.add_argument("--no-vlm", action="store_true")
    ap.add_argument("--vlm-model", default="qwen3-vl-plus")
    ap.add_argument("--legacy-vlm-calls", action="store_true",
                    help="stage0-pre uses the original 7-call Qwen path instead of the "
                         "combined 1-call one")
    ap.add_argument("--skip-stages", default="",
                    help="same codes as run_davis_batch_v3 (p,0 handled here; "
                         "t,1,tt,2,b,3..7 delegated). 'g' = GT front-end (this runner)")
    ap.add_argument("--recon", choices=("sam3d", "fastsam3d"), default="sam3d",
                    help="stage1 重建路线(转发给 run_davis_batch_v3):fastsam3d=Fast-SAM3D "
                         "加速 fork(同权重,官宣 2.67x)。论文的 Stereo4D 全量批用 fastsam3d。")
    ap.add_argument("--gt-path2", choices=("auto", "on", "off"), default="off",
                    help="转发给 stage1。**默认 off = Path 1**(每实例 mesh 待在自己 t* 的 SAM3D "
                         "相机系,不做场景级公制 layout)—— ⑦ 本来就只需要逐实例位姿,而 Path 2 的"
                         "公制缩放是唯一会把 mesh 尺寸做错的环节。auto/on 才走 Path 2")
    ap.add_argument("--front", choices=("gt", "t4w"), default="gt",
                    help="dense front-end: 'gt' (default, Stereo4D GT-trajectory attribution, "
                         "pure CPU — skips T4W t/tt downstream) or 't4w' (the tracker front-end)")
    ap.add_argument("--gt-cap", type=int, default=800,
                    help="[--front gt] max tracks per instance (spatial FPS)")
    ap.add_argument("--gt-min-tracks", type=int, default=8,
                    help="[--front gt] drop DYNAMIC instances with fewer attributed tracks")
    ap.add_argument("--gt-min-tracks-static", type=int, default=4,
                    help="[--front gt] lower min-tracks for motion=='static' instances "
                         "(they need only ~4 representative points)")
    ap.add_argument("--gt-min-vis-frames", type=int, default=3,
                    help="[--front gt] a GT track needs >= this many visible frames to be "
                         "attribution-eligible (drops coincidental 1-2 frame tracks)")
    ap.add_argument("--gt-workers", type=int, default=int(os.environ.get("S4D_GT_WORKERS", "1")),
                    help="[--front gt] GT 稠密前端按 scene 并发的进程数(默认 1 = 旧行为;"
                         "也可用环境变量 S4D_GT_WORKERS 设)。这一步纯 CPU 单线程,"
                         "2026-08-20 实测它让 GPU 全程 0%% 而 180 核只用 1 核 —— 是真瓶颈。"
                         "每 worker RSS 仅 0.3~0.4 GB。")
    ap.add_argument("--gt-motion-thresh", type=float, default=0.05,
                    help="[--front gt] GT world-frame motion magnitude above which an instance is "
                         "dynamic; the GT verdict OVERRIDES Qwen concept motion in video_input.pkl")
    ap.add_argument("--n-candidates", type=int, default=None,
                    help="forwarded to run_davis_batch_v3 (stage1); None = default (3,2026-08-19 从 6 改)")
    ap.add_argument("--accept-onmask", type=float, default=None,
                    help="forwarded to run_davis_batch_v3 (stage1 early-stop)")
    ap.add_argument("--onmask-gate", type=float, default=None,
                    help="forwarded to run_davis_batch_v3 (stage1 提前放弃第二道门);"
                         "None = 用引擎模型自带的值(sam3d 关 / fastsam3d 0.65)")
    ap.add_argument("--no-static-skip", action="store_true",
                    help="forwarded to run_davis_batch_v3 (V3 static routing off)")
    ap.add_argument("--no-batch-stage0", action="store_true",
                    help="revert stage0 to one SAM 3 process per clip (reloads the weights "
                         "every clip; only for debugging a batch-runner regression)")
    ap.add_argument("--reuse-frames-dir", action="store_true",
                    help="⚠️ 别开,除非你就是想换输入像素。stage0 batch 直接把数据集帧目录给 SAM 3,"
                         "不再重编码到临时目录。实测(2026-08-12,3 clip):**省不到时间**(23.8s vs "
                         "23.5s,重编码只占 ~0.1s/clip),但**会改结果** —— 旧链路一直是把帧按 "
                         "JPEG q95 重编码后再喂 SAM 3,直接读原图等于换了输入,个别实例掩码面积差 "
                         "28%%。与已产出的 895 场景批次不可比。")
    ap.add_argument("--abandon-frac", type=float, default=0.2,
                    help="转发到 stage1(经 run_davis_batch_v3):提前放弃注定失败的实例。"
                         "**默认 0.2 = 开启**;传 0 关闭。实测省 11.9%% 调用、1.5%% 实例丢达标")
    ap.add_argument("--abandon-cap", type=int, default=2)
    ap.add_argument("--stage2a-workers", type=int, default=1,
                    help="转发给 run_davis_batch_v3:stage2a 同卡并发分片 worker 数。"
                         "PartSAM 是本链路唯一没喂满的阶段(58.4%% TDP / 19.2GB),46GB 卡最多 2")
    ap.add_argument("--qwen-workers", type=int, default=1,
                    help="stage0-pre concurrency. The Qwen call is a DashScope round trip with "
                         "the GPU idle, so >1 overlaps them. 1 = the old serial behaviour.")
                    # 2026-08-15 起 PURE-2a 是【默认】(motion-part 已于 2026-08-13 退场,见 memory
                    # project_motion_part_retired)。--no-motion-groups 保留为向后兼容的 no-op,
                    # 想跑回旧的运动臂用 --motion-groups。
    ap.add_argument("--no-motion-groups", action="store_true", default=True,
                    help="转发给 run_davis_batch_v3:PURE-2a 臂 —— 整条链忽略 Stage-2b 的运动部件"
                         "分解,部件结构一律用 2a 语义 part。详见该脚本同名 flag 的说明。")
    ap.add_argument("--motion-groups", dest="no_motion_groups", action="store_false",
                    help="退回 2026-08-13 之前的运动臂(motion-part 参与部件结构)。仅用于\n"
                         "复现旧产物/对照实验;实测 62.3%% 的非刚体实例 motion_part 恒为 0。")
    ap.add_argument("--no-static-clip-filter", action="store_true",
                    help="关掉『无动态实例的 clip 直接剔除』(默认开)。开启时分两刀:①Qwen 说整个 "
                         "clip 没有任何会动的概念 -> 不跑 SAM 3 就剔除(省算力的主力,实测 "
                         "~93s/clip);②运动门+发射闸跑完后一个 dynamic 实例都不剩 -> 剔除"
                         "(为数据集干净度,只省 ~5s)。被剔除的 scene 目录搬到 _skipped_static/,"
                         f"名单写 {SKIPPED_JSON};搬回上一级即可恢复。"
                         "⚠️ ① 的正确性依赖【单向运动门】(永不把 static 升级成 dynamic),"
                         "改回双向必须重新评估。")
    ap.add_argument("--no-rigid-route", action="store_true",
                    help="转发给 run_davis_batch_v3:关掉 RIGID 路由(默认开)。开启时 Qwen 判 rigid "
                         "且过双闸的实例跳过 Stage 1/2a,表示由稠密轨迹直接建。详见该脚本同名 flag。")
    args = ap.parse_args()
    use_vlm = not args.no_vlm

    skip = {s.strip() for s in args.skip_stages.split(",") if s.strip()}
    clips = _parse_clips(args.clips, args.frames_root)
    br = args.batch_root
    (_ROOT / br).mkdir(parents=True, exist_ok=True)
    # 已被"无动态实例"过滤剔除过的 clip:直接不再处理。**放在最前面**才能让重跑幂等且不重复
    # 烧 Qwen API —— 它们的 scene 目录(含 stage0_pre.json)被移进了 _skipped_static/。
    prev_skipped = _load_skipped(br)
    if prev_skipped and not args.no_static_clip_filter:
        n0 = len(clips)
        clips = [c for c in clips if c not in prev_skipped]
        if n0 != len(clips):
            print(f"[static-filter] 沿用已有剔除名单:{n0 - len(clips)} 个 clip 跳过 "
                  f"(见 {br}/{SKIPPED_JSON})", flush=True)
    print(f"[stereo4d-batch] clips={len(clips)} -> {br}", flush=True)

    # 1) scene setup (numpy-free, runs in this process)
    metas = {}
    for clip in clips:
        m = _prepare_scene(clip, args.frames_root, br)
        if m is not None:
            metas[clip] = m
    clips = [c for c in clips if c in metas]

    # 2a) stage0-pre (Qwen VLM) — per-clip, soft, skip-if-exists. A "p-only" invocation
    # (--skip-stages 0,t,1,tt,2,b,3,4,5,6,7) works for CPU/API prefetch, mirroring tapvid.
    # An EXISTING stage0_pre.json counts regardless of whether we ran stage p this invocation.
    # Before this, has_pre was only filled inside the `"p" not in skip` branch, so any run with
    # --skip-stages p (i.e. the whole point of a decoupled Qwen prefetch pass, and equally a
    # plain resume) silently fell through to --concepts person and threw away the VLM instance
    # list / rigid-nonrigid types for every clip. Quality loss with no error message.
    has_pre = {}
    if use_vlm:
        for clip in clips:
            if (_ROOT / br / f"scene_{clip}" / "stage0_pre.json").exists():
                has_pre[clip] = True
    if use_vlm and "p" not in skip:
        pending = []
        for clip in clips:
            sd = _ROOT / br / f"scene_{clip}"
            sp = sd / "stage0_pre.json"
            if sp.exists():
                has_pre[clip] = True
                continue
            a = ["--frames-dir", metas[clip]["frames_dir"], "--out", str(sp),
                 "--model", args.vlm_model, "--stride", str(args.stride)]
            if args.max_frames:
                a += ["--max-frames", str(args.max_frames)]
            if args.legacy_vlm_calls:
                a += ["--legacy-calls"]
            pending.append((clip, a))

        # Each job is a DashScope round trip (~17 s measured) with the GPU doing nothing, so
        # they overlap freely. Serial, this block is what left 3 of 6 lanes at 0 % util for the
        # first ~25 min of the 1k run (GPU利用率.md §3). Output is captured per job and printed
        # whole, so concurrent stdout can't interleave into unreadable soup.
        nw = max(1, int(args.qwen_workers))
        if pending:
            print(f"[stage0-pre] {len(pending)} clips, {nw} concurrent", flush=True)
        if nw == 1 or len(pending) <= 1:
            for clip, a in pending:
                has_pre[clip] = _run_soft(f"stage0-pre {clip}", "qwen",
                                          "src/stage0/run_stage0_pre.py", a)
        else:
            def _pre(item):
                clip, a = item
                r = subprocess.run([PY["qwen"], str(_ROOT / "src/stage0/run_stage0_pre.py"), *a],
                                   env={**os.environ, **_BASE_ENV}, cwd=str(_ROOT),
                                   capture_output=True, text=True)
                return clip, r.returncode == 0, r.stdout, r.stderr
            with ThreadPoolExecutor(max_workers=nw) as ex:
                for clip, ok, so, se in ex.map(_pre, pending):
                    print(f"\n----- [stage0-pre {clip}] {'ok' if ok else 'FAILED'}", flush=True)
                    if so:
                        print(so, end="", flush=True)
                    if not ok and se:
                        print(se, end="", flush=True)
                    has_pre[clip] = ok

    # 2a-2) 【过滤器 1 / 省算力的主力】Qwen 说"整个 clip 没有任何会动的东西" -> 直接剔除,
    #       **不跑 SAM 3**。放在这里而不是运动门之后,是因为对一个全静态 clip 而言:
    #         stage0-pre(Qwen) ~17s(API,不占 GPU)  ← 到这里为止已花掉的
    #         stage0(SAM 3)    ~85s GPU            ← 这一刀能省的大头
    #         GT 稠密前端       ~3s
    #         stage1/2a        ≈0(静态实例本来就跳过 mesh 链 —— 所以运动门之后再筛,
    #                            该省的早就没花,只剩下面这 ~5s)
    #         2b/pass1/⑦/pass2/finalize ~5s
    #       实测 96 场景:这一刀命中 17 个(18%),每个省 ~93s;运动门之后那一刀命中 30 个(31%)
    #       但每个只省 ~5s。总收益 26min vs 2.5min。
    if use_vlm and not args.no_static_clip_filter and 0 not in {int(s) for s in skip if s.isdigit()}:
        drop = {c: "qwen_all_static" for c in clips if _qwen_all_static(br, c)}
        if drop:
            _record_skipped(br, drop)
            clips = [c for c in clips if c not in drop]
            print(f"[static-filter] Qwen 判定无任何动态概念 -> 剔除 {len(drop)} 个 clip"
                  f"(不跑 SAM 3),剩 {len(clips)} 个:{sorted(drop)[:6]}"
                  f"{' …' if len(drop) > 6 else ''}", flush=True)
        if not clips:
            print("[stereo4d-batch] 所有 clip 都被静态过滤剔除 —— 无事可做", flush=True)
            return

    # 2b) stage0 SAM 3 with GT intrinsics -> video_input.pkl
    #     ONE batch process for the whole sub-batch (SAM 3 weights loaded once) instead of one
    #     process per clip. The per-clip loop cost 123 reloads x ~25 s = 3.9 % of the 2026-08-01
    #     1k run's lane time (see GPU利用率.md §2) — pure idle. --no-batch-stage0 restores the
    #     old path; per-clip skip-if-exists / soft-fail semantics are unchanged either way.
    failed = []
    if "0" not in skip:
        jobs = []
        for clip in clips:
            sd = _ROOT / br / f"scene_{clip}"
            vip = sd / "video_input.pkl"
            if vip.exists():
                print(f"[stage0 sam3:{clip}] skip (video_input.pkl exists)", flush=True)
                continue
            sd.mkdir(parents=True, exist_ok=True)
            sp = sd / "stage0_pre.json"
            j = {"clip": clip, "frames_dir": metas[clip]["frames_dir"], "out": str(vip),
                 "intrinsics": metas[clip]["intrinsics"], "stride": args.stride,
                 "max_frames": args.max_frames, "prompt_frame": args.prompt_frame,
                 "min_area_frac": args.min_area_frac}
            if use_vlm and has_pre.get(clip) and sp.exists():
                j["stage0_pre"] = str(sp)
            else:
                j["concepts"] = _FALLBACK_CONCEPT
            jobs.append(j)

        if jobs and not args.no_batch_stage0:
            jf = _ROOT / br / ".stage0_jobs.jsonl"
            jf.write_text("".join(json.dumps(j) + "\n" for j in jobs))
            a = ["--jobs", str(jf)]
            if args.reuse_frames_dir:
                a.append("--reuse-frames-dir")
            _run_soft(f"stage0 sam3 BATCH n={len(jobs)}", "sam3",
                      "src/stage0/run_stage0_batch.py", a)
            # Truth is on disk, not the exit code: the batch runner soft-fails per clip.
            failed = [j["clip"] for j in jobs if not Path(j["out"]).exists()]
            # stereo4d_lane.sh finishes a sub-batch with `rmdir "$SB"`; anything left in the
            # sub-root makes that fail and strands an empty _laneN_subM dir.
            jf.unlink(missing_ok=True)
        else:
            for j in jobs:
                a = ["--frames-dir", j["frames_dir"], "--stride", str(args.stride),
                     "--prompt-frame", str(args.prompt_frame),
                     "--min-area-frac", str(args.min_area_frac),
                     "--intrinsics", j["intrinsics"], "--out", j["out"]]
                if "stage0_pre" in j:
                    a += ["--stage0-pre", j["stage0_pre"]]
                    tag = f"stage0 sam3:{j['clip']}(VLM)"
                else:
                    a += ["--concepts", _FALLBACK_CONCEPT]
                    tag = f"stage0 sam3:{j['clip']}({_FALLBACK_CONCEPT})"
                if args.max_frames:
                    a += ["--max-frames", str(args.max_frames)]
                if not _run_soft(tag, "sam3", "src/stage0/run_stage0.py", a):
                    failed.append(j["clip"])
        if failed:
            print(f"[stage0] {len(failed)} clips failed: {failed}", flush=True)

    # 2c) GT DENSE FRONT-END (--front gt, default): attribute Stereo4D GT world-frame 3D tracks to
    #     the SAM3 masklets -> contract-identical dense_tracks_<iid>.npz. Pure CPU (numpy+scipy+cv2);
    #     runs AFTER stage0 (needs masklets+gt only), BEFORE Stage 1 (same slot as the T4W front-end).
    #     Since GT covers ALL frames (incl t*), the downstream T4W steps t + tt are force-skipped.
    if args.front == "gt" and "g" not in skip and "t" not in skip:
        # --clips 必须转发下去:前端自己是按 batch-root 下的 scene_* 枚举的,不转发的话
        # 「只修几个 clip」这种调用会把整个根重跑一遍(2026-08-18 实测踩到:508 个场景全被重跑)。
        # 队列批量跑时每个 chunk 有自己的子 root,所以这个坑一直没暴露。
        # ⚠️ 必须写成单个 token 的 `--clips=...`:Stereo4D 的 clip 名以 '-' 开头
        # (如 -cS8QtOfaB4_205205205),拆成 ["--clips", "-cS8Qt..."] 两个 argv 时 argparse 会把
        # 值当成另一个 flag → "expected one argument"。同一个减号陷阱也咬过 stereo4d_queue.sh
        # 的 `grep -qxF "$c"`(要写 `grep -qxF -e "$c"`)。
        gargs = ["--batch-root", str(_ROOT / br), f"--clips={','.join(clips)}",
                 "--cap", str(args.gt_cap), "--min-tracks", str(args.gt_min_tracks),
                 "--min-tracks-static", str(args.gt_min_tracks_static),
                 "--min-vis-frames", str(args.gt_min_vis_frames),
                 "--motion-thresh", str(args.gt_motion_thresh),
                 # 纯 CPU 单线程 numpy,实测是全量跑的真瓶颈(87k~93k 轨迹的 clip 单个几分钟、
                 # 期间 GPU 0%)。按 scene 并发,默认 1 保持旧行为;生产由 S4D_GT_WORKERS 调。
                 "--workers", str(args.gt_workers)]
        _run("gt dense-front", "sam3d-objects", "src/stage4/gt_dense_frontend.py", gargs)

        # 2d) 【过滤器 2 / 数据集干净度,不是为了省算力】运动门 + 发射闸跑完后才能知道的那一批:
        #     Qwen 说有动、但实例要么被单向门降级成 static、要么被 no_own_tracks/few_tracks 挡掉,
        #     最终一个 dynamic 实例都不剩。这种 clip 进不了数据集也没意义,就地剔除。
        #     它只省下游 ~5s(静态实例本就跳过 mesh 链),价值在"别让没用的 clip 混进产物"。
        #     实测 96 场景里这类有 13 个(全静 30 个 - Qwen 阶段已剔的 17 个)。
        if not args.no_static_clip_filter:
            drop = {}
            for clip in clips:
                sp = _ROOT / br / f"scene_{clip}" / "gt_dense_stats.json"
                if not sp.exists():
                    continue                     # 没跑到/没产物 -> 不动它(未知≠静态)
                try:
                    st = json.load(open(sp))
                except Exception:
                    continue
                n_emit, n_static = st.get("n_emit"), st.get("n_emit_static")
                if n_emit is None or n_static is None:
                    continue
                if n_emit == 0:
                    # 前端自己知道更具体的原因就用它(目前只有 "no_masklets" = SAM 3 一个
                    # masklet 都没产出),便于 110k 规模上按原因分桶审计。
                    drop[clip] = st.get("skip_reason") or "no_emitted_instance"
                elif n_emit == n_static:
                    drop[clip] = "all_static_after_gate"
            if drop:
                _record_skipped(br, drop)
                clips = [c for c in clips if c not in drop]
                print(f"[static-filter] 运动门后无 dynamic 实例 -> 剔除 {len(drop)} 个 clip,"
                      f"剩 {len(clips)} 个", flush=True)
            if not clips:
                print("[stereo4d-batch] 所有 clip 都被静态过滤剔除 —— 无事可做", flush=True)
                return

    # 3) stages t,1..7: delegate to the DAVIS V3 orchestrator unchanged (batch stages only
    #    glob scene_*/ under --batch-root; p,0 skipped so --davis-root is never touched)
    # --front gt: the GT front-end already produced dense tracks -> skip T4W (t) and its t*-topup (tt).
    downstream_extra = {"t", "tt"} if args.front == "gt" else set()
    downstream_skip = ",".join(sorted({"p", "0"} | downstream_extra | (skip - {"p", "0"})))
    if all(s in skip for s in ("t", "r", "1", "tt", "2", "b", "3", "4", "5", "6", "7")):
        print("[stereo4d-batch] all downstream stages skipped — done (p/0-only run)")
        return
    cmd = [sys.executable, str(_ROOT / "src" / "run_davis_batch_v3.py"),
           "--sequences", "ALL",
           "--davis-root", args.frames_root,
           "--batch-root", br,
           "--max-frames", str(args.max_frames), "--stride", str(args.stride),
           "--skip-stages", downstream_skip]
    if args.n_candidates is not None:
        cmd += ["--n-candidates", str(args.n_candidates)]
    if args.accept_onmask is not None:
        cmd += ["--accept-onmask", str(args.accept_onmask)]
    if args.onmask_gate is not None:
        cmd += ["--onmask-gate", str(args.onmask_gate)]
    if args.no_static_skip:
        cmd += ["--no-static-skip"]
    if args.recon != "sam3d":
        cmd += ["--recon", args.recon]
    if args.stage2a_workers != 1:
        cmd += ["--stage2a-workers", str(args.stage2a_workers)]
    # 无条件转发(同 run_davis_batch_v3 的理由:默认非 0 后,条件转发会让显式的 0 关不掉)
    cmd += ["--abandon-frac", str(args.abandon_frac), "--abandon-cap", str(args.abandon_cap)]
    if not args.no_motion_groups:      # 默认已是 PURE-2a,只需转发"关掉"那一侧
        cmd += ["--motion-groups"]
    if args.no_rigid_route:
        cmd += ["--no-rigid-route"]
    cmd += ["--gt-path2", args.gt_path2]
    print(f"\n===== [delegate t,1..7] {' '.join(cmd[1:])}\n", flush=True)
    r = subprocess.run(cmd, env={**os.environ, **_BASE_ENV}, cwd=str(_ROOT))
    if r.returncode != 0:
        raise SystemExit(f"[stereo4d-batch] downstream stages FAILED (exit {r.returncode})")


if __name__ == "__main__":
    main()
