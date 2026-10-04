"""DAVIS batch orchestrator: the FULL v2 chain on real DAVIS video, end to end to the final
sparse-trajectory representation.

DAVIS = real monocular video with NO GT camera -> Stage 1 runs Path 1 (SAM 3D native layout).
Stage 4 projects keypoints with SAM 3D's OWN camera (object_pose_cam + sam3d_intrinsics, produced
on both paths), so the whole GT-free chain works. Every downstream batch stage (1, 2a, 3, 4, 2b,
pass2, §4) is REUSED UNCHANGED — they all glob scene_*/ ; this driver only adds the SAM 3 Stage 0
per-sequence loop (DAVIS frames + a text concept -> masklets) and chains the rest.

ROUTING (what never enters the mesh chain at all): masklets the Stage-0-pre VLM judged
motion=='static', plus — since 2026-08-13 — the RIGID instances that pass both gates in
src/stage1/rigid_route.py (stage 'r'). Both get their final representation built at finalize
straight from the dense tracks; for a rigid instance that is all the mesh chain ever produced
(median 4 points + SE(3)), so this skips Stage 1 + Stage 2a = 96.9% of the chain for ~30% of the
mesh-bearing instances. --no-rigid-route / --skip-stages r reverts bit-for-bit.

Per sequence -> <batch-root>/scene_<seqname>/ (2026-07-30 restructure: 2b-early before pass1):
  video_input.pkl (SAM 3) -> dense_tracks_*.npz (t4w) -> rigid_route.json -> inst_*.npz (SAM 3D) -> parts_*.npz
  (PartSAM) -> stage2b_early.json/.npz (2b-early: dense motion analysis, sem->motion map)
  -> pass1_keypoints.pkl (per MOTION part) -> tracks_d4rt.npz (⑦ link) -> stage2b.npz
  (materialized) -> pass2.npz -> final_repr.pkl

Each stage runs as ONE subprocess in its own conda env. Stage 0 is a per-sequence loop (SAM 3 is
single-video); stages 1-§4 are single batch calls (model loaded once, loops all scene dirs).

Example:
    python src/run_davis_batch_v3.py --sequences blackswan:swan,bear:bear,car-turn:car \
        --batch-root outputs/mono_batch --max-frames 24
    # concept defaults to the sequence name if ":concept" omitted (e.g. "bear" -> concept "bear")
    # resume:  --skip-stages 0,1
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]                 # stage0_4
# conda envs (one per stage): override with CONDA_ENVS_DIR or the per-env *_PY variables below
_ENVS = Path(os.environ.get("CONDA_ENVS_DIR", str(Path.home() / "miniconda3" / "envs")))
PY = {
    "sam3":          os.environ.get("SAM3_PY",   str(_ENVS / "sam3" / "bin" / "python")),
    "sam3d-objects": os.environ.get("SAM3D_PY",  str(_ENVS / "sam3d-objects" / "bin" / "python")),
    "PartSAM":       os.environ.get("PARTSAM_PY", str(_ENVS / "PartSAM" / "bin" / "python")),
    "track4world":   os.environ.get("T4W_PY",    str(_ENVS / "track4world" / "bin" / "python")),
    # Stage 0-pre (Qwen via DashScope) only needs numpy+PIL+stdlib urllib — reuse any env that has
    # them (default: sam3d-objects). Needs DASHSCOPE_API_KEY in the environment (export from ~/.bashrc).
    "qwen":          os.environ.get("QWEN_PY",   str(_ENVS / "sam3d-objects" / "bin" / "python")),
}
_BASE_ENV = {
    "TMPDIR": os.environ.get("TMPDIR", "/tmp"),
    "HF_HUB_OFFLINE": "1",
    "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
    # 两个名字都要设(2026-08-12):`PYTORCH_ALLOC_CONF` 是新版 torch 才引入的通用名,
    # torch<2.7 只认 `PYTORCH_CUDA_ALLOC_CONF`。本机 PartSAM env(2.4.1)和 sam3d-objects
    # env(2.5.1)的 libtorch_cuda.so 里只有后者、sam3 env(2.10)两个都有 —— 所以在此之前
    # 只设新名字 = stage1/stage2a 的 expandable_segments 【静默失效】,只有 stage0 生效。
    # 验证办法:strings <env>/lib/python*/site-packages/torch/lib/libtorch_cuda.so | grep -x 名字
    "PYTORCH_ALLOC_CONF": "expandable_segments:True",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}

# SAM 3 text concept per DAVIS sequence (the main foreground object). A few are best-guess
# (india/libby/stunt/varanus/tuk-tuk); SAM 3 is open-vocab so most segment fine. Sequences not
# listed fall back to the first hyphen-token of the name.
_DAVIS_CONCEPTS = {
    "bear": "bear", "bike-packing": "person", "blackswan": "swan", "bmx-bumps": "person",
    "bmx-trees": "person", "boat": "boat", "boxing-fisheye": "person", "breakdance": "person",
    "breakdance-flare": "person", "bus": "bus", "camel": "camel", "car-roundabout": "car",
    "car-shadow": "car", "car-turn": "car", "cat-girl": "cat", "classic-car": "car",
    "color-run": "person", "cows": "cow", "crossing": "person", "dance-jump": "person",
    "dance-twirl": "person", "dancing": "person", "disc-jockey": "person", "dog": "dog",
    "dog-agility": "dog", "dog-gooses": "dog", "dogs-jump": "dog", "dogs-scale": "dog",
    "drift-chicane": "car", "drift-straight": "car", "drift-turn": "car", "drone": "drone",
    "elephant": "elephant", "flamingo": "flamingo", "goat": "goat", "gold-fish": "fish",
    "hike": "person", "hockey": "person", "horsejump-high": "horse", "horsejump-low": "horse",
    "india": "elephant", "judo": "person", "kid-football": "person", "kite-surf": "person",
    "kite-walk": "person", "koala": "koala", "lab-coat": "person", "lady-running": "person",
    "libby": "dog", "lindy-hop": "person", "loading": "person", "longboard": "person",
    "lucia": "person", "mallard-fly": "duck", "mallard-water": "duck", "mbike-trick": "motorcycle",
    "miami-surf": "person", "motocross-bumps": "motorcycle", "motocross-jump": "motorcycle",
    "motorbike": "motorcycle", "night-race": "car", "paragliding": "person",
    "paragliding-launch": "person", "parkour": "person", "pigs": "pig", "planes-water": "airplane",
    "rallye": "car", "rhino": "rhino", "rollerblade": "person", "schoolgirls": "person",
    "scooter-black": "person", "scooter-board": "person", "scooter-gray": "person", "sheep": "sheep",
    "shooting": "person", "skate-park": "person", "snowboard": "person", "soapbox": "person",
    "soccerball": "ball", "stroller": "person", "stunt": "car", "surf": "person", "swing": "person",
    "tennis": "person", "tractor-sand": "tractor", "train": "train", "tuk-tuk": "vehicle",
    "upside-down": "person", "varanus-cage": "lizard", "walking": "person",
}


_TIMINGS = []


def _run(tag, env_key, script, script_args):
    py = PY[env_key]
    if not Path(py).exists():
        raise SystemExit(f"[{tag}] env python not found: {py}")
    cmd = [py, str(_ROOT / script), *script_args]
    print(f"\n===== [{tag}] env={env_key}\n      {script} {' '.join(script_args)}\n", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, env={**os.environ, **_BASE_ENV}, cwd=str(_ROOT))
    dt = time.time() - t0
    _TIMINGS.append((tag, dt))
    if r.returncode != 0:
        raise SystemExit(f"[{tag}] FAILED (exit {r.returncode}) after {dt:.0f}s")
    print(f"===== [{tag}] DONE in {dt:.0f}s ({dt/60:.1f} min)", flush=True)


def _run_parallel(tag, env_key, script, arg_lists):
    """像 _run,但把同一个脚本按不同参数并发起 N 个进程,全部成功才算成功。
    用于同卡多 worker(目前只有 stage2a PartSAM 有这个余量 —— 见 --stage2a-workers)。
    任一 worker 非零退出就抛,和 _run 的语义一致:半成品比明确失败更难查。"""
    py = PY[env_key]
    if not Path(py).exists():
        raise SystemExit(f"[{tag}] env python not found: {py}")
    print(f"\n===== [{tag}] env={env_key} x{len(arg_lists)} workers\n      {script}\n", flush=True)
    t0 = time.time()
    procs = []
    for a in arg_lists:
        cmd = [py, str(_ROOT / script), *a]
        procs.append((a, subprocess.Popen(cmd, env={**os.environ, **_BASE_ENV}, cwd=str(_ROOT))))
    bad = []
    for a, p in procs:
        if p.wait() != 0:
            bad.append((a, p.returncode))
    dt = time.time() - t0
    _TIMINGS.append((tag, dt))
    if bad:
        raise SystemExit(f"[{tag}] FAILED after {dt:.0f}s — workers {bad}")
    print(f"===== [{tag}] DONE in {dt:.0f}s ({dt / 60:.1f} min, {len(arg_lists)} workers)",
          flush=True)


def _parse_sequences(spec, davis_root):
    """'blackswan:swan,bear' -> [('blackswan','swan'), ('bear','bear')]; concept defaults to the
    _DAVIS_CONCEPTS map then the first hyphen-token. 'ALL' -> every sequence dir under davis_root."""
    if spec.strip().upper() == "ALL":
        seqs = sorted(d.name for d in Path(davis_root).iterdir() if d.is_dir())
        return [(s, _DAVIS_CONCEPTS.get(s, s.split("-")[0])) for s in seqs]
    out = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        seq, sep, concept = item.partition(":")
        if not concept:
            concept = _DAVIS_CONCEPTS.get(seq, seq.split("-")[0])
        out.append((seq, concept))
    return out


def _run_soft(tag, env_key, script, script_args):
    """Like _run but returns True/False instead of raising — for the per-sequence Stage 0 loop so
    one bad sequence can't kill a 90-sequence sweep."""
    py = PY[env_key]
    cmd = [py, str(_ROOT / script), *script_args]
    print(f"\n----- [{tag}] {' '.join(script_args[-2:])}", flush=True)
    r = subprocess.run(cmd, env={**os.environ, **_BASE_ENV}, cwd=str(_ROOT))
    if r.returncode != 0:
        print(f"[{tag}] FAILED (exit {r.returncode}) — skipping", flush=True)
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequences", required=True,
                    help="'ALL' for every sequence under --davis-root, or comma list of seq[:concept] "
                         "(concept defaults to the built-in DAVIS map, then the first hyphen-token)")
    ap.add_argument("--davis-root", default="data/DAVIS/JPEGImages/480p",
                    help="dir holding per-sequence frame folders")
    ap.add_argument("--batch-root", default="outputs/mono_batch")
    ap.add_argument("--max-frames", type=int, default=24)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--prompt-frame", type=int, default=0)
    ap.add_argument("--min-area-frac", type=float, default=0.003,
                    help="drop masklets whose largest single-frame mask < frac*H*W (0.003=0.3%%≈35px "
                         "object); kills tiny distant clutter/over-enumeration, keeps substantial "
                         "objects incl. static rigid. Set 0 to disable.")
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--n-candidates", type=int, default=3,
                    help="Stage1 Path1 候选参考帧上限。**2026-08-19 用户拍板 6 -> 3**。"
                         "实测(逐实例回放完整观测的候选序列,零 GPU):官方 SAM3D 2337 实例 "
                         "k=6->3 调用/实例 2.72->1.98(-27%%)、达标率 56.1%%->50.4%%、"
                         "best-on-mask 0.8498->0.8369;Fast-SAM3D 142 实例 "
                         "4.21->2.56(-39%%)、29.6%%->21.8%%、0.8041->0.7848。"
                         "(旧默认 6 的依据是 on-mask 90.8%%,那是 t* 上的口径,不是达标率)")
    ap.add_argument("--accept-onmask", type=float, default=0.9,
                    help="Stage1 multi-candidate early-stop on-mask threshold.")
    ap.add_argument("--no-vlm", action="store_true",
                    help="disable the Stage 0-pre VLM (Qwen) front end and fall back to the built-in "
                         "_DAVIS_CONCEPTS text concepts (one per sequence, no box, no object_type)")
    ap.add_argument("--vlm-model", default="qwen3-vl-plus")
    ap.add_argument("--no-static-skip", action="store_true",
                    help="ctrl/escape hatch: run the FULL chain for every instance, ignoring the "
                         "VLM motion=='static' routing (forwarded to stage1 + finalize)")
    ap.add_argument("--skip-stages", default="",
                    help="comma list of stage idx to skip (p=stage0-pre VLM, 0=sam3, "
                         "t=dense front-end, r=rigid routing (2026-08-13), 1=sam3d, "
                         "tt=t* top-up (A'), 2=partsam, "
                         "b=stage2b-early (dense motion analysis, 2026-07-30 restructure), "
                         "3=pass1, 4=d4rt, 5=stage2b materialize, 6=pass2, 7=finalize)")
    ap.add_argument("--recon", choices=("sam3d", "fastsam3d"), default="sam3d",
                    help="stage1 重建路线:sam3d=官方 SAM 3D(通用);fastsam3d=Fast-SAM3D "
                         "(ICML26 training-free 加速 fork,官宣 2.67x,同权重)。"
                         "论文中的 Stereo4D 全量批使用 fastsam3d。")
    ap.add_argument("--gt-path2", choices=("auto", "on", "off"), default="off",
                    help="转发给 stage1(sam3d/fastsam3d):有 gt_stereo4d.npz 就补 GT 外参+"
                         "gt_object_extent 走 Path 2,mesh 统一落在 W_gt。**默认 off** —— 受控实测"
                         "Path 1 匹配率 0.7149 > Path 2 0.7083,且 Path 2 的公制缩放是唯一会把"
                         "mesh 尺寸做错的地方(实测 −14.4%%);详见 run_stage1.py 的说明")
    ap.add_argument("--abandon-frac", type=float, default=0.2,
                    help="转发给 stage1:提前放弃注定达不到 --accept-onmask 的实例。"
                         "**默认 0.2 = 开启**;传 0 关闭。三臂 A/B 实测:省 11.9%% 调用,"
                         "代价 on-mask -0.0099(显著)、1.5%% 实例丢掉达标")
    ap.add_argument("--abandon-cap", type=int, default=2,
                    help="转发给 stage1:被判'注定失败'的实例最多试几个候选。"
                         "⚠️ --n-candidates 改成 3 后,cap=2 每个命中实例最多只省 1 次调用")
    ap.add_argument("--onmask-gate", type=float, default=None,
                    help="转发给 stage1:提前放弃的第二道门(rank-1 实测 on_mask < 此值就停)。"
                         "默认不转发 = 用引擎模型自带的值(sam3d 关 / fastsam3d 0.65);传 0 强制关")
    ap.add_argument("--stage2a-workers", type=int, default=1,
                    help="stage2a 在同一张卡上并发起几个分片 worker。默认 1。"
                         "实测(2026-08-12,独占 L20):PartSAM 单 worker 只跑到 58.4%% TDP / "
                         "19.2GB —— 是本链路唯一'kernel 在跑但 SM 没喂满'的阶段;SAM3(23GB/84.5%%) "
                         "与 SAM3D(26GB/84.5%%)都已喂饱且放不下第二份,别对它们用这招。"
                         "显存按 workers x 19.2GB 估,46GB 的卡最多 2。")
    ap.add_argument("--no-rigid-route", action="store_true",
                    help="关掉 RIGID 路由(2026-08-13):默认开 —— Qwen 判 rigid 且通过双闸"
                         "(单在场段 + 无 mesh 保守刚体门附议)的实例跳过 Stage 1 与 Stage 2a,"
                         "表示在 finalize 直接由稠密轨迹建(k 点 + SE(3))。实测 rigid 占带 mesh "
                         "实例的 30%%,而 stage1+2a 占全链 96.9%%,且刚体最终只留 median 4 个点。"
                         "关掉 = 与本功能引入前逐位一致。等价写法:--skip-stages r")
                    # 2026-08-15 起 PURE-2a 是【默认】(motion-part 已于 2026-08-13 退场,见 memory
                    # project_motion_part_retired)。--no-motion-groups 保留为向后兼容的 no-op,
                    # 想跑回旧的运动臂用 --motion-groups。
    ap.add_argument("--no-motion-groups", action="store_true", default=True,
                    help="PURE-2a 臂(2026-08-13):整条链忽略 Stage-2b 的运动部件分解 —— pass1 按 "
                         "2a 语义 part 采样(且不读 parts_motion_*.npz),stage5 把 motion_part 全写 "
                         "0 并标 motion_part_source='2a',pass2 对非刚体改成每个语义 part 取 1 个 "
                         "medoid keypoint。⑦ 不变(查不到映射键 -> peer 层退化成整实例 = legacy)。"
                         "本 flag 不跳过 stage b:2b-early 跑一次即可,之后 pass1 起两条臂对跑;"
                         "确定不要运动臂时另加 --skip-stages b 省掉那步。")
    ap.add_argument("--motion-groups", dest="no_motion_groups", action="store_false",
                    help="退回 2026-08-13 之前的运动臂(motion-part 参与部件结构)。仅用于\n"
                         "复现旧产物/对照实验;实测 62.3%% 的非刚体实例 motion_part 恒为 0。")
    args = ap.parse_args()
    # 默认已是 PURE-2a(2026-08-15),所以要转发的是【关掉它】那一侧;转发 --no-motion-groups
    # 现在是 no-op,写反了会静默退回运动臂。
    nmg = [] if args.no_motion_groups else ["--motion-groups"]
    use_vlm = not args.no_vlm

    skip = {s.strip() for s in args.skip_stages.split(",") if s.strip()}     # e.g. {"p","0","1"}
    skip_int = {int(s) for s in skip if s.lstrip("-").isdigit()}
    seqs = _parse_sequences(args.sequences, str(_ROOT / args.davis_root))
    br = args.batch_root
    (_ROOT / br).mkdir(parents=True, exist_ok=True)

    # 0-pre: Stage 0-pre (Qwen) — per-sequence, writes scene_<seq>/stage0_pre.json (instance list +
    # box + rigid/non-rigid). Soft + skip-if-exists. A sequence whose VLM step fails falls back to the
    # built-in text concept at Stage 0 (no box, no object_type) so the sweep still completes.
    has_pre = {}
    if use_vlm and "p" not in skip and 0 not in skip_int:
        print(f"[stage0-pre] {len(seqs)} sequences (model={args.vlm_model})", flush=True)
        for seq, _concept in seqs:
            frames_dir = _ROOT / args.davis_root / seq
            sd = _ROOT / br / f"scene_{seq}"
            sp = sd / "stage0_pre.json"
            if sp.exists():
                print(f"[stage0-pre {seq}] skip (stage0_pre.json exists)", flush=True)
                has_pre[seq] = True; continue
            if not frames_dir.exists():
                continue
            sd.mkdir(parents=True, exist_ok=True)
            a = ["--frames-dir", str(frames_dir), "--out", str(sp), "--model", args.vlm_model,
                 "--stride", str(args.stride)]
            if args.max_frames:
                a += ["--max-frames", str(args.max_frames)]
            has_pre[seq] = _run_soft(f"stage0-pre {seq}", "qwen", "src/stage0/run_stage0_pre.py", a)

    # 0: Stage 0 SAM 3 — per-sequence (SAM 3 is single-video); writes scene_<seq>/video_input.pkl.
    # Soft per-sequence (skip-if-exists + tolerate failures) so a 90-sequence sweep is resumable.
    failed = []
    if 0 not in skip_int:
        print(f"[stage0] {len(seqs)} sequences", flush=True)
        for seq, concept in seqs:
            frames_dir = _ROOT / args.davis_root / seq
            sd = _ROOT / br / f"scene_{seq}"
            vip = sd / "video_input.pkl"
            if vip.exists():
                print(f"[stage0 sam3:{seq}] skip (video_input.pkl exists)", flush=True)
                continue
            if not frames_dir.exists():
                print(f"[stage0 sam3:{seq}] frames dir not found: {frames_dir} — skipping", flush=True)
                failed.append(seq); continue
            sd.mkdir(parents=True, exist_ok=True)
            a = ["--frames-dir", str(frames_dir),
                 "--stride", str(args.stride), "--prompt-frame", str(args.prompt_frame),
                 "--min-area-frac", str(args.min_area_frac),
                 "--out", str(vip)]
            sp = sd / "stage0_pre.json"
            if use_vlm and has_pre.get(seq) and sp.exists():
                a += ["--stage0-pre", str(sp)]            # VLM instances + box + object_type
                tag = f"stage0 sam3:{seq}(VLM)"
            else:
                a += ["--concepts", concept]              # fallback: built-in text concept
                tag = f"stage0 sam3:{seq}({concept})"
            if args.max_frames:
                a += ["--max-frames", str(args.max_frames)]
            if not _run_soft(tag, "sam3", "src/stage0/run_stage0.py", a):
                failed.append(seq)
        if failed:
            print(f"[stage0] {len(failed)} sequences failed/skipped: {failed}", flush=True)

    # ③ V3 DENSE FRONT-END tracker (SpaTrackerV2). Seeds a dense grid in each instance's mask across
    #    MULTIPLE frames -> dense_tracks_<iid>.npz (W_stv 3D trajectories + vis for the whole clip).
    #    Runs after Stage 0 (needs masklets only). BA ON by default (Ego4D large-camera-motion).
    #    Skip with --skip-stages t.
    #    Front-end = Track4World (bake-off v2 winner 2026-07-11: PO EPE2D 10.4 vs STV2 16.9,
    #    relock@20 0.35 vs 0.19; see src/bakeoff2/). Stereo4D batches skip this stage and use the
    #    released tracks instead (src/stage4/gt_dense_frontend.py, called by run_stereo4d_batch_v3.py).
    if "t" not in skip:
        _run("v3 dense-front", "track4world", "src/stage4/run_dense_frontend_t4w.py", ["--batch-root", br])

    # r / RIGID ROUTING (2026-08-13): decide which rigid instances skip the mesh chain entirely.
    #    Must run AFTER the dense front-end (gate 2 judges the raw dense tracks) and BEFORE Stage 1
    #    (that is what it gates). Writes scene_*/rigid_route.json; Stage 1 skips the routed
    #    instances (so Stage 2a never sees them either — no inst_*.npz), and finalize builds their
    #    entry straight from the tracks. Skip with --skip-stages r (or --no-rigid-route) -> nothing
    #    routed -> the chain is bit-for-bit what it was before. Same thresholds as 2b-early.
    if "r" not in skip and not args.no_rigid_route:
        _run("rigid-route", "sam3", "src/stage1/rigid_route.py",
             ["--batch-root", br, "--tau-rigid", "0.05", "--min-kp", "6"])

    # 1: Stage 1 SAM 3D (Path 1 — no GT camera). --max-position-error is a Path-2-only grade; omit.
    if 1 not in skip_int:
        s1_args = ["--batch-root", br, "--n-candidates", str(args.n_candidates),
                   "--accept-onmask", str(args.accept_onmask),
                   "--engine", args.recon, "--gt-path2", args.gt_path2]
        # 无条件转发:默认已非 0,若写成 `if args.abandon_frac:` 则显式传 0 时不转发,
        # stage1 会退回它自己的默认 0.2 —— 用户的关闭意图被静默忽略。
        s1_args += ["--abandon-frac", str(args.abandon_frac),
                    "--abandon-cap", str(args.abandon_cap)]
        if args.onmask_gate is not None:      # None = 交给引擎模型自带的默认值
            s1_args += ["--onmask-gate", str(args.onmask_gate)]
        if args.no_static_skip:
            s1_args.append("--no-static-skip")
        if args.no_rigid_route:
            s1_args.append("--no-rigid-route")
        _run(f"stage1 {args.recon}", "sam3d-objects", "src/stage1/run_stage1.py", s1_args)
    # tt / A' t* TOP-UP (2026-07-29): the dense front-end (step t) runs BEFORE Stage 1 (branch
    #    order), so the instance t* frames (inst_<iid>.npz reference_frame_index) are unknown
    #    when its pass frames are picked — the "MANDATORY t* pass" fell through for every
    #    instance (MOVi v4 regression root cause #1). This light step re-runs the T4W front-end
    #    in --tstar-topup mode: only instances whose existing dense tracks have < threshold
    #    visible tracks at t* get ONE extra pass at t*, merged into dense_tracks_<iid>.npz in the
    #    file's own world (Umeyama + guard, c2w fallback). MUST run after Stage 1 and before ⑦
    #    (run_link_dense pairs strictly at t*). no-op for the stv2 front-end (per-point query
    #    frames make t* seeding native there). Skip with --skip-stages tt.
    if "tt" not in skip:
        _run("v3 t*-topup", "track4world", "src/stage4/run_dense_frontend_t4w.py",
             ["--batch-root", br, "--tstar-topup"])
    # 2: Stage 2a PartSAM — 可在同一张卡上并发多个分片 worker(见 --stage2a-workers)
    if 2 not in skip_int:
        nw2 = max(1, int(args.stage2a_workers))
        if nw2 == 1:
            _run("stage2a partsam", "PartSAM", "src/stage2/stage2a_runner.py", ["--batch-root", br])
        else:
            _run_parallel("stage2a partsam", "PartSAM", "src/stage2/stage2a_runner.py",
                          [["--batch-root", br, "--shard", f"{i}/{nw2}"] for i in range(nw2)])
    # b / 2b-EARLY (2026-07-30 restructure): dense-trajectory motion analysis BEFORE pass1 —
    #    associates the t4w dense tracks to the Stage-2a semantic parts at t* (shared 4x-fix
    #    projection) and runs the EXISTING stage2b analysis on them -> stage2b_early.json
    #    (sem->motion map). pass1 then samples per MOTION part; ⑦ gains a motion-part propagation
    #    peer tier; stage 5 below materializes the map keypoint-aligned. Skipping this (or any
    #    missing product) falls every consumer back to its legacy per-semantic-part behavior.
    #    Thresholds tau_rigid/min_kp: same V3-recalibrated values as stage 5 (0.05/6).
    if "b" not in skip:
        _run("stage2b-early", "sam3", "src/stage2/run_stage2b_dense.py",
             ["--batch-root", br, "--tau-rigid", "0.05", "--min-kp", "6"])
    # 3 / ⑥: Stage 3 pass1 — V3 whole-mesh keypoints (default; --front-facing-only reverts to V2).
    #    Sampling unit = MOTION part when scene_*/stage2b_early.json exists (stage b), else semantic.
    if 3 not in skip_int:
        _run("stage3 pass1", "sam3", "src/stage3/stage3_pass1.py",
             ["--batch-root", br, "--alpha", str(args.alpha)] + nmg)
    # 4 / ⑦: V3 coordinate unification (Umeyama T: W_stv->W_mesh) + keypoint↔dense-track linking
    #    (visible: t* pixel+3D gate; occluded: per-part SE(3) propagation). Emits tracks_d4rt.npz
    #    (SAME contract as V2), so Stage 2b/pass2/finalize below are unchanged. Replaces V2 stage4 stv2.
    if 4 not in skip_int:
        _run("v3 link-dense", "sam3d-objects", "src/stage4/run_link_dense.py", ["--batch-root", br])
    # 5: Stage 2b MATERIALIZE (2026-07-30 restructure): the motion analysis itself moved EARLY
    #    (stage b, on dense trajectories); this call now only materializes the sem->motion map
    #    keypoint-aligned into stage2b.npz (contract unchanged — pass2 reads it as before). The
    #    keypoint-level analysis still runs unchanged as the e_smooth/audit source, and scenes
    #    WITHOUT a 2b-early product keep the full legacy keypoint-clustering behavior (skip code
    #    "5" unchanged). V3-recalibrated thresholds (2026-07-02, 9-scene sweep): tau_rigid=0.05,
    #    min_kp 4->6.
    if 5 not in skip_int:
        _run("stage2b", "sam3", "src/stage2/run_stage2b_batch.py",
             ["--batch-root", br, "--scenes-root", str(_ROOT / br / "_nogt"),
              "--tau-rigid", "0.05", "--min-kp", "6",
              "--out", str(_ROOT / br / "davis_stage2b_eval.json")] + nmg)
    # 6: Stage 3 pass2
    if 6 not in skip_int:
        _run("stage3 pass2", "sam3", "src/stage3/stage3_pass2.py", ["--batch-root", br] + nmg)
    # 7: §4 finalize -> final_repr.pkl
    if 7 not in skip_int:
        fin_args = ["--batch-root", br]
        if args.no_static_skip:
            fin_args.append("--no-static-skip")
        if args.no_rigid_route:
            fin_args.append("--no-rigid-route")
        _run("stage4 finalize", "sam3", "src/stage4/stage4_finalize.py", fin_args)

    if _TIMINGS:
        tot = sum(d for _, d in _TIMINGS)
        print(f"\n===== 各 stage 耗时 (合计 {tot/60:.1f} min) =====")
        for tg, d in _TIMINGS:
            print(f"  {tg:<28} {d:>7.0f}s  {d/max(tot,1e-9):>5.1%}")
    print(f"\n===== DAVIS BATCH DONE -> {br}")
    for sd in sorted((_ROOT / br).glob("scene_*")):
        files = {p.name for p in sd.glob("*")}
        done = ("✓final" if "final_repr.pkl" in files else
                "✓track" if "tracks_d4rt.npz" in files else
                "kp" if "pass1_keypoints.pkl" in files else
                "mesh" if any(f.startswith("inst_") for f in files) else "...")
        print(f"  {sd.name}: {done}")


if __name__ == "__main__":
    main()
