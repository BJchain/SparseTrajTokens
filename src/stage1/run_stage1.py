"""M0 Stage 1 runner (env: sam3d-objects).

VideoInput pickle (from Stage 0)  ->  SCHEMA.stage0_to_stage1  ->  SceneReconstruction:
per-instance STATIC mesh placed in one shared scene frame. Writes each InstanceMesh as
  inst_<id>.obj   (viewable)  +  inst_<id>.npz  (lossless verts/faces/transforms)
plus scene_manifest.json.

两个可切换的维度(2026-08-09 加):
  --engine  sam3d | fastsam3d   重建后端。fastsam3d = Fast-SAM3D(ICML26,training-free
            加速 fork,官宣 2.67x)。两个仓库顶层包同名,只能靠 sys.path 顺序二选一,
            所以后端在 src/stage1/sam3d_engine.py 里【延迟】构建 —— 本文件不再在模块
            顶层 import inference。
  --gt-path2 auto | on | off    有 gt_stereo4d.npz 的场景(Stereo4D)把 GT 外参 +
            gt_object_extent 补进 VideoInput,使 compose_scene 走 **Path 2**:所有 mesh
            落进【同一个】GT 世界系 W_gt,而不是 Path 1 每实例各自的 sam3d_layout。
            细节与 gt_object_extent 的定义标定见 src/stage1/gt_path2.py。

            **默认 off(2026-08-10 定案)**。受控实测(同一份 mesh,只改坐标系,127 实例):
            Path 1 匹配率 0.7149 vs Path 2 0.7083。而且 ⑦ 的每一个 3D 门限都按该实例自己的
            mesh bbox 归一化、投影又用 object_pose_cam@inv(object_to_scene) 抵消世界系,所以
            **⑦ 只需要"该实例 canonical -> 自己 t* 相机"这一个位姿,mesh 落在哪个世界系是
            自由选择**;而 Path 2 为了拼出场景级公制系要额外做一次 scale=gt_ext/placed_ext 的
            缩放,那正是唯一可能把尺寸做错的地方(实测偏小/偏近 14.4%)。不要那个场景系,就
            不会有那个缩放,也就不会错。Path 2 换来的 quality 打分只是诊断、公制 mesh 只有
            可视化用得上,本链路都不消费。

Run:
  TMPDIR=/tmp HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  <sam3d-objects-env-python> src/stage1/run_stage1.py \
      --video-input outputs/m0_runs/blackswan/video_input.pkl \
      --out-dir    outputs/m0_runs/blackswan
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# --- env bring-up BEFORE importing the SAM 3D inference module ------------------
# inference.py top-level does os.environ["CUDA_HOME"] = os.environ["CONDA_PREFIX"];
# running <env>/bin/python directly does NOT set CONDA_PREFIX, so set it from the
# interpreter location.
os.environ.setdefault("CONDA_PREFIX", str(Path(sys.executable).resolve().parents[1]))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch  # noqa: E402

# DINOv2 is pulled via torch.hub.load without trust_repo; the Dino class can't pass it,
# so the caller must inject trust_repo=True (+ keep cached weights). Proper def, not a
# lambda: Dino calls torch.hub.load(repo_or_dir=..., model=...) with KEYWORDS.
_orig_hub_load = torch.hub.load
def _patched_hub_load(repo_or_dir, model=None, *a, **k):  # noqa: E305
    k.setdefault("trust_repo", True)
    k.setdefault("force_reload", False)
    # 仓库已缓存就改走 source='local',彻底不碰网络。
    # 起因(2026-08-12):即使 repo 和权重都在 ~/.cache/torch/hub 里,source='github' 仍会去
    # GitHub API 做仓库校验,而本机到 github 经常超时 —— 实测 stage1 在
    # `Dino.__init__` 里挂掉:InstantiationException(RemoteDisconnected)。
    # 这不是"偶尔重试一下"的小事:模型加载在每个 sub-batch 开头,一次 110k 长跑要加载几千次,
    # 任何一次网络抖动都会让整个 chunk 失败重排。HF_HUB_OFFLINE 管不到 torch.hub。
    try:
        if k.get("source", "github") == "github" and isinstance(repo_or_dir, str) \
                and "/" in repo_or_dir and not os.path.isdir(repo_or_dir):
            owner, repo = repo_or_dir.split(":")[0].split("/")[:2]
            import glob as _glob
            hits = sorted(_glob.glob(os.path.join(torch.hub.get_dir(), f"{owner}_{repo}_*")))
            if hits:
                repo_or_dir = hits[0]
                k["source"] = "local"
                for dead in ("trust_repo", "force_reload", "skip_validation"):
                    k.pop(dead, None)          # source='local' 不接受这几个参数
    except Exception:
        pass                                    # 解析失败就照原样走网络路径
    return _orig_hub_load(repo_or_dir, model, *a, **k)
torch.hub.load = _patched_hub_load

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# NOTE: `from inference import Inference` 已移进 src/stage1/sam3d_engine.build_inference() —
# 官方仓库和 Fast-SAM3D 的顶层包/模块同名,谁在 sys.path 前面谁生效,必须等 --engine 定了
# 再决定。SCHEMA / io_videoinput 是纯 numpy,不碰 sam3d_objects,可以照常顶层 import。
from SCHEMA import stage0_to_stage1, QualityThresholds  # noqa: E402
from src.stage0.io_videoinput import load_video_input, summarize_video_input  # noqa: E402
from src.stage1.gt_coverage import load_gt_track_coverage  # noqa: E402
from src.stage1.gt_path2 import inject_gt_path2  # noqa: E402
from src.stage1.sam3d_engine import build_inference  # noqa: E402

_DEFAULT_YAML = _ROOT / "third_party" / "sam-3d-objects" / "checkpoints" / "hf" / "pipeline.yaml"


class _TimedInference:
    """透明包一层,记录每次 SAM 3D 调用的墙钟时间 —— 换后端要做 A/B,逐实例耗时是主指标。
    (multi-candidate 模式下一个实例可能调多次,所以记的是"调用"不是"实例"。)"""

    def __init__(self, inner):
        self._inner = inner
        self.calls: list[float] = []

    def __call__(self, *a, **k):
        t0 = time.time()
        try:
            return self._inner(*a, **k)
        finally:
            self.calls.append(time.time() - t0)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def write_obj(path: Path, verts: np.ndarray, faces: np.ndarray) -> None:
    """Minimal ASCII OBJ (faces are 0-indexed in -> 1-indexed out)."""
    with open(path, "w") as f:
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for tri in faces:
            f.write(f"f {tri[0] + 1} {tri[1] + 1} {tri[2] + 1}\n")


def _routed_ids(sd):
    """instance_ids that src/stage1/rigid_route.py routed away from the mesh path.
    Missing/unreadable file -> empty set (nothing routed), so this is a no-op on old products."""
    p = os.path.join(str(sd), "rigid_route.json")
    if not os.path.exists(p):
        return set()
    try:
        with open(p) as f:
            recs = json.load(f).get("instances", {})
        return {int(k) for k, v in recs.items() if v.get("routed")}
    except Exception as e:
        print(f"[stage1] {p}: unreadable ({e}) -> nothing rigid-routed", flush=True)
        return set()


def process_scene(video, out_dir, inference, qt, seed, n_candidates=1, accept_onmask=0.9,
                  static_skip=True, gt_path2="auto", scene_dir=None, gt_tstar=True,
                  cand_log=True, abandon_frac=0.0, abandon_cap=2, rigid_route=True,
                  engine="sam3d", onmask_gate=None):
    """Run the Stage 0->1 bridge for one scene's VideoInput and persist inst_*.npz +
    manifest. (Algorithm unchanged; factored out so batch mode can reuse one Inference.)

    STATIC-INSTANCE ROUTING (V3, 2026-07): masklets the Stage 0-pre VLM judged
    motion=='static' skip SAM 3D entirely (their final representation is 4 representative
    points from their dense tracks, built at finalize). motion None (old pickles / no VLM)
    is treated as DYNAMIC — zero regression on pre-motion data. --no-static-skip escapes.

    PATH 2 (2026-08-09): `gt_path2` != "off" 时,先从场景目录的 gt_stereo4d.npz +
    dense_tracks_*.npz 把 GT 外参 / gt_object_extent 注入 VideoInput(仅内存),于是
    compose_scene 自动改走 _compose_path2_gt_camera —— mesh 统一落在 W_gt。注入必须在
    static-skip 过滤【之前】做:extent 是按 instance_id 建的字典,多几个键无害,少了就得
    退回 cam_origin_rescale。

    GT-COVERAGE t* (2026-08-10):`gt_tstar` 时把"该帧该实例有 >= 30 条可见 GT 稠密轨迹"
    作为选帧的**候选过滤**(排序不变)。⑦ 严格在 t* 配对,没覆盖的 t* 结构性废掉;实测
    28~29% 的实例踩在这上面。见 src/stage1/gt_coverage.py。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sd = str(scene_dir or out_dir)
    gt_cov = load_gt_track_coverage(sd) if gt_tstar else None
    path2_info = None
    if gt_path2 != "off":
        path2_info = inject_gt_path2(video, sd)
        if path2_info is None and gt_path2 == "on":
            raise RuntimeError(f"{out_dir}: --gt-path2 on 但注入失败(缺 gt_stereo4d.npz 或形状不符)")
    if static_skip:
        static = [m for m in video.masklets if getattr(m, "motion", None) == "static"]
        if static:
            print(f"[stage1] {out_dir.name}: static-skip {len(static)} instance(s): "
                  + ", ".join(f"{m.instance_id}({m.concept})" for m in static), flush=True)
            video.masklets = [m for m in video.masklets
                              if getattr(m, "motion", None) != "static"]
    # RIGID ROUTING (2026-08-13): same treatment for the rigid instances that passed BOTH gates in
    # src/stage1/rigid_route.py (single presence segment + qwen and the mesh-free rigid gate
    # agreeing). Their final representation is built at finalize straight from the dense tracks —
    # k points + SE(3) — which is all the mesh path ever produced for a rigid instance anyway.
    # Absent rigid_route.json (stage 'r' not run) = nothing routed = old behavior bit-for-bit.
    if rigid_route:
        routed = _routed_ids(sd)
        hit = [m for m in video.masklets if int(m.instance_id) in routed]
        if hit:
            print(f"[stage1] {out_dir.name}: rigid-route skip {len(hit)} instance(s): "
                  + ", ".join(f"{m.instance_id}({m.concept})" for m in hit), flush=True)
            video.masklets = [m for m in video.masklets if int(m.instance_id) not in routed]
    if static_skip or rigid_route:
        if not video.masklets:
            manifest = {"scene_frame": "none", "flagged_ids": [], "instances": [],
                        "note": "all instances static/rigid-routed — SAM 3D skipped"}
            with open(out_dir / "scene_manifest.json", "w") as f:
                json.dump(manifest, f, indent=2)
            print(f"[stage1] {out_dir.name}: ALL instances static/rigid-routed -> no meshes",
                  flush=True)
            return None, manifest
    calls0 = len(getattr(inference, "calls", ()))
    t_scene = time.time()
    # P0 插桩:每候选一行 JSONL,落在场景目录里(跟着场景走,多 lane 并发也不会互相写同一个文件)
    clog = str(out_dir / "stage1_candidates.jsonl") if cand_log else None
    scene = stage0_to_stage1(video, inference, qt=qt, seed=seed,
                             n_candidates=n_candidates, accept_onmask=accept_onmask,
                             gt_cov=gt_cov, cand_log=clog, group_prefix=f"{out_dir.name}:",
                             abandon_frac=abandon_frac, abandon_cap=abandon_cap,
                             engine=engine, onmask_gate=onmask_gate)
    t_scene = time.time() - t_scene
    call_times = list(getattr(inference, "calls", ())[calls0:])

    manifest = {
        "scene_frame": scene.scene_frame,
        "flagged_ids": list(scene.flagged_ids or []),
        "timing": {"scene_sec": round(t_scene, 2), "n_recon_calls": len(call_times),
                   "recon_sec": [round(c, 2) for c in call_times]},
        "instances": [],
    }
    if path2_info is not None:
        manifest["path2"] = path2_info
    # ⑦ 严格在 t* 上把 keypoint 和 dense track 配对 —— t* 那一帧没有可见 GT 轨迹的实例,
    # 后面必然链接不上,而且 Path 2 的 position_error 也是拿最近有数据帧外推的(被物体运动
    # 撑大),不能当精度看。把这个数记进 manifest,分析时才有得筛。
    cov_all = gt_cov if gt_cov is not None else load_gt_track_coverage(sd)

    def _gt_at(iid, t):
        c = cov_all.get(int(iid))
        return int(c[t]) if c is not None and 0 <= t < len(c) else None

    for inst in scene.instances:
        verts = np.asarray(inst.verts)
        faces = np.asarray(inst.faces)
        write_obj(out_dir / f"inst_{inst.instance_id}.obj", verts, faces)
        save_kw = dict(
            verts=verts, faces=faces,
            object_to_scene=np.asarray(inst.object_to_scene),
            reference_frame_index=inst.reference_frame_index,
            scale_correction=inst.scale_correction,
        )
        # carry the SAM 3D camera (+ GT camera) so Stage 4 build_tracker_query can project
        for k, v in [("object_pose_cam", inst.object_pose_cam),
                     ("sam3d_intrinsics", inst.sam3d_intrinsics),
                     ("camera_to_scene", inst.camera_to_scene),
                     ("intrinsics", inst.intrinsics)]:
            if v is not None:
                save_kw[k] = np.asarray(v)
        np.savez_compressed(out_dir / f"inst_{inst.instance_id}.npz", **save_kw)
        q = inst.quality
        manifest["instances"].append({
            "instance_id": inst.instance_id,
            "reference_frame_index": int(inst.reference_frame_index),
            "n_verts": int(verts.shape[0]),
            "n_faces": int(faces.shape[0]),
            "scale_correction": float(inst.scale_correction),
            "gt_tracks_at_tstar": _gt_at(inst.instance_id, int(inst.reference_frame_index)),
            "quality": None if q is None else {
                "position_error": q.position_error,
                "scale_ratio": q.scale_ratio,
                "rotation_error_deg": q.rotation_error_deg,
                "flagged": q.flagged,
                "notes": q.notes,
            },
        })

    with open(out_dir / "scene_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    per = (t_scene / max(len(scene.instances), 1))
    print(f"[stage1] {out_dir.name}: scene_frame={scene.scene_frame} "
          f"instances={len(scene.instances)} flagged={manifest['flagged_ids']} "
          f"{t_scene:.1f}s ({per:.1f}s/inst, {len(call_times)} recon calls)", flush=True)
    return scene, manifest


def main():
    import glob
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-input", help="VideoInput pickle (single-scene mode)")
    ap.add_argument("--out-dir", help="output dir (single-scene mode)")
    ap.add_argument("--batch-root", help="dir of scene_*/ subdirs each with video_input.pkl "
                                         "(batch mode: load SAM 3D once, loop scenes; outputs in-place)")
    ap.add_argument("--pipeline-yaml", default=str(_DEFAULT_YAML))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-candidates", type=int, default=3,
                    help="每实例最多用 N 个候选参考帧重建,保留 on-mask 最好的那个(1 = 单帧,"
                         "即 best-of-k 之前的老行为)。**默认 2026-08-19 从 1 改成 3**,与前端"
                         "(run_davis_batch_v3.py)对齐 —— 此前直跑 stage1 会静默退化成单候选。"
                         "注意 N>1 才会走 best-of-k 分支,提前放弃(--abandon-frac / "
                         "--onmask-gate)也只在该分支生效。")
    ap.add_argument("--accept-onmask", type=float, default=0.9,
                    help="multi-candidate early-stop: accept the first frame reaching this on-mask.")
    ap.add_argument("--abandon-frac", type=float, default=0.2,
                    help="提前放弃:用 top-1 候选的 mask 特征预判'这个实例注定达不到 "
                         "--accept-onmask',命中就把候选数砍到 --abandon-cap。"
                         "**默认 0.2 = 开启**(2026-08-13 用户定案);传 0 关闭,关闭时与本功能"
                         "引入前等价。三臂 A/B 实测(51 clip / 264 实例,含同配置复跑对照臂):"
                         "省 11.9%% 调用(噪声底 1.1%%),代价是 on-mask 配对差 -0.0099±0.0036"
                         "(显著,噪声底 -0.0001)、264 个实例里 4 个(1.5%%)丢掉达标。"
                         "依据见 src/stage1/abandon.py 与 STAGE1_FRAME_SCORER_PLAN.md §10")
    ap.add_argument("--abandon-cap", type=int, default=2,
                    help="被判'注定失败'的实例最多试几个候选(默认 2 —— 达标实例里 79%% 本来"
                         "就在前 2 个候选内命中,所以 cap=2 的实际伤害远小于名义误伤率)。"
                         "⚠️ 2026-08-19 候选上限从 6 改成 3 之后,cap=2 每个命中实例最多只省 "
                         "1 次调用(上限 ~ 判死率/(调用数/实例)),预判门的杠杆被摊薄 —— "
                         "省调用的主力已经换成 --onmask-gate")
    ap.add_argument("--onmask-gate", type=float, default=None,
                    help="提前放弃的【第二道门】:第 1 个候选实测 on_mask 低于此值就不再试后面的"
                         "候选。默认 None = 用该引擎模型里带的值(官方 sam3d = 0 关闭;"
                         "fastsam3d = 0.65);显式传 0 可强制关掉。"
                         "Fast-SAM3D 实测(29 场景/142 实例,k=3 回放):省 15.7%% 调用且"
                         "**一个真达标实例都不丢**(rank-1 on_mask<0.65 的 33 个实例没有一个被"
                         "后续候选救回来);官方 SAM3D 同阈值要丢 2.8%% —— 所以这是引擎特异的,"
                         "别跨引擎搬。依据见 src/stage1/abandon.py")
    ap.add_argument("--no-cand-log", action="store_true",
                    help="关掉每候选 JSONL 插桩(默认开)。这个日志记的是早停判据 on_mask 本身,"
                         "过去算完就丢,导致所有'候选数该怎么调'的分析只能事后反算。"
                         "体量 ~60 字节/候选,零 GPU 成本")
    ap.add_argument("--max-position-error", type=float, default=None,
                    help="Path 2 flag threshold (metric); raise to match scene scale (MOVi-E ~5m).")
    ap.add_argument("--no-static-skip", action="store_true",
                    help="escape hatch: also reconstruct instances the VLM judged motion=='static' "
                         "(default: static instances skip SAM 3D — V3 static routing)")
    ap.add_argument("--no-rigid-route", action="store_true",
                    help="escape hatch: also reconstruct the rigid instances that rigid_route.json "
                         "marked routed (default: they skip SAM 3D + PartSAM and get their "
                         "representation built from the dense tracks at finalize). No-op when "
                         "rigid_route.json is absent (stage 'r' not run).")
    ap.add_argument("--engine", choices=("sam3d", "fastsam3d"), default="sam3d",
                    help="重建后端:sam3d=官方 sam-3d-objects;fastsam3d=Fast-SAM3D "
                         "(ICML26 training-free 加速 fork,官宣 2.67x)")
    ap.add_argument("--fs-layout-opt", choices=("mesh", "off"), default="mesh",
                    help="[--engine fastsam3d] fork 基于更旧的 SAM3D 快照,layout 后优化默认是 "
                         "None(=不做);'mesh' 把它自带的那版接回去(默认),'off' 保持 fork 原状")
    ap.add_argument("--no-gt-tstar", action="store_true",
                    help="逃生开关:关掉『t* 必须有 GT 稠密轨迹覆盖』的候选过滤(默认开)。"
                         "⑦ 严格在 t* 配对,没覆盖的 t* 必然 0 链接 —— 实测 28~29%% 的实例踩在"
                         "这上面。没有 GT 轨迹的数据集本来就不受影响。")
    ap.add_argument("--gt-path2", choices=("auto", "on", "off"), default="off",
                    help="有 gt_stereo4d.npz 时补 GT 外参+gt_object_extent 走 Path 2(mesh 统一到 "
                         "W_gt)。**默认 off**(2026-08-10 定,见下);auto=有就走;on=必须成功否则报错。"
                         "只控制【注入】—— MOVi 等 video_input.pkl 自带 camera_extrinsics 的数据集"
                         "不受本开关影响,照常走 Path 2。")
    args = ap.parse_args()
    static_skip = not args.no_static_skip
    rigid_route = not args.no_rigid_route

    qt = QualityThresholds()
    if args.max_position_error is not None:
        qt.max_position_error = args.max_position_error

    print(f"[stage1] building {args.engine} Inference from {args.pipeline_yaml} ...", flush=True)
    inference = _TimedInference(build_inference(
        args.engine, pipeline_yaml=args.pipeline_yaml, compile=False,
        layout_opt=args.fs_layout_opt))

    if args.batch_root:
        scene_dirs = sorted(d for d in glob.glob(os.path.join(args.batch_root, "scene_*"))
                            if os.path.exists(os.path.join(d, "video_input.pkl")))
        print(f"[stage1] BATCH: {len(scene_dirs)} scenes under {args.batch_root}", flush=True)
        for sd in scene_dirs:
            if os.path.exists(os.path.join(sd, "scene_manifest.json")):
                print(f"[stage1] {os.path.basename(sd)}: skip (already done)", flush=True)
                continue
            try:
                video = load_video_input(os.path.join(sd, "video_input.pkl"))
                process_scene(video, sd, inference, qt, args.seed,
                              n_candidates=args.n_candidates, accept_onmask=args.accept_onmask,
                              static_skip=static_skip, gt_path2=args.gt_path2, scene_dir=sd,
                              gt_tstar=not args.no_gt_tstar, cand_log=not args.no_cand_log,
                              abandon_frac=args.abandon_frac, abandon_cap=args.abandon_cap,
                              rigid_route=rigid_route, engine=args.engine,
                              onmask_gate=args.onmask_gate)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[stage1] {os.path.basename(sd)}: FAILED {type(e).__name__}: {e}", flush=True)
    else:
        if not args.video_input or not args.out_dir:
            ap.error("single-scene mode needs --video-input and --out-dir (or use --batch-root)")
        video = load_video_input(args.video_input)
        print(f"[stage1] loaded {args.video_input}\n" + summarize_video_input(video), flush=True)
        process_scene(video, args.out_dir, inference, qt, args.seed,
                      n_candidates=args.n_candidates, accept_onmask=args.accept_onmask,
                      static_skip=static_skip, gt_path2=args.gt_path2,
                      scene_dir=os.path.dirname(os.path.abspath(args.video_input)),
                      gt_tstar=not args.no_gt_tstar, cand_log=not args.no_cand_log,
                      abandon_frac=args.abandon_frac, abandon_cap=args.abandon_cap,
                      rigid_route=rigid_route, engine=args.engine,
                      onmask_gate=args.onmask_gate)


if __name__ == "__main__":
    main()
