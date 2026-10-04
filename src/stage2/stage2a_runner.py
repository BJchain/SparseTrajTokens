"""Stage 2a (v2): PartSAM semantic over-segmentation on an M0 InstanceMesh.

Thin driver that reuses the validated `src/stage2/partsam_runner.py` (official
PartSAM model + preprocessing + predict_masks + NMS + point->face->vertex) to turn
one InstanceMesh (verts/faces, from Stage 1 / m0_runs/inst_*.npz) into a per-vertex
SEMANTIC part label (over-segmentation). No motion correction here — that's Stage 2b
(after Stage 4). Part labels are in OUR vertex order (frame-invariant).

Two pipeline-level additions on top of the official path (2026-08-08; auditable per instance in
the npz: `iou_threshold` / `n_parts_raw` / `n_masks` / `n_merged`, and a `parts_<id>.FAILED.json`
sidecar when PartSAM returns nothing):
  1. PER-RECON THRESHOLD — PartSAM's confidence tracks how much geometric detail a mesh has
     (measured Pearson(log10 n_verts, n_parts) = 0.633). The official 0.65 is right for SAM 3D's
     ~120k-vertex meshes (~9 parts) but starves the TripoSR fast path (~4 parts, and K=0 hard
     failure on 17% of instances). So the threshold is chosen per scene from
     scene_manifest.json's `scene_frame`: fast path -> 0.50, everything else -> official 0.65.
     The THRESHOLD is thus unchanged for SAM 3D.
  2. TINY-PART MERGE — a lower threshold also admits few-dozen-vertex crumbs; they are pure waste
     for pass1, which hands every part a keypoint budget. Parts under `--min-part-frac` of the
     vertices are merged into their largest MESH-ADJACENT neighbour (see merge_tiny_parts).
     NOTE this one applies to BOTH routes — SAM 3D loses its (few) crumb parts too.
     `--min-part-frac 0` disables it entirely.
Measured on the fast path (STAGE1_FASTPATH_QUALITY_FINDINGS.md §6/§7): parts/instance 4 -> 10
(matching SAM 3D), pass1 keypoints +94%.

DON'T try to bit-compare parts_*.npz across runs to validate a change here: PARTSAM ITSELF IS
NON-DETERMINISTIC. Measured 2026-08-08 — the same mesh, same loaded model, same process, run
twice back to back gives the same part COUNT but different part BOUNDARIES (neither bit-equal
nor even partition-equivalent). The likely source is the float atomic-add in PVCNN voxelization
(`scatter_(..., reduce='add')`), whose accumulation order is not fixed on GPU: tiny feature
deltas flip argmax on boundary vertices. Validate changes with INVARIANTS instead — that is what
the merge was checked against: no part below min_part_frac survives, labels are contiguous
0..K-1, and total connected components drop by at least n_merged (merging is along mesh edges,
so A∪B has at most comp(A)+comp(B)-1 components).

Run in env PartSAM (env sam3d-objects/sam3 can't import the compiled PartSAM ops):
  TMPDIR=/tmp CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  <PartSAM-env-python> src/stage2/stage2a_runner.py \
      --inst-dir outputs/m0_runs/movi_scene0 --out-dir outputs/m1_runs/movi_scene0
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import trimesh

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from src.stage2.partsam_runner import (  # noqa: E402
    load_partsam_model, run_partsam_on_mesh, partsam_output_to_soft_assign, IOU_THRESHOLD,
)

# --- PartSAM 置信度阈值:按 Stage 1 的重建路线选(2026-08-08) -------------------------------
# PartSAM 对低顶点/被简化平滑的表面系统性不自信(实测 Pearson(log10 顶点数, n_parts)=0.633),
# 官方 0.65 在 TripoSR fast-path 上留下 17% 的 K=0 硬失败、且只给 ~4 个部件(SAM 3D ~9)。
# 0.50 实测:部件中位 4->10(追平 SAM 3D)、硬失败 17%->8%、pass1 keypoint +94%。
# SAM 3D 路的 mesh 本来就够密(部件中位 9),没有理由动它 -> 保持官方 0.65。
IOU_THR_FASTPATH = 0.50
FASTPATH_SCENE_FRAMES = {"gt_camera_fast"}   # run_stage1_fast.py 写的 scene_manifest 标记
MIN_PART_FRAC = 0.02                         # 小于此顶点占比的 part 视为碎屑 -> 合并
# K=0 兜底阶梯(2026-08-11):主阈值一个 mask 都没过线时,按此表降阈值重试。
# 阈值是【纯后处理】(前向已经跑完,scores 在手上),所以零额外算力,而且用的是同一批 scores。
# 主阈值本来就有 mask 时这段完全不执行 -> 现有能成功的实例走逐位相同的路径。
# 取值依据:实测失败 mesh 的 score_max 落在 0.48~0.64(成功的 0.84~0.91)。低于 0.40 不再降,
# 那已经是"模型自己都不信"的区间,救回来大概率是垃圾部件。--no-iou-fallback 可关。
IOU_THR_FALLBACKS = (0.50, 0.40)


def resolve_iou_threshold(scene_dir: str, override: float | None = None) -> tuple[float, str]:
    """选本场景该用的 PartSAM 阈值。override 优先;否则按 scene_manifest.json 的 scene_frame
    自动判定(fast-path -> 0.50,其余 -> 官方 0.65)。返回 (阈值, 判定依据) 便于日志/审计。"""
    if override is not None:
        return float(override), "cli"
    try:
        with open(os.path.join(scene_dir, "scene_manifest.json")) as f:
            sf = str(json.load(f).get("scene_frame", ""))
    except Exception:
        return float(IOU_THRESHOLD), "default(no manifest)"
    if sf in FASTPATH_SCENE_FRAMES:
        return IOU_THR_FASTPATH, f"fastpath({sf})"
    return float(IOU_THRESHOLD), f"default({sf or 'unknown'})"


def merge_tiny_parts(faces: np.ndarray, labels: np.ndarray, min_frac: float = MIN_PART_FRAC,
                     n_labels: int | None = None):
    """把顶点占比 < min_frac 的"碎屑 part"并入其在 MESH 上相邻的最大 part。

    为什么:降低 IOU_THRESHOLD 换来更细的真部件,但也会带出一批几十个顶点的碎屑(实测 fast-path
    @0.50 碎屑 part 中位 1->5)。碎屑对 pass1 是纯浪费——它按 part 分配 keypoint 预算。
    合并沿 mesh 邻接做(不是空间最近邻),所以不会把身体两侧隔空的碎片错并到一起;孤立碎片
    (无任何异部件邻边)并入全局最大 part。每轮只并当前最小的一个,直到没有碎屑。

    `n_labels` = 标签空间大小(默认 lab.max()+1)。传 PartSAM 的 mask 数 K,可让 old2new 覆盖那些
    "有 mask 但 argmax 后没抢到任何顶点"的空列,soft_assign 才能同步合并。

    返回 (labels 重映射为 0..K-1, n_merged, old2new) —— old2new 供 soft_assign 同步合并。"""
    lab = np.asarray(labels, np.int64).copy()
    n = len(lab)
    K0 = max(int(lab.max()) + 1 if n else 0, int(n_labels or 0))
    ident = np.arange(max(K0, 1), dtype=np.int64)
    if n == 0 or min_frac <= 0 or K0 <= 1:
        return lab, 0, ident
    F = np.asarray(faces, np.int64)
    e = (np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [0, 2]]]) if len(F)
         else np.zeros((0, 2), np.int64))
    chain, n_merged = {}, 0
    while True:
        uniq, cnt = np.unique(lab, return_counts=True)
        if len(uniq) <= 1:
            break
        j = int(np.argmin(cnt))
        if cnt[j] / n >= min_frac:
            break
        src = int(uniq[j])
        size_of = {int(u): int(c) for u, c in zip(uniq, cnt)}
        tgt = None
        if len(e):
            la, lb = lab[e[:, 0]], lab[e[:, 1]]
            nb = np.concatenate([lb[(la == src) & (lb != src)], la[(lb == src) & (la != src)]])
            if len(nb):
                cand = np.unique(nb)
                tgt = int(cand[int(np.argmax([size_of.get(int(c), 0) for c in cand]))])
        if tgt is None:                                   # 孤立碎屑 -> 全局最大 part
            order = np.argsort(-cnt)
            tgt = int(uniq[order[0]])
            if tgt == src:
                tgt = int(uniq[order[1]])
        lab[lab == src] = tgt
        chain[src] = tgt
        n_merged += 1
    if n_merged == 0:
        # 没合并任何东西就原样返回:PartSAM 的标签本来可能跳号(有 mask 没抢到顶点),
        # 无条件重编号会让"什么都没做"的产物也和旧产物逐位不同,毁掉回归比对。
        return lab, 0, ident
    uniq = np.unique(lab)
    to_new = np.full(int(uniq.max()) + 1, 0, np.int64)
    to_new[uniq] = np.arange(len(uniq))
    old2new = np.zeros(K0, np.int64)
    for p in range(K0):                                   # 沿合并链回溯到根,再映射到连续标签
        q = p
        seen = 0
        while q in chain and seen <= K0:
            q = chain[q]; seen += 1
        old2new[p] = to_new[q] if q < len(to_new) else 0
    return to_new[lab], n_merged, old2new


def segment_instance_mesh(verts: np.ndarray, faces: np.ndarray, model,
                          use_graph_cut: bool = False, iou_threshold: float | None = None,
                          min_part_frac: float = MIN_PART_FRAC,
                          iou_fallbacks: "tuple | list" = IOU_THR_FALLBACKS) -> dict:
    """verts (V,3), faces (F,3) -> dict with per-vertex semantic part labels.
    PartSAM normalizes internally, so scene-frame coords are fine.
    `iou_threshold` None = PartSAM 官方 0.65;`min_part_frac` 0 = 关闭碎屑合并;
    `iou_fallbacks` = K=0 时的降阈值阶梯(见 IOU_THR_FALLBACKS),传 () 关闭。"""
    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    tmpd = tempfile.mkdtemp(prefix="stage2a_", dir=os.environ.get("TMPDIR", "/tmp"))
    mesh_path = os.path.join(tmpd, "mesh.obj")
    # process=False: keep our exact verts/faces so point_to_face -> our face order holds.
    trimesh.Trimesh(vertices=verts, faces=faces, process=False).export(mesh_path)
    try:
        raw = run_partsam_on_mesh(model, mesh_path, use_graph_cut=use_graph_cut,
                                  iou_threshold=iou_threshold, iou_fallbacks=iou_fallbacks)
        assign = partsam_output_to_soft_assign(raw, verts, faces)
    finally:
        shutil.rmtree(tmpd, ignore_errors=True)
    lab = assign["hard_label_per_vertex"].astype(np.int64)
    soft = assign["soft_assign"]
    n_masks = int(assign["K"])                       # NMS 后的 mask 数(可 > 实际出现的部件数)
    k_raw = int(len(np.unique(lab)))                 # 合并前实际出现的部件数(与 n_parts 同口径)
    lab, n_merged, old2new = merge_tiny_parts(faces, lab, min_frac=min_part_frac,
                                              n_labels=n_masks)
    if n_merged and soft is not None and soft.ndim == 2 and soft.shape[1] == len(old2new):
        s2 = np.zeros((soft.shape[0], int(old2new.max()) + 1), soft.dtype)
        np.add.at(s2.T, old2new, soft.T)      # 合并的列概率相加,行和仍为 1
        soft = s2
    return {
        "part_labels": lab,                                                # (V,)
        "n_parts": int(len(np.unique(lab))),
        "soft_assign": soft,                                               # (V,K)
        "ptf_match": bool(raw["ptf_resample_matches"]),
        "n_empty_faces": int(assign["n_empty_faces"]),
        "iou_threshold": float(raw["iou_threshold"]),                      # 实际生效的
        "iou_threshold_requested": float(raw.get("iou_threshold_requested",
                                                 raw["iou_threshold"])),   # 两者不等=兜底救回
        "n_parts_raw": k_raw,                                              # 合并前(同口径)
        "n_masks": n_masks,                                                # NMS 后 mask 数
        "n_merged": int(n_merged),
    }


def _save_colored_ply(path, verts, faces, labels):
    import matplotlib
    K = int(labels.max()) + 1
    colors = (matplotlib.colormaps["tab20"](np.arange(K) % 20)[:, :3] * 255).astype(np.uint8)
    vc = np.zeros((len(verts), 4), np.uint8); vc[:, 3] = 255
    vc[:, :3] = colors[np.clip(labels, 0, K - 1)]
    trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=vc, process=False).export(path)


def process_dir(inst_dir, out_dir, model, graph_cut=False, ply=False, force=False,
                iou_threshold=None, min_part_frac=MIN_PART_FRAC,
                iou_fallbacks=IOU_THR_FALLBACKS):
    """Segment every inst_*.npz in `inst_dir` -> parts_*.npz in `out_dir` (model reused).
    ALWAYS recomputes (no skip-if-exists): SAM 3D meshing is NON-deterministic — the vertex
    count AND order drift every Stage 1 run — so a cached parts_*.npz from an earlier mesh would
    silently misalign (part_labels are per-vertex, index-matched to inst verts). Recomputing on
    the current mesh is the only safe contract. (PartSAM is non-deterministic too, so re-running
    this stage reshuffles part BOUNDARIES even on an identical mesh — see the module docstring;
    that is a reproducibility caveat for the whole pipeline, not something caching would fix.)
    Per-instance try/except so one mesh PartSAM can't segment (degenerate -> empty masks) does
    NOT kill the rest of the scene; failures leave a parts_<id>.FAILED.json sidecar so a batch
    run's failure rate is countable afterwards instead of being inferred from missing files.
    (`force` is accepted for call-compat but no longer needed — recompute is unconditional.)"""
    import glob
    os.makedirs(out_dir, exist_ok=True)
    thr, why = resolve_iou_threshold(inst_dir, iou_threshold)
    fb = tuple(f for f in (iou_fallbacks or ()) if float(f) < thr)   # 只保留比主阈值低的
    print(f"[stage2a] {os.path.basename(out_dir)}: iou_threshold={thr} ({why}), "
          f"min_part_frac={min_part_frac}, K=0 兜底阶梯={list(fb) or '关'}", flush=True)
    for p in sorted(glob.glob(os.path.join(inst_dir, "inst_*.npz"))):
        iid = os.path.splitext(os.path.basename(p))[0].replace("inst_", "")
        out = os.path.join(out_dir, f"parts_{iid}.npz")
        verts = None
        try:
            d = np.load(p)
            verts, faces = d["verts"], d["faces"]
            res = segment_instance_mesh(verts, faces, model, use_graph_cut=graph_cut,
                                        iou_threshold=thr, min_part_frac=min_part_frac,
                                        iou_fallbacks=fb)
            np.savez_compressed(out, part_labels=res["part_labels"], n_parts=res["n_parts"],
                                soft_assign=res["soft_assign"],
                                iou_threshold=np.float32(res["iou_threshold"]),
                                iou_threshold_requested=np.float32(res["iou_threshold_requested"]),
                                n_parts_raw=np.int64(res["n_parts_raw"]),
                                n_masks=np.int64(res["n_masks"]),
                                n_merged=np.int64(res["n_merged"]))
            # 成功 -> 清掉上一次留下的失败痕迹。stage2a 是无条件重算的,不清的话"一次失败、
            # 二次成功"的实例会永远带着误导性的 sidecar,而这个 sidecar 存在的唯一目的就是
            # 让批量跑完能数出失败率(2026-08-11)。
            _f = os.path.join(out_dir, f"parts_{iid}.FAILED.json")
            if os.path.exists(_f):
                os.remove(_f)
            if ply:
                _save_colored_ply(os.path.join(out_dir, f"parts_{iid}.ply"), verts, faces, res["part_labels"])
            rescued = ("" if res["iou_threshold"] == res["iou_threshold_requested"]
                       else f" [K=0 兜底: {res['iou_threshold_requested']} -> {res['iou_threshold']}]")
            print(f"[stage2a] {os.path.basename(out_dir)}/inst {iid}: V={len(verts)} -> "
                  f"{res['n_parts']} parts (raw {res['n_parts_raw']}, merged {res['n_merged']} tiny; "
                  f"thr={res['iou_threshold']}, ptf_match={res['ptf_match']}){rescued}", flush=True)
        except Exception as e:
            # 失败留痕:下游只能靠"文件不存在"推断,批量跑完无从统计 -> 落一个 sidecar。
            # 同时必须删掉上一次成功留下的 parts_<id>.npz:那份 per-vertex 标签是按【上一个
            # mesh】的顶点算的,而 SAM3D 重跑会改顶点数与顺序,留着会被下游当成当前 mesh 的
            # 标签静默用错(模块 docstring 里警告过的同一个坑)。
            print(f"[stage2a] {os.path.basename(out_dir)}/inst {iid}: SKIP "
                  f"{type(e).__name__}: {e}", flush=True)
            for _st in (out, os.path.join(out_dir, f"parts_{iid}.ply")):
                if os.path.exists(_st):
                    os.remove(_st)
                    print(f"[stage2a]   (删掉上一次成功留下的 {os.path.basename(_st)},"
                          f"它属于另一个 mesh)", flush=True)
            try:
                json.dump({"instance_id": iid, "error": f"{type(e).__name__}: {e}",
                           "iou_threshold": thr, "iou_fallbacks": [float(x) for x in fb],
                           "n_verts": (-1 if verts is None else int(len(verts)))},
                          open(os.path.join(out_dir, f"parts_{iid}.FAILED.json"), "w"), indent=2)
            except Exception:
                pass


def main():
    import glob
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--inst", help="single InstanceMesh npz")
    g.add_argument("--inst-dir", help="dir of inst_*.npz (single-scene)")
    g.add_argument("--batch-root", help="dir of scene_*/ each with inst_*.npz "
                                        "(batch: load PartSAM once, parts saved in-place)")
    ap.add_argument("--out-dir", help="output dir (single-scene; default = inst-dir)")
    ap.add_argument("--graph-cut", action="store_true")
    ap.add_argument("--ply", action="store_true")
    ap.add_argument("--iou-threshold", type=float, default=None,
                    help=f"PartSAM mask 置信度阈值。默认按 scene_manifest.json 的 scene_frame 自动:"
                         f"fast-path(TripoSR)={IOU_THR_FASTPATH},其余(含 SAM 3D)={IOU_THRESHOLD}。"
                         f"显式给值则对所有场景生效。")
    ap.add_argument("--min-part-frac", type=float, default=MIN_PART_FRAC,
                    help=f"顶点占比小于此值的 part 视为碎屑,沿 mesh 邻接并入最大邻部件"
                         f"(默认 {MIN_PART_FRAC};传 0 关闭合并)")
    ap.add_argument("--shard", default=None, metavar="i/n",
                    help="[--batch-root] 只处理场景列表里下标 %% n == i 的那一份,用于在同一张卡上"
                         "并发起 n 个 worker。实测(2026-08-12,独占 L20):PartSAM 单 worker 只有 "
                         "10.9GB 显存 / 53.7%% TDP —— util 读 81%% 但 SM 没喂满,是本链路里唯一"
                         "能靠多 worker 提升的阶段(SAM3 23GB / SAM3D 26GB 且都 >84%% TDP,上不了)。"
                         "取模作用在【排序后的完整场景列表】上,各 worker 所有权互斥且与进度无关"
                         "(别再犯 stereo4d_lane.sh 那个对剩余清单取模的抢跑坑)。")
    ap.add_argument("--no-iou-fallback", action="store_true",
                    help=f"关掉 K=0 兜底阶梯(默认开:主阈值一个 mask 都没过线时按 "
                         f"{list(IOU_THR_FALLBACKS)} 依次降阈值重筛。阈值是纯后处理,零额外算力,"
                         f"且主阈值有 mask 时该逻辑完全不执行)")
    args = ap.parse_args()
    fallbacks = () if args.no_iou_fallback else IOU_THR_FALLBACKS

    print("[stage2a] loading PartSAM once …", flush=True)
    model = load_partsam_model()

    if args.batch_root:
        scene_dirs = sorted(d for d in glob.glob(os.path.join(args.batch_root, "scene_*"))
                            if glob.glob(os.path.join(d, "inst_*.npz")))
        shard_tag = ""
        if args.shard:
            try:
                si, sn = (int(x) for x in args.shard.split("/"))
            except ValueError:
                raise SystemExit(f"--shard 要写成 i/n,收到 {args.shard!r}")
            if not (0 <= si < sn):
                raise SystemExit(f"--shard {args.shard}: 需要 0 <= i < n")
            scene_dirs = [d for k, d in enumerate(scene_dirs) if k % sn == si]
            shard_tag = f" [shard {si}/{sn}]"
        print(f"[stage2a] BATCH: {len(scene_dirs)} scenes under {args.batch_root}{shard_tag}",
              flush=True)
        for sd in scene_dirs:
            try:
                process_dir(sd, sd, model, args.graph_cut, args.ply,
                            iou_threshold=args.iou_threshold, min_part_frac=args.min_part_frac,
                            iou_fallbacks=fallbacks)
            except Exception as e:
                print(f"[stage2a] {os.path.basename(sd)}: FAILED {type(e).__name__}: {e}", flush=True)
    elif args.inst:
        out = args.out_dir or os.path.dirname(args.inst)
        d = np.load(args.inst); verts, faces = d["verts"], d["faces"]
        iid = os.path.splitext(os.path.basename(args.inst))[0].replace("inst_", "")
        os.makedirs(out, exist_ok=True)
        thr, why = resolve_iou_threshold(os.path.dirname(os.path.abspath(args.inst)),
                                         args.iou_threshold)
        res = segment_instance_mesh(verts, faces, model, use_graph_cut=args.graph_cut,
                                    iou_threshold=thr, min_part_frac=args.min_part_frac,
                                    iou_fallbacks=tuple(f for f in fallbacks if float(f) < thr))
        np.savez_compressed(os.path.join(out, f"parts_{iid}.npz"), part_labels=res["part_labels"],
                            n_parts=res["n_parts"], soft_assign=res["soft_assign"],
                            iou_threshold=np.float32(res["iou_threshold"]),
                            iou_threshold_requested=np.float32(res["iou_threshold_requested"]),
                            n_parts_raw=np.int64(res["n_parts_raw"]),
                            n_masks=np.int64(res["n_masks"]),
                            n_merged=np.int64(res["n_merged"]))
        if args.ply:
            _save_colored_ply(os.path.join(out, f"parts_{iid}.ply"), verts, faces, res["part_labels"])
        print(f"[stage2a] inst {iid}: {res['n_parts']} parts (raw {res['n_parts_raw']}, "
              f"merged {res['n_merged']}; thr={thr} [{why}])", flush=True)
    else:
        process_dir(args.inst_dir, args.out_dir or args.inst_dir, model, args.graph_cut, args.ply,
                    iou_threshold=args.iou_threshold, min_part_frac=args.min_part_frac,
                    iou_fallbacks=fallbacks)


if __name__ == "__main__":
    main()
