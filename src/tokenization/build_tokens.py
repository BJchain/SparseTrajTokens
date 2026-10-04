"""批量把一个 batch_root 下的 scene_* 目录 tokenize 成 SceneTokens pkl。

用法(stage0_4 下):
  <py> src/tokenization/build_tokens.py --root data/scenes \
       --out outputs/tokens_v4td --camera [--compare-to data/tokens_v4td]
  # MOVi 是嵌套布局(<out>/scene_val_XXXXX/scene_tokens.pkl):
  <py> src/tokenization/build_tokens.py --root outputs/movi_gt \
       --out outputs/movi_gt_v4t --layout nested --compare-to outputs/movi_gt

--compare-to 是**对照臂**:逐字段比新旧两批,只允许 `frame_times_s`/`fps_effective`
这两个新增字段不同,其余任何差异都算回归并列出来。2026-08-22 给"token 补绝对秒"用。
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from dataclasses import fields, is_dataclass

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.abspath(os.path.join(_HERE, "..", "..")), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import tokenizer as tk                                        # noqa: E402
from schema import SceneTokens                                # noqa: E402

# token 数用**训练侧真正的序列化器**来数(纯 numpy,不引 torch),避免自己再数一套口径
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "training")))
from g1.traj_serialize import serialize_scene                 # noqa: E402

NEW_FIELDS = ("frame_times_s", "fps_effective")   # 允许与旧批不同的字段


def _diff(a, b, path, out, allow):
    """递归比对两个 SceneTokens(dataclass/ndarray/list/标量),差异写进 out。"""
    if path.split(".")[-1].split("[")[0] in allow:
        return
    if is_dataclass(a) and is_dataclass(b):
        for f in fields(a):
            _diff(getattr(a, f.name, None), getattr(b, f.name, None), f"{path}.{f.name}", out, allow)
        return
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            out.append(f"{path}: 长度 {len(a)} vs {len(b)}"); return
        for i, (x, y) in enumerate(zip(a, b)):
            _diff(x, y, f"{path}[{i}]", out, allow)
        return
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        x, y = np.asarray(a), np.asarray(b)
        if x.shape != y.shape:
            out.append(f"{path}: 形状 {x.shape} vs {y.shape}")
        elif not np.array_equal(x, y):
            out.append(f"{path}: 数值不同(max|Δ|={np.abs(x.astype(float)-y.astype(float)).max():.3g})")
        return
    if a != b:
        out.append(f"{path}: {a!r} vs {b!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="batch root,内含 scene_* 目录")
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=tk.DEFAULT_ALPHA)
    ap.add_argument("--curve", choices=["bspline", "pl"], default=tk.DEFAULT_CURVE)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--compare-to", default="", help="对照臂:已有 token 目录,逐字段比")
    ap.add_argument("--rigid-mode", choices=["joint", "split"], default=tk.DEFAULT_RIGID_MODE,
                    help="刚体路径:joint=§2.2 共节点联合拟合+keypoint 空间预算(默认);"
                         "split=旧口径对照臂(两流独立、绝对 rotvec)")
    ap.add_argument("--camera", action="store_true", help="相机作为刚体物体组进 token(2026-09-13)")
    ap.add_argument("--fresh", action="store_true",
                    help="重建全部;默认断点续跑(已有 token 的场景直接读回,只补缺的)")
    ap.add_argument("--layout", choices=["flat", "nested"], default="flat",
                    help="flat=<out>/<scene>.pkl(Stereo4D);nested=<out>/<scene>/scene_tokens.pkl(MOVi)")
    args = ap.parse_args()

    scenes = sorted(d for d in os.listdir(args.root)
                    if d.startswith("scene_")
                    and os.path.exists(os.path.join(args.root, d, "final_repr.pkl")))
    if args.limit:
        scenes = scenes[:args.limit]
    os.makedirs(args.out, exist_ok=True)

    def tok_path(root, name):
        return (os.path.join(root, f"{name}.pkl") if args.layout == "flat"
                else os.path.join(root, name, "scene_tokens.pkl"))
    # 断点续跑护栏:同一个 out 目录里的 token 必须是同一套配置产的,否则会静默混批。
    mpath = os.path.join(args.out, "_meta.json")
    if not args.fresh and os.path.exists(mpath):
        try:
            om = json.load(open(mpath))
        except Exception:
            om = {}
        cur = (args.curve, args.alpha, args.rigid_mode, bool(args.camera))
        prev = (om.get("curve"), om.get("alpha"), om.get("rigid_mode"), bool(om.get("camera", False)))
        if None not in prev and prev != cur:
            sys.exit(f"[build] 拒绝续跑:{args.out} 里已有 {prev} 的 token,本次是 {cur};"
                     f"换目录或加 --fresh")
    print(f"[build] {len(scenes)} 个场景 -> {args.out} (curve={args.curve}, alpha={args.alpha})", flush=True)

    t0 = time.time()
    n_tok, miss, runs, no_time, diffs_total, cmp_scenes = [], 0, 0, [], [], 0
    fails = []
    n_reused = 0
    for i, name in enumerate(scenes):
        dst = tok_path(args.out, name)
        st = None
        if not args.fresh and os.path.exists(dst):
            try:                                   # 已有产物直接读回(仍进统计)
                with open(dst, "rb") as f:
                    st = pickle.load(f)
                n_reused += 1
            except Exception:
                st = None                          # 半截/坏产物 -> 重建
        if st is None:
            try:
                st = tk.tokenize_scene(os.path.join(args.root, name), alpha=args.alpha,
                                       curve=args.curve, rigid_mode=args.rigid_mode, camera=args.camera)
            except Exception as e:                 # 单个场景的坏产物不能带走整批(实测:
                fails.append((name, f"{type(e).__name__}: {e}"))   # 一个 0 字节 gt npz)
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            tmp = dst + ".tmp"                     # 原子写:中断也不会留半截 pkl
            with open(tmp, "wb") as f:
                pickle.dump(st, f)
            os.replace(tmp, dst)
        # 统计:token 数 = 序列化后的记录条数(训练真正看到的口径)
        n_tok.append(len(serialize_scene(st)))
        for o in st.objects:
            for owner in ([o.se3_track] if o.kind == "rigid" else (o.keypoint_tracks or [])):
                for attr in ("rot_segments", "trans_segments", "pl_segments"):
                    for seg in (getattr(owner, attr, None) or []):
                        runs += 1
                        miss += int(bool(getattr(seg, "budget_miss", False)))
        if getattr(st, "frame_times_s", None) is None:
            no_time.append(name)
        if args.compare_to:
            p_old = tok_path(args.compare_to, name)
            if os.path.exists(p_old):
                cmp_scenes += 1
                d = []
                _diff(pickle.load(open(p_old, "rb")), st, name, d, NEW_FIELDS)
                diffs_total.extend(d[:5])
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(scenes)}", flush=True)

    if fails:
        with open(os.path.join(args.out, "_failures.txt"), "w") as f:
            for n, e in fails:
                f.write(f"{n}\t{e}\n")
    a = np.asarray(n_tok, float) if n_tok else np.zeros(1)
    meta = dict(n_scenes=len(scenes), n_built=len(n_tok), n_reused=n_reused,
                n_failed=len(fails), failures=[n for n, _ in fails][:10],
                curve=args.curve, alpha=args.alpha, camera=bool(args.camera),
                rigid_mode=args.rigid_mode,
                tokens_per_scene=dict(median=float(np.median(a)), mean=float(a.mean()),
                                      p90=float(np.percentile(a, 90)), max=float(a.max())),
                budget_miss_runs=miss, total_runs=runs,
                scenes_without_timeaxis=len(no_time), examples_without_timeaxis=no_time[:10],
                elapsed_s=round(time.time() - t0, 1))
    with open(os.path.join(args.out, "_meta.json"), "w") as f:
        json.dump(meta, f, indent=1, ensure_ascii=False)
    print(json.dumps(meta, ensure_ascii=False, indent=1), flush=True)
    if args.compare_to:
        print(f"\n[对照臂] 与 {args.compare_to} 比了 {cmp_scenes} 个场景,"
              f"除 {NEW_FIELDS} 外的差异:{len(diffs_total)} 处", flush=True)
        for d in diffs_total[:20]:
            print("   ", d, flush=True)


if __name__ == "__main__":
    main()
