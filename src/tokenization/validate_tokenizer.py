"""Tokenizer 验证(v4 兼容):探针 B(token 预算)+ round-trip 保真 + 示例序列。

v4(2026-08-20):加 --curve {bspline,pl}(默认随 tokenizer.DEFAULT_CURVE = bspline);
round-trip 重建走 tk.reconstruct_*(已兼容 SplineRun/PLSegment);示例序列打印两族通用字段。
--root 默认改指 s4d_1k_0816(stereo4d_v3 为 08-08 前旧契约批,已 stale)。

方案 .claude/skills/4d-vlm-pipeline/references/Tokenization.md §4 探针 B + round-trip。
- 探针 B:100 随机场景 tokenize,报 per-scene token 数分布(连续/离散),对比 RGB(32×256=8192);
  分层静态/动态贡献。
- round-trip:从 SceneTokens 反重建归一化世界轨迹,与 final_repr 原轨迹(同归一化)比
  T3 反转 recall / T2 speed MAE / T5 timing(复用 events),确认分段+归一化无额外丢信息。
- 抽 1 场景打印完整 token 序列结构示例。

纯 CPU。运行:
  <py> src/tokenization/validate_tokenizer.py --n-scenes 100
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import random
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import curve_fit as cf          # noqa: E402
import events as ev             # noqa: E402
import tokenizer as tk          # noqa: E402
from schema import CHANNELS      # noqa: E402

STEREO4D_ROOT = os.environ.get("STEREO4D_SCENES", "data/scenes")
OUT_DIR = os.environ.get("TOKENIZER_VAL_OUT", "outputs/tokenizer_val")
RGB_TOKENS = 32 * 256           # 方案对照基线:32 帧 × 每帧 ~256 视觉 token


def pct(x, p):
    return round(float(np.percentile(x, p)), 2) if len(x) else None


# ================================================================== round-trip
def _roundtrip_track(gt_pos_n, vis, rec_pos, bd):
    """一条归一化世界轨迹:GT vs 重建,逐可见 run 算 T3/T2/T5。返回累计计数。"""
    frames_all = np.arange(len(vis))
    R = dict(gt_rev=0, tp_rev=0, pred_rev=0,
             gt_imp=0, tp_imp=0, pred_imp=0, imp_timing=[],
             t2_speed=[], t2_accel=[])
    for (a, b) in ev.visible_runs(vis):
        fr = frames_all[a:b + 1]
        if len(fr) < 3:
            continue
        gp = gt_pos_n[a:b + 1]
        rp = rec_pos[a:b + 1]
        if np.isnan(rp).any():
            continue
        gt_rev = ev.detect_reversals(fr, gp, bd)
        rc_rev = ev.detect_reversals(fr, rp, bd)
        tp, ng, npd = ev.match_events(gt_rev, rc_rev)
        R["gt_rev"] += ng; R["tp_rev"] += tp; R["pred_rev"] += npd
        gt_imp = ev.detect_impacts(fr, gp, bd)
        rc_imp = ev.detect_impacts(fr, rp, bd)
        itp, nig, ipd = ev.match_events(gt_imp, rc_imp)
        R["gt_imp"] += nig; R["tp_imp"] += itp; R["pred_imp"] += ipd
        for g in gt_imp:
            if rc_imp:
                R["imp_timing"].append(min(abs(x - g) for x in rc_imp))
        gs, ga = ev.finite_diff_kin(fr, gp)
        rs, ra = ev.finite_diff_kin(fr, rp)

        def mae(x, y):
            m = ~np.isnan(x) & ~np.isnan(y)
            return float(np.mean(np.abs(x[m] - y[m])) / bd) if (m.sum() and bd > 0) else None
        v = mae(rs, gs)
        if v is not None:
            R["t2_speed"].append(v)
        v = mae(ra, ga)
        if v is not None:
            R["t2_accel"].append(v)
    return R


def roundtrip_scene(scene_dir, scene_tok, alpha):
    """对一个场景:重建每个动态实例的归一化世界轨迹,与 final_repr 原始(同归一化)对比。"""
    reprs = pickle.load(open(os.path.join(scene_dir, "final_repr.pkl"), "rb"))
    reprs_by_id = {int(i["instance_id"]): i for i in reprs}
    center, scale = scene_tok.norm_center, scene_tok.norm_scale
    T = scene_tok.T
    agg = dict(gt_rev=0, tp_rev=0, pred_rev=0, gt_imp=0, tp_imp=0, pred_imp=0,
               imp_timing=[], t2_speed=[], t2_accel=[])
    for o in scene_tok.objects:
        if o.kind == "static":
            continue
        inst = reprs_by_id[o.instance_id]
        if o.kind == "nonrigid":
            traj = np.asarray(inst["trajectories"], float)
            vis = np.asarray(inst["visibility"], bool)
            traj_n = (traj - center) / scale
            allvis = traj_n[vis]
            bd = tk._cloud_diag(allvis) if len(allvis) >= 2 else 0.0
            for kt in o.keypoint_tracks:
                m = o.keypoint_tracks.index(kt)
                rec = tk.reconstruct_keypoint(kt, T)
                r = _roundtrip_track(traj_n[m], vis[m], rec, bd)
                for k, v in r.items():
                    (agg[k].extend(v) if isinstance(v, list) else agg.__setitem__(k, agg[k] + v))
        else:  # rigid
            world, vis = tk._rigid_world_traj(inst)
            world_n = (world - center) / scale
            allvis = world_n[vis]
            bd = tk._cloud_diag(allvis) if len(allvis) >= 2 else 0.0
            rec = tk.reconstruct_rigid(o.se3_track, T)     # (M,T,3)
            for m in range(world_n.shape[0]):
                r = _roundtrip_track(world_n[m], vis[m], rec[m], bd)
                for k, v in r.items():
                    (agg[k].extend(v) if isinstance(v, list) else agg.__setitem__(k, agg[k] + v))
    return agg


# ================================================================== 示例序列打印
def _seg_lines(s):
    """一段的逐节点内容。PLSegment 用 vframes/verts,SplineRun(v4)用 fnodes/nodes ——
    dump_example 原来只认 PL,v4 默认切成 B 样条后会 AttributeError(2026-08-22 修)。"""
    if hasattr(s, "fnodes"):
        out = [f"          fnodes={np.round(s.fnodes, 3).tolist()}",
               f"          nodes={np.round(s.nodes, 3).tolist()}",
               f"          degrees={np.asarray(s.degrees).tolist()} "
               f"budget_miss={s.budget_miss} max_err={round(float(s.max_err), 5)}"]
        rr = getattr(s, "rot_ref", None)
        if rr is not None:      # §2.2 联合拟合:nodes 是相对 run 首帧的旋转向量
            out.append(f"          rot_ref(元信息,不进 token)={np.round(rr, 4).tolist()}")
        return out
    return [f"          vframes={s.vframes.tolist()}",
            f"          verts={np.round(s.verts, 3).tolist()}"]


def dump_example(scene_tok, path):
    lines = []
    L = lines.append
    st = scene_tok
    L(f"SCENE {st.scene_id}")
    L(f"  T={st.T}  frame_convention={st.frame_convention[:60]!r}")
    L(f"  [continuous] norm_center={np.round(st.norm_center,4).tolist()}  norm_scale={round(st.norm_scale,5)}")
    L(f"  [continuous/metadata] camera_K={'set '+str(st.camera_K.shape) if st.camera_K is not None else None}"
      f"  camera_w2c={'set '+str(st.camera_w2c.shape) if st.camera_w2c is not None else None}")
    L(f"  n_objects={len(st.objects)}")
    for o in st.objects:
        L(f"  OBJ iid={o.instance_id} label={o.label!r} object_type={o.object_type} "
          f"kind={o.kind} [discrete]  motion_part={o.motion_part_summary}")
        if o.kind == "static":
            L(f"      [continuous] static_points{np.asarray(o.static_points).shape} = "
              f"{np.round(o.static_points,3).tolist()}")
        elif o.kind == "nonrigid":
            for kt in o.keypoint_tracks:
                L(f"      KP id={kt.keypoint_id} part={kt.part_id} vis={kt.visibility_summary} "
                  f"n_segments={len(kt.pl_segments)} occlusion_gaps={[tuple(round(x,3) for x in g) for g in kt.occlusion_gaps]}")
                for s in kt.pl_segments:
                    L(f"        SEG u=[{round(s.u_start,3)},{round(s.u_end,3)}] "
                      f"n_verts={s.n_verts()} [continuous verts] knot_types(discrete)={s.knot_types}")
                    for ln in _seg_lines(s):
                        L(ln)
        else:  # rigid
            se = o.se3_track
            L(f"      SE3 ref_norm{se.ref_norm.shape} keypoint_ids={se.keypoint_ids} "
              f"part_ids={se.part_ids} vis={se.visibility_summary} "
              f"occlusion_gaps={[tuple(round(x,3) for x in g) for g in se.occlusion_gaps]}")
            for tag, segs in [("ROT(rotvec)", se.rot_segments), ("TRANS", se.trans_segments)]:
                for s in segs:
                    L(f"        {tag} SEG u=[{round(s.u_start,3)},{round(s.u_end,3)}] "
                      f"n_verts={s.n_verts()} knot_types={s.knot_types}")
                    for ln in _seg_lines(s):
                        L(ln)
    tc = st.token_counts()
    L(f"  TOKEN_COUNTS = {json.dumps(tc)}")
    txt = "\n".join(lines)
    with open(path, "w") as f:
        f.write(txt)
    return txt


# ================================================================== 主流程
def run(n_scenes, seed, alpha, curve=None):
    curve = curve or tk.DEFAULT_CURVE
    os.makedirs(OUT_DIR, exist_ok=True)
    all_scenes = sorted(glob.glob(os.path.join(STEREO4D_ROOT, "scene_*")))
    all_scenes = [s for s in all_scenes if os.path.exists(os.path.join(s, "final_repr.pkl"))]
    random.seed(seed)
    sample = sorted(random.sample(all_scenes, min(n_scenes, len(all_scenes))))
    print(f"[sample] {len(sample)}/{len(all_scenes)} scenes seed={seed} alpha={alpha} curve={curve}", flush=True)

    # ---- 探针 B:token 预算 ----
    per_scene = []
    cont_list, disc_list, tot_list, camera_list = [], [], [], []
    cont_static_list, cont_dyn_list, disc_static_list, disc_dyn_list = [], [], [], []
    n_static_tot = n_rigid_tot = n_nonrigid_tot = 0
    example_tok = None
    example_dir = None
    for i, sd in enumerate(sample):
        st = tk.tokenize_scene(sd, alpha=alpha, curve=curve)
        tc = st.token_counts()
        per_scene.append(dict(scene=st.scene_id, **tc))
        cont_list.append(tc["continuous_total"]); disc_list.append(tc["discrete_total"])
        tot_list.append(tc["total_no_camera"]); camera_list.append(tc["camera"])
        cont_static_list.append(tc["cont_static"]); cont_dyn_list.append(tc["cont_dyn"])
        disc_static_list.append(tc["disc_static"]); disc_dyn_list.append(tc["disc_dyn"])
        n_static_tot += tc["n_static"]; n_rigid_tot += tc["n_rigid"]; n_nonrigid_tot += tc["n_nonrigid"]
        # 选一个"有动态实例"的场景当示例
        if example_tok is None and (tc["n_rigid"] + tc["n_nonrigid"]) >= 1:
            example_tok, example_dir = st, sd

    def dist(x):
        x = np.asarray(x, float)
        return dict(mean=round(float(x.mean()), 2), p10=pct(x, 10), p50=pct(x, 50),
                    p90=pct(x, 90), max=round(float(x.max()), 2), min=round(float(x.min()), 2))

    budget = dict(
        rgb_baseline=RGB_TOKENS,
        total_no_camera=dist(tot_list),
        continuous=dist(cont_list),
        discrete=dist(disc_list),
        camera_metadata=dist(camera_list),
        # 静/动分层(连续+离散贡献)
        static_continuous=dist(cont_static_list),
        static_discrete=dist(disc_static_list),
        dynamic_continuous=dist(cont_dyn_list),
        dynamic_discrete=dist(disc_dyn_list),
        instance_counts=dict(static=n_static_tot, rigid=n_rigid_tot, nonrigid=n_nonrigid_tot),
        compression_vs_rgb=round(RGB_TOKENS / max(np.mean(tot_list), 1), 1),
    )
    print(f"[probeB] total/scene mean={budget['total_no_camera']['mean']} "
          f"p50={budget['total_no_camera']['p50']} p90={budget['total_no_camera']['p90']} "
          f"vs RGB {RGB_TOKENS}  (~{budget['compression_vs_rgb']}x shorter)", flush=True)
    print(f"[probeB] static inst={n_static_tot} rigid={n_rigid_tot} nonrigid={n_nonrigid_tot}", flush=True)

    # ---- round-trip 保真 ----
    agg = dict(gt_rev=0, tp_rev=0, pred_rev=0, gt_imp=0, tp_imp=0, pred_imp=0,
               imp_timing=[], t2_speed=[], t2_accel=[])
    for sd in sample:
        st = tk.tokenize_scene(sd, alpha=alpha, curve=curve)
        r = roundtrip_scene(sd, st, alpha)
        for k, v in r.items():
            (agg[k].extend(v) if isinstance(v, list) else agg.__setitem__(k, agg[k] + v))

    def rc(tp, n):
        return round(tp / n, 4) if n else None

    def mn(x):
        return round(float(np.mean(x)), 5) if len(x) else None
    roundtrip = dict(
        T3_reversal_recall=rc(agg["tp_rev"], agg["gt_rev"]),
        T3_reversal_precision=rc(agg["tp_rev"], agg["pred_rev"]),
        gt_reversals=agg["gt_rev"],
        T5_impact_recall=rc(agg["tp_imp"], agg["gt_imp"]),
        T5_impact_precision=rc(agg["tp_imp"], agg["pred_imp"]),
        T5_impact_timing_meanabs=mn(agg["imp_timing"]),
        gt_impacts=agg["gt_imp"],
        T2_speed_MAE=mn(agg["t2_speed"]),
        T2_accel_MAE=mn(agg["t2_accel"]),
        note="round-trip 在归一化世界系;PL 事件对齐(D3),应 >= 探针 A 纯 DP 同 α 的 PL 保真。",
    )
    print(f"[roundtrip] T3 recall={roundtrip['T3_reversal_recall']} "
          f"T2 speed_MAE={roundtrip['T2_speed_MAE']} T5 timing={roundtrip['T5_impact_timing_meanabs']}",
          flush=True)

    # ---- 示例序列 ----
    example_path = None
    if example_tok is not None:
        example_path = os.path.join(OUT_DIR, "example_scene_tokens.txt")
        dump_example(example_tok, example_path)
        # 落一个 pkl 证明序列化可用
        from schema import save_scene_tokens, load_scene_tokens
        pk = os.path.join(OUT_DIR, "example_scene_tokens.pkl")
        save_scene_tokens(example_tok, pk)
        _ = load_scene_tokens(pk)     # round-trip 序列化自检
        print(f"[example] {example_tok.scene_id} -> {example_path} (+pkl serialize OK)", flush=True)

    out = dict(n_scenes=len(sample), seed=seed, alpha=alpha,
               rgb_baseline=RGB_TOKENS, budget=budget, roundtrip=roundtrip,
               channels=CHANNELS, example_scene=example_tok.scene_id if example_tok else None,
               per_scene=per_scene)
    with open(os.path.join(OUT_DIR, "validate_results.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[done] wrote {OUT_DIR}/validate_results.json", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-scenes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alpha", type=float, default=tk.DEFAULT_ALPHA)
    ap.add_argument("--curve", choices=["bspline", "pl"], default=tk.DEFAULT_CURVE)
    args = ap.parse_args()
    run(args.n_scenes, args.seed, args.alpha, args.curve)
