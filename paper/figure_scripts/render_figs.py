"""论文插图渲染(只读生产数据,只写本目录):SAM 3 重跑得到的实例掩码 + 数据管线降采样后的关键点轨迹(彩虹色 = 时间)。

数据来源
- 掩码:data/figure_scenes/scene_<clip>/video_input.pkl(按生产参数重跑 SAM 3 所得;
  生产成品里 video_input.pkl 已被瘦身删除)。
- 关键点轨迹:data/scenes/scene_<clip>/final_repr.pkl(世界系,生产产物,未改动),
  用同目录 gt_stereo4d.npz 的 extrs_w2c / intrinsics_K 逐帧投影到画面;非刚体按 visibility 只画可见帧,
  刚体点 = R(t)·ref + t(t)(只画 se3_ok 帧)。
- 哪些物体算"在动":以训练用 token(data/tokens_v4td)的 kind 为准(nonrigid / rigid;相机除外)。
- 新掩码 ↔ final_repr 实例:SAM 3 重跑会重新编号,用该实例的 dense_tracks_<iid>.npz(生产时归属到它的 GT 轨迹)
  落进新掩码的比例做一一匹配(≥ 0.5 才算匹配上)。

用法(stereo4d 或 sam3d-objects 环境;cwd 任意):
  <python> render_figs.py            # 渲染全部 6 个 clip
输出:每个 clip 的图写进 paper/figure_assets/<clip>/ 子文件夹;色条与 render_summary.json 在 paper/figure_assets/。见 README.md。
"""
from __future__ import annotations

import colorsys
import json
import os
import pickle
import sys

import cv2
import numpy as np

S = str(__import__("pathlib").Path(__file__).resolve().parents[2])          # repo root(本文件在 paper/figure_scripts/)
OUT = os.path.join(S, "paper", "figure_assets")          # 逐 clip 帧图/拼图输出目录
for p in (S, os.path.join(S, "src", "tokenization")):
    if p not in sys.path:
        sys.path.insert(0, p)
import SCHEMA  # noqa: E402,F401  (VideoInput / Masklet 反序列化)

RERUN = os.path.join(S, "data", "figure_scenes")
OK = os.path.join(S, "data", "scenes")
TOK = os.path.join(S, "data", "tokens_v4td")
FR = os.path.join(S, "data", "stereo4d", "frames")

# (输出前缀, clip, 选帧)
CLIPS = [
    ("fig1_teaser", "pwBorgoDJh0_141875209", [0, 8, 12, 16, 24]),
    ("fig4_row1_complex_interaction", "sbcUC3xaXfE_475542209", [0, 16, 33]),
    ("fig4_row2_hand_object", "QE_pA0S3ttc_161895229", [12, 16, 26]),
    ("fig4_row3_multiple_people", "2ekr1IfOn5c_101835169", [0, 16, 33]),
    ("fig4_row4_swapping_positions", "Kh42LoZOS2Y_102502503", [0, 20, 33]),
    ("fig4_row5_complex_video", "yRsUHBgOt8k_61795128", [0, 16, 33]),
]
# 物体颜色(BGR,彼此区分明显)
PAL = [(214, 118, 31), (14, 127, 255), (44, 160, 44), (40, 39, 214), (189, 103, 148), (75, 86, 140),
       (194, 119, 227), (34, 189, 188), (207, 190, 23), (127, 127, 127), (180, 119, 31), (120, 187, 255)]


def rainbow_bgr(x):
    """x∈[0,1]:紫(开始)→ 红(结束)。"""
    r, g, b = colorsys.hsv_to_rgb(0.78 * (1.0 - float(x)), 1.0, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def obj_points(o):
    """(M,T,3) 世界点 + (M,T) 可见。"""
    if "se3_R" in o:
        R = np.asarray(o["se3_R"], float); t = np.asarray(o["se3_t"], float)
        ref = np.asarray(o["ref_positions"], float); ref = ref[np.abs(ref).sum(1) > 0]
        P = (np.einsum("tij,mj->tmi", R, ref) + t[:, None]).transpose(1, 0, 2)
        V = np.broadcast_to(np.asarray(o["se3_ok"], bool)[None], P.shape[:2]).copy()
        return P, V
    if "trajectories" in o:
        return np.asarray(o["trajectories"], float), np.asarray(o["visibility"], bool)
    return None, None


def project(P, E, K, W, H):
    """P (M,T,3) 世界 → uv (M,T,2),ok (M,T)。"""
    R, t = E[:, :3, :3], E[:, :3, 3]
    cam = np.einsum("tij,mtj->mti", R, P) + t[None]
    z = cam[..., 2]
    zs = np.where(np.abs(z) > 1e-9, z, 1.0)
    uv = np.stack([K[0, 0] * cam[..., 0] / zs + K[0, 2], K[1, 1] * cam[..., 1] / zs + K[1, 2]], -1)
    ok = (z > 1e-3) & (uv[..., 0] > -50) & (uv[..., 0] < W + 50) & (uv[..., 1] > -50) & (uv[..., 1] < H + 50)
    return uv, ok


def match_masks(vi, sd, iids, T, H, W):
    """final_repr 实例 → 新 masklet(用该实例的 dense_tracks 落入比例,一一贪心)。返回 {iid: (masklet, frac)}。"""
    cand = []
    for iid in iids:
        p = os.path.join(sd, f"dense_tracks_{iid}.npz")
        if not os.path.exists(p):
            continue
        d = np.load(p)
        uv = np.asarray(d["uv_norm"], float) * np.array([W, H]); vis = np.asarray(d["vis"], bool)
        n, TT = vis.shape
        if TT != T:
            continue
        xs = np.clip(np.round(uv[..., 0]).astype(int), 0, W - 1); ys = np.clip(np.round(uv[..., 1]).astype(int), 0, H - 1)
        tt = np.broadcast_to(np.arange(T)[None], (n, T))
        tot = int(vis.sum())
        if tot == 0:
            continue
        for mi, m in enumerate(vi.masklets):
            inside = m.masks[tt[vis], ys[vis], xs[vis]]
            cand.append((float(inside.mean()), iid, mi))
    cand.sort(reverse=True)
    used_i, used_m, out = set(), set(), {}
    for frac, iid, mi in cand:
        if frac < 0.5 or iid in used_i or mi in used_m:
            continue
        out[iid] = (vi.masklets[mi], frac); used_i.add(iid); used_m.add(mi)
    return out


def centroid_track(P, V):
    c = np.full((P.shape[1], 3), np.nan)
    for f in range(P.shape[1]):
        if V[:, f].any():
            c[f] = np.median(P[V[:, f], f], axis=0)
    return c


def draw(frame, t, T, objs, show_mask=True, show_traj=True):
    img = frame.copy()
    if show_mask:
        ov = img.copy()
        for o in objs:
            if o["mask"] is not None and o["mask"][t].any():
                ov[o["mask"][t]] = o["color"]
        img = cv2.addWeighted(ov, 0.28, img, 0.72, 0)
        for o in objs:
            if o["mask"] is not None and o["mask"][t].any():
                cs, _ = cv2.findContours(o["mask"][t].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(img, cs, -1, o["color"], 2, cv2.LINE_AA)
    if show_traj:
        for o in objs:
            uv, ok = o["uv"], o["ok"]
            for k in range(uv.shape[0]):
                for f in range(1, t + 1):
                    if ok[k, f] and ok[k, f - 1]:
                        p0 = tuple(int(round(v)) for v in uv[k, f - 1]); p1 = tuple(int(round(v)) for v in uv[k, f])
                        cv2.line(img, p0, p1, rainbow_bgr(f / max(T - 1, 1)), 2, cv2.LINE_AA)
                if ok[k, t]:
                    c = tuple(int(round(v)) for v in uv[k, t])
                    cv2.circle(img, c, 4, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(img, c, 4, o["color"], 1, cv2.LINE_AA)
    return img


def hstack(imgs, gap=8):
    H = imgs[0].shape[0]
    out = [imgs[0]]
    for im in imgs[1:]:
        out += [np.full((H, gap, 3), 255, np.uint8), im]
    return np.concatenate(out, 1)


def colorbar(path, w=600, h=28):
    bar = np.zeros((h, w, 3), np.uint8)
    for x in range(w):
        bar[:, x] = rainbow_bgr(x / (w - 1))
    canvas = np.full((h + 30, w + 120, 3), 255, np.uint8)
    canvas[5:5 + h, 60:60 + w] = bar
    cv2.putText(canvas, "start", (5, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, "end", (w + 66, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(path, canvas)


def render(prefix, clip, sel):
    sd_rerun = os.path.join(RERUN, f"scene_{clip}")
    sd = os.path.join(OK, f"scene_{clip}")
    vi = pickle.load(open(os.path.join(sd_rerun, "video_input.pkl"), "rb"))
    fr = pickle.load(open(os.path.join(sd, "final_repr.pkl"), "rb"))
    g = np.load(os.path.join(sd, "gt_stereo4d.npz"), allow_pickle=True)
    E = np.asarray(g["extrs_w2c"], float); K = np.asarray(g["intrinsics_K"], float)
    sc = pickle.load(open(os.path.join(TOK, f"scene_{clip}.pkl"), "rb"))
    frames = [cv2.imread(os.path.join(FR, clip, f"{i:05d}.jpg")) for i in range(E.shape[0])]
    T = len(frames); H, W = frames[0].shape[:2]
    assert vi.masklets and vi.masklets[0].masks.shape == (T, H, W), (clip, vi.masklets[0].masks.shape, T, H, W)
    moving = {o.instance_id for o in sc.objects if o.kind in ("nonrigid", "rigid") and o.instance_id != 1_000_000}
    iids = [int(o["instance_id"]) for o in fr]
    mm = match_masks(vi, sd, iids, T, H, W)
    objs, report = [], []
    for o in fr:
        iid = int(o["instance_id"])
        if iid not in moving:
            continue
        P, V = obj_points(o)
        if P is None:
            continue
        uv, okp = project(P, E, K, W, H)
        m = mm.get(iid)
        objs.append(dict(iid=iid, label=str(o.get("label")), P=P, V=V, uv=uv, ok=okp & V,
                         mask=(m[0].masks if m else None), color=None))
        report.append(dict(iid=iid, label=str(o.get("label")), kind=("rigid" if "se3_R" in o else "nonrigid"),
                           n_keypoints=int(P.shape[0]), mask_matched=bool(m), mask_overlap=(round(m[1], 3) if m else None),
                           mask_concept=(m[0].concept if m else None)))
    # 配色:teaser 按"骑车人+自行车 / 背包+背包的人 / 购物袋+拎袋的人"成组着色,其余物体灰色系;其它 clip 逐物体配色
    if prefix.startswith("fig1"):
        cen = {o["iid"]: centroid_track(o["P"], o["V"]) for o in objs}
        def nearest_person(iid):
            best, bd = None, 1e9
            for o in objs:
                if o["label"] != "person" or o["iid"] in groups:   # 已分组的人不再分配
                    continue
                d = np.linalg.norm(cen[o["iid"]] - cen[iid], axis=1)
                d = d[np.isfinite(d)]
                if len(d) >= 3 and np.median(d) < bd:
                    best, bd = o["iid"], float(np.median(d))
            return best, bd
        groups = {}
        for lab, col in (("bicycle", (214, 118, 31)), ("backpack", (14, 127, 255)), ("shopping bag", (44, 160, 44))):
            o = next((x for x in objs if x["label"] == lab), None)
            if o is None:
                continue
            groups[o["iid"]] = col
            pid, dist = nearest_person(o["iid"])
            if pid is not None and dist <= 1.0:          # 超过 1 m 不算携带 / 骑行
                groups[pid] = col
                report.append(dict(note=f"{lab} 的携带者/骑行者 = person iid {pid}(质心中位距离 {dist:.2f} m)"))
            else:
                report.append(dict(note=f"{lab}: 没有 1 m 内的被跟踪行人(最近 {dist:.2f} m),只给物体本身着色"))
        for o in objs:
            o["color"] = groups.get(o["iid"], (150, 150, 150))
        key = [o for o in objs if o["iid"] in groups]
    else:
        for i, o in enumerate(objs):
            o["color"] = PAL[i % len(PAL)]
        key = objs
    od = os.path.join(OUT, clip)          # 每个 clip 一个子文件夹
    os.makedirs(od, exist_ok=True)
    outs = []
    variants = [("", objs)] + ([("_key_objects", key)] if prefix.startswith("fig1") else [])
    for suf, os_ in variants:
        panel = [draw(frames[t], t, T, os_) for t in sel]
        for t, im in zip(sel, panel):
            fn = f"{prefix}{suf}_f{t:02d}.png"; cv2.imwrite(os.path.join(od, fn), im); outs.append(fn)
        fn = f"{prefix}{suf}_panel.png"; cv2.imwrite(os.path.join(od, fn), hstack(panel)); outs.append(fn)
        fn = f"{prefix}{suf}_panel_masks_only.png"
        cv2.imwrite(os.path.join(od, fn), hstack([draw(frames[t], t, T, os_, show_traj=False) for t in sel])); outs.append(fn)
        fn = f"{prefix}{suf}_panel_traj_only.png"
        cv2.imwrite(os.path.join(od, fn), hstack([draw(frames[t], t, T, os_, show_mask=False) for t in sel])); outs.append(fn)
    strip = [int(round(x)) for x in np.linspace(0, T - 1, 6)]
    fn = f"{prefix}_strip6.png"
    cv2.imwrite(os.path.join(od, fn), hstack([draw(frames[t], t, T, objs) for t in strip])); outs.append(fn)
    n_mask_total = len(vi.masklets)
    return dict(prefix=prefix, clip=clip, frames=sel, strip6=strip, T=T, n_masklets_rerun=n_mask_total,
                n_moving=len(objs), n_moving_with_mask=sum(1 for o in objs if o["mask"] is not None),
                objects=report, files=outs)


def main():
    colorbar(os.path.join(OUT, "rainbow_colorbar.png"))
    summary = []
    for prefix, clip, sel in CLIPS:
        r = render(prefix, clip, sel)
        summary.append(r)
        print(f"[{prefix}] {clip}: masklets={r['n_masklets_rerun']} moving={r['n_moving']} "
              f"with_mask={r['n_moving_with_mask']} files={len(r['files'])}", flush=True)
    json.dump(summary, open(os.path.join(OUT, "render_summary.json"), "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
