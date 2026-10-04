"""独立核对:paper/figure_assets/<clip>/ 下每个文件确实属于该 clip。输出 核对表.md(本目录)。
判据
  - PNG 渲染图 / 管线 lastframe:未被标注覆盖的像素与该 clip 原始帧逐像素完全相同(占比 ≥ 0.3),
    同帧号的其它主选 clip 完全相同像素 < 1%。
  - JPG / MP4(有损):128×128 灰度 MAE,与候选库(6 个主选 + 全部同 YouTube 视频的兄弟 clip)逐帧比;
    全库最小者必须是本 clip(给定帧号时必须是该帧),其它 clip 最小 MAE ≥ 1.25 × 本 clip。
    MP4 例外处理:抽样帧全库最佳必须精确命中本 clip 的该帧,且末帧与已精确核对的 lastframe PNG 的 MAE ≤ 4。
用法:<sam3d-objects python> verify_folders.py
"""
import glob
import json
import os
import re
import sys

import cv2
import numpy as np

S = str(__import__("pathlib").Path(__file__).resolve().parents[2])          # repo root
FIG = os.path.join(S, "paper", "figure_assets")
FR = os.path.join(S, "data", "stereo4d", "frames")
OK = os.path.join(S, "data", "scenes")
R, RATIO = 128, 1.25
SUMMARY = {r["clip"]: r for r in json.load(open(os.path.join(FIG, "render_summary.json")))}
CLIPS = sorted(SUMMARY)
yt = lambda c: c.rsplit("_", 1)[0]
bank = sorted(n[6:] for n in os.listdir(OK) if n.startswith("scene_") and yt(n[6:]) in {yt(c) for c in CLIPS})
G, own, fid, RAW = [], [], [], {}
for c in bank:
    for i, f in enumerate(sorted(glob.glob(os.path.join(FR, c, "*.jpg")))):
        im = cv2.imread(f)
        if c in SUMMARY:
            RAW[(c, i)] = im
        G.append(cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (R, R), interpolation=cv2.INTER_AREA).astype(np.int16))
        own.append(c); fid.append(i)
G = np.stack(G); own = np.array(own); fid = np.array(fid)


def exact(tile, clip, f):
    ref = RAW[(clip, f)]
    eq = float((tile == ref).all(axis=2).mean())
    alt = max(float((tile == RAW[(c2, f)]).all(axis=2).mean()) for c2 in CLIPS if c2 != clip and (c2, f) in RAW)
    return eq >= 0.3 and alt < 0.01, f"逐像素相同 {eq:.0%}(其它 clip 同帧 {alt:.1%})"


def rank(tile, clip, f=None, need_ratio=True):
    t = cv2.resize(cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY), (R, R), interpolation=cv2.INTER_AREA).astype(np.int16)
    e = np.concatenate([np.abs(G[a:a + 512] - t[None]).mean(axis=(1, 2)) for a in range(0, len(G), 512)])
    b = int(np.argmin(e))
    ce = float(e[(own == clip) & (fid == f)][0]) if f is not None else float(e[own == clip].min())
    oth = float(e[own != clip].min())
    hit = own[b] == clip and (f is None or fid[b] == f)
    ok = hit and (not need_ratio or oth >= RATIO * max(ce, 1e-6))
    return ok, f"全库最佳 {own[b]}#{fid[b]}(MAE {e[b]:.1f});最近兄弟 clip MAE {oth:.1f}(×{oth / max(ce, 1e-6):.2f})"


def tiles(img, frames, w, gap, y0):
    return [img[y0:y0 + w, k * (w + gap):k * (w + gap) + w] for k in range(len(frames))]


rows, bad = [], 0
for clip in CLIPS:
    r = SUMMARY[clip]; pre = r["prefix"]
    for fn in sorted(os.listdir(os.path.join(FIG, clip))):
        p = os.path.join(FIG, clip, fn)
        res = []
        if fn.endswith("_sam3_all_masklets.mp4"):
            cap = cv2.VideoCapture(p)
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for k in (0, n // 2, n - 1):
                cap.set(cv2.CAP_PROP_POS_FRAMES, k); _, im = cap.read()
                res.append(rank(im, clip, k, need_ratio=False))
            lf = cv2.imread(os.path.join(FIG, clip, pre + "_sam3_all_masklets_lastframe.png"))
            d = float(np.abs(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(int) - cv2.cvtColor(lf, cv2.COLOR_BGR2GRAY).astype(int)).mean())
            cap.release()
            res.append((d <= 4.0 and n == r["T"], f"末帧 vs 已核 lastframe MAE {d:.1f};帧数 {n}/{r['T']}"))
        elif fn.endswith("_sam3_all_masklets_lastframe.png"):
            res.append(exact(cv2.imread(p), clip, r["T"] - 1))
        elif fn.endswith(".png"):
            img = cv2.imread(p)
            if "_panel" in fn:
                frames = r["frames"]
            elif fn.endswith("_strip6.png"):
                frames = r["strip6"]
            else:
                frames = [int(re.search(r"_f(\d+)\.png$", fn).group(1))]
            res += [exact(t, clip, f) for t, f in zip(tiles(img, frames, 512, 8, 0), frames)]
        elif fn.startswith("scout_frames_strip_"):
            frames = [int(x) for x in re.findall(r"\d+", fn[len("scout_frames_strip_"):])]
            res += [rank(t, clip, f) for t, f in zip(tiles(cv2.imread(p), frames, 240, 0, 18), frames)]
        elif fn.startswith("scout_frames_"):
            frames = [int(x) for x in re.findall(r"\d+", fn[len("scout_frames_"):])]
            res += [rank(t, clip, f) for t, f in zip(tiles(cv2.imread(p), frames, 256, 0, 20), frames)]
        elif fn.startswith("scout_contactsheet_"):
            img = cv2.imread(p); w = 300 if img.shape[1] == 900 else 240
            res += [rank(img[34:34 + w, 0:w], clip, 0), rank(img[34:34 + w, w:2 * w], clip, None)]
        else:
            res.append((False, "未知文件类型"))
        ok = all(x[0] for x in res)
        bad += (not ok)
        rows.append((clip, fn, ok, " | ".join(x[1] for x in res)))
        print(("✅" if ok else "❌"), clip, fn, flush=True)

with open(os.path.join(FIG, "核对表.md"), "w", encoding="utf-8") as f:
    f.write("# 核对表:每个子文件夹里的文件确实属于该 clip\n\n")
    f.write(f"候选库 = 6 个主选 clip + 全部同 YouTube 视频的兄弟 clip,共 {len(bank)} 段 / {len(own)} 帧(兄弟 clip 是最容易混淆的对象)。判据见 `verify_folders.py` 文件头。\n\n")
    f.write(f"**结果:{len(rows)} 个文件,{len(rows) - bad} 个通过,{bad} 个未通过。**\n\n")
    f.write("| clip | 文件 | 结论 | 依据(每个面板 / 抽样帧一条) |\n|---|---|---|---|\n")
    for clip, fn, ok, why in rows:
        f.write(f"| {clip} | {fn} | {'✅' if ok else '❌'} | {why} |\n")
print(f"共 {len(rows)} 个文件,未通过 {bad} 个", flush=True)
sys.exit(1 if bad else 0)
