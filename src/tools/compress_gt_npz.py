"""把 data/stereo4d/frames/<clip>/gt_stereo4d.npz 压成【只留相机】的切片。

依据(2026-08-19 110k 磁盘核算):整个下游只消费 `intrinsics_K` / `extrs_w2c` / `frame_indices`
(tokenizer 的 GT 相机、QA 硬挖的相机);占地方的 `track3d_world`(均 47,299 条轨迹 × 帧数 × 3)
只在 GT 稠密前端跑的时候用一次,轨迹跑完就已经归属进各实例的 dense_tracks_*.npz / final_repr。

实测 9.245 MiB -> 3.95 KiB(**2235 倍**)。108k clip 上就是 993 GiB -> 0.4 GiB,是本项目
单条收益最大的磁盘杠杆。

⚠️ 代价:压完就没法原地重跑 GT 稠密前端 / 曲线族的 GT 评测。官方 npz 可以从
   storage.googleapis.com/stereo4d/{train,test}/<clip>.npz 重下,所以这是可逆的。
⚠️ 只压【已经彻底跑完】的 clip(有 scene_manifest.json,或在 skipped 名单里),否则重跑会缺字段。

用法:
  python3 src/tools/compress_gt_npz.py --batch-root data/scenes --frames-root data/stereo4d/frames
  加 --dry-run 只报告不动手
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

KEEP = ("intrinsics_K", "extrs_w2c", "frame_indices", "timestamps_us", "stride",
        "imh", "imw", "hfov_deg", "video_id", "clip", "convention")
MARK = "_camera_only"          # 压过的文件里会多这个键,幂等判据


def compress_one(path: str, dry: bool = False) -> tuple[int, int]:
    """返回 (压缩前字节, 压缩后字节);已经压过或读不动就返回 (0,0)。"""
    before = os.path.getsize(path)
    try:
        d = np.load(path, allow_pickle=True)
        if MARK in d.files:
            return 0, 0                                   # 幂等:压过了
        if "track3d_world" not in d.files:
            return 0, 0                                   # 没有大字段,不用动
        out = {k: d[k] for k in KEEP if k in d.files}
        out[MARK] = np.array(True)
    except Exception as e:
        print(f"  [skip] {path}: {type(e).__name__}: {e}", flush=True)
        return 0, 0
    if dry:
        return before, 4096
    # ⚠️ np.savez_compressed 会给不以 .npz 结尾的路径自动补上 .npz,所以临时名必须自带 .npz,
    # 否则它写的是 <tmp>.npz 而 os.replace 去找 <tmp> —— FileNotFoundError。
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **out)
    os.replace(tmp, path)                                  # 同分区 rename,原子
    return before, os.path.getsize(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-root", required=True, help="扫这个批次里【已跑完】的 clip")
    ap.add_argument("--frames-root", default="data/stereo4d/frames")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # 🔴 2026-08-29:判据从「manifest 或 skipped.txt」收紧成「manifest 或 _skipped_static 目录」。
    #    原因:skipped.txt 是【只增不减的累计名单】。一个 clip 早先被剔除记了一笔,后来因为
    #    重跑/搁浅回收又回到队列里等着处理 —— 它仍在 skipped.txt 里,于是被当成"已跑完"
    #    把 gt npz 压成只剩相机字段的切片(丢掉 track3d_world)。等真的跑到它时,GT 稠密前端
    #    直接 `KeyError: track3d_world is not a file in the archive` 而失败。
    #    实测代价:6,702 个队列剩余里有 3,665 个(超过一半)被提前压掉,只能删帧重取。
    # 🔑 `_skipped_static/scene_<clip>/` 是【当前真的被剔除】的硬证据(目录会随重跑被搬走),
    #    比只增不减的名单可靠。压缩是破坏性的,判据必须用硬证据。
    done = set()
    for name in os.listdir(args.batch_root):
        if name.startswith("scene_") and os.path.exists(
                os.path.join(args.batch_root, name, "scene_manifest.json")):
            done.add(name[len("scene_"):])
    skdir = os.path.join(args.batch_root, "_skipped_static")
    if os.path.isdir(skdir):
        for name in os.listdir(skdir):
            if name.startswith("scene_"):
                done.add(name[len("scene_"):])
    # 兜底闸:凡是还在待跑清单里、又没有上面两种硬证据的,一律不压
    ready = os.path.join(args.batch_root, "clips_ready.txt")
    if os.path.exists(ready):
        pending = {l.strip() for l in open(ready) if l.strip()} - done
        if pending:
            done -= pending
    done = sorted(done)

    tot_b = tot_a = n = 0
    for c in done:
        p = os.path.join(args.frames_root, c, "gt_stereo4d.npz")
        if not os.path.exists(p):
            continue
        b, a = compress_one(p, args.dry_run)
        if b:
            tot_b += b; tot_a += a; n += 1
            if n % 200 == 0:
                print(f"  ... {n} 个,已省 {(tot_b-tot_a)/2**30:.1f} GiB", flush=True)
    print(f"[gt-compress] {'(dry) ' if args.dry_run else ''}压了 {n}/{len(done)} 个 clip:"
          f" {tot_b/2**30:.2f} GiB -> {tot_a/2**30:.4f} GiB,省 {(tot_b-tot_a)/2**30:.2f} GiB"
          + (f",压缩比 {tot_b/max(tot_a,1):.0f}x" if tot_a else ""))


if __name__ == "__main__":
    main()
