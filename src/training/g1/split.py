"""train/val/test 划分。种子固定、与运行顺序无关,各臂共用同一划分。

两种粒度:
  mode="scene"  片段级(G1 原行为,逐位不变):按 video_id 哈希。**旧 run 的复算只能用它。**
  mode="video"  **来源视频级**(110k 默认):Stereo4D 的 clip 名是 `<ytid>_<us>`,同一个
                YouTube 视频会切出多个 clip —— 实测 42,152 个成品只来自 4,079 个视频
                (均 10.3 个/视频,最多 160),按片段划会把同视频的不同时间段分到两边,
                同场景/同物体/同机位 → test 虚高。故 110k 一律按 ytid 分组。
"""
from __future__ import annotations

import hashlib


def group_key(video_id: str, mode: str = "video") -> str:
    """划分分组键。mode="scene" 时就是 video_id 本身。

    mode="video": Stereo4D `scene_<ytid>_<us>` → `<ytid>`(ytid 自身可含下划线,故只剥最后
    一段、且要求它是纯数字);MOVi 的 `val_XXXXX` 各自独立,原样返回。
    """
    if mode == "scene":
        return video_id
    if mode != "video":
        raise ValueError(f"未知划分粒度: {mode}")
    if video_id.startswith("val_"):          # MOVi:一个 scene 就是一个来源
        return video_id
    clip = video_id[len("scene_"):] if video_id.startswith("scene_") else video_id
    head, sep, tail = clip.rpartition("_")
    return head if (sep and tail.isdigit() and head) else clip


def _u(key: str) -> float:
    """把分组键确定性地映到 [0,1)。"""
    return (int(hashlib.md5(key.encode()).hexdigest(), 16) % 10000) / 10000.0


def scene_in_val(video_id: str, val_frac: float = 0.1, mode: str = "scene") -> bool:
    """确定性哈希划分:同一分组恒定落同一侧。默认 mode="scene" = G1 原行为。"""
    return _u(group_key(video_id, mode)) < val_frac


def split_items(items, val_frac: float = 0.1, mode: str = "scene"):
    """两分:train / val。默认粒度为片段级,保持旧 run 复算逐位不变。"""
    train, val = [], []
    for it in items:
        (val if scene_in_val(it["video_id"], val_frac, mode) else train).append(it)
    return train, val


def split3_items(items, val_frac: float = 0.1, test_frac: float = 0.1, mode: str = "video"):
    """三分:train / val / test。区间 [0,test) / [test,test+val) / 其余,分组不跨集。"""
    if val_frac < 0 or test_frac < 0 or val_frac + test_frac >= 1.0:
        raise ValueError(f"划分比例非法: val={val_frac} test={test_frac}")
    train, val, test = [], [], []
    for it in items:
        u = _u(group_key(it["video_id"], mode))
        (test if u < test_frac else val if u < test_frac + val_frac else train).append(it)
    return train, val, test
