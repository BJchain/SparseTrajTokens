"""Q1「能力上限」口径的抽帧件。**不动 common.py** —— 那个是 Q2 受控口径的冻结基准。

Q2(common.py):整段 clip 均匀抽 8 帧、max_side=384、不给时间信息。
                与我方 RGB 臂逐像素对齐,用于横向可比。
Q1(本文件):  按各模型**官方视频推理配置**给到最强,用于测能力上限。
                帧数放开到整段 clip 全帧,分辨率保持源图 512(不降采样),时间信息给。

各模型官方口径(查各自 README 的官方示例代码得到):
  InternVL2.5/3.5/InternVideo2.5
      def load_video(video_path, input_size=448, max_num=1, num_segments=32)
      -> 每帧 1 tile(与 Q2 一致 ✓)、448、默认 32 段。
         Q2 的链路是 512→384(common.py)→448(run_hf 的 Resize),**先降后升白丢一道**;
         Q1 直接 512→448。
  LLaVA-NeXT-Video
      video 通道,按 32 帧训练;Q2 只给了 8 帧。
  Qwen2.5-VL / Qwen3-VL
      官方走 video 通道(qwen-vl-utils),自行按 fps 抽帧 + 每帧像素预算;
      Q2 走的是逐帧 image。注意 temporal_patch_size=2 会两帧并一个时间 patch,
      帧数够密时这是设计意图,不是损失。
"""
from __future__ import annotations
import glob, json, os
from PIL import Image

from common import frames_dir           # 复用同一套 clip 路径解析


def load_manifest(pkg_root: str):
    with open(os.path.join(pkg_root, "qa", "clips_manifest.json")) as f:
        return json.load(f)


def clip_meta(man, video_id: str, default_fps: float = 5.0):
    """-> (fps, 是否用了兜底值)。全集 110/3103 个 clip 的 manifest fps 为 null。"""
    m = man.get(video_id) or man.get(video_id.replace("scene_", ""))
    f = (m or {}).get("fps")
    return (float(f), False) if f else (default_fps, True)


def sample_frames_native(pkg_root: str, video_id: str, n_frames: int | None = None,
                         max_side: int | None = None):
    """整段 clip 取帧,默认**全帧、源分辨率不降采样**。

    n_frames=None -> 全部帧;否则整段均匀取 n_frames 帧(与 Q2 同算法,只是 k 不同)。
    max_side=None -> 不缩放(源图 512);给数值则等比缩放。
    返回 (List[PIL.Image], 帧在原 clip 中的 0-based 下标)。
    """
    fs = sorted(glob.glob(os.path.join(frames_dir(pkg_root, video_id), "*.jpg")))
    if not fs:
        raise FileNotFoundError(f"no frames for {video_id}")
    if n_frames is None or n_frames >= len(fs):
        idx = list(range(len(fs)))
    else:
        import numpy as np
        idx = np.linspace(0, len(fs) - 1, n_frames).round().astype(int).tolist()
    out = []
    for i in idx:
        im = Image.open(fs[i]).convert("RGB")
        if max_side:
            w, h = im.size
            if max(w, h) > max_side:
                s = max_side / max(w, h)
                im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
        out.append(im)
    return out, idx


def temporal_header(idx, total_frames: int, fps: float) -> str:
    """给不自带时间戳的家族(InternVL / LLaVA)补一行时间锚点。

    用户 2026-09-21 决定:时间信息可以给 —— 目的是诚实测出能力上限,
    所以把 fps / 时长 / 所见帧在原 clip 中的真实编号都交代清楚。
    """
    dur = total_frames / fps if fps else None
    shown = ", ".join(str(i + 1) for i in idx)
    s = (f"This video has {total_frames} frames recorded at {fps:.2f} fps"
         + (f" (total duration {dur:.2f} seconds)." if dur else "."))
    s += f" You are shown frame(s) {shown} of {total_frames}, in order."
    return s
