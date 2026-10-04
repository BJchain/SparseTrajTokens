"""Env-agnostic (de)serialization of SCHEMA.VideoInput.

Stage 0 (env sam3, torch2.10+cu128) and Stage 1 (env sam3d-objects, torch2.5.1+cu121)
live in DIFFERENT conda envs and cannot share a process. The M0 pipeline therefore hands
a VideoInput between them as a file on disk. Both envs have numpy and can import SCHEMA
(pure python + numpy, no torch), so a pickle of the dataclass round-trips cleanly — the
class path SCHEMA.VideoInput / SCHEMA.Masklet is identical on both sides (same repo file).
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

# Make SCHEMA importable so pickle can resolve VideoInput / Masklet on load.
_STAGE04_ROOT = Path(__file__).resolve().parents[2]
if str(_STAGE04_ROOT) not in sys.path:
    sys.path.insert(0, str(_STAGE04_ROOT))
import SCHEMA  # noqa: E402,F401  (registers the dataclasses for unpickling)


def save_video_input(video, path) -> None:
    """Pickle a SCHEMA.VideoInput (frames + masklets + optional camera/GT) to `path`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(video, f, protocol=4)


def load_video_input(path):
    """Load a SCHEMA.VideoInput previously written by save_video_input."""
    with open(path, "rb") as f:
        return pickle.load(f)


def summarize_video_input(video) -> str:
    """One-line-per-field human summary (for logs / smoke output)."""
    T, H, W = video.frames.shape[:3]
    lines = [
        f"frames: ({T},{H},{W},3) {video.frames.dtype}",
        f"masklets: {len(video.masklets)}",
    ]
    for m in video.masklets:
        cover = int(m.masks.reshape(m.masks.shape[0], -1).any(axis=1).sum())
        lines.append(
            f"  id={m.instance_id} concept={m.concept!r} present={cover}/{T} px={int(m.masks.sum())}"
        )
    lines.append(f"camera_extrinsics: {None if video.camera_extrinsics is None else video.camera_extrinsics.shape}")
    lines.append(f"camera_intrinsics: {None if video.camera_intrinsics is None else video.camera_intrinsics.shape}")
    lines.append(f"gt_object_poses: {None if video.gt_object_poses is None else list(video.gt_object_poses)}")
    return "\n".join(lines)
