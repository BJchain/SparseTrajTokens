"""Stereo4D npz + rectified left-eye video -> frames dir + per-scene GT npz (V3 ingest adapter).

Stereo4D (CVPR25, gs://stereo4d) stores per clip <video_id>_<timestamp_us>.npz:
  timestamps            (T,)   original-video frame timestamps in MICROSECONDS
  camera2world          (T,3,4) c2w pose of the RECTIFIED LEFT camera per frame
  track_lengths/track_indices/track_coordinates   sparse-packed 3D tracks (world frame),
                        unpacked by third_party/stereo4d-code utils.load_dataset_npz()
                        into track3d (npt,T,3) with NaN = point absent at that frame
  rectified2rig         (2,3,3) rig calibration (consumed by rectify.py, not needed here)
  fov_bounds            (4,)   VR180 fov -> meta_fov (consumed by rectify.py)

RGB is NOT in the npz: the raw VR180 video comes from YouTube, then
third_party/stereo4d-code/rectify.py (README step 2, JAX_PLATFORMS=cpu) turns it into
<processed>/<clip>/<clip>-left_rectified.mp4  (512x512, 60deg hfov, T frames, one per
timestamp). This adapter consumes that left-eye video.

COORDINATE CONVENTION (empirically verified, see data/stereo4d/check_left_eye.py +
viz_leftcheck/: on-image rate 1.000, tracks land on the correct objects in the LEFT eye,
right eye shows the expected stereo-baseline offset => GT lives in the left rectified cam):
  - camera = OpenCV pinhole: x right, y down, z forward (depth > 0 in front).
  - extrs_w2c = inverse of npz camera2world -> world-to-LEFT-RECTIFIED-camera, (T,3,4).
  - intrinsics for the 512x512 60deg-hfov perspective frames:
        fx = fy = 0.5 * W / tan(30deg) = 443.405 px, cx = cy = W/2 = 256.
  - project: uv_hom = K @ (R_w2c @ X_world + t_w2c);  uv = uv_hom[:2] / uv_hom[2].

Like TAPVid-3D there are no instance masks -> masklets come from Stage 0 (SAM 3 + VLM
concepts). Per clip this adapter:
  1. decode <clip>-left_rectified.mp4, take frames [::stride], write
     <frames_root>/<clip>/%05d.jpg  (default stride 6: 199 frames -> 34, ~5 fps)
  2. write <scene_dir>/gt_stereo4d.npz with tracks/camera/intrinsics sliced by the SAME
     stride (gt frame i  <->  frames_dir %05i.jpg, exact correspondence)
  3. write <scene_dir>/stereo4d_meta.json (counts, stride, fps, intrinsics string for
     run_stage0.py --intrinsics)

Deps: numpy + opencv (env `stereo4d`). Usage:
  python src/stage0/stereo4d_adapter.py --clip 9876543210b_66957 \
      --npz-dir data/stereo4d/demo_npz --scene-dir outputs/stereo4d_demo/scene_9876543210b_66957
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import cv2
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_NPZ_DIR = os.path.join(_ROOT, "data", "stereo4d", "npz")
DEFAULT_PROCESSED_DIR = os.path.join(_ROOT, "data", "stereo4d", "processed")
DEFAULT_FRAMES_ROOT = os.path.join(_ROOT, "data", "stereo4d", "frames")
GT_FILENAME = "gt_stereo4d.npz"
META_FILENAME = "stereo4d_meta.json"
HFOV_DEG = 60.0  # rectify.py NewCropFlags.output_hfov
IMH = IMW = 512  # rectify.py NewCropFlags.imh/imw


def load_clip_npz(path: str) -> dict:
    """Unpack the released npz (same math as stereo4d-code utils.load_dataset_npz,
    inlined so the adapter only needs numpy)."""
    data = dict(np.load(path))
    c2w = data["camera2world"]  # (T,3,4)
    r_inv = np.transpose(c2w[:, :, :3], (0, 2, 1))
    t_inv = -np.matmul(r_inv, c2w[:, :, 3:])
    extrs_w2c = np.concatenate([r_inv, t_inv], axis=-1)  # (T,3,4) world->left-rect cam

    lengths = data["track_lengths"]
    track3d = np.full((len(lengths), len(data["timestamps"]), 3), np.nan, np.float32)
    track3d[np.repeat(np.arange(len(lengths)), lengths), data["track_indices"], :] = (
        data["track_coordinates"])
    return {
        "timestamps_us": data["timestamps"],
        "extrs_w2c": extrs_w2c.astype(np.float64),
        "track3d_world": track3d,
        "video_id": str(data["video_id"]),
    }


def intrinsics_K() -> np.ndarray:
    fx = 0.5 * IMW / math.tan(math.radians(HFOV_DEG / 2.0))
    return np.array([[fx, 0.0, IMW / 2.0], [0.0, fx, IMH / 2.0], [0.0, 0.0, 1.0]])


def intrinsics_arg() -> str:
    k = intrinsics_K()
    return f"{k[0,0]:.6f},{k[1,1]:.6f},{k[0,2]:.6f},{k[1,2]:.6f}"


def extract_frames(left_video: str, frames_dir: str, stride: int, jpg_quality: int = 95
                   ) -> list[int]:
    """left_rectified.mp4 -> frames_dir/%05d.jpg for frames [::stride].
    Returns the kept original frame indices. Skips decode if already complete."""
    cap = cv2.VideoCapture(left_video)
    if not cap.isOpened():
        raise FileNotFoundError(left_video)
    t_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    keep = list(range(0, t_total, stride))
    have = ([f for f in os.listdir(frames_dir) if f.endswith(".jpg")]
            if os.path.isdir(frames_dir) else [])
    if len(have) == len(keep):
        cap.release()
        return keep
    os.makedirs(frames_dir, exist_ok=True)
    keep_set = set(keep)
    out_i = 0
    for src_i in range(t_total):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"decode failed at frame {src_i}/{t_total} of {left_video}")
        if src_i in keep_set:
            cv2.imwrite(os.path.join(frames_dir, f"{out_i:05d}.jpg"), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, jpg_quality])
            out_i += 1
    cap.release()
    return keep


def adaptive_stride(t_total: int, target_min: int = 20, target_max: int = 34) -> int:
    """Per-clip stride so the subsampled length lands in [target_min, target_max]
    (2026-07-31, user-approved): clips range 25..199 frames @30fps; a fixed stride 6
    leaves short clips with as few as 5 frames. Pick the LARGEST stride (lowest fps,
    pipeline-cheapest) that still yields >= target_min frames; clips shorter than
    target_min keep every frame (stride 1). ceil-division keeps the count <= target_max
    for every t_total <= 199 (199/6 -> 34)."""
    if t_total <= target_min:
        return 1
    # exact largest stride with ceil(t_total/stride) >= target_min:
    #   ceil(t/s) >= m  <=>  s <= (t-1) // (m-1)
    # t in [35,38] cannot land in [20,34] with any integer stride (stride1 -> 35-38,
    # stride2 -> 18-19); we accept the small overshoot (<= 38 frames) there.
    s = max(1, (t_total - 1) // (target_min - 1))
    return min(s, 6)                                  # never sparser than the legacy 6


def prepare_clip(clip: str, npz_dir: str, processed_dir: str, frames_root: str,
                 scene_dir: str | None, stride: int | None, src_fps: float = 30.0) -> dict:
    npz_path = os.path.join(npz_dir, clip + ".npz")
    left_video = os.path.join(processed_dir, clip, f"{clip}-left_rectified.mp4")
    dp = load_clip_npz(npz_path)
    t_npz = len(dp["timestamps_us"])

    if stride is None:                                # adaptive (default): ~20-34 frames per clip
        stride = adaptive_stride(t_npz)
    frames_dir = os.path.join(frames_root, clip)
    keep = extract_frames(left_video, frames_dir, stride)
    if len(keep) == 0 or keep[-1] >= t_npz:
        raise RuntimeError(f"frame/npz length mismatch: video keeps {keep[-1] + 1}+ frames, "
                           f"npz has {t_npz}")

    idx = np.asarray(keep)
    track3d = dp["track3d_world"][:, idx]          # (npt, T_sub, 3)
    visible = ~np.isnan(track3d).any(axis=-1)      # (npt, T_sub); Stereo4D releases only
    # frames where the (filtered, visibility-checked) track exists, so NaN==not visible
    meta = {
        "clip": clip,
        "video_id": dp["video_id"],
        "n_frames": len(idx),
        "n_frames_full": t_npz,
        "stride": stride,
        "fps_effective": src_fps / stride,
        "n_tracks": int(track3d.shape[0]),
        "imh": IMH, "imw": IMW, "hfov_deg": HFOV_DEG,
        "intrinsics": intrinsics_arg(),
        "frames_dir": frames_dir,
    }
    if scene_dir is not None:
        os.makedirs(scene_dir, exist_ok=True)
        gt_path = os.path.join(scene_dir, GT_FILENAME)
        np.savez_compressed(
            gt_path,
            track3d_world=track3d.astype(np.float32),
            visible=visible,
            extrs_w2c=dp["extrs_w2c"][idx].astype(np.float32),
            intrinsics_K=intrinsics_K().astype(np.float32),
            frame_indices=idx.astype(np.int32),
            timestamps_us=np.asarray(dp["timestamps_us"])[idx],
            stride=np.int32(stride),
            imh=np.int32(IMH), imw=np.int32(IMW), hfov_deg=np.float32(HFOV_DEG),
            video_id=np.str_(dp["video_id"]), clip=np.str_(clip),
            convention=np.str_(
                "track3d_world: (npt,T,3) world frame, NaN=absent; extrs_w2c: (T,3,4) "
                "world->LEFT-rectified-camera (OpenCV x-right y-down z-forward); "
                "uv = K @ (R X + t), divide by z; frames %05d.jpg <-> gt index, "
                "original clip frame = frame_indices[i]"),
        )
        meta["gt_path"] = gt_path
        with open(os.path.join(scene_dir, META_FILENAME), "w") as f:
            json.dump(meta, f, indent=2)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True, help="<video_id>_<timestamp>, npz basename")
    ap.add_argument("--npz-dir", default=DEFAULT_NPZ_DIR)
    ap.add_argument("--processed-dir", default=DEFAULT_PROCESSED_DIR,
                    help="rectify.py output root (contains <clip>/<clip>-left_rectified.mp4)")
    ap.add_argument("--frames-root", default=DEFAULT_FRAMES_ROOT)
    ap.add_argument("--scene-dir", default=None, help="if set, write gt_stereo4d.npz + meta there")
    ap.add_argument("--stride", type=int, default=None,
                    help="temporal stride on the ~199x30fps clip; default None = per-clip "
                         "adaptive (largest stride <=6 giving >=20 frames; short clips keep "
                         "every frame). Pass an int to force the legacy fixed behavior.")
    args = ap.parse_args()

    meta = prepare_clip(args.clip, args.npz_dir, args.processed_dir, args.frames_root,
                        args.scene_dir, args.stride)
    print(f"[stereo4d-adapter] {args.clip}: {meta['n_frames']}/{meta['n_frames_full']} frames "
          f"(stride {meta['stride']}, {meta['fps_effective']:.1f} fps) -> {meta['frames_dir']}, "
          f"{meta['n_tracks']} GT tracks, K={meta['intrinsics']}"
          + (f", gt={meta['gt_path']}" if "gt_path" in meta else ""))


if __name__ == "__main__":
    main()
