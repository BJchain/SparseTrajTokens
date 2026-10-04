"""M0 Stage 0 runner (env: sam3).

frames directory + noun-phrase concepts  ->  SAM 3.1 masklets  ->  VideoInput pickle.

Run:
  TMPDIR=/tmp HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  <sam3-env-python> src/stage0/run_stage0.py \
      --frames-dir data/DAVIS/JPEGImages/480p/blackswan \
      --concepts swan --max-frames 8 --out outputs/m0_runs/blackswan/video_input.pkl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from src.stage0 import (build_predictor, concept_prompts_from_stage0_pre,  # noqa: E402
                        video_input_from_frames)
from src.stage0.io_videoinput import save_video_input, summarize_video_input  # noqa: E402

_IMG_EXT = (".jpg", ".jpeg", ".png")


def load_frames_dir(frames_dir: Path, max_frames: int | None, stride: int) -> np.ndarray:
    paths = [p for p in frames_dir.iterdir() if p.suffix.lower() in _IMG_EXT]
    if not paths:
        raise FileNotFoundError(f"no images ({_IMG_EXT}) under {frames_dir}")
    try:  # NNNNN.jpg -> int order
        paths.sort(key=lambda p: int(p.stem))
    except ValueError:
        paths.sort()
    paths = paths[::stride]
    if max_frames is not None:
        paths = paths[:max_frames]
    return np.stack([np.asarray(Image.open(p).convert("RGB")) for p in paths])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--concepts", default=None,
                    help="comma-separated noun phrases, e.g. 'swan' or 'person,ball' "
                         "(text-only; mutually exclusive with --stage0-pre)")
    ap.add_argument("--stage0-pre", default=None,
                    help="stage0_pre.json from run_stage0_pre.py (VLM concepts + box + type)")
    ap.add_argument("--out", required=True, help="output VideoInput pickle path")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--prompt-frame", type=int, default=0)
    ap.add_argument("--min-area-frac", type=float, default=0.0)
    ap.add_argument("--dedup-iou", type=float, default=0.5,
                    help="merge cross-concept masklets above this temporal IoU (0 disables)")
    ap.add_argument("--intrinsics", default=None,
                    help="'fx,fy,cx,cy' GT pinhole intrinsics (dataset-provided, e.g. TAPVid-3D); "
                         "stored as VideoInput.camera_intrinsics (3,3). Temporal subsampling does "
                         "not change K, so this is safe with --stride/--max-frames.")
    args = ap.parse_args()

    if (args.concepts is None) == (args.stage0_pre is None):
        ap.error("provide exactly one of --concepts or --stage0-pre")
    if args.stage0_pre:
        concepts = concept_prompts_from_stage0_pre(args.stage0_pre)
        desc = [f"{c.concept}({c.object_type},{c.motion or 'dyn?'},box={'Y' if c.bbox_xywh else 'N'})"
                for c in concepts]
    else:
        concepts = [c.strip() for c in args.concepts.split(",") if c.strip()]
        desc = concepts
    frames = load_frames_dir(Path(args.frames_dir), args.max_frames, args.stride)
    print(f"[stage0] frames {frames.shape} from {args.frames_dir}; concepts={desc}")

    predictor = build_predictor()
    video = video_input_from_frames(
        frames, concepts, predictor,
        prompt_frame_index=args.prompt_frame,
        min_area_frac=args.min_area_frac,
        dedup_iou=args.dedup_iou,
    )
    if args.intrinsics:
        fx, fy, cx, cy = (float(v) for v in args.intrinsics.split(","))
        video.camera_intrinsics = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        print(f"[stage0] camera_intrinsics <- GT fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")
    save_video_input(video, args.out)
    print(f"[stage0] wrote {args.out}\n" + summarize_video_input(video))


if __name__ == "__main__":
    main()
