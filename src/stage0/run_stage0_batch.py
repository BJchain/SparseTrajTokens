"""Stage 0 BATCH runner (env: sam3) — SAM 3 model loaded ONCE for N clips.

Why this exists
---------------
`run_stage0.py` is one process per clip, so `build_predictor()` runs per clip. Measured on
the 2026-08-01 Stereo4D 1k production run: **123 reloads, median 25 s each = 0.88 h of the
22.80 lane-h covered window (3.9 %)** — pure GPU idle, no algorithmic content. At 110k clips
that is ~764 lane-hours thrown away. This runner keeps the predictor resident and loops.

Contract is IDENTICAL to run_stage0.py per clip (same VideoInput pickle, same masklets, same
camera_intrinsics injection) — it only changes *how many times the weights are loaded*. Each
clip is independently soft-failed and skip-if-exists, so a batch is resumable exactly like
the per-clip loop it replaces.

Input: a JSONL job file, one object per line —
    {"clip": "-3Sx...", "frames_dir": "...", "out": ".../video_input.pkl",
     "stage0_pre": ".../stage0_pre.json",     # optional; else "concepts" is required
     "concepts": "person",                    # optional fallback (comma string)
     "intrinsics": "fx,fy,cx,cy",             # optional
     "max_frames": 0, "stride": 1}            # optional, default 0/1

Run:
  TMPDIR=/tmp HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
  <sam3-env-python> src/stage0/run_stage0_batch.py --jobs /tmp/jobs.jsonl --reuse-frames-dir
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from src.stage0 import (build_predictor, concept_prompts_from_stage0_pre,  # noqa: E402
                        video_input_from_frames)
from src.stage0.io_videoinput import save_video_input, summarize_video_input  # noqa: E402
from src.stage0.sam3_masklets import _empty_cuda_cache  # noqa: E402

_IMG_EXT = (".jpg", ".jpeg", ".png")


def load_frames_dir(frames_dir: Path, max_frames: int | None, stride: int) -> np.ndarray:
    """Byte-identical to run_stage0.load_frames_dir (kept in sync deliberately)."""
    paths = [p for p in frames_dir.iterdir() if p.suffix.lower() in _IMG_EXT]
    if not paths:
        raise FileNotFoundError(f"no images ({_IMG_EXT}) under {frames_dir}")
    try:
        paths.sort(key=lambda p: int(p.stem))
    except ValueError:
        paths.sort()
    paths = paths[::stride]
    if max_frames:
        paths = paths[:max_frames]
    return np.stack([np.asarray(Image.open(p).convert("RGB")) for p in paths])


def _one(job: dict, predictor, args) -> tuple[bool, float]:
    """Process one clip. Returns (ok, seconds). Never raises — a bad clip must not kill
    the batch (same guarantee as the _run_soft per-clip loop this replaces)."""
    t0 = time.time()
    clip = job.get("clip", "?")
    out = Path(job["out"])
    if out.exists():
        print(f"[stage0-batch {clip}] skip (exists)", flush=True)
        return True, 0.0
    try:
        frames_dir = Path(job["frames_dir"])
        stride = int(job.get("stride", 1))
        max_frames = int(job.get("max_frames", 0))

        if job.get("stage0_pre") and Path(job["stage0_pre"]).exists():
            concepts = concept_prompts_from_stage0_pre(job["stage0_pre"])
            desc = [f"{c.concept}({c.object_type},{c.motion or 'dyn?'})" for c in concepts]
        elif job.get("concepts"):
            concepts = [c.strip() for c in job["concepts"].split(",") if c.strip()]
            desc = concepts
        else:
            print(f"[stage0-batch {clip}] FAILED: neither stage0_pre nor concepts", flush=True)
            return False, time.time() - t0

        frames = load_frames_dir(frames_dir, max_frames, stride)
        # Hand SAM 3 the dataset's own frame dir instead of re-encoding all T frames into a
        # temp dir — only legal when we took every frame in order (stride 1, no cap), i.e.
        # the on-disk NNNNN.jpg set IS `frames`. Otherwise fall back to materializing.
        verbatim = bool(args.reuse_frames_dir) and stride == 1 and not max_frames
        seg = dict(prompt_frame_index=int(job.get("prompt_frame", args.prompt_frame)),
                   min_area_frac=float(job.get("min_area_frac", args.min_area_frac)),
                   dedup_iou=float(job.get("dedup_iou", args.dedup_iou)))
        if verbatim:
            seg.update(frames_dir=str(frames_dir), materialize=False, keep_frames_dir=True)

        print(f"[stage0-batch {clip}] frames {frames.shape} concepts={desc} "
              f"{'(frames-dir verbatim)' if verbatim else ''}", flush=True)
        video = video_input_from_frames(frames, concepts, predictor, **seg)
        if job.get("intrinsics"):
            fx, fy, cx, cy = (float(v) for v in job["intrinsics"].split(","))
            video.camera_intrinsics = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])

        out.parent.mkdir(parents=True, exist_ok=True)
        save_video_input(video, str(out))
        dt = time.time() - t0
        print(f"[stage0-batch {clip}] wrote {out} in {dt:.1f}s\n"
              + summarize_video_input(video), flush=True)
        return True, dt
    except Exception:
        traceback.print_exc()
        dt = time.time() - t0
        print(f"[stage0-batch {clip}] FAILED after {dt:.1f}s — skipping", flush=True)
        return False, dt
    finally:
        # The per-clip-process design freed all VRAM at exit; a resident model does not. Without
        # this the caching allocator's high-water mark only grows, and on a shared card the next
        # clip starts OOMing (measured 2026-08-12 on GPU6: 10 of 49 masklets lost).
        _empty_cuda_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True, help="JSONL, one clip job per line")
    ap.add_argument("--reuse-frames-dir", action="store_true",
                    help="⚠️ NOT a free win — measured 2026-08-12 on 3 clips: saves ~0.1 s/clip "
                         "(nothing) but CHANGES RESULTS. The legacy path re-encodes every frame "
                         "at JPEG q95 before SAM 3 sees it; reading the originals feeds different "
                         "pixels and moved mask areas by up to 28 % on some instances. Batching "
                         "alone is bit-identical to the per-clip runner — this flag is the only "
                         "part that isn't. Ignored unless stride==1 and no max-frames.")
    ap.add_argument("--prompt-frame", type=int, default=0)
    ap.add_argument("--min-area-frac", type=float, default=0.0)
    ap.add_argument("--dedup-iou", type=float, default=0.5)
    args = ap.parse_args()

    jobs = [json.loads(l) for l in Path(args.jobs).read_text().splitlines() if l.strip()]
    todo = [j for j in jobs if not Path(j["out"]).exists()]
    print(f"[stage0-batch] {len(jobs)} jobs ({len(todo)} to run) — loading SAM 3 once", flush=True)
    if not todo:
        print("[stage0-batch] nothing to do (all outputs exist) — not loading the model", flush=True)
        return

    t_load = time.time()
    predictor = build_predictor()
    load_s = time.time() - t_load
    print(f"[stage0-batch] predictor ready in {load_s:.1f}s", flush=True)

    ok = fail = 0
    durs = []
    for j in jobs:
        good, dt = _one(j, predictor, args)
        if dt > 0:
            durs.append(dt)
        ok, fail = (ok + 1, fail) if good else (ok, fail + 1)

    tot = sum(durs)
    med = sorted(durs)[len(durs) // 2] if durs else 0.0
    # One machine-parseable summary line — this is what the timing harness greps.
    print(f"[stage0-batch] DONE ok={ok} fail={fail} ran={len(durs)} "
          f"load={load_s:.1f}s work={tot:.1f}s median={med:.1f}s/clip "
          f"load_share={100 * load_s / max(load_s + tot, 1e-9):.1f}%", flush=True)
    if fail:
        sys.exit(0)      # soft: caller decides, mirrors _run_soft semantics per clip


if __name__ == "__main__":
    main()
