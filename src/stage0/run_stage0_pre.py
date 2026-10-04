"""Stage 0-pre runner (env: qwen).

frames directory  ->  Qwen-VL  ->  stage0_pre.json  (the instance list + rigid/non-rigid
type that drives Stage 0 SAM 3 prompting and rides downstream as the authoritative
object_type, replacing Stage 2b's tracker-dependent residual gate).

Pipeline (all VLM calls validated — see memory project_qwen_dashscope_setup):
  1. adaptively sample K frames  (K = clamp(round(T/15), 8, 32))
  2. enumerate_video  : sampled frames as ONE video block -> every label that appears
                        anywhere in the clip (catches enter/leave/occlusion)  [coverage]
  3. detect_instances : per sampled frame -> labels + boxes                    [localisation]
  4. MERGE -> canonical instances: token-subset synonym union ("swan"⊆"black swan",
     "car"⊆"silver car"); pick a seed frame + representative box per instance.
  5. classify_types   : rigid / non-rigid per canonical label.

Synonyms we CAN'T merge lexically (e.g. person/pedestrian) are left as separate concepts
on purpose — they are category-level identical in type, and Stage 0's cross-concept mask-IoU
dedup is the real safety net for "same physical object, two names".

Run:
  <qwen-env-python> src/stage0/run_stage0_pre.py \
      --frames-dir data/DAVIS/JPEGImages/480p/blackswan \
      --out outputs/m0_runs/blackswan/stage0_pre.json
  (DASHSCOPE_API_KEY must be exported; a non-login shell does NOT auto-source ~/.bashrc.)
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from src.stage0 import qwen_client as qc  # noqa: E402

_IMG_EXT = (".jpg", ".jpeg", ".png")


# =============================================================================
# frame sampling
# =============================================================================
def list_frame_paths(frames_dir: Path) -> list:
    paths = [p for p in frames_dir.iterdir() if p.suffix.lower() in _IMG_EXT]
    if not paths:
        raise FileNotFoundError(f"no images ({_IMG_EXT}) under {frames_dir}")
    try:
        paths.sort(key=lambda p: int(p.stem))
    except ValueError:
        paths.sort()
    return paths


def adaptive_k(n_frames: int, k_min: int = 8, k_max: int = 32) -> int:
    """Density-adaptive sample count: ~1 frame / 15, clamped to [k_min, k_max]."""
    return int(max(k_min, min(k_max, round(n_frames / 15))))


# =============================================================================
# synonym merge (token-subset union-find)
# =============================================================================
def _tokens(label: str) -> frozenset:
    return frozenset(label.split())


def merge_labels(labels) -> dict:
    """Group labels where one's token set is a subset of another's (a refinement of the
    same concept). Returns {label: canonical_label}; canonical = shortest (most general)."""
    labels = sorted(set(labels))
    parent = {l: l for l in labels}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    for i, a in enumerate(labels):
        ta = _tokens(a)
        for b in labels[i + 1:]:
            tb = _tokens(b)
            if ta <= tb or tb <= ta:  # one refines the other
                union(a, b)

    # canonical per group = shortest token count, then shortest string, then alpha
    groups: dict = {}
    for l in labels:
        groups.setdefault(find(l), []).append(l)
    mapping = {}
    for members in groups.values():
        canon = min(members, key=lambda m: (len(_tokens(m)), len(m), m))
        for m in members:
            mapping[m] = canon
    return mapping


# =============================================================================
# core
# =============================================================================
def _run_combined(frames, idx, model) -> list | None:
    """SINGLE-call path (2026-07-31 speedup #3): one qc.analyze_video round trip yields
    label+count+seed box+object_type+motion for every instance. Returns the assembled
    `instances` list (same schema as the legacy path) or None on VLM/parse failure so the
    caller can fall back to the validated multi-call path."""
    try:
        items = qc.analyze_video(frames, model=model)
    except qc.QwenError as e:
        print(f"[stage0-pre] combined call FAILED ({e}) — falling back to legacy multi-call")
        return None
    if not items:
        print("[stage0-pre] combined call returned no instances — falling back to legacy")
        return None
    print(f"[stage0-pre] combined -> { {it['label']: it['count'] for it in items} }")

    # canonicalise labels exactly as the legacy path does (token-subset synonym union);
    # a single reply rarely repeats a concept, but the merge keeps behavior aligned.
    mapping = merge_labels([it["label"] for it in items])
    groups: dict = {}
    for it in items:
        groups.setdefault(mapping[it["label"]], []).append(it)

    mid_frame = int(idx[len(idx) // 2])
    instances = []
    for c, members in sorted(groups.items()):
        # representative member: prefer one with a box (largest area), then highest count
        def _area(it):
            b = it["bbox"]
            return (b[2] - b[0]) * (b[3] - b[1]) if b else -1.0
        rep = max(members, key=lambda it: (_area(it), it["count"]))
        if rep["bbox"] is not None:
            seed_frame = int(idx[rep["frame_pos"]])
            bbox_xywh = qc.xyxy_to_xywh(rep["bbox"])
        else:
            seed_frame, bbox_xywh = mid_frame, None
        # conservative defaults for omitted fields — same as the legacy path
        typed = [m for m in members if m["object_type"]]
        moved = [m for m in members if m["motion"]]
        t = (max(typed, key=lambda m: m["type_confidence"]) if typed
             else {"object_type": "non-rigid", "type_confidence": 0.0})
        mo = (max(moved, key=lambda m: m["motion_confidence"]) if moved
              else {"motion": "dynamic", "motion_confidence": 0.0})
        instances.append({
            "concept": c,
            "object_type": t["object_type"],
            "confidence": round(float(t["type_confidence"]), 3),
            "motion": mo["motion"],
            "motion_confidence": round(float(mo["motion_confidence"]), 3),
            "count": int(max(m["count"] for m in members)),
            "seed_frame": int(seed_frame),
            "bbox_xywh": [round(v, 4) for v in bbox_xywh] if bbox_xywh else None,
            "aliases": sorted({m["label"] for m in members}),
            "source": "combined",
        })
    return instances


def run(frames_dir: Path, out_path: Path, *, model: str = qc.DEFAULT_MODEL,
        k: int | None = None, detect_frames: int = 4, min_box_area: float = 0.002,
        workers: int = 6, stride: int = 1, max_frames: int | None = None,
        overwrite: bool = False, legacy_calls: bool = False) -> dict:
    if out_path.exists() and not overwrite:
        print(f"[stage0-pre] {out_path} exists — skip (use --overwrite to redo)")
        return json.loads(out_path.read_text())

    # Select frames EXACTLY as run_stage0.load_frames_dir does (paths[::stride][:max_frames]) so
    # seed_frame is in the SAME coordinate as the frames Stage 0 / segment_video will receive.
    paths = list_frame_paths(frames_dir)[::stride]
    if max_frames:
        paths = paths[:max_frames]
    T = len(paths)
    K = k or adaptive_k(T)
    idx = np.linspace(0, T - 1, K).round().astype(int)
    idx = sorted(set(int(i) for i in idx))  # de-dup for very short clips
    frames = [np.asarray(Image.open(paths[i]).convert("RGB")) for i in idx]
    H, W = frames[0].shape[:2]
    print(f"[stage0-pre] {frames_dir.name}: T={T}, sampling K={len(idx)} at {idx} ({W}x{H})")

    # ---- COMBINED single-call path (default; --legacy-calls restores the multi-call one) ----
    if not legacy_calls:
        instances = _run_combined(frames, idx, model)
        if instances is not None:
            instances.sort(key=lambda r: (r["bbox_xywh"] is None, r["seed_frame"], r["concept"]))
            result = {
                "sequence": frames_dir.name,
                "model": model,
                "frame_size": [W, H],
                "n_frames": T,
                "sampled_indices": idx,
                "instances": instances,
            }
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
            n_nr = sum(1 for r in instances if r["object_type"] == "non-rigid")
            n_st = sum(1 for r in instances if r["motion"] == "static")
            print(f"[stage0-pre] wrote {out_path}: {len(instances)} instances "
                  f"({len(instances) - n_nr} rigid / {n_nr} non-rigid; "
                  f"{n_st} static / {len(instances) - n_st} dynamic) [combined 1-call]")
            return result
        # else: fall through to the legacy multi-call path below

    # 2) enumerate over the whole clip
    enum = qc.enumerate_video(frames, model=model)
    enum_counts = {e["label"]: e["count"] for e in enum}
    print(f"[stage0-pre] enumerate -> {enum_counts}")

    # 3) detect (for SEED BOXES) on a spread subset of the sampled frames — enumerate
    #    already handles cross-clip coverage, so a few frames suffice to box each concept.
    #    PARALLEL (network-bound). record the ACTUAL video frame index for each box.
    det_pos = sorted(set(np.linspace(0, len(idx) - 1, min(detect_frames, len(idx)))
                         .round().astype(int).tolist()))

    def _detect(pos):
        return idx[pos], qc.detect_instances(frames[pos], model=model)

    detections = []  # (video_frame_idx, label, bbox_xyxy or None)
    with ThreadPoolExecutor(max_workers=min(workers, len(det_pos))) as ex:
        for vid_idx, dets in ex.map(_detect, det_pos):
            for d in dets:
                detections.append((vid_idx, d["label"], d["bbox"]))
    det_labels = [lab for _, lab, _ in detections]
    print(f"[stage0-pre] detect on frames {[idx[p] for p in det_pos]}")
    print(f"[stage0-pre] detect -> {len(detections)} boxes, "
          f"{len(set(det_labels))} distinct labels")

    # 4) merge all labels (enumerate ∪ detect) into canonical concepts
    mapping = merge_labels(list(enum_counts) + det_labels)
    canon_labels = sorted(set(mapping.values()))

    # gather per-canonical evidence
    inst = {c: {"aliases": set(), "boxes": [], "from_enum": False, "from_det": False}
            for c in canon_labels}
    for lab, c in mapping.items():
        inst[c]["aliases"].add(lab)
        if lab in enum_counts:
            inst[c]["from_enum"] = True
    # boxes-per-frame tally (for count when enumerate is silent) + seed selection
    per_frame_canon_count: dict = {}
    for vid_idx, lab, box in detections:
        c = mapping[lab]
        inst[c]["from_det"] = True
        key = (c, vid_idx)
        per_frame_canon_count[key] = per_frame_canon_count.get(key, 0) + 1
        if box is not None:
            w, h = box[2] - box[0], box[3] - box[1]
            inst[c]["boxes"].append((vid_idx, box, w * h))

    # 5) classify rigid / non-rigid on the most instance-dense sampled frame
    dense_pos = int(np.argmax([
        sum(1 for v, _, _ in detections if v == vid_idx) for vid_idx in idx
    ])) if detections else len(frames) // 2
    types = qc.classify_types(frames[dense_pos], canon_labels, model=model)
    print(f"[stage0-pre] classify (on frame {idx[dense_pos]}) -> "
          f"{ {k: v['object_type'] for k, v in types.items()} }")

    # 5b) classify dynamic / static per canonical label on 8 frames spread ACROSS TIME (the
    #     motion question needs the temporal axis; a single frame cannot answer it). Prompt
    #     explicitly discounts camera motion. Labels the VLM leaves out default to DYNAMIC
    #     (conservative: dynamic instances run the full mesh pipeline, so a miss costs compute,
    #     never information).
    mot_pos = sorted(set(np.linspace(0, len(idx) - 1, min(8, len(idx))).round().astype(int).tolist()))
    try:
        motions = qc.classify_motion([frames[p] for p in mot_pos], canon_labels, model=model)
    except qc.QwenError as e:
        print(f"[stage0-pre] classify_motion FAILED ({e}) — all instances default to dynamic")
        motions = {}
    print(f"[stage0-pre] motion (on frames {[idx[p] for p in mot_pos]}) -> "
          f"{ {k: v['motion'] for k, v in motions.items()} }")

    # assemble instances
    mid_frame = int(idx[len(idx) // 2])
    instances = []
    for c in canon_labels:
        info = inst[c]
        big_boxes = [b for b in info["boxes"] if b[2] >= min_box_area]
        seed = max(big_boxes or info["boxes"], key=lambda b: b[2], default=None)
        if seed is not None:
            seed_frame, bbox_xyxy, _ = seed
            bbox_xywh = qc.xyxy_to_xywh(bbox_xyxy)
        else:
            seed_frame, bbox_xywh = mid_frame, None
        max_det_per_frame = max(
            (per_frame_canon_count[(c, v)] for v in idx if (c, v) in per_frame_canon_count),
            default=0)
        count = max(enum_counts.get(c, 0), max_det_per_frame, 1)
        t = types.get(c, {"object_type": "non-rigid", "confidence": 0.0})  # conservative default
        mo = motions.get(c, {"motion": "dynamic", "confidence": 0.0})      # conservative default
        src = "+".join(s for s, on in (("enum", info["from_enum"]), ("detect", info["from_det"])) if on)
        instances.append({
            "concept": c,
            "object_type": t["object_type"],
            "confidence": round(float(t["confidence"]), 3),
            "motion": mo["motion"],
            "motion_confidence": round(float(mo["confidence"]), 3),
            "count": int(count),
            "seed_frame": int(seed_frame),
            "bbox_xywh": [round(v, 4) for v in bbox_xywh] if bbox_xywh else None,
            "aliases": sorted(info["aliases"]),
            "source": src,
        })
    # stable order: located instances first (by seed frame), then text-only
    instances.sort(key=lambda r: (r["bbox_xywh"] is None, r["seed_frame"], r["concept"]))

    result = {
        "sequence": frames_dir.name,
        "model": model,
        "frame_size": [W, H],
        "n_frames": T,
        "sampled_indices": idx,
        "instances": instances,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    n_nr = sum(1 for r in instances if r["object_type"] == "non-rigid")
    n_st = sum(1 for r in instances if r["motion"] == "static")
    print(f"[stage0-pre] wrote {out_path}: {len(instances)} instances "
          f"({len(instances) - n_nr} rigid / {n_nr} non-rigid; "
          f"{n_st} static / {len(instances) - n_st} dynamic)")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--out", required=True, help="output stage0_pre.json path")
    ap.add_argument("--model", default=qc.DEFAULT_MODEL)
    ap.add_argument("--k", type=int, default=None, help="override adaptive enumerate frame count")
    ap.add_argument("--detect-frames", type=int, default=4,
                    help="how many sampled frames to run box-detection on (for seed boxes)")
    ap.add_argument("--min-box-area", type=float, default=0.002,
                    help="drop detect boxes below this normalised area when picking a seed")
    ap.add_argument("--workers", type=int, default=6, help="parallel detect requests")
    ap.add_argument("--stride", type=int, default=1, help="match run_stage0 frame selection")
    ap.add_argument("--max-frames", type=int, default=None, help="match run_stage0 frame selection")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--legacy-calls", action="store_true",
                    help="use the original multi-call VLM path (enumerate + detect x4 + "
                         "type + motion = 7 API calls) instead of the combined 1-call one")
    args = ap.parse_args()
    run(Path(args.frames_dir), Path(args.out), model=args.model, k=args.k,
        detect_frames=args.detect_frames, min_box_area=args.min_box_area,
        workers=args.workers, stride=args.stride, max_frames=args.max_frames,
        overwrite=args.overwrite, legacy_calls=args.legacy_calls)


if __name__ == "__main__":
    main()
