"""Qwen-VL semantic labeling for adapter-built Masklets (PointOdyssey first).

GT adapters that bypass Stage 0 (po_to_videoinput) ship no text labels: PO's own
scene_info.json asset names are Blender object names ('model.003', 'Cube.005'), many
scenes lack the file entirely, and the mask-value -> asset mapping is undocumented
(observed mask ids exceed the asset list). So the semantic label comes from Qwen-VL
instead: crop the instance at its largest-mask frame, outline its mask in red, ask for
a 1-3 word English noun phrase. PO splits characters into part meshes, hence the
"part + owner" phrasing in the prompt ('horse leg', not just 'leg').

Entry points:
  label_video_input(vi, ...)  in-process: sets Masklet.concept in place
                              (po_to_videoinput --vlm-labels uses this for NEW scenes)
  CLI batch                   patch EXISTING video_input.pkl files. Follows symlinks and
                              writes the REAL target (e.g. po_reg_base behind
                              po_v3_reg_full) so every root sharing the base updates
                              together. A vlm_labels.json sidecar next to the real pkl
                              makes re-runs resume without new API calls. Afterwards run
                              stage4 backfill_labels.py to propagate into final_repr.

    python src/stage0/vlm_labeler.py --batch-root outputs/po_v3 \
        [--workers 8] [--limit N] [--dry-run] [--relabel]

Idempotent: only instances whose concept is missing or a placeholder ('po:<n>') are
labeled; real text labels are left alone unless --relabel. Needs DASHSCOPE_API_KEY.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import re
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from src.stage0 import qwen_client  # noqa: E402
from src.stage0.schema import Masklet, VideoInput  # noqa: E402,F401  (unpickle registration)

_PLACEHOLDER = re.compile(r"po:\d+")            # adapter's non-semantic fallback concepts
_PROMPT = ("The red outline marks ONE object, or ONE body part of a person/animal/creature, "
           "in this image. Name it with a short lowercase English noun phrase (1-3 words). "
           "If it is a body part, include the owner, e.g. 'horse leg', 'human hand'. "
           "Reply with ONLY the noun phrase — no punctuation, no explanation.")
_WORDS = re.compile(r"[a-z][a-z'\-]*")


def _needs_label(concept) -> bool:
    return concept is None or bool(_PLACEHOLDER.fullmatch(str(concept)))


def _crop_with_outline(frame, mask, margin=0.3, max_side=768, min_side=32):
    """Crop the mask bbox (+margin for context) and paint the mask edge red."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    my, mx = int((y1 - y0) * margin) + 8, int((x1 - x0) * margin) + 8
    Y0, Y1 = max(0, y0 - my), min(frame.shape[0], y1 + my)
    X0, X1 = max(0, x0 - mx), min(frame.shape[1], x1 + mx)
    crop = frame[Y0:Y1, X0:X1].copy()
    sub = mask[Y0:Y1, X0:X1]
    er = sub & np.roll(sub, 1, 0) & np.roll(sub, -1, 0) & np.roll(sub, 1, 1) & np.roll(sub, -1, 1)
    edge = sub & ~er
    edge = edge | np.roll(edge, 1, 0) | np.roll(edge, -1, 0) | np.roll(edge, 1, 1) | np.roll(edge, -1, 1)
    crop[edge] = (255, 0, 0)
    img = Image.fromarray(crop)
    if max(img.size) > max_side:
        s = max_side / max(img.size)
        img = img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))))
    if min(img.size) < min_side:
        s = min_side / min(img.size)
        img = img.resize((int(img.width * s), int(img.height * s)))
    return img


def _clean(text):
    t = (text or "").strip().lower()
    t = t.splitlines()[0] if t else ""
    words = _WORDS.findall(t.replace("_", " "))
    out = " ".join(words[:3])
    return out or None


def label_instance(frame, mask, model=qwen_client.DEFAULT_MODEL):
    """One (frame, bool mask) -> cleaned noun phrase, or None if the mask is empty."""
    img = _crop_with_outline(frame, mask)
    if img is None:
        return None
    content = [{"type": "image_url", "image_url": {"url": qwen_client._to_data_url(img)}},
               {"type": "text", "text": _PROMPT}]
    reply = qwen_client._chat([{"role": "user", "content": content}], model, max_tokens=32)
    return _clean(reply)


def label_video_input(vi, model=qwen_client.DEFAULT_MODEL, workers=8, force=False, log=print):
    """Label every placeholder-concept masklet of a VideoInput IN PLACE (threaded; API-bound).
    Per-instance failures keep the old concept. Returns {instance_id: concept} after."""
    todo = [m for m in vi.masklets if force or _needs_label(m.concept)]

    def work(m):
        t = int(m.masks.reshape(m.masks.shape[0], -1).sum(1).argmax())
        try:
            return m, label_instance(vi.frames[t], m.masks[t], model)
        except Exception as e:
            log(f"  [vlm-label] instance {m.instance_id}: FAILED {type(e).__name__}: {e}")
            return m, None

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for m, lab in ex.map(work, todo):
            if lab:
                m.concept = lab
    return {int(m.instance_id): m.concept for m in vi.masklets}


def _atomic_pickle(obj, path):
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(obj, f, protocol=4)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def patch_scene(sd, model, workers, relabel=False, dry_run=False):
    """Label one scene dir's video_input.pkl (through its symlink). Returns
    {instance_id: concept} or None if the scene has no video_input.pkl."""
    link = os.path.join(sd, "video_input.pkl")
    if not os.path.exists(link):
        return None
    real = os.path.realpath(link)
    sidecar = os.path.join(os.path.dirname(real), "vlm_labels.json")

    with open(real, "rb") as f:
        vi = pickle.load(f)
    before = {int(m.instance_id): m.concept for m in vi.masklets}

    if os.path.exists(sidecar) and not relabel:                # resume: no API calls
        saved = {int(k): v for k, v in json.load(open(sidecar))["labels"].items()}
        for m in vi.masklets:
            lab = saved.get(int(m.instance_id))
            if lab and _needs_label(m.concept):
                m.concept = lab
    else:
        label_video_input(vi, model=model, workers=workers, force=relabel)
        if not dry_run:
            json.dump({"labels": {int(m.instance_id): m.concept for m in vi.masklets},
                       "model": model, "prompt": _PROMPT},
                      open(sidecar, "w"), indent=2)

    after = {int(m.instance_id): m.concept for m in vi.masklets}
    if after != before and not dry_run:
        _atomic_pickle(vi, real)
    return after


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-root", required=True, help="dir with scene_*/video_input.pkl")
    ap.add_argument("--model", default=qwen_client.DEFAULT_MODEL)
    ap.add_argument("--workers", type=int, default=8, help="parallel API calls per scene")
    ap.add_argument("--limit", type=int, default=0, help="only the first N scenes (smoke)")
    ap.add_argument("--dry-run", action="store_true", help="call the VLM but write nothing")
    ap.add_argument("--relabel", action="store_true",
                    help="re-query even instances that already have a real text label")
    args = ap.parse_args()

    sds = sorted(glob.glob(os.path.join(args.batch_root, "scene_*")))
    if args.limit:
        sds = sds[:args.limit]
    print(f"[vlm-label] {len(sds)} scene dirs under {args.batch_root}"
          + (" (DRY-RUN)" if args.dry_run else ""), flush=True)
    n_done = n_lab = n_inst = 0
    for sd in sds:
        try:
            res = patch_scene(sd, args.model, args.workers, args.relabel, args.dry_run)
        except Exception as e:
            print(f"[vlm-label] {os.path.basename(sd)}: FAILED {type(e).__name__}: {e}", flush=True)
            continue
        if res is None:
            continue
        n_done += 1; n_inst += len(res)
        n_lab += sum(1 for c in res.values() if not _needs_label(c))
        print(f"[vlm-label] {os.path.basename(sd)}: "
              f"{ {k: v for k, v in sorted(res.items())} }", flush=True)
    print(f"[vlm-label] DONE: {n_done} scenes, {n_lab}/{n_inst} instances labeled", flush=True)


if __name__ == "__main__":
    main()
