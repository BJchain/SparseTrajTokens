"""PointOdyssey scene dir -> SCHEMA.VideoInput  (Path 2: GT camera).

PointOdyssey (ICCV23): long synthetic videos (849-4000 frames, 960x540 @30fps) with GT instance
masks, GT per-frame camera, GT depth, and GT point trajectories. Per scene dir:
  rgbs/rgb_%05d.jpg   masks/mask_%05d.png (palette; value = instance id)   depths/depth_%05d.png
  anno.npz {trajs_2d (T,N,2), trajs_3d (T,N,3) world, visibs/valids (T,N),
            intrinsics (T,3,3), extrinsics (T,4,4)}   scene_info.json {assets, character}

Conventions (PINNED numerically on sample/r4_new_f, see project memory):
  - anno extrinsics are WORLD->CAMERA (projecting trajs_3d with E gives 0.117px median error vs
    trajs_2d; inv(E) gives garbage). SCHEMA.VideoInput.camera_extrinsics wants camera->world -> invert.
  - depth png is uint16 with depth_m = png / 65535 * 1000  (ratio check: 65.3 ~= 65.535).

Frame subsampling: PO clips are FAR longer than the pipeline's working length, so we take a
CENTERED CONTIGUOUS window of max_frames*stride raw frames sampled every `stride` (keeps motion
smooth for the tracker, unlike whole-clip uniform sampling which makes huge jumps).

Instance selection (mask value semantics are asset-ish but not fully documented; chosen by area
statistics, background-name-agnostic):
  - drop value 0 (void) and any value whose MEAN area fraction > bg-frac (walls/floor/sky blobs)
  - keep values whose max-over-time area >= min_area_px; cap at --max-instances by area.
  NOTE: PO splits characters into part meshes (hands/head...) with separate mask ids; first full
  run treats each id as one instance (like MOVi per-asset). A character-prefix merge can be added
  later once the id->asset mapping is trusted.

gt_object_extent: unprojected GT depth at masked pixels (3 probe frames, median bbox diagonal) —
metric, powers Path 2 GT-ray placement. No per-object pose in PO -> Path2 grading = "ungraded"
(placement itself unaffected).

Per-scene GT self-check: reproject anno trajs_3d at the window's first frame; abort the scene if
median error > 2px (bad/corrupt annotation).

  <python+numpy+PIL> src/stage0/po_to_videoinput.py --scene-dir <.../scene_x> --out .../video_input.pkl
  <...>              --scenes-glob 'data/pointodyssey/{train,val,test}/*' --batch-root outputs/po_v3
"""
from __future__ import annotations
import argparse, glob, json, os, pickle, sys
import numpy as np
from PIL import Image

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from src.stage0.schema import Masklet, VideoInput  # noqa: E402

DEPTH_SCALE = 1000.0 / 65535.0                     # png -> meters (pinned, see docstring)


def _frame_ids(scene_dir):
    fs = sorted(glob.glob(os.path.join(scene_dir, "rgbs", "rgb_*.jpg")))
    return [int(os.path.basename(f)[4:-4]) for f in fs]


def _window(ids, max_frames, stride, window_index=0, n_windows=1):
    """Contiguous window of max_frames sampled every stride. n_windows=1: centered (original).
    n_windows>1 (Plan A multi-window, 2026-07-10): the window_index-th of up to n_windows
    evenly-spaced NON-overlapping windows over the clip; returns None if the clip is too short
    to fit this window index (caller skips that scene-window)."""
    n = len(ids)
    while stride > 1 and max_frames * stride > n:
        stride -= 1
    take = min(max_frames, n // max(stride, 1))
    span = take * stride
    if n_windows <= 1:
        start = max(0, (n - span) // 2)
        return [ids[start + i * stride] for i in range(take)]
    m_eff = max(1, min(n_windows, n // span))
    if window_index >= m_eff:
        return None
    gap = (n - m_eff * span) // max(m_eff, 1)
    start = window_index * (span + gap) + gap // 2
    return [ids[start + i * stride] for i in range(take)]


def build_video_input(scene_dir, max_frames=48, stride=8, min_area_px=256, bg_frac=0.20,
                      max_instances=24, px_gate=2.0, window_index=0, n_windows=1):
    anno = np.load(os.path.join(scene_dir, "anno.npz"))
    ids_all = _frame_ids(scene_dir)
    if not ids_all:
        raise RuntimeError("no rgbs")
    T_anno = anno["intrinsics"].shape[0]
    ids_all = [i for i in ids_all if i < T_anno]       # keep frames covered by the annotation
    sel = _window(ids_all, max_frames, stride, window_index, n_windows)
    if sel is None:
        raise RuntimeError(f"window {window_index}/{n_windows} does not fit this clip")

    # ---- GT self-check at the window's first frame (annotation sanity) ----
    # Some PO scenes (character*, gso_*) ship trajs_3d as a scalar NaN (no 3D GT) — the pipeline
    # doesn't need trajs_3d (masks/camera/depth only), so just skip the reprojection self-check.
    t0 = sel[0]
    K0, E0 = anno["intrinsics"][t0], anno["extrinsics"][t0]
    has_3d = anno["trajs_3d"].ndim == 3
    ok = (anno["valids"][t0] & anno["visibs"][t0]) if has_3d else np.zeros(1, bool)
    X = anno["trajs_3d"][t0][ok] if has_3d else np.zeros((0, 3))
    if len(X) >= 50:
        Xc = (np.c_[X, np.ones(len(X))] @ E0.T)[:, :3]
        z = Xc[:, 2]; good = z > 1e-6
        u = np.stack([K0[0, 0] * Xc[good, 0] / z[good] + K0[0, 2],
                      K0[1, 1] * Xc[good, 1] / z[good] + K0[1, 2]], 1)
        err = float(np.median(np.linalg.norm(u - anno["trajs_2d"][t0][ok][good], axis=1)))
        if err > px_gate:
            raise RuntimeError(f"GT reprojection self-check failed: {err:.2f}px > {px_gate}")
    # ---- frames + masks over the window ----
    frames, masks = [], []
    for t in sel:
        frames.append(np.asarray(Image.open(os.path.join(scene_dir, "rgbs", f"rgb_{t:05d}.jpg")).convert("RGB")))
        masks.append(np.asarray(Image.open(os.path.join(scene_dir, "masks", f"mask_{t:05d}.png"))))
    frames = np.stack(frames)                          # (T,H,W,3) uint8
    seg = np.stack(masks)                              # (T,H,W) int
    T, H, W = seg.shape

    # ---- instance selection by area statistics ----
    vals = np.unique(seg)
    keep = []
    for v in vals:
        if v == 0:
            continue
        area = (seg == v).reshape(T, -1).sum(1)
        if area.mean() / (H * W) > bg_frac:            # background-scale blob
            continue
        if area.max() < min_area_px:
            continue
        keep.append((int(v), int(area.max())))
    keep = [v for v, _ in sorted(keep, key=lambda x: -x[1])[:max_instances]]

    # ---- camera: w2c (pinned) -> c2w for SCHEMA ----
    K = anno["intrinsics"][sel].astype(np.float64)     # (T,3,3)
    E_w2c = anno["extrinsics"][sel].astype(np.float64)
    E_c2w = np.linalg.inv(E_w2c)                       # (T,4,4) camera->world

    # ---- per-instance metric extent from GT depth (3 probe frames, biggest-mask first) ----
    gt_ext = {}
    for v in keep:
        area = (seg == v).reshape(T, -1).sum(1)
        probes = np.argsort(area)[::-1][:3]
        diags = []
        for pi in probes:
            t = sel[pi]
            dpth = np.asarray(Image.open(os.path.join(scene_dir, "depths", f"depth_{t:05d}.png"))
                              ).astype(np.float64) * DEPTH_SCALE
            m = (seg[pi] == v) & (dpth > 1e-3)
            if m.sum() < 20:
                continue
            ys, xs = np.nonzero(m)
            if len(ys) > 4000:
                sub = np.linspace(0, len(ys) - 1, 4000).astype(int); ys, xs = ys[sub], xs[sub]
            z = dpth[ys, xs]
            Kt = anno["intrinsics"][t]
            pc = np.stack([(xs - Kt[0, 2]) * z / Kt[0, 0], (ys - Kt[1, 2]) * z / Kt[1, 1], z], 1)
            pw = (np.c_[pc, np.ones(len(pc))] @ E_c2w[pi].T)[:, :3]
            diags.append(float(np.linalg.norm(pw.max(0) - pw.min(0))))
        if diags:
            gt_ext[v] = float(np.median(diags))

    masklets = [Masklet(instance_id=int(v), masks=(seg == v), concept=f"po:{v}") for v in keep]
    return VideoInput(frames=frames, masklets=masklets,
                      camera_intrinsics=K, camera_extrinsics=E_c2w,
                      gt_object_poses=None, gt_object_extent=gt_ext or None,
                      gt_object_bbox=None), sel


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--scene-dir"); g.add_argument("--scenes-glob")
    ap.add_argument("--out"); ap.add_argument("--batch-root")
    ap.add_argument("--max-frames", type=int, default=48)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--min-area-px", type=int, default=256)
    ap.add_argument("--bg-frac", type=float, default=0.20)
    ap.add_argument("--max-instances", type=int, default=24)
    ap.add_argument("--window-index", type=int, default=0)
    ap.add_argument("--n-windows", type=int, default=1)
    ap.add_argument("--scene-suffix", default="", help="appended to scene_<name> in batch mode")
    ap.add_argument("--vlm-labels", action="store_true",
                    help="replace the po:<id> placeholder concepts with Qwen-VL semantic labels "
                         "(needs DASHSCOPE_API_KEY; per-instance failures keep the placeholder)")
    args = ap.parse_args()

    def _label(vi, name):
        if not args.vlm_labels:
            return
        try:
            from src.stage0.vlm_labeler import label_video_input
            label_video_input(vi)
        except Exception as e:
            print(f"[po->vi] {name}: VLM labels FAILED ({type(e).__name__}: {e}); "
                  f"keeping po:<id> placeholders", flush=True)

    if args.scene_dir:
        vi, sel = build_video_input(args.scene_dir, args.max_frames, args.stride,
                                    args.min_area_px, args.bg_frac, args.max_instances,
                                    window_index=args.window_index, n_windows=args.n_windows)
        _label(vi, os.path.basename(args.scene_dir))
        out = args.out or os.path.join(args.scene_dir, "video_input.pkl")
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        pickle.dump(vi, open(out, "wb"))
        print(f"[po->vi] {os.path.basename(args.scene_dir)}: {len(vi.masklets)} instances, "
              f"T={len(sel)} (raw {sel[0]}..{sel[-1]}) -> {out}", flush=True)
        return
    scenes = sorted(d for d in glob.glob(args.scenes_glob) if os.path.isdir(d)
                    and os.path.exists(os.path.join(d, "anno.npz")))
    print(f"[po->vi] BATCH: {len(scenes)} scenes -> {args.batch_root}", flush=True)
    n_ok = 0
    for sd in scenes:
        name = os.path.basename(sd.rstrip("/"))
        out = os.path.join(args.batch_root, f"scene_{name}{args.scene_suffix}", "video_input.pkl")
        try:
            vi, sel = build_video_input(sd, args.max_frames, args.stride,
                                        args.min_area_px, args.bg_frac, args.max_instances,
                                        window_index=args.window_index, n_windows=args.n_windows)
        except Exception as e:
            print(f"[po->vi] {name}: SKIP {type(e).__name__}: {str(e)[:80]}", flush=True)
            continue
        _label(vi, name)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        pickle.dump(vi, open(out, "wb"))
        n_ok += 1
        print(f"[po->vi] {name}: {len(vi.masklets)} inst, T={len(sel)}", flush=True)
    print(f"[po->vi] DONE {n_ok}/{len(scenes)}", flush=True)


if __name__ == "__main__":
    main()
