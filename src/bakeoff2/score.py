"""Bake-off v2 scorer. Metrics on the SAME eval points for every tracker:
  GT sets (PO/MOVi): EPE-2D px (GT-visible frames), delta@8px accuracy, OA (vis prediction acc),
    EPE-3D after per-scene similarity alignment (pred world -> GT world, Umeyama on visible pairs;
    monocular trackers have arbitrary gauge — alignment is standard TAP-Vid-3D practice),
    RELOCK@20: among points with a GT absence run >=20 frames, fraction whose mean EPE-2D over the
    first 5 GT-visible frames AFTER the gap is < 8px (direct R7 measurement).
  DAVIS (no GT): on-mask rate (pred-visible points inside their instance mask) — weak proxy, labeled.
4RC note: scored only on its meta frames_idx columns (native 30f protocol, labeled).
  python src/bakeoff2/score.py --pred outputs/bakeoff2_pred --eval outputs/bakeoff2_eval
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)
from src.geometry import umeyama_similarity  # noqa: E402

GAP, THR = 20, 8.0


def _gaps(v):
    w = np.nonzero(v)[0]
    if len(w) < 2:
        return []
    d = np.diff(w)
    return [(w[i], w[i + 1]) for i in np.nonzero(d > GAP)[0]]


def score_pair(ev, pr):
    res = {}
    fidx = None
    meta = json.loads(str(pr["meta"])) if "meta" in pr.files else {}
    if meta.get("native_30f") and "frames_idx" in meta:
        fidx = np.array(meta["frames_idx"], int)
    gv = ev["gt_vis"] if "gt_vis" in ev.files else None
    p2, vis = pr["p2d"], pr["vis"]
    if gv is not None:
        g2 = ev["gt_2d"]
        M = gv.copy()
        if fidx is not None:
            keep = np.zeros(gv.shape[1], bool); keep[fidx] = True
            M &= keep[None, :]
        err = np.linalg.norm(p2 - g2, axis=2)
        res["epe2d"] = float(np.median(err[M])) if M.any() else float("nan")
        res["delta8"] = float((err[M] < THR).mean()) if M.any() else float("nan")
        Mv = np.ones_like(gv, bool) if fidx is None else np.tile(keep, (len(gv), 1))
        res["oa"] = float((vis == gv)[Mv].mean())
        # relock@20
        hits = tot = 0
        for i in range(len(gv)):
            for (a, b) in _gaps(gv[i] if fidx is None else (gv[i] & keep)):
                post = np.nonzero(gv[i, b:])[0][:5] + b
                if len(post) == 0:
                    continue
                tot += 1
                hits += int(np.mean(err[i, post]) < THR)
        res["relock20"] = float(hits / tot) if tot else float("nan")
        res["n_gap"] = tot
        # 3D (similarity-aligned)
        if int(pr["has_3d"]) and "gt_3d" in ev.files and int(ev.get("has_3d", 1)):
            g3, p3 = ev["gt_3d"], pr["p3d"]
            sel = M & vis
            if sel.sum() >= 16:
                src = p3[sel]; dst = g3[sel]
                ss = np.linspace(0, len(src) - 1, min(4000, len(src))).astype(int)
                s, R, t = umeyama_similarity(src[ss], dst[ss])
                e3 = np.linalg.norm((s * (p3.reshape(-1, 3) @ R.T) + t).reshape(p3.shape) - g3, axis=2)
                res["epe3d"] = float(np.median(e3[sel]))
    else:  # DAVIS proxy
        masks = ev["masks"]; H, W = masks.shape[1:]
        T = masks.shape[1] if False else masks.shape[0]
        onm = []
        for i in range(len(p2)):
            for t in range(masks.shape[0]):
                if not vis[i, t]:
                    continue
                x, y = int(round(p2[i, t, 0])), int(round(p2[i, t, 1]))
                if 0 <= x < W and 0 <= y < H:
                    onm.append(masks[t, y, x] > 0)
                else:
                    onm.append(False)
        res["onmask"] = float(np.mean(onm)) if onm else float("nan")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="outputs/bakeoff2_pred")
    ap.add_argument("--eval", default="outputs/bakeoff2_eval")
    args = ap.parse_args()
    rows = {}
    for pp in sorted(glob.glob(os.path.join(args.pred, "pred_*.npz"))):
        base = os.path.basename(pp)[5:-4]
        stem, tracker = base.rsplit("__", 1)
        epath = os.path.join(args.eval, stem + ".npz")
        if not os.path.exists(epath):
            continue
        ev, pr = np.load(epath, allow_pickle=True), np.load(pp, allow_pickle=True)
        r = score_pair(ev, pr)
        ds = stem.split("_")[1]
        rows.setdefault((tracker, ds), []).append(r)
    # aggregate
    print(f"{'tracker':10s}{'ds':7s}{'EPE2D':>8}{'δ<8px':>8}{'OA':>7}{'relock@20':>11}{'EPE3D':>8}{'onmask':>8}{'n':>4}")
    agg_out = {}
    for (tk, ds), rs in sorted(rows.items()):
        def m(k):
            v = [r[k] for r in rs if k in r and r[k] == r[k]]
            return float(np.mean(v)) if v else float("nan")
        print(f"{tk:10s}{ds:7s}{m('epe2d'):>8.2f}{m('delta8'):>8.2f}{m('oa'):>7.2f}"
              f"{m('relock20'):>11.2f}{m('epe3d'):>8.3f}{m('onmask'):>8.2f}{len(rs):>4}")
        agg_out[f"{tk}|{ds}"] = {k: m(k) for k in ["epe2d", "delta8", "oa", "relock20", "epe3d", "onmask"]}
    json.dump(agg_out, open(os.path.join(args.pred, "scoreboard.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
