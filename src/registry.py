"""V3 mesh-registry helpers: presence segments + point-cloud->mesh re-capture registration.

THE MESH IS THE REGISTRY (V3 multi-presence extension, 2026-07-11): an instance's identity and
geometry are carried by its Stage-1 mesh. Long ego videos (PointOdyssey lt96) have objects leaving
and re-entering the view (mask presence ~0.16), which breaks POINT-level tracker identity across
absences even though the masklet identity survives. So each PRESENCE SEGMENT (run of non-empty
masklet frames) solves its own pose against the same mesh: within a segment the tracker provides
correspondence; across segments the mesh does.

Shared by src/stage4/run_link_dense.py (segment re-capture + keypoint hooking) and
src/stage4/stage4_finalize.py (per-segment SE(3) + mesh bridging). Pure numpy + scipy.cKDTree.
"""
from __future__ import annotations

import itertools

import numpy as np


def presence_segments(present, max_hole=2):
    """Bool (T,) presence -> list of CLOSED intervals [(s, e)] (inclusive).
    Holes of <= max_hole consecutive absent frames are merged into the surrounding segment
    (brief occlusions / mask flicker are not identity breaks)."""
    present = np.asarray(present, bool)
    idx = np.nonzero(present)[0]
    if len(idx) == 0:
        return []
    segs = []
    s = p = int(idx[0])
    for i in idx[1:]:
        i = int(i)
        if i - p <= max_hole + 1:          # gap of (i-p-1) absent frames <= max_hole
            p = i
        else:
            segs.append((s, p))
            s = p = i
    segs.append((s, p))
    return segs


def se3_inv(R, t):
    """Inverse of x -> R@x + t."""
    return R.T, -R.T @ t


def se3_compose(R2, t2, R1, t1):
    """Composition (R2,t2) ∘ (R1,t1): apply (R1,t1) FIRST. Returns (R, t)."""
    return R2 @ R1, R2 @ t1 + t2


def _kabsch(A, B):
    """Rigid R,t with B ~= R@A + t. A,B (n,3)."""
    cA, cB = A.mean(0), B.mean(0)
    H = (A - cA).T @ (B - cB)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, cB - R @ cA


def _pca_axes(pts):
    """(3,3) right-handed principal-axis matrix, columns sorted by descending spread."""
    c = pts.mean(0)
    cov = (pts - c).T @ (pts - c) / max(len(pts), 1)
    _, evecs = np.linalg.eigh(cov)         # ascending
    A = evecs[:, ::-1].copy()
    if np.linalg.det(A) < 0:
        A[:, 2] *= -1.0
    return A


def register_pointset_to_mesh(pts, mesh_verts, inlier_thr=None, min_inlier_frac=0.3,
                              max_pts=800, max_mesh=4000, icp_iters=30,
                              inplace_margin=0.03):
    """Re-capture registration of a (partial-view) point cloud onto an instance mesh.

    SCALE IS KNOWN AND ALREADY APPLIED by the caller: `pts` must be in the mesh's metric frame
    (mapped through the instance's t*-solved similarity, or the scene-scale refit) — only the
    RIGID part is solved here (a free scale on a partial view is unidentifiable).

    Pragmatic pipeline: multi-start (IN-PLACE identity + centroid + all 24 proper PCA axis
    permutation×flip combinations) -> quantile-trimmed ICP per start. START SELECTION uses a FINE
    score (inlier fraction at 0.025*bbox minus median-NN-distance/bbox): the coarse 0.1*bbox
    threshold cannot discriminate poses (measured: nearly every converged start clears it).

    MEASURED FUNDAMENTAL AMBIGUITY: a partial view on a smooth closed mesh fits many poses almost
    equally well (true vs flipped/slid pose scored 0.993 vs 0.988 on a real SAM3D mesh) — no
    geometric score resolves that. Two honesty measures instead of pretending: (1) STATIC-WORLD
    PRIOR — the tracker world is globally consistent, so an object that left the view and
    returned usually sits where it was; the IN-PLACE start (R=I, t=0: pts already live in mesh
    coordinates) is preferred whenever its refined score is within `inplace_margin` of the global
    best. (2) The returned info dict carries `ambiguous`/`margin` (score gap between the chosen
    basin and the best rotation-distinct basin) so callers can report it.

    The ok GATE stays at the spec's coarse threshold: inlier fraction at ~0.1*mesh-bbox >=
    min_inlier_frac ("this cloud plausibly lies on this mesh").

    Returns (R (3,3), t (3,), inlier_frac float [coarse], ok bool, info dict)
    with  mesh ≈ R @ pts + t;  info = {score, basin, margin, ambiguous}.
    """
    from scipy.spatial import cKDTree

    pts = np.asarray(pts, np.float64).reshape(-1, 3)
    mesh = np.asarray(mesh_verts, np.float64).reshape(-1, 3)
    if len(pts) < 3 or len(mesh) < 3:
        return np.eye(3), np.zeros(3), 0.0, False, {"score": -1.0, "basin": "none",
                                                    "margin": 0.0, "ambiguous": True}
    if len(pts) > max_pts:
        pts = pts[np.linspace(0, len(pts) - 1, max_pts).astype(int)]
    if len(mesh) > max_mesh:
        mesh = mesh[np.linspace(0, len(mesh) - 1, max_mesh).astype(int)]
    bbox = float(np.linalg.norm(mesh.max(0) - mesh.min(0)) + 1e-9)
    thr = float(inlier_thr) if inlier_thr is not None else 0.1 * bbox
    tree = cKDTree(mesh)
    Ap, Am = _pca_axes(pts), _pca_axes(mesh)
    cp, cm = pts.mean(0), mesh.mean(0)

    # starts: IN-PLACE identity (the static-world case: pts already ~on the mesh), centroid
    # identity, and the 24 proper PCA permutation×flip alignments (moved/rotated objects).
    starts = [("inplace", np.eye(3), np.zeros(3)), ("centroid", np.eye(3), cm - cp)]
    for perm in itertools.permutations(range(3)):
        P = np.zeros((3, 3))
        P[list(perm), np.arange(3)] = 1.0
        for sx, sy, sz in itertools.product((1, -1), repeat=3):
            R0 = Am @ P @ np.diag([sx, sy, sz]) @ Ap.T
            if np.linalg.det(R0) < 0.5:    # proper rotations only
                continue
            starts.append(("pca", R0, cm - R0 @ cp))

    fine = 0.025 * bbox                                 # start-selection resolution

    def _icp(R, t, iters, keep):
        """Quantile-trimmed ICP (keep the closest `keep` fraction of NN pairs each iteration):
        adapts its own scale, unlike an absolute-threshold trim which — measured on a thin box —
        keeps wrong-face correspondences and stalls ~35deg from the true pose."""
        for _ in range(iters):
            d, j = tree.query(pts @ R.T + t)
            cut = np.quantile(d, keep)
            sel = d <= cut
            if int(sel.sum()) < 3:
                break
            Rn, tn = _kabsch(pts[sel], mesh[j[sel]])
            if np.abs(Rn - R).max() < 1e-7 and np.abs(tn - t).max() < 1e-9:
                return Rn, tn
            R, t = Rn, tn
        return R, t

    def _score(R, t):
        d, _ = tree.query(pts @ R.T + t)
        return float((d < fine).mean()) - float(np.median(d)) / bbox

    results = []
    for name, R0, t0 in starts:
        R, t = _icp(R0, t0, icp_iters, keep=0.8)
        R, t = _icp(R, t, 10, keep=0.6)                 # tight trim locks the true contact set
        results.append((name, R, t, _score(R, t)))
    best = max(results, key=lambda r: r[3])
    # STATIC-WORLD PRIOR: prefer the in-place basin when its score is (near-)as-good OR its
    # absolute fit is tight — a partial view on a smooth mesh scores many poses equally, and noise
    # can even push a WRONG slid pose above the true one (measured), so "the object is where it
    # was" is the right disambiguation for a re-appearing object in a consistent tracker world.
    inp = results[0]
    d_inp, _ = tree.query(pts @ inp[1].T + inp[2])
    inp_med = float(np.median(d_inp)) / bbox
    chosen = inp if (inp[3] >= best[3] - inplace_margin or inp_med < 0.04) else best
    # ambiguity: score gap to the best ROTATION-DISTINCT (>15deg away) alternative basin
    margin = float("inf")
    for name, R, t, sc in results:
        cosang = np.clip((np.trace(R @ chosen[1].T) - 1.0) / 2.0, -1.0, 1.0)
        if np.degrees(np.arccos(cosang)) > 15.0:
            margin = min(margin, chosen[3] - sc)
    margin = float(margin) if margin != float("inf") else 1.0
    name, R, t, sc = chosen
    d, _ = tree.query(pts @ R.T + t)
    frac = float((d < thr).mean())                      # coarse gate fraction (spec)
    info = {"score": round(sc, 4), "basin": name, "margin": round(margin, 4),
            "ambiguous": bool(margin < 0.05), "med_d_frac": round(float(np.median(d)) / bbox, 4),
            "inplace_med_d_frac": round(inp_med, 4)}
    return R, t, frac, bool(frac >= min_inlier_frac), info
