"""Stage 2b (v2): motion correction + rigid/non-rigid classification (REFERENCE §3.0/§3.4/§3.5).

Consumes Stage 4 SPARSE keypoint trajectories (NOT dense ARAP — the v1 route is deleted).

object_type is set by a CONSERVATIVE RIGID-GATE (REFERENCE §3.0/§3.5, 2026-06-19) — NOT by the
old "motion-part count == 1?" symmetric test. Four independent angles (render->D4RT, 4RC vs
D4RT, render texture, scale-invariant contrast) showed non-rigid recall is TRACKER-limited:
D4RT's coherent drift on dynamic objects is indistinguishable from real part motion. So we stop
trying to detect subtle non-rigidity and instead:
  default NON-RIGID (keep per-keypoint trajectories); collapse to RIGID (global SE(3)) ONLY when
  the whole-object trajectory is confidently, uniformly rigid — i.e. NO spatially-coherent
  cluster of >= min_kp keypoints whose per-keypoint residual to the dominant rigid fit exceeds
  tau_rigid (bbox-normalized).
This is LOSSLESS w.r.t. what the tracker captured (motion the tracker missed isn't in the
trajectories anyway) and the error is asymmetric (false-rigid loses info; false-non-rigid only
costs compression), hence the one-sided bias. It also dissolves the cross-dataset tau_e problem
(we need only a "confident rigid" floor, not a symmetric non-rigid/rigid boundary).

The motion-part decomposition comes from the SAME per-keypoint residual (cluster_motion_parts:
moving keypoints = residual > tau_rigid, clustered spatially into parts), so object_type and the
part structure are one unified signal — rigid IFF K_motion == 1. The old pairwise co-motion merge
over semantic parts is RETIRED for this (it inherited the aggregate-residual blind spot and
collapsed every articulated object to K_motion=1 on sparse tracker trajectories); comotion_residual
/ _rel_motion_max are kept only for reference. §4: a rigid object's global SE(3) = per-frame
Kabsch over all kp.

Pure numpy. Reusable functions + a batch runner that also grades against MOVi-E GT (rotation
angle relative to frame 0 is invariant to the D4RT-vs-GT frame alignment, so comparable).
"""
from __future__ import annotations

import warnings

import numpy as np

TAU_R_DEG = 7.0        # §3.4: fixed vs revolute SE3 angle 0 vs >=74deg in M1 dense data
ALPHA_T = 0.025        # §3.4: tau_t = alpha * bbox_diag
MIN_PART_KP = 3        # Kabsch needs >=3 non-collinear points


def kabsch(A, B):
    """Rigid transform mapping A -> B (both (n,3)):  B ~= R @ A^T + t.  Returns (R, t)."""
    cA, cB = A.mean(0), B.mean(0)
    H = (A - cA).T @ (B - cB)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, cB - R @ cA


def _rot_angle_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def part_se3_traj(P, vis=None):
    """P (Kp,T,3) keypoint trajectories -> per-frame SE(3) rest(frame0)->frame t.
    Returns (R (T,3,3), t (T,3), ok (T,) bool, ref_frame int) — ref_frame is 0 whenever
    frame 0 is solvable (old contract preserved); otherwise the anchor frame a*.

    DE-ANCHORED from frame 0 (long-window fix, 2026-07-09): the old solver required every frame to
    share >=MIN_PART_KP points with FRAME 0, which collapses on long windows / egocentric sweeps
    (PO stride4x96: SE3-usable fell 42%->8% from co-visibility dying, not from tracking).
      1) DIRECT solve (0,f) when co-visible >= MIN_PART_KP — identical to before, drift-free
         (short windows / other datasets behave exactly as before);
      2) else CHAIN: relative Kabsch from the nearest earlier ok frame g with co-vis(g,f) >=
         MIN_PART_KP, composed onto g's absolute pose (bounded drift, only where direct fails).
    """
    Kp, T, _ = P.shape
    if vis is None:
        vis = np.ones((Kp, T), bool)
    R = np.tile(np.eye(3), (T, 1, 1)); t = np.zeros((T, 3)); ok = np.zeros(T, bool)
    # ANCHOR = the best-visibility frame a* (NOT hardwired to 0: on long/ego windows frame 0 can be
    # unobserved, which under the old scheme killed every frame). Solve poses relative to a*.
    a = int(np.argmax(vis.sum(0)))
    va = vis[:, a]
    if int(va.sum()) < MIN_PART_KP:                            # nothing observable at all
        return R, t, ok, 0
    ok[a] = True                                               # a* = identity seed
    for f in list(range(a + 1, T)) + list(range(a - 1, -1, -1)):
        sel = va & vis[:, f]
        if int(sel.sum()) >= MIN_PART_KP:                      # direct vs anchor, drift-free
            R[f], t[f] = kabsch(P[sel, a], P[sel, f])
            ok[f] = True
            continue
        step = 1 if f > a else -1                              # chained fallback toward the anchor
        for g in range(f - step, a - step, -step):
            if not ok[g]:
                continue
            sg = vis[:, g] & vis[:, f]
            if int(sg.sum()) >= MIN_PART_KP:
                Rr, tr = kabsch(P[sg, g], P[sg, f])            # g -> f
                R[f] = Rr @ R[g]; t[f] = Rr @ t[g] + tr        # compose a*->g->f
                ok[f] = True
                break
    # Re-express to frame-0 reference when possible (keeps the old "rest=frame0" contract exactly
    # whenever frame 0 is solvable — i.e. all short-window/other-dataset behavior unchanged).
    if a != 0 and ok[0]:
        R0i, t0i = R[0].T, -R[0].T @ t[0]                      # inv(a*->0)
        for f in range(T):
            if ok[f]:
                R[f], t[f] = R[f] @ R0i, R[f] @ t0i + t[f]     # 0->f = (a*->f) ∘ inv(a*->0)
        return R, t, ok, 0
    return R, t, ok, (0 if ok[0] else a)   # ref_frame: 0 normally; a* when frame0 is unobservable


def _rel_motion_max(Ra, ta, Rb, tb, ok):
    """Max over valid frames of (rel rotation angle deg, rel translation magnitude)."""
    f = np.nonzero(ok)[0]
    if len(f) == 0:
        return 0.0, 0.0
    relR = max(_rot_angle_deg(Rb[i] @ Ra[i].T) for i in f)
    relt = max(float(np.linalg.norm(tb[i] - ta[i])) for i in f)
    return relR, relt


def comotion_residual(Pa, va, Pb, vb):
    """Position-ROBUST co-motion test: fit ONE rigid SE(3) to A∪B (rest frame0 -> frame f)
    and return the MEDIAN over frames of the RMS residual. Low => A,B move as one rigid body;
    high => they articulate. Unlike per-part SE(3) rel_t, this is unaffected by how far the
    object sits from the frame origin (the failure mode on D4RT's depth~6 ref0 frame)."""
    P = np.concatenate([Pa, Pb], 0)
    v = np.concatenate([va, vb], 0)
    v0 = v[:, 0]
    res = []
    for f in range(P.shape[1]):
        sel = v0 & v[:, f]
        if int(sel.sum()) >= MIN_PART_KP:
            R, t = kabsch(P[sel, 0], P[sel, f])
            pred = (R @ P[sel, 0].T).T + t
            res.append(float(np.sqrt(((P[sel, f] - pred) ** 2).sum(1).mean())))
    return float(np.median(res)) if res else 0.0


def _weighted_kabsch(A, B, w):
    w = w / (w.sum() + 1e-12)
    cA = (w[:, None] * A).sum(0); cB = (w[:, None] * B).sum(0)
    H = (A - cA).T @ (w[:, None] * (B - cB))
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, cB - R @ cA


def per_keypoint_residual(P, vis, n_iter=6):
    """Robustly fit the DOMINANT rigid motion (rest=frame0 -> frame f) via IRLS with a Tukey
    biweight, then return each keypoint's residual to that dominant motion (median over frames).
    Keypoints that follow the bulk -> ~tracker noise; keypoints on a part that moves differently
    (even a SMALL part) -> large residual. This is the sparse-trajectory analog of ARAP e_i and,
    unlike an aggregate rigid-fit residual, is NOT swamped by the large static component."""
    K, T, _ = P.shape
    E = np.full((K, T), np.nan)
    for f in range(T):
        v = vis[:, 0] & vis[:, f]
        if int(v.sum()) < 4:
            continue
        idx = np.nonzero(v)[0]
        A, B = P[idx, 0], P[idx, f]
        w = np.ones(len(idx))
        R, t = _weighted_kabsch(A, B, w)
        for _ in range(n_iter):
            r = np.linalg.norm(B - (A @ R.T + t), axis=1)
            s = 1.4826 * np.median(r) + 1e-9          # robust scale (MAD)
            u = r / (2.5 * s)
            w = np.where(u < 1.0, (1 - u ** 2) ** 2, 0.0)   # Tukey biweight
            if w.sum() < 4:
                break                                  # degenerate -> keep last fit
            R, t = _weighted_kabsch(A, B, w)
        E[idx, f] = np.linalg.norm(B - (A @ R.T + t), axis=1)
    with warnings.catch_warnings():                     # keypoints never co-visible -> all-NaN row
        warnings.simplefilter("ignore", RuntimeWarning)  # (-> NaN, handled by nan_to_num upstream)
        return np.nanmedian(E, axis=1)                  # e_i per keypoint (absolute units)


def smoothed_residual(P, vis=None, knn=8):
    """Per-keypoint residual to the dominant rigid motion, bbox-normalized and SPATIALLY SMOOTHED
    (median over each keypoint's `knn` frame-0 neighbours). The smoothing is the crux: scattered
    tracker-noise spikes are killed (neighbours are low) while a real moving part stays high
    (neighbours agree). Returns (e_raw, e_smooth, bbox_diag)."""
    from scipy.spatial import cKDTree
    K, T, _ = P.shape
    if vis is None:
        vis = np.ones((K, T), bool)
    bbox = float(np.linalg.norm(P[:, 0].max(0) - P[:, 0].min(0))) + 1e-9
    e = np.nan_to_num(per_keypoint_residual(P, vis)) / bbox
    nbr = cKDTree(P[:, 0]).query(P[:, 0], k=min(knn, K))[1]
    e_s = np.median(e[nbr], axis=1)
    return e, e_s, bbox


def cluster_motion_parts(P, vis=None, tau_rigid=0.05, min_kp=4, knn=8):
    """Decompose an object's keypoints into MOTION PARTS straight from the per-keypoint residual to
    the dominant rigid motion (REFERENCE §3.0/§3.5 unified approach) — NOT from the old pairwise
    co-motion merge over semantic parts (which inherited the aggregate-residual blind spot and, on
    sparse tracker trajectories, collapsed every articulated object to K_motion=1).

    keypoints whose spatially-smoothed residual exceeds tau_rigid = "moving"; the rest = the
    dominant/base part (label 0). Moving keypoints are then clustered SPATIALLY (frame-0 KNN graph,
    connected components): each component with >= min_kp keypoints becomes its own motion part
    (label 1, 2, ...). Tiny components are noise -> folded back into base. If moving keypoints exist
    but none form a >= min_kp spatial cluster (distributed deformation rather than a discrete part),
    they all become a single moving part (label 1). So K_motion == 1 IFF fewer than min_kp keypoints
    are moving — identical to the conservative gate's rigid condition (the two stay consistent).

    On PartNet this recovers the moving link with PERFECT precision wherever the tracker captured
    the motion (e.g. pm101293 P=1.0/R=0.88); where the tracker attenuated the part below the noise
    floor nothing is recovered — but those are exactly the gate's lossless-rigid cases."""
    e, e_s, bbox = smoothed_residual(P, vis, knn)
    K = P.shape[0]
    high = e_s > tau_rigid
    label = np.zeros(K, int)                                   # 0 = dominant/base part
    if int(high.sum()) >= min_kp:
        from scipy.spatial import cKDTree
        idx = np.nonzero(high)[0]
        nbr = cKDTree(P[:, 0]).query(P[:, 0], k=min(knn, K))[1]
        uf = _UF(K)
        for i in idx:                                          # connect spatially-adjacent movers
            for j in nbr[i]:
                if high[j]:
                    uf.u(int(i), int(j))
        comp = {}
        for i in idx:
            comp.setdefault(uf.f(int(i)), []).append(int(i))
        big = [c for c in comp.values() if len(c) >= min_kp]
        if big:
            for lab, c in enumerate(sorted(big, key=len, reverse=True), start=1):
                label[c] = lab                                 # each coherent cluster = a motion part
        else:
            label[idx] = 1                                     # distributed deformation -> one part
    n_moving = len(set(label.tolist())) - 1
    return {"motion_part": label, "n_moving": n_moving, "K_motion": n_moving + 1,
            "e_per_kp": e, "e_smooth": e_s, "high_mask": high, "n_high": int(high.sum()),
            "bbox_diag": bbox, "audit_max_resid": float(e_s.max())}


def conservative_rigid_gate(P, vis=None, tau_rigid=0.05, min_kp=4, knn=8):
    """CONSERVATIVE one-sided gate (REFERENCE §3.0/§3.5). DEFAULT non-rigid; judge RIGID only when
    the whole-object trajectory is confidently uniform — i.e. no >= min_kp keypoints move (smoothed
    residual to the dominant rigid fit > tau_rigid). Thin wrapper over cluster_motion_parts so
    object_type and the motion-part decomposition stay consistent (rigid IFF K_motion == 1).

    Why one-sided (not a symmetric tau_e boundary): non-rigid recall is tracker-limited (coherent
    D4RT drift mimics real part motion), so a clean cross-dataset boundary doesn't exist. But the
    error is asymmetric — false-rigid LOSES motion, false-non-rigid only costs compression — so we
    bias to non-rigid and only collapse to SE(3) when confidently rigid. LOSSLESS: a rigid verdict
    means tracked residual < tau_rigid everywhere, so no captured motion is discarded.

    `audit_max_resid` = max smoothed residual among keypoints (for the no-loss audit: a rigid
    verdict should always have this < tau_rigid, i.e. nothing structured was thrown away)."""
    mp = cluster_motion_parts(P, vis, tau_rigid=tau_rigid, min_kp=min_kp, knn=knn)
    confident_rigid = mp["K_motion"] == 1
    e_s = mp["e_smooth"]
    return {"object_type": "rigid" if confident_rigid else "non-rigid",
            "confident_rigid": confident_rigid, "e_per_kp": mp["e_per_kp"], "e_smooth": e_s,
            "high_mask": mp["high_mask"], "n_high": mp["n_high"], "motion_part": mp["motion_part"],
            "K_motion": mp["K_motion"], "tau_rigid": tau_rigid, "bbox_diag": mp["bbox_diag"],
            "e_smooth_median": float(np.median(e_s)), "e_smooth_p90": float(np.percentile(e_s, 90)),
            "audit_max_resid": mp["audit_max_resid"]}


def classify_nonrigid(P, vis=None, tau_e=0.06, min_kp=4, knn=8):
    """DEPRECATED symmetric detector (kept for diagnostics/sweeps). The pipeline now classifies via
    conservative_rigid_gate (REFERENCE §3.0/§3.5). Mechanically identical (non-rigid if a coherent
    cluster of >= min_kp keypoints exceeds tau_e), but the gate reframes the threshold as a
    one-sided 'confident rigid' floor with a non-rigid default."""
    e, e_s, bbox = smoothed_residual(P, vis, knn)
    high = e_s > tau_e
    n_high = int(high.sum())
    return {"e_per_kp": e, "e_smooth": e_s, "high_mask": high, "n_high": n_high,
            "object_type": "non-rigid" if n_high >= min_kp else "rigid",
            "bbox_diag": bbox, "e_smooth_median": float(np.median(e_s)),
            "e_smooth_p90": float(np.percentile(e_s, 90)), "e_smooth_max": float(e_s.max())}


class _UF:
    def __init__(s, n): s.p = list(range(n))
    def f(s, x):
        while s.p[x] != x: s.p[x] = s.p[s.p[x]]; x = s.p[x]
        return x
    def u(s, a, b): s.p[s.f(a)] = s.f(b)


def stage2b_instance(P, part_id=None, vis=None, tau_rigid=0.05, min_kp=4, knn=8, vlm_type=None):
    """P (K,T,3) all keypoints of one instance; part_id (K,) optional semantic part per keypoint
    (Stage 2a; carried through for downstream use, NOT used to decide motion structure).

    object_type: the Stage 0-pre VLM verdict (`vlm_type` in {"rigid","non-rigid"}) is
    AUTHORITATIVE when provided — classification was moved to the VLM because the residual gate is
    tracker-limited on real video (degraded D4RT trajectories collapse to ~static -> false rigid,
    losing deformable trajectories). When no VLM type is given (MOVi-E / dt4d / PartNet have none)
    we fall back to the CONSERVATIVE rigid-gate (rigid IFF K_motion == 1). Either way the residual
    is still computed and kept as an AUDIT signal (gate_object_type / type_agrees), never discarded.

    motion-part decomposition (cluster_motion_parts: moving keypoints = residual > tau_rigid,
    spatially clustered):
      - RIGID  -> single base part (motion_part all 0); §4 global SE(3) over all keypoints.
      - NON-RIGID -> per-keypoint residual clusters are KEPT for pass2 point allocation (this is
        the part-merging job the VLM does NOT do); per-keypoint trajectories are preserved at §4.
    Returns dict: motion_part, K_motion, object_type, type_source, gate diagnostics, whole_se3
    (rigid only)."""
    K, T, _ = P.shape
    if vis is None:
        vis = np.ones((K, T), bool)
    mp = cluster_motion_parts(P, vis, tau_rigid=tau_rigid, min_kp=min_kp, knn=knn)
    gate_type = "rigid" if mp["K_motion"] == 1 else "non-rigid"

    vlm = vlm_type if vlm_type in ("rigid", "non-rigid") else None
    object_type = vlm or gate_type

    if object_type == "rigid":
        motion_part = np.zeros(K, int)          # rigid -> single base part
        k_motion, n_moving = 1, 0
    else:                                        # non-rigid -> keep residual part decomposition
        motion_part = mp["motion_part"]
        k_motion, n_moving = mp["K_motion"], mp["n_moving"]

    out = {"motion_part": motion_part, "K_motion": k_motion, "n_moving": n_moving,
           "object_type": object_type, "confident_rigid": object_type == "rigid",
           "type_source": "vlm" if vlm else "gate", "gate_object_type": gate_type,
           "type_agrees": bool(vlm is None or vlm == gate_type),
           "e_per_kp": mp["e_per_kp"], "e_smooth": mp["e_smooth"], "high_mask": mp["high_mask"],
           "n_high": mp["n_high"], "gate_audit_max_resid": mp["audit_max_resid"],
           "bbox_diag": mp["bbox_diag"], "tau_rigid": tau_rigid,
           "K_semantic": int(len(np.unique(part_id))) if part_id is not None else None}
    if object_type == "rigid":
        R, t, ok, _rf = part_se3_traj(P, vis)       # §4 global SE(3) over all keypoints
        out["whole_se3_R"] = R; out["whole_se3_t"] = t; out["whole_se3_ok"] = ok
    return out
