"""V3 Stage-⑦: coordinate unification + keypoint↔dense-track linking.

Consumes:
  dense_tracks_<iid>.npz   (Stage③: xyz_world in W_track, uv_norm, vis, seed_frame, c2w, ...)
  inst_<iid>.npz           (Stage1: mesh verts in scene frame W_mesh, object_pose_cam, sam3d_intrinsics, t*)
  pass1_keypoints.pkl      (Stage3⑥: keypoints = mesh vertices, per SEMANTIC part)
  video_input.pkl          (masklets for mask-consistency; Path2 GT is already baked into inst)
Produces:
  tracks_d4rt.npz          (SAME field contract as V2 -> Stage2b/pass2/finalize unchanged)

OUTPUT FRAME = W_track (2026-08-08 fix; the `track_frame` npz key records it).
`xyz_ref0` carries the dense front-end's OWN world coordinates verbatim — for Stereo4D that is the
METRIC GT world W_gt, for DAVIS/MOVi the tracker world. It used to carry sim(W_track) = W_mesh, but
W_mesh is NOT a scene frame: on Path 1 `scene_frame=="sam3d_layout"` means each instance's mesh sits
in SAM 3D's OWN per-object layout frame (and at its own t*), so every instance had a DIFFERENT
target frame. Measured on Stereo4D: per-instance scale spread 1.58x and cross-instance placement
error vs the pooled scene_T up to 0.29 scene-diagonals (7.4 m) — i.e. the "unified scene frame" the
static branch, the tokenizer's scene-level normalization and the 4D viz all assume never existed.
Emitting W_track fixes that at the root: every instance (dynamic AND static-routed) lands in the one
frame the dense front-end already guarantees is scene-consistent.

The per-instance similarity is still SOLVED (it is what pairs keypoints with tracks) — it is just no
longer applied to the output, and is persisted as `inst_sim` so consumers can move the MESH into
W_track when they need it (viz, registration). A useful consequence: the exported coordinates no
longer depend on the fit quality at all — a bad/failed T degrades WHICH track a keypoint gets, never
WHERE the trajectory is.

Method (per instance):
 1) COORD UNIFY: at t*, match dense tracks (W_track 3D + t* pixel) to mesh verts (W_mesh 3D + t*
    pixel via SAM3D round-trip) by nearest t* pixel -> robust Umeyama(scale+R+t) T: W_track→W_mesh.
    (Path1 target=W_mesh; Path2 target=W_gt — same code, object_to_scene already maps to the target.)
    All of the geometry below (SAM3D projection, mesh registry, 3D gates) runs in W_mesh; only the
    ASSEMBLED output arrays are taken from the raw W_track tracks.
 2) LINK visible keypoints: kp t* pixel -> nearest visible dense track within radius -> that track's
    (T-aligned) trajectory IS the kp trajectory.
 3) PROPAGATE occluded-at-t* keypoints: fit per-SEMANTIC-part SE(3)(t) from the linked kps, predict
    the occluded kp's position at each frame, link the nearest visible dense track over frames.
    (This is how V3 recovers parts occluded at t* — the reason V3 exists.)
"""
from __future__ import annotations
import argparse, glob, json, os, pickle, sys
from pathlib import Path
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import SCHEMA  # noqa: F401  (register classes for unpickle)
from src.stage3.schema import Keypoint  # noqa: F401
from src.geometry import umeyama_similarity, k_grid_wh, sam3d_project_norm
from src.registry import presence_segments, register_pointset_to_mesh


def _apply_sim(s, R, t, X):
    return (s * (X @ R.T)) + t


def _ransac_umeyama(src, dst, iters=200, thr=None, min_inl=8, rng_seed=0):
    """Robust similarity src->dst. thr auto = 5% of dst bbox diag."""
    n = len(src)
    if n < 3:
        return None
    if thr is None:
        thr = 0.05 * float(np.linalg.norm(dst.max(0) - dst.min(0)) + 1e-9)
    best_inl, best = None, None
    # deterministic pseudo-random triplets (no Math.random needs; index-varied)
    for it in range(iters):
        i = (np.array([it * 7 + 1, it * 13 + 3, it * 29 + 5]) % n)
        if len(set(i.tolist())) < 3:
            continue
        try:
            s, R, t = umeyama_similarity(src[i], dst[i])
        except Exception:
            continue
        err = np.linalg.norm(_apply_sim(s, R, t, src) - dst, axis=1)
        inl = err < thr
        if best_inl is None or inl.sum() > best_inl:
            best_inl, best = int(inl.sum()), inl
    if best is None or best_inl < min_inl:
        s, R, t = umeyama_similarity(src, dst)                 # fallback: plain fit
        return s, R, t, np.ones(n, bool)
    s, R, t = umeyama_similarity(src[best], dst[best])          # refit on inliers
    return s, R, t, best


# 4x-mismatch projection logic (2026-07-23) now lives in src/geometry.py so Stage-2b-early can
# reuse the EXACT same code (2026-07-30 restructure). Verbatim move; behavior unchanged.
_k_grid_wh = k_grid_wh
_project_norm = sam3d_project_norm


def _scale_align(src, dst):
    """PATH 2 scale-only alignment: dst ≈ s*src + t with R=I (the GT camera already pinned world
    R,t in Stage③, so only the VGGT-depth scale + a residual translation remain). Robust: s = median
    per-pair distance ratio about the (median) centroids."""
    cs, cd = np.median(src, 0), np.median(dst, 0)
    ds = np.linalg.norm(src - cs, axis=1); dd = np.linalg.norm(dst - cd, axis=1)
    ok = ds > 1e-6
    s = float(np.median(dd[ok] / ds[ok])) if ok.any() else 1.0
    return s, np.eye(3), cd - s * cs


def _kabsch(A, B):
    """rigid R,t: B ~= R@A + t.  A,B (n,3)."""
    cA, cB = A.mean(0), B.mean(0)
    Hm = (A - cA).T @ (B - cB)
    U, _, Vt = np.linalg.svd(Hm)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, cB - R @ cA


def _point_in_mask(uvn, masks, t, tol=4):
    """uvn (2,) normalized -> inside instance mask at frame t (dilated by tol px)?"""
    if masks is None or not (0 <= t < masks.shape[0]):
        return False
    Hh, Ww = masks.shape[1:]
    x = int(round(uvn[0] * Ww)); y = int(round(uvn[1] * Hh))
    x0, x1 = max(0, x - tol), min(Ww, x + tol + 1); y0, y1 = max(0, y - tol), min(Hh, y + tol + 1)
    if x0 >= x1 or y0 >= y1:
        return False
    return bool(masks[t, y0:y1, x0:x1].any())


def _solve_T(dense, inst, W, H, tstar, pix_r=0.03, path2=False, pair_frames=None):
    """Coordinate unification: match dense tracks (W_stv 3D + t* pixel) to mesh verts (W_mesh 3D +
    SAM3D-round-trip t* pixel) by nearest t* pixel -> robust Umeyama(scale+R+t) W_stv->W_mesh.
    Returns {s,R,t, src,dst (the 3D pairs), n (inliers), resid (median inlier residual)} — process_scene
    then pools a robust SCENE scale and refits degenerate instances (guards camel-inst8 scale blow-up).

    pair_frames (方案C 级2): pool pixel-pairs from these frames instead of [t*] only. The dense side
    uses its OWN frame-f pixel+3D; the mesh side stays the t* projection (short-time-static
    approximation) — callers MUST validate the fit with _reproj_consistency (t*) afterwards."""
    dw = dense["xyz_world"].astype(np.float64)
    duv = dense["uv_norm"].astype(np.float32)
    dvis = dense["vis"].astype(bool)
    verts = inst["verts"].astype(np.float64)
    vpx_arr, ok_v = _project_norm(verts, inst, W, H)
    vi = np.nonzero(ok_v)[0]
    if len(vi) > 5000:
        vi = vi[np.linspace(0, len(vi) - 1, 5000).astype(int)]
    src, dst = [], []
    for f in (pair_frames if pair_frames is not None else [tstar]):
        vt = dvis[:, f] & np.isfinite(dw[:, f]).all(1)
        tk_px = duv[:, f]
        ti = np.nonzero(vt)[0]
        if len(ti) < 3 or len(vi) == 0:
            continue
        vpx_sub = vpx_arr[vi]
        for k in ti:
            d2 = np.sum((vpx_sub - tk_px[k]) ** 2, axis=1)
            j = int(np.argmin(d2))
            if d2[j] < (pix_r ** 2):
                src.append(dw[k, f]); dst.append(verts[vi[j]])
    src = np.array(src) if src else np.zeros((0, 3))
    dst = np.array(dst) if dst else np.zeros((0, 3))
    if len(src) >= 3:
        if path2:                                          # GT camera pinned R,t -> only scale
            s, R, t = _scale_align(src, dst)
            resid = float(np.median(np.linalg.norm(_apply_sim(s, R, t, src) - dst, axis=1)))
            n = len(src)
        else:
            s, R, t, inl = _ransac_umeyama(src, dst)
            resid = (float(np.median(np.linalg.norm(_apply_sim(s, R, t, src[inl]) - dst[inl], axis=1)))
                     if inl.any() else 1e9)
            n = int(inl.sum())
    else:
        s, R, t, resid, n = 1.0, np.eye(3), np.zeros(3), 1e9, len(src)
    return dict(s=float(s), R=R, t=t, src=src, dst=dst, n=n, resid=resid)


# ---------------------------------------------------------------------------
# 方案 C (2026-07-29): ⑦ pairing-robustness fallback ladder.
# For instances whose DEFAULT t* pairing fails (T-unreliable) or links < 3 keypoints, try a
# graded ladder of alternative pairings — every level is quality-gated (fit guards + a t*
# reprojection-consistency check); if nothing passes, the instance honestly stays as-is.
#   lvl1: t* pairing with relaxed pixel radius (x2, x3) -> same RANSAC-Umeyama + scale guard;
#   lvl2: neighbor-frame pairing (|f-t*|<=2; dense pixel/3D at f, mesh projection at t* —
#         short-time-static approximation) -> MANDATORY t* reprojection guard;
#   lvl3: scene_T (if solvable) maps mesh verts into the T4W world -> direct 3D-NN pairing
#         (registry-style), inlier-fraction gate; plus raw scene_T as the very last candidate.
# Additive only: no new REQUIRED tracks_d4rt fields (audit sidecar `link_fallback` json).
# Escape hatch: --no-link-fallback.
# ---------------------------------------------------------------------------
C_REPROJ_THR = 0.04       # max median t*-reprojection error (normalized px) for a fallback fit
C_MIN_REPROJ_PTS = 5      # need at least this many t*-visible tracks to run the reproj check
C_L3_MIN_INL_FRAC = 0.4   # lvl3 RANSAC inlier-fraction gate


def _reproj_consistency(sim, dense, inst, W, H, tstar):
    """Validate a candidate sim: map dense@t* (visible) into W_mesh, project via the instance's own
    SAM3D t* camera, compare with the tracks' observed t* pixels. Returns (median_err, n) —
    err in normalized-pixel units; (inf, n) when too few points."""
    dw = dense["xyz_world"].astype(np.float64)
    duv = dense["uv_norm"].astype(np.float32)
    dvis = dense["vis"].astype(bool)
    vt = dvis[:, tstar] & np.isfinite(dw[:, tstar]).all(1)
    ti = np.nonzero(vt)[0]
    if len(ti) < C_MIN_REPROJ_PTS:
        return float("inf"), int(len(ti))
    s, R, t = sim
    xm = _apply_sim(s, R, t, dw[ti, tstar])
    prj, okz = _project_norm(xm, inst, W, H)
    if okz.sum() < C_MIN_REPROJ_PTS:
        return float("inf"), int(okz.sum())
    err = np.linalg.norm(prj[okz] - duv[ti][okz, tstar], axis=1)
    return float(np.median(err)), int(okz.sum())


def _fit_guard(Tsol, s_scene, bbox, min_pairs):
    """PER-INSTANCE quality gates for a candidate fit (pair count / residual). Returns None if OK
    else a short failure reason.

    ⚠️ 2026-08-10 试过删掉 `0.5 <= s/s_scene <= 2.0` 这一项(理由是 s_scene 是**跨实例**中位数,
    而每个实例的 mesh 是在自己的 t* 重建的、尺度本就该各管各的 —— Path 1 下逐实例尺度合法地
    横跨 6 倍),**实测两条 path 都变差**:Path2 0.7083→0.7031、Path1 0.7149→0.7142。**已回退。**
    原因在 `_cands`:raw fit 一旦通过本函数就 `return`,**不再产出 s_scene 版候选**;放松门槛
    让更多 raw fit 通过,反而把后面更好的候选挤掉了。所以这一项在这里的实际作用不是"过滤",
    而是"决定该产出哪一版候选" —— 不要因为它看起来是跨实例耦合就删。"""
    if Tsol["n"] < min_pairs:
        return f"pairs={Tsol['n']}<{min_pairs}"
    if s_scene is not None and not (0.5 <= Tsol["s"] / s_scene <= 2.0):
        return f"scale_out s={Tsol['s']:.3f} vs s_scene={s_scene:.3f}"
    if Tsol["resid"] > 0.15 * bbox:
        return f"resid={Tsol['resid']:.4f}>{0.15 * bbox:.4f}"
    return None


def _solve_T_scene_nn(dense, inst, tstar, scene_T, bbox):
    """方案C 级3: scene_T maps mesh verts into the T4W world; pair each t*-visible dense point with
    its 3D-NN mesh vertex there (radius = link_thr3d in T4W units); refit RANSAC-Umeyama on the
    pairs (mesh-frame targets). Returns (Tsol, inlier_frac) or (None, reason)."""
    sT = float(np.linalg.norm(scene_T[:3, 0]))          # scene_T[:3,:3] = s*R
    if not (1e-6 < sT < 1e6):
        return None, "bad_sceneT_scale"
    RT = scene_T[:3, :3] / sT
    tT = scene_T[:3, 3]
    verts = inst["verts"].astype(np.float64)
    vi = np.arange(len(verts))
    if len(vi) > 5000:
        vi = np.linspace(0, len(vi) - 1, 5000).astype(int)
    v_t4w = ((verts[vi] - tT) @ RT) / sT                 # inverse scene_T (row form: R^T x = x @ R)
    dw = dense["xyz_world"].astype(np.float64)
    dvis = dense["vis"].astype(bool)
    vt = dvis[:, tstar] & np.isfinite(dw[:, tstar]).all(1)
    ti = np.nonzero(vt)[0]
    if len(ti) < 3:
        return None, "few_tstar_tracks"
    r3d = (0.06 * bbox) / sT                             # link_thr3d expressed in T4W units
    src, dst = [], []
    for k in ti:
        d = np.linalg.norm(v_t4w - dw[k, tstar], axis=1)
        j = int(np.argmin(d))
        if d[j] < r3d:
            src.append(dw[k, tstar]); dst.append(verts[vi[j]])
    if len(src) < 3:
        return None, f"few_nn_pairs={len(src)}"
    src, dst = np.array(src), np.array(dst)
    fit = _ransac_umeyama(src, dst)
    if fit is None:
        return None, "ransac_fail"
    s, R, t, inl = fit
    frac = float(inl.sum()) / len(src)
    resid = (float(np.median(np.linalg.norm(_apply_sim(s, R, t, src[inl]) - dst[inl], axis=1)))
             if inl.any() else 1e9)
    return dict(s=float(s), R=R, t=t, src=src, dst=dst, n=int(inl.sum()), resid=resid), frac


def _fallback_candidates(dense, inst, W, H, tstar, s_scene, scene_T, bbox, path2, min_pairs,
                         rejects):
    """Yield (tag, sim, audit) for guard-PASSED fallback fits, cheapest level first. Every
    candidate has already passed: fit guards (_fit_guard) + t* reprojection consistency. Levels
    that fail append a reason to the CALLER-OWNED `rejects` list (audit)."""

    def _gate(tag, Tsol, extra=""):
        why = _fit_guard(Tsol, s_scene, bbox, min_pairs)
        if why is not None:
            rejects.append(f"{tag}: {why}")
            return None
        err, n = _reproj_consistency((Tsol["s"], Tsol["R"], Tsol["t"]), dense, inst, W, H, tstar)
        if not (err <= C_REPROJ_THR):
            rejects.append(f"{tag}: reproj={err:.4f}>{C_REPROJ_THR} (n={n})")
            return None
        return dict(tag=tag, n_pairs=int(Tsol["n"]), resid=round(float(Tsol["resid"]), 5),
                    s=round(float(Tsol["s"]), 4), reproj=round(err, 5), extra=extra)

    def _cands(tag, Tsol):
        """A pairing gives up to TWO gated fit candidates: the raw RANSAC-Umeyama fit, and — when
        that fails the guards and a scene scale exists — the SAME degenerate-fit remedy the
        default path uses (Kabsch with the scale pinned to s_scene), re-gated on resid+reproj."""
        a = _gate(tag, Tsol)
        if a:
            yield a["tag"], (Tsol["s"], Tsol["R"], Tsol["t"]), a
            return
        if path2 or s_scene is None or len(Tsol["src"]) < max(3, min_pairs):
            return
        R2, t2 = _kabsch(s_scene * Tsol["src"], Tsol["dst"])
        resid = float(np.median(np.linalg.norm(
            _apply_sim(s_scene, R2, t2, Tsol["src"]) - Tsol["dst"], axis=1)))
        Tsol2 = dict(s=float(s_scene), R=R2, t=t2, src=Tsol["src"], dst=Tsol["dst"],
                     n=len(Tsol["src"]), resid=resid)
        a2 = _gate(tag + "+refit", Tsol2)
        if a2:
            yield a2["tag"], (Tsol2["s"], Tsol2["R"], Tsol2["t"]), a2

    # ---- lvl1: relaxed t* pixel radius ----
    for pr in (0.06, 0.09):
        Tsol = _solve_T(dense, inst, W, H, tstar, pix_r=pr, path2=path2)
        yield from _cands(f"C-lvl1(pix_r={pr})", Tsol)
    # ---- lvl2: neighbor-frame pairing (short-time-static approximation) ----
    T = dense["vis"].shape[1]
    nbrs = [f for f in range(max(0, tstar - 2), min(T, tstar + 3))]
    for pr in (0.03, 0.06):
        Tsol = _solve_T(dense, inst, W, H, tstar, pix_r=pr, path2=path2, pair_frames=nbrs)
        yield from _cands(f"C-lvl2(|f-t*|<=2,pix_r={pr})", Tsol)
    # ---- lvl3: scene_T-guided 3D-NN direct pairing (+ raw scene_T last) ----
    if scene_T is not None:
        Tsol, frac = _solve_T_scene_nn(dense, inst, tstar, scene_T, bbox)
        if Tsol is None:
            rejects.append(f"C-lvl3(nn): {frac}")
        elif frac < C_L3_MIN_INL_FRAC:
            rejects.append(f"C-lvl3(nn): inl_frac={frac:.2f}<{C_L3_MIN_INL_FRAC}")
        else:
            a = _gate("C-lvl3(nn)", Tsol, extra=f"inl_frac={frac:.2f}")
            if a:
                yield a["tag"], (Tsol["s"], Tsol["R"], Tsol["t"]), a
        sT = float(np.linalg.norm(scene_T[:3, 0]))
        if 1e-6 < sT < 1e6:
            simT = (sT, scene_T[:3, :3] / sT, scene_T[:3, 3])
            err, n = _reproj_consistency(simT, dense, inst, W, H, tstar)
            if err <= C_REPROJ_THR:
                yield ("C-lvl3(scene_T)", simT, dict(
                    tag="C-lvl3(scene_T)", n_pairs=-1, resid=-1.0, s=round(sT, 4),
                    reproj=round(err, 5), extra=""))
            else:
                rejects.append(f"C-lvl3(scene_T): reproj={err:.4f}>{C_REPROJ_THR} (n={n})")


def _link_segments(dense, dm, verts, kp_pos, link, arrs, masks, segs, tstar, sim, W, H,
                   link_thr3d, pix_r=0.05, min_pts=15):
    """MESH-REGISTRY re-capture for NON-t* presence segments (V3 multi-presence, 2026-07-11).

    Cross-segment tracker identity is broken by construction (tracks seeded in a later presence
    segment never existed at t*, so the t*-pixel pairing above cannot hook them). Per segment not
    containing t*:
      anchor = the segment frame with the most in-mask visible dense points; register the anchor's
      visible cloud (already in W_mesh via `sim` -> scale known, rigid-only) onto the mesh ->
      re-capture pose (mesh ≈ R@p(anchor) + t); hook keypoints to that segment's tracks by
      mesh-3D NN at the anchor (+ pixel gate when the camera roundtrip validates); merge the hooked
      track's frames into the kp arrays. STRICTLY ADDITIVE per frame: a segment-track frame is
      taken only when it is visible AND in-mask (authoritative, mesh-verified); otherwise the
      primary(t*)-link value is kept — single-segment instances and the t* segment are untouched.

    Returns per-segment meta (aligned 1:1 with `segs`) for the tracks_d4rt.npz `segments` sidecar.
    """
    dw = dense["xyz_world"].astype(np.float64)
    duv = dense["uv_norm"].astype(np.float32)
    dvis = dense["vis"].astype(bool)
    dconf = dense["conf"].astype(np.float32)
    dcam = dense["xyz_cam"].astype(np.float32)
    c2w = dense["c2w"].astype(np.float64)
    intrs = dense["intrs"].astype(np.float64)
    finite = np.isfinite(dw).all(2)
    s_g, R_g, t_g = sim
    M = len(kp_pos)

    def _inmask_vec(uvn, f):
        if masks is None or not (0 <= f < masks.shape[0]):
            return np.zeros(len(uvn), bool)
        Hh, Ww = masks.shape[1:]
        x = np.clip(np.round(uvn[:, 0] * Ww).astype(int), 0, Ww - 1)
        y = np.clip(np.round(uvn[:, 1] * Hh).astype(int), 0, Hh - 1)
        return masks[f, y, x]

    def _proj_t4w(x_w, f):
        """W_mesh points -> normalized pixels of frame f (inverse-sim -> tracker camera)."""
        x = ((x_w - t_g) @ R_g) / s_g                        # W_mesh -> W_t4w
        w2c = np.linalg.inv(c2w[f])
        cam = x @ w2c[:3, :3].T + w2c[:3, 3]
        z = np.where(np.abs(cam[:, 2]) > 1e-9, cam[:, 2], 1.0)
        K = intrs[f]
        uvn = np.stack([(K[0, 0] * cam[:, 0] / z + K[0, 2]) / W,
                        (K[1, 1] * cam[:, 1] / z + K[1, 2]) / H], 1)
        return uvn.astype(np.float32), cam[:, 2] > 1e-9

    tseg = next((k for k, (a, b) in enumerate(segs) if a <= tstar <= b), None)
    meta = []
    pix_gate = None                                          # camera-roundtrip validity (lazy)
    seg_linked = set()
    for k, (a0, a1) in enumerate(segs):
        if k == tseg:
            # the t* segment IS the mesh's own frame: registration = identity at t* by construction
            meta.append(dict(s=a0, e=a1, anchor=int(tstar), tstar_segment=True, registered=True,
                             inlier_frac=1.0, n_pts=-1, n_kp=int((link >= 0).sum()),
                             reg_R=np.eye(3).tolist(), reg_t=[0.0, 0.0, 0.0]))
            continue
        # ---- segment anchor = in-mask-densest frame ----
        best_f, best_sel = -1, None
        for f in range(a0, a1 + 1):
            selv = dvis[:, f] & finite[:, f]
            if not selv.any():
                continue
            sub = np.nonzero(selv)[0]
            selv[sub] = _inmask_vec(duv[sub, f], f)
            if best_sel is None or selv.sum() > best_sel.sum():
                best_f, best_sel = f, selv
        n_pts = int(best_sel.sum()) if best_sel is not None else 0
        rec = dict(s=a0, e=a1, anchor=int(best_f), tstar_segment=False, n_pts=n_pts,
                   registered=False, inlier_frac=0.0, n_kp=0)
        if n_pts < min_pts:
            rec["fail"] = "few_points"
            meta.append(rec)
            continue
        # ---- rigid re-capture onto the mesh (scale already in `sim`) ----
        R_k, t_k, frac, reg_ok, reg_info = register_pointset_to_mesh(dm[best_sel, best_f], verts)
        rec["inlier_frac"] = round(frac, 4)
        rec["reg_basin"] = reg_info["basin"]
        rec["reg_score"] = reg_info["score"]
        rec["reg_ambiguous"] = reg_info["ambiguous"]
        rec["reg_med_d_frac"] = reg_info["med_d_frac"]
        if not reg_ok:
            rec["fail"] = "low_inliers"
            meta.append(rec)
            continue
        # HONESTY GATE: an ambiguous NON-inplace pose is a guess (several rotation-distinct poses
        # fit the mesh equally well); hooking keypoints to it would fabricate plausible-but-wrong
        # trajectories. Accept only the static-world (inplace) basin or an unambiguous winner.
        if reg_info["ambiguous"] and reg_info["basin"] != "inplace":
            rec["fail"] = "ambiguous_pose"
            meta.append(rec)
            continue
        rec["registered"] = True
        rec["reg_R"] = np.asarray(R_k).tolist()
        rec["reg_t"] = np.asarray(t_k).tolist()
        if pix_gate is None:                                 # validate camera roundtrip ONCE
            sub = np.nonzero(best_sel)[0][:500]
            prj, _ = _proj_t4w(dm[sub, best_f], best_f)
            err = np.linalg.norm(prj - duv[sub, best_f], axis=1)
            pix_gate = bool(np.median(err) < 0.02)
            if not pix_gate:
                print(f"    [seg] camera roundtrip err med={np.median(err):.3f} -> pixel gate OFF",
                      flush=True)
        # ---- hook keypoints: mesh->anchor positions, 3D NN + (validated) pixel gate ----
        kpa = (kp_pos - t_k) @ R_k                           # R_k^T (kp - t_k): kp at anchor frame
        cand = np.nonzero(best_sel)[0]
        if pix_gate:
            kp_px, kp_front = _proj_t4w(kpa, best_f)
        hooks = []                                           # (m, tj)
        for m in range(M):
            d = np.linalg.norm(dm[cand, best_f] - kpa[m], axis=1)
            j = int(np.argmin(d))
            if d[j] >= link_thr3d:
                continue
            if pix_gate and (not kp_front[m]
                             or np.sum((duv[cand[j], best_f] - kp_px[m]) ** 2) >= pix_r ** 2):
                continue
            hooks.append((m, int(cand[j])))
        # ---- SURVIVING-EVIDENCE VERIFICATION: where a keypoint's PRIMARY (t*-linked) track
        # genuinely survives into this segment (visible AND in-mask), the re-capture pose must
        # AGREE with it (same physical point). A registration can be geometrically perfect yet
        # wrong (ambiguity); contradiction with surviving direct evidence is the one independent
        # check we have — measured to fire exactly on the polluted char5f instances while ego
        # scenes (no cross-gap survivors, vis~0) are untouched. On conflict: reject the whole
        # segment (keep baseline behavior) rather than inject plausible-but-wrong trajectories.
        dis = []
        for m, tj in hooks:
            pj = link[m]
            if pj < 0 or pj == tj:
                continue
            for f in range(a0, a1 + 1):
                if (dvis[tj, f] and dvis[pj, f] and finite[tj, f] and finite[pj, f]
                        and arrs["vis"][m, f]):              # primary was mask-verified here
                    dis.append(float(np.linalg.norm(dm[tj, f] - dm[pj, f])))
        rec["n_overlap"] = len(dis)
        if len(dis) >= 10:
            med = float(np.median(dis))
            rec["overlap_med_frac"] = round(med / (link_thr3d / 0.06), 4)   # /bbox_diag
            if med > link_thr3d:
                rec["fail"] = "contradicts_surviving_tracks"
                meta.append(rec)
                continue
        n_kp = 0
        for m, tj in hooks:
            took = False
            for f in range(a0, a1 + 1):                      # merge: seg track wins where valid
                if not (dvis[tj, f] and finite[tj, f]):
                    continue
                if not _point_in_mask(duv[tj, f], masks, f):
                    continue
                arrs["xyz"][m, f] = dw[tj, f]        # OUTPUT = raw W_track (dm is W_mesh, geometry only)
                arrs["xyzl"][m, f] = dcam[tj, f]
                arrs["uv"][m, f] = duv[tj, f]
                arrs["vraw"][m, f] = True
                arrs["conf"][m, f] = dconf[tj, f]
                arrs["vis"][m, f] = True
                took = True
            if took:
                n_kp += 1
                if link[m] < 0 and m not in seg_linked:      # kp had NO t* link: seed its query here
                    arrs["qt"][m] = best_f
                    arrs["quv"][m] = duv[tj, best_f]
                seg_linked.add(m)
        rec["n_kp"] = n_kp
        meta.append(rec)
    return meta, seg_linked


def link_instance(dense, inst, kps, masks, tstar, W, H, sim, pix_r=0.03, prop_thr_frac=0.06,
                  segs=None, mot_of=None):
    """Return per-keypoint dict arrays (M,T,*) linked from dense tracks. `sim`=(s,R,t) W_stv->W_mesh
    (from _solve_T + process_scene's scene-scale guard). kps: list[Keypoint].
    `mot_of` (M,) optional per-keypoint MOTION-part label (Stage-2b-early sem->motion map,
    2026-07-30 restructure): with motion-part pass1 sampling a semantic part may hold <3 keypoints,
    so the occluded-kp SE(3) propagation gains a middle peer tier — semantic part (>=3 peers, as
    before) -> motion part (the rigidly-co-moving unit, exactly what a per-part SE(3) assumes)
    -> whole instance. mot_of=None (no 2b-early product) keeps the old behavior bit-for-bit."""
    dw = dense["xyz_world"].astype(np.float64)          # (N,T,3) W_stv
    dcam = dense["xyz_cam"].astype(np.float32)          # (N,T,3)
    duv = dense["uv_norm"].astype(np.float32)           # (N,T,2) normalized
    dvis = dense["vis"].astype(bool)                    # (N,T)
    dconf = dense["conf"].astype(np.float32)
    N, T = dvis.shape

    s, R, t = sim
    vt = dvis[:, tstar] & np.isfinite(dw[:, tstar]).all(1)
    tk_px = duv[:, tstar]                                # track t* pixel (normalized)
    ti = np.nonzero(vt)[0]
    verts = inst["verts"].astype(np.float64)
    dm = _apply_sim(s, R, t, dw.reshape(-1, 3)).reshape(N, T, 3)   # dense tracks in W_mesh

    # --- (2) link visible keypoints by t* pixel + 3D-consistency gate ---
    # A t*-OCCLUDED (back-facing) keypoint projects to the SAME t* pixel as the front surface, so
    # pixel-NN alone would wrongly bind it to a FRONT track. Require the keypoint's 3D pos to also be
    # near the matched track's 3D at t* (both in W_mesh); else leave it for SE(3) propagation (3).
    bbox_diag = float(np.linalg.norm(verts.max(0) - verts.min(0)) + 1e-9)
    link_thr3d = prop_thr_frac * bbox_diag
    M = len(kps)
    link = np.full(M, -1, np.int64)                      # keypoint -> dense track index
    kp_pos = np.array([kp.position for kp in kps], np.float64)
    kp_px, kp_ok = _project_norm(kp_pos, inst, W, H)     # (M,2), (M,)
    for m in range(M):
        if not kp_ok[m] or len(ti) == 0:
            continue
        d2 = np.sum((tk_px[ti] - kp_px[m]) ** 2, axis=1)
        j = int(np.argmin(d2))
        if d2[j] < (pix_r ** 2) and np.linalg.norm(kp_pos[m] - dm[ti[j], tstar]) < link_thr3d:
            link[m] = ti[j]

    n_direct = int((link >= 0).sum())                    # linked directly (visible at t*)
    # --- (3) propagate occluded keypoints via per-part SE(3) ---
    part_of = np.array([(-1 if kp.part_id is None else kp.part_id) for kp in kps])
    ref_pos = kp_pos                                                 # (M,3) W_mesh (t* ref)
    prop_thr = link_thr3d
    occ = np.nonzero(link < 0)[0]
    for m in occ:
        pid = part_of[m]
        # linked kps of same part (or whole instance if part too sparse) to fit SE(3)(t)
        peers = [i for i in range(M) if link[i] >= 0 and part_of[i] == pid]
        if len(peers) < 3 and mot_of is not None:      # middle tier: same MOTION part (2b-early)
            peers = [i for i in range(M) if link[i] >= 0 and mot_of[i] == mot_of[m]]
        if len(peers) < 3:
            peers = [i for i in range(M) if link[i] >= 0]
        if len(peers) < 3:
            continue
        A = ref_pos[peers]                              # ref positions (static)
        Bt = dm[link[peers]]                            # (P,T,3) trajectories
        vpeer = dvis[link[peers]]                       # (P,T)
        best = (prop_thr, -1, -1)                       # (dist, frame, track)
        for tt in range(T):
            sel = vpeer[:, tt]
            if sel.sum() < 3:
                continue
            R_t, tvec = _kabsch(A[sel], Bt[sel, tt])
            pred = R_t @ ref_pos[m] + tvec             # predicted occluded-kp pos at frame tt
            cand = np.nonzero(dvis[:, tt])[0]
            if len(cand) == 0:
                continue
            d = np.linalg.norm(dm[cand, tt] - pred, axis=1)
            j = int(np.argmin(d))
            if d[j] < best[0]:
                best = (float(d[j]), tt, int(cand[j]))
        if best[2] >= 0:
            link[m] = best[2]

    # --- assemble per-keypoint arrays (M,T,*) ; mask-consistency visibility ---
    # xyz is emitted in W_TRACK (the dense front-end's own world), NOT in W_mesh: `dm` exists only to
    # do the geometry above (3D gates / SE(3) propagation / mesh registry) in the mesh's frame. Since
    # dm = sim(dw) exactly, taking dw here is precisely sim⁻¹(dm) — bit-exact, no extra arithmetic.
    xyz = np.zeros((M, T, 3), np.float32); xyzl = np.zeros((M, T, 3), np.float32)
    uv = np.zeros((M, T, 2), np.float32); vis = np.zeros((M, T), bool)
    vraw = np.zeros((M, T), bool); conf = np.zeros((M, T), np.float32)
    qt = np.zeros(M, np.int64); quv = np.zeros((M, 2), np.float32)
    for m in range(M):
        j = link[m]
        if j < 0:
            qt[m] = tstar
            continue
        xyz[m] = dw[j].astype(np.float32); xyzl[m] = dcam[j]; uv[m] = duv[j]
        vraw[m] = dvis[j]; conf[m] = dconf[j]
        qt[m] = int(dense["seed_frame"][j]); quv[m] = duv[j, int(dense["seed_frame"][j])]
        vis[m] = np.array([dvis[j, tt] and _point_in_mask(duv[j, tt], masks, tt) for tt in range(T)])

    # --- (4) MESH-REGISTRY multi-presence re-capture (V3, 2026-07-11): hook the NON-t* presence
    # segments' tracks (identity-broken vs t*) to the same keypoints via the mesh. The t* segment
    # and single-segment instances go through the path above COMPLETELY UNCHANGED. ---
    seg_meta, seg_linked = [], set()
    if segs:
        arrs = dict(xyz=xyz, xyzl=xyzl, uv=uv, vis=vis, vraw=vraw, conf=conf, qt=qt, quv=quv)
        seg_meta, seg_linked = _link_segments(dense, dm, verts, kp_pos, link, arrs, masks, segs,
                                              tstar, (s, R, t), W, H, link_thr3d)
    return dict(xyz=xyz, xyzl=xyzl, uv=uv, vis=vis, vraw=vraw, conf=conf, qt=qt, quv=quv,
                link=link, sim=(s, R, t), M=M, T=T, n_direct=n_direct,
                seg_meta=seg_meta, n_seg_kp=len(seg_linked))


def process_scene(scene_dir, link_fallback=True):
    vip = os.path.join(scene_dir, "video_input.pkl")
    kpf = os.path.join(scene_dir, "pass1_keypoints.pkl")
    if not (os.path.exists(vip) and os.path.exists(kpf)):
        print(f"  [skip] {scene_dir}: missing video_input/pass1_keypoints", flush=True); return
    vid = pickle.load(open(vip, "rb"))
    T, H, W = vid.frames.shape[:3]
    masks_by = {ml.instance_id: np.asarray(ml.masks).astype(bool) for ml in vid.masklets}
    kps_all = pickle.load(open(kpf, "rb"))
    kps_by = {}
    for kp in kps_all:
        kps_by.setdefault(kp.instance_id, []).append(kp)
    # Stage-2b-early sem->motion part map (2026-07-30 restructure; ADDITIVE — absent = old behavior)
    early_map = {}
    ep = os.path.join(scene_dir, "stage2b_early.json")
    if os.path.exists(ep):
        try:
            early_map = {int(k): {int(s): int(v) for s, v in rec.get("map", {}).items()}
                         for k, rec in json.load(open(ep)).get("instances", {}).items()}
            print(f"  [2b-early] motion-part map loaded for {len(early_map)} instance(s) "
                  f"(propagation peer tier: sem -> motion -> instance)", flush=True)
        except Exception as e:
            print(f"  [2b-early] failed to read {ep}: {e} — ignoring", flush=True)
    # PATH 2 scale-only (R=I) is OPT-IN (V3_PATH2_GTCAM=1) and paired with Stage③ GT-camera feeding.
    # Currently DISABLED: STV2's world after GT-extrinsic feeding has a ~150deg convention rotation vs
    # W_gt, so R=I is invalid -> use the general 7-DOF Umeyama (default), which absorbs it. See Stage③.
    path2 = (getattr(vid, "camera_extrinsics", None) is not None) and os.environ.get("V3_PATH2_GTCAM") == "1"
    if path2:
        print("  [Path2] GT-camera tracks -> scale-only alignment (R=I)", flush=True)

    # ---- PASS 1: solve raw T per instance (collect for scene-scale guard) ----
    MIN_PAIRS = 8
    inst_data = {}   # iid -> (dense, inst, tstar, kps, Tsol)
    for dpath in sorted(glob.glob(os.path.join(scene_dir, "dense_tracks_*.npz"))):
        iid = int(os.path.basename(dpath).split("_")[-1].split(".")[0])
        ipath = os.path.join(scene_dir, f"inst_{iid}.npz")
        if iid not in kps_by or not os.path.exists(ipath):
            continue
        dense = np.load(dpath); inst = {k: np.load(ipath)[k] for k in np.load(ipath).files}
        tstar = int(inst["reference_frame_index"])
        Tsol = _solve_T(dense, inst, W, H, tstar, path2=path2)
        inst_data[iid] = (dense, inst, tstar, kps_by[iid], Tsol)
    # robust SCENE scale from well-solved instances (W_stv->W_mesh scale is ~scene-consistent)
    good_s = [d[4]["s"] for d in inst_data.values() if d[4]["n"] >= MIN_PAIRS and 1e-2 < d[4]["s"] < 1e3]
    s_scene = float(np.median(good_s)) if good_s else None

    # ---- SCENE-LEVEL transform scene_T: W_t4w -> the unified scene frame (V3 static routing,
    # 2026-07). Static instances have dense tracks but NO mesh/keypoints, so finalize needs one
    # scene-wide (s,R,t) to place their representative points in the SAME frame as the dynamic
    # instances. Fit: pool the t* correspondence pairs (dense-track W_t4w <-> mesh W_scene) of the
    # WELL-SOLVED instances (same criteria as the scene-scale guard) and refit one robust
    # similarity on the pooled pairs. Written as an ADDITIVE npz key (4x4, [s*R | t]) — existing
    # consumers (2b/pass2/finalize dynamic path) load by key and are untouched.
    # (Computed BEFORE the linking pass since 方案C lvl3 uses it; pooled from the DEFAULT PASS-1
    # fits only — fallback fits never feed it.) ----
    pooled_src, pooled_dst = [], []
    for iid, (dense, inst, tstar, kps, Tsol) in inst_data.items():
        if Tsol["n"] >= MIN_PAIRS and 1e-2 < Tsol["s"] < 1e3 and len(Tsol["src"]):
            if s_scene is not None and not (0.5 <= Tsol["s"] / s_scene <= 2.0):
                continue
            pooled_src.append(Tsol["src"]); pooled_dst.append(Tsol["dst"])
    scene_T = None
    if pooled_src:
        ps, pd = np.concatenate(pooled_src), np.concatenate(pooled_dst)
        if len(ps) > 20000:
            sel = np.linspace(0, len(ps) - 1, 20000).astype(int)
            ps, pd = ps[sel], pd[sel]
        if len(ps) >= MIN_PAIRS:
            if path2:
                sT, RT, tT = _scale_align(ps, pd)
            else:
                fit = _ransac_umeyama(ps, pd)
                if fit is not None:
                    sT, RT, tT = fit[0], fit[1], fit[2]
                else:
                    sT = None
            if sT is not None:
                scene_T = np.eye(4)
                scene_T[:3, :3] = sT * RT
                scene_T[:3, 3] = tT
                print(f"  scene_T: s={sT:.3f} (pooled {len(ps)} pairs from "
                      f"{len(pooled_src)} instances)", flush=True)
    if scene_T is None:
        print("  scene_T: UNAVAILABLE (no well-solved instance pairs) — finalize will fall back "
              "to identity (frame=t4w_raw)", flush=True)

    # ---- PASS 2: guard/refit T, then link ----
    XYZ, XYZL, UV, VIS, VRAW, CONF, QT, QUV, KID, IID, PID = ([] for _ in range(11))
    n_link = n_kp = 0
    segments_all = {}                     # iid -> presence segments + per-segment re-capture meta
    fallback_all = {}                     # iid -> 方案C audit (additive sidecar)
    # iid -> 链接统计。**匹配率(n_linked/n_kp)是换 stage1 重建路线时的头号指标**,以前只
    # 打在 stdout 里,重跑一次日志没留就再也算不回来了(2026-08-09 加,ADDITIVE)。
    link_stats = {}
    SIM_M, SIM_IID = [], []               # per-instance W_track->W_mesh similarity (4x4 [s*R|t])
    for iid, (dense, inst, tstar, kps, Tsol) in inst_data.items():
        s, R, t = Tsol["s"], Tsol["R"], Tsol["t"]
        bbox = float(np.linalg.norm(inst["verts"].max(0) - inst["verts"].min(0)) + 1e-9)
        # bad T: too few inliers, OR scale is an outlier vs the scene, OR high fit residual
        bad = (Tsol["n"] < MIN_PAIRS
               or (s_scene is not None and not (0.5 <= s / s_scene <= 2.0))
               or Tsol["resid"] > 0.15 * bbox)
        note = ""
        refit = None
        if bad and s_scene is not None and len(Tsol["src"]) >= 3:
            # refit with the SCENE scale fixed. Path2: R=I (scale-only), t from centroids; Path1:
            # Kabsch(s_scene*src -> dst). Fixes the sparse-correspondence scale blow-up.
            if path2:
                Rr = np.eye(3); tr = np.median(Tsol["dst"], 0) - s_scene * np.median(Tsol["src"], 0)
            else:
                Rr, tr = _kabsch(s_scene * Tsol["src"], Tsol["dst"])
            refit = (s_scene, Rr, tr)
        elif bad:
            note = "[T-unreliable: few/no pairs -> kept raw]"
        masks = masks_by.get(iid)
        segs = presence_segments(masks.any(axis=(1, 2)), max_hole=2) if masks is not None else []
        # per-keypoint MOTION-part labels (2b-early map over the kp's SEMANTIC part_id); None
        # (no 2b-early product) keeps the propagation peer logic bit-identical to before.
        emap = early_map.get(int(iid))
        mot_of = (np.array([emap.get(int(kp.part_id), 0) if kp.part_id is not None else -1
                            for kp in kps]) if emap else None)
        r = link_instance(dense, inst, kps, masks, tstar, W, H, sim=(s, R, t), segs=segs,
                          mot_of=mot_of)
        n_total = int((r["link"] >= 0).sum())

        # ---- HONEST GATE on the scene-scale refit (2026-08-10) ----
        # The refit above used to be applied UNCONDITIONALLY whenever `bad` fired, with no check
        # that it helped — unlike 方案C's ladder below, which requires `n_fb > n_total`. Measured
        # cost of that asymmetry (official SAM3D, 127 instances, Path 2 vs the SAME meshes placed
        # Path-1 style): the guard tripped on a DIFFERENT 6 instances under Path 2 and the forced
        # refit destroyed them — 769 of the 1089-link gap (71%), worst case 212 -> 1 links. The
        # guard's `s/s_scene` test is a CROSS-instance heuristic, so which instances it flags moves
        # with the mesh frame; the refit itself pins s to a pooled median that need not fit this
        # object at all. So: try it, keep it only if it actually links more.
        if refit is not None:
            r_rf = link_instance(dense, inst, kps, masks, tstar, W, H, sim=refit, segs=segs,
                                 mot_of=mot_of)
            n_rf = int((r_rf["link"] >= 0).sum())
            if n_rf > n_total:
                r, n_total = r_rf, n_rf
                s, R, t = refit
                note = f"[refit->s_scene={s_scene:.2f} linked {n_rf}]"
            else:
                note = f"[refit rejected: {n_rf}<={n_total} -> kept raw fit]"

        # ---- 方案C: pairing-robustness fallback ladder (only when the default path failed) ----
        # Trigger widened 2026-08-10 from `n_total<3 or T-unreliable` to `n_total<3 or bad`:
        # the honest gate above can now KEEP a raw fit that the old code would have overwritten
        # with the scene-scale refit, and if that refit had landed under 3 links the ladder used
        # to fire and rescue the instance. Keeping the (better) raw fit was silently taking that
        # rescue away — measured as −1.1% on the Path-1 arm. `bad` is exactly "the default solve
        # failed a quality gate", which is what the ladder is for; every candidate is still gated
        # by `n_fb > n_total`, so a wider trigger can only add links.
        if link_fallback and (n_total < 3 or bad):
            rejects = []
            audit = dict(trigger=f"linked={n_total}" + (" bad-fit" if bad else ""),
                         tried=[], accepted=None)
            # ⚠️ 2026-08-10 试过改成"全评比最优"(不 break,评完所有候选取链接最多的),理由是
            # 现在的"第一个能改善的就采纳"让候选**顺序**决定结果。**实测不值得**:
            # Path1 0.7149→0.7164(+0.2%)、Path2 0.7083→0.7083(0),而 ⑦ 从 65s 涨到 151s
            # (2.3x)。已回退,保持 first-improving。
            for tag, sim_fb, a in _fallback_candidates(
                    dense, inst, W, H, tstar, s_scene, scene_T, bbox, path2, MIN_PAIRS, rejects):
                r_fb = link_instance(dense, inst, kps, masks, tstar, W, H, sim=sim_fb, segs=segs,
                                     mot_of=mot_of)
                n_fb = int((r_fb["link"] >= 0).sum())
                a["linked"] = n_fb
                audit["tried"].append(a)
                if n_fb >= 3 and n_fb > n_total:            # honest gate: must actually improve
                    r, n_total = r_fb, n_fb
                    s, R, t = sim_fb
                    note += (f" [{tag} n_pairs={a['n_pairs']} reproj={a['reproj']:.4f} "
                             f"linked->{n_fb}]")
                    audit["accepted"] = a
                    break
            audit["rejected"] = rejects
            fallback_all[str(int(iid))] = audit
            if audit["accepted"] is None:
                note += " [C-none: honest keep]"

        n_dir = r["n_direct"]; n_prop = n_total - n_dir
        n_link += n_total; n_kp += r["M"]
        link_stats[str(int(iid))] = dict(n_kp=int(r["M"]), n_linked=int(n_total),
                                         n_direct=int(n_dir), n_propagated=int(n_prop),
                                         s=float(s), tstar=int(tstar),
                                         # 尺度守卫的结局:没触发 / 接受 refit / 拒绝 refit
                                         refit=(None if refit is None else
                                                ("applied" if "refit->s_scene" in note
                                                 else "rejected")))
        segments_all[str(int(iid))] = {"tstar": int(tstar),
                                       "segments": [[int(a), int(b)] for a, b in segs],
                                       "per_seg": r["seg_meta"]}
        if len(segs) > 1:
            reg = [sm for sm in r["seg_meta"] if not sm["tstar_segment"]]
            n_ok = sum(1 for sm in reg if sm["registered"])
            note += (f" [segs={len(segs)} reg_ok={n_ok}/{len(reg)} +segkp={r['n_seg_kp']}]")
        # persist the ACCEPTED per-instance similarity (W_track -> W_mesh). The output tracks are in
        # W_track, so this is what a consumer needs to move the MESH the other way (viz / registration).
        M4 = np.eye(4); M4[:3, :3] = s * R; M4[:3, 3] = t
        SIM_M.append(M4); SIM_IID.append(int(iid))
        XYZ.append(r["xyz"]); XYZL.append(r["xyzl"]); UV.append(r["uv"]); VIS.append(r["vis"])
        VRAW.append(r["vraw"]); CONF.append(r["conf"]); QT.append(r["qt"]); QUV.append(r["quv"])
        KID.append(np.array([kp.keypoint_id for kp in kps], np.int64))
        IID.append(np.full(len(kps), iid, np.int64))
        PID.append(np.array([(-1 if kp.part_id is None else kp.part_id) for kp in kps], np.int64))
        print(f"  inst {iid}: {r['M']} kps -> {n_total} linked ({n_dir} direct + {n_prop} propagated), "
              f"{r['M']-n_total} unlinked | s={s:.3f} vis={r['vis'].mean():.2f} {note}", flush=True)

    if not XYZ:
        print(f"  [skip] {scene_dir}: nothing linked", flush=True); return

    out = os.path.join(scene_dir, "tracks_d4rt.npz")
    extra = {} if scene_T is None else {"scene_T": scene_T.astype(np.float64)}
    # CONTRACT TAG (2026-08-08): which frame xyz_ref0 is in. Absent == the pre-fix products
    # (W_mesh) — finalize/viz branch on this so old scene dirs keep loading correctly.
    extra["track_frame"] = np.str_("W_track")
    if SIM_M:
        extra["inst_sim"] = np.stack(SIM_M).astype(np.float64)      # (n,4,4) W_track -> W_mesh
        extra["inst_sim_iid"] = np.asarray(SIM_IID, np.int64)       # (n,)
    np.savez_compressed(
        out,
        **extra,
        xyz_ref0=np.concatenate(XYZ), xyz_local=np.concatenate(XYZL),
        visibility=np.concatenate(VIS), visibility_d4rt=np.concatenate(VRAW),
        confidence=np.concatenate(CONF),
        query_t=np.concatenate(QT), query_uv_norm=np.concatenate(QUV),
        uv_track=np.concatenate(UV),
        keypoint_id=np.concatenate(KID), instance_id=np.concatenate(IID),
        part_id=np.concatenate(PID),
        # ADDITIVE sidecar (json str; consumers load by key, so this cannot break them):
        # per-instance presence segments + per-segment mesh re-capture poses (V3 multi-presence).
        segments=json.dumps(segments_all),
        # ADDITIVE 方案C audit sidecar: per-instance fallback-ladder trigger/attempts/acceptance.
        link_fallback=json.dumps(fallback_all),
        # ADDITIVE: per-instance link stats (匹配率的唯一可复现来源,见 link_stats 定义处)
        link_stats=json.dumps(link_stats),
    )
    print(f"  -> {os.path.basename(out)}: {n_kp} kps, {n_link} linked-at-t* "
          f"(+propagated) | frame=W_track, inst_sim x{len(SIM_M)} | contract fields OK", flush=True)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--scene-dir"); g.add_argument("--batch-root")
    ap.add_argument("--no-link-fallback", action="store_true",
                    help="escape hatch: disable the 方案C pairing-robustness fallback ladder "
                         "(lvl1 relaxed-pix_r / lvl2 neighbor-frame / lvl3 scene_T 3D-NN) and "
                         "keep the strict default-t*-pairing-only behavior")
    args = ap.parse_args()
    scenes = ([args.scene_dir] if args.scene_dir
              else sorted(glob.glob(os.path.join(args.batch_root, "scene_*"))))
    print(f"[link-dense] {len(scenes)} scene(s) "
          f"(fallback={'OFF' if args.no_link_fallback else 'ON'})", flush=True)
    for sd in scenes:
        print(f"[link-dense] {os.path.basename(sd)}", flush=True)
        process_scene(sd, link_fallback=not args.no_link_fallback)
    print("[link-dense] done", flush=True)


if __name__ == "__main__":
    main()
