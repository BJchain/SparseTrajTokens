"""Shared geometry / convention utilities (split out of the former monolithic SCHEMA.py).
Umeyama alignment, homogeneous transforms, PCA/OBB axes, quaternion<->matrix, SAM 3D mesh/pose
extraction. Pure-numpy, no stage/env deps; imported by stage1/stage4 schema. Re-exported by SCHEMA.
"""
from __future__ import annotations
import numpy as np


def umeyama_similarity(src: np.ndarray, dst: np.ndarray):
    """Closed-form similarity (Umeyama 1991): find s, R, t with dst ≈ s * R @ src + t.
    Use this when GT mesh correspondences ARE available, to recover the SAM3D-canonical
    -> GT-canonical alignment and thus a real rotation/scale error. src,dst: (N,3).
    Returns (s: float, R: (3,3), t: (3,)).
    """
    src = np.asarray(src, float); dst = np.asarray(dst, float)
    n = src.shape[0]
    mu_s, mu_d = src.mean(0), dst.mean(0)
    Sc, Dc = src - mu_s, dst - mu_d
    cov = (Dc.T @ Sc) / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1.0
    R = U @ S @ Vt
    var_s = (Sc ** 2).sum() / n
    s = float(np.trace(np.diag(D) @ S) / var_s) if var_s > 1e-12 else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def static_point_residual_px(uv, vis, E, K, X0=None, min_frames=8, chunk=4000):
    """【运动可观测性判据】每条轨迹:用它逐帧的视线拟合 *单个固定世界点*,返回重投影残差
    中位数(px)。静止 -> 所有观测被同一个 3D 点解释,残差 = 2D 跟踪+位姿噪声(亚像素);
    真在动 -> 没有任何静止点能解释,残差 = 该运动在图像上的位移(几十 px)。

    为什么要有这个东西(2026-08-13):世界系位移/速度这类判据在 Stereo4D 上不可用 ——
    官方 GT 的 track3d_world 自带米级深度噪声(静态场景实测:<5m 处 0.011m,10-20m 处 0.525m,
    20-40m 处 0.884m),且噪声是与相机运动相关的【平滑低频漂移】而非零均值白噪声,所以
    最小二乘斜率(速度)照样被它带偏:实测一棵静止的树世界"速度"0.24 m/s,远超任何阈值。
    本判据不吃深度这个自由度,所以对【沿视线】那一大半噪声免疫 —— 同一棵树的本残差只有
    0.40 px。实测判别力(415 实例):person 3.08px / horse 5.36px vs tree 0.54 / sign 0.78
    / pole 0.59 / car 0.17 / bench 0.27。

    输入 uv 允许是"把(带噪的)GT 3D 投影回图像"得到的:重建误差主要沿视线(实测 沿/垂 = 5:1),
    而沿视线的偏移投影到图像上 ≈ 0。实测:静态场景 300 条随机 GT 轨迹按静止点拟合,残差中位
    0.80px / p90 1.62px。

    ⚠️ 两个必须知道的局限(2026-08-14 实测补充,别再说本判据"完全不受深度影响"):
    (1) **垂直分量会漏进来,且随深度增长。** GT 误差的垂直(非视线)分量直接投进图像,量级
        ≈ f·垂直误差/深度。实测 scene_-JXwdsdJKHI:静态背景轨迹残差中位 0-3m 处 0.28px、
        5-7m 1.12px、7-9m **6.55px**、9-12m **20.02px**(同 clip 内 70× 跨度);对照
        scene_-6R40KeF9V8 在 12-20m 处仍只有 ~0.6px —— 所以它**既随深度也随 clip 变**。
        后果是【单侧安全】的:噪声只会抬高残差 -> 判据说"在动" -> 不降级 -> 少省算力,
        绝不会造成误降级或误升级。但也意味着**残差绝对不能反过来用作"在动"的证据**
        (2026-08-14 用户目检 9 个高残差实例,Qwen 判静全部正确)。
    (2) 盲区:物体恰好【沿视线方向】运动 -> 2D 不动 -> 残差为 0,本判据看不见。调用方应
        再并一个世界速度闸(两者互补:速度闸看得见沿视线的平动,本判据看得见世界速度盲区的
        原地往复/摆动运动)。

    uv (N,T,2) px; vis (N,T) bool; E (T,3,4) world->cam [R|t]; K (3,3) 或 (T,3,3);
    X0 (N,3) = 锚点,**强烈建议传**(取该轨迹可见帧的 3D 中位)。它不是"弱先验"而是候选解之一,
    也是低视差退化时唯一可靠的兜底(见下方实现注释)。不传则只能用自由拟合,过不了合理性
    检查就返回 nan(弃权)。
    返回 (N,) 残差中位数;可见帧 < min_frames 或无法判断的为 nan(调用方须按"弃权"处理)。
    """
    E = np.asarray(E, float)
    R = E[:, :, :3]                                          # (T,3,3) world->cam
    tv = E[:, :, 3]                                          # (T,3)
    C = -np.einsum("tji,tj->ti", R, tv)                      # (T,3) 相机中心(世界系)
    K = np.asarray(K, float)
    Kt = np.repeat(K[None], len(E), 0) if K.ndim == 2 else K  # (T,3,3)
    Kin = np.linalg.inv(Kt)
    uv = np.asarray(uv, float)
    vis = np.asarray(vis, bool)
    N, T = vis.shape
    out = np.full(N, np.nan)
    I3 = np.eye(3)
    for a in range(0, N, chunk):
        b = min(a + chunk, N)
        w = vis[a:b].astype(float)                           # (n,T)
        k = w.sum(1)
        h = np.concatenate([uv[a:b], np.ones((b - a, T, 1))], -1)
        d = np.einsum("tij,ntj->nti", Kin, h)                # 相机系视线
        d = np.einsum("tji,ntj->nti", R, d)                  # -> 世界系 (R^T d)
        d /= np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-12)
        dC = np.einsum("ntl,tl->nt", d, C)
        has_anchor = X0 is not None
        anchor = (np.nan_to_num(np.asarray(X0, float)[a:b]) if has_anchor
                  else np.zeros((b - a, 3)))

        def resid_of(P):
            """把候选静止点 P (n,3) 投到每帧,返回逐帧像素残差 (n,T) 与有效掩码。"""
            cm = np.einsum("tij,nj->nti", R, P) + tv[None]
            zz = cm[..., 2]
            pp = np.einsum("tij,ntj->nti", Kt, cm)
            zzs = np.where(np.abs(zz) > 1e-9, zz, 1.0)
            return (np.linalg.norm(pp[..., :2] / zzs[..., None] - uv[a:b], axis=-1),
                    w.astype(bool) & (zz > 1e-6))

        # 候选 1:锚点本身(= 该轨迹可见帧的 3D 中位)。**永远合法、永不退化**,是最终的安全网。
        # 它已经足够好:实测静态场景 300 条 GT 轨迹,不做任何拟合直接投影,残差中位就只有 0.80px
        # —— 因为重建误差几乎全部沿视线,投到图像上≈0。
        r_best, g_best = resid_of(anchor)
        # 候选 2:自由拟合(最小化点到视线距离)。⚠️ 它在【低视差】下会**塌到相机中心** ——
        # 那里所有 3D 距离都≈0,是个代价极低的退化极小点。实测一条 2D 只动 18px、相机只平移
        # 4mm 的轨迹,拟合点跑到相机后方(cam z=-0.03),残差虚高到 108px,把静止物误判成运动。
        # 改 IRLS(1/d² 加权成角度目标)也救不回来。所以:拟合结果必须过合理性检查才采纳,
        # 且最终取两个候选中残差更小的那个 —— 我们问的是"是否**存在**一个静止点能解释观测",
        # 任何合法候选给出更低残差都是更好的答案,取 min 单调、不会比锚点更差。
        S = np.einsum("ntl,ntm->nlm", d * w[..., None], d)
        A = k[:, None, None] * I3[None] - S
        rhs = np.einsum("nt,tl->nl", w, C) - np.einsum("ntl,nt->nl", d, w * dC)
        lam = 1e-6 * np.maximum(k, 1.0)
        Xf = np.linalg.solve(A + lam[:, None, None] * I3[None],
                             (rhs + lam[:, None] * anchor)[..., None])[..., 0]
        r_fit, g_fit = resid_of(Xf)
        sane = g_fit.sum(1) >= min_frames                     # 至少要投在相机前方足够多帧
        if has_anchor:                                        # 有锚点时再加"离锚点不能太远"
            cbar = (w[..., None] * C[None]).sum(1) / np.maximum(k, 1.0)[:, None]  # 平均相机中心
            far = np.linalg.norm(anchor - cbar, axis=1)                           # 锚点到相机距离
            sane = sane & (np.linalg.norm(Xf - anchor, axis=1) <= 0.5 * far)
        for i in range(b - a):
            cands = []
            if has_anchor and g_best[i].sum() >= min_frames:      # 候选 1 只在传了锚点时才有效
                cands.append(float(np.median(r_best[i][g_best[i]])))
            if sane[i]:
                cands.append(float(np.median(r_fit[i][g_fit[i]])))
            if cands:                                             # 都没有 -> 留 nan = 弃权
                out[a + i] = min(cands)
    return out


def k_grid_wh(K: np.ndarray, W: float, H: float):
    """The pixel grid `sam3d_intrinsics` actually lives in (4x-mismatch fix, 2026-07-23; moved
    here from src/stage4/run_link_dense.py so Stage-⑦ and Stage-2b-early share ONE copy).
    SAM3D's pipeline returns a pointmap that may be an ISOTROPIC power-of-two DOWNSAMPLE of the
    input frame (_down_sample_img: x0.5 for max_side>1200, x0.25 for >1900 — e.g. 1920x1280 ->
    480x320), and K is least-squares-fit IN THAT GRID (its principal point ~ grid center).
    Normalizing K-projections by the video (W,H) then shrinks them by the factor -> zero t* pixel
    pairs -> "T-unreliable" (the TAPVid-3D drivetrack failure). Recover the factor SELF-ADAPTIVELY
    from the stored K (2*cx ~ grid width), snapped to the pipeline's power-of-two set; frames that
    were not downsampled (DAVIS 854, PO 960) snap to 1 and keep (W,H) EXACTLY -> bit-identical
    behavior there."""
    fac = np.array([1.0, 2.0, 4.0, 8.0])
    gw, gh = max(2.0 * float(K[0, 2]), 1e-9), max(2.0 * float(K[1, 2]), 1e-9)
    err = np.abs(np.log(gw * fac / W)) + np.abs(np.log(gh * fac / H))
    c = float(fac[int(np.argmin(err))])
    return W / c, H / c


def sam3d_project_norm(pts: np.ndarray, inst, W: float, H: float):
    """VECTORIZED: pts (n,3) scene frame -> (n,2) normalized [0,1] t* pixels via SAM3D's OWN
    camera (inst mapping with object_to_scene / object_pose_cam / sam3d_intrinsics — an inst_*.npz
    handle works), + (n,) valid mask (z in front). Handles ~300k verts in one matmul.
    Normalization uses K's OWN pixel grid via k_grid_wh (the 4x fix): == (W,H) unless SAM3D
    downsampled its pointmap; the downsample is an isotropic resize of the full frame, so
    normalized coords stay comparable with the full-frame-normalized dense uv."""
    pts = np.asarray(pts, float).reshape(-1, 3)
    P = np.c_[pts, np.ones(len(pts))]
    canon = P @ np.linalg.inv(np.asarray(inst["object_to_scene"], float)).T      # (n,4)
    v = canon @ np.asarray(inst["object_pose_cam"], float).T                     # (n,4) SAM3D camera
    z = v[:, 2]; ok = np.abs(z) > 1e-9
    zz = np.where(ok, z, 1.0)
    K = np.asarray(inst["sam3d_intrinsics"], float)
    Wk, Hk = k_grid_wh(K, W, H)
    x = K[0, 0] * v[:, 0] / zz + K[0, 2]
    y = K[1, 1] * v[:, 1] / zz + K[1, 2]
    return np.stack([x / Wk, y / Hk], 1).astype(np.float32), ok


def _apply_transform(verts: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Apply a (4,4) homogeneous transform to (V,3) vertices."""
    h = np.hstack([verts, np.ones((verts.shape[0], 1))])
    return (h @ M.T)[:, :3]


def _centroid(verts: np.ndarray) -> np.ndarray:
    return verts.mean(0)


def _extent(verts: np.ndarray) -> float:
    """Axis-aligned bbox diagonal length — cheap, rotation-sensitive proxy for size."""
    return float(np.linalg.norm(verts.max(0) - verts.min(0)))


def _scale_about_centroid(verts: np.ndarray, s: float) -> np.ndarray:
    c = _centroid(verts)
    return (verts - c) * s + c


def _scale_about_point(p: np.ndarray, s: float) -> np.ndarray:
    """(4,4) homogeneous uniform scale by `s` about point `p`:  v -> p + s*(v - p)."""
    p = np.asarray(p, float).reshape(3)
    M = np.eye(4)
    M[:3, :3] = s * np.eye(3)
    M[:3, 3] = (1.0 - s) * p
    return M


def _principal_axes(points: np.ndarray):
    """PCA of a point cloud -> (axes (3,3) unit columns, extent (3,)), both sorted by
    descending spread. `extent[i]` = full range of the points projected on axis i."""
    pts = np.asarray(points, float)
    c = pts.mean(0)
    cov = (pts - c).T @ (pts - c) / max(len(pts), 1)
    evals, evecs = np.linalg.eigh(cov)            # ascending
    order = np.argsort(evals)[::-1]
    axes = evecs[:, order]                         # columns = principal directions
    proj = (pts - c) @ axes
    extent = proj.max(0) - proj.min(0)
    return axes, extent


def _obb_axes_from_corners(corners: np.ndarray):
    """GT oriented-bbox 8 corners -> (axes (3,3) unit cols, extent (3,)), sorted desc.
    PCA of the 8 corners recovers the box edge directions regardless of corner ordering."""
    return _principal_axes(np.asarray(corners, float).reshape(-1, 3))


def _rotation_angle_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))


def _axis_alignment(A_sam: np.ndarray, A_gt: np.ndarray):
    """Smallest proper rotation mapping the (sorted) SAM axes to the GT axes, resolving the
    per-axis sign ambiguity of PCA. Returns (R (3,3), angle_deg). Assumes both axis sets are
    already ordered by descending extent so axis i<->i (valid when extents are non-degenerate).
    """
    best_R, best_ang = np.eye(3), 180.0
    for sx in (1, -1):
        for sy in (1, -1):
            for sz in (1, -1):
                R = A_gt @ np.diag([sx, sy, sz]) @ A_sam.T
                if np.linalg.det(R) < 0.5:          # keep proper rotations only (det≈+1)
                    continue
                ang = _rotation_angle_deg(R)
                if ang < best_ang:
                    best_ang, best_R = ang, R
    return best_R, best_ang


def _rotation_check(verts_world: np.ndarray, corners_gt: np.ndarray, degen_ratio: float = 1.15):
    """Compare a placed mesh's principal axes to the GT OBB axes.
    Returns (R_align (3,3) SAM->GT, rotation_error_deg, degenerate: bool). `R_align` reorients
    the mesh to GT; rotation_error_deg = SAM's raw orientation error. `degenerate=True` when two
    GT extents are within `degen_ratio` (axis order ambiguous, e.g. symmetric objects) -> caller
    should NOT trust the result (skip grade + skip re-orient)."""
    A_sam, _ = _principal_axes(verts_world)
    A_gt, ext_gt = _obb_axes_from_corners(corners_gt)
    e = np.sort(ext_gt)[::-1]
    degenerate = (e[1] > 1e-9 and e[0] / e[1] < degen_ratio) or \
                 (e[2] > 1e-9 and e[1] / e[2] < degen_ratio) or e[2] < 1e-9
    R_align, ang = _axis_alignment(A_sam, A_gt)
    return R_align, ang, bool(degenerate)


def _to_np(x) -> np.ndarray:
    """torch.Tensor (any device) or array-like -> numpy, without importing torch here."""
    if hasattr(x, "detach"):                       # torch.Tensor
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _quat_wxyz_to_matrix(q) -> np.ndarray:
    """Unit-normalized (w,x,y,z) quaternion (pytorch3d convention, what SAM 3D returns)
    -> (3,3) rotation matrix."""
    q = np.asarray(q, float)
    w, x, y, z = q / (np.linalg.norm(q) + 1e-12)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ], float)


def _similarity_from_quat(quat_wxyz, translation, scale) -> np.ndarray:
    """Compose SAM 3D's object(canonical)->camera similarity into a (4,4):
        v_cam = R(quat)^T @ (scale ⊙ v_canonical) + t
    NOTE the TRANSPOSE on R. SAM 3D poses the mesh with pytorch3d
    `Transform3d().scale().rotate(quaternion_to_matrix(q)).translate()` applied to ROW vectors
    (`v @ R`), which equals `R^T @ v` in our column convention. Numerically verified against SAM 3D's
    own SceneVisualizer.object_pointcloud: ours-with-R disagrees (~0.4), ours-with-R^T matches exactly.
    Using R (not R^T) here transposed every object's rotation -> consistent wrong mesh orientation
    while on-mask still passed (silhouette is rotation-insensitive), so it went undetected.
    scale may be per-axis (3,)."""
    R = _quat_wxyz_to_matrix(quat_wxyz)
    M = np.eye(4)
    M[:3, :3] = R.T @ np.diag(np.asarray(scale, float).reshape(3))
    M[:3, 3] = np.asarray(translation, float).reshape(3)
    return M


def _extract_mesh(mesh_obj):
    """Pull (verts (V,3) float64, faces (F,3) int64) out of SAM 3D's mesh container.
    run_sam3d_single stores a single `MeshExtractResult` (.vertices / .faces). This also
    accepts a Trimesh (.vertices/.faces), a [list][0] of either, or a raw (verts, faces) pair.
    """
    if isinstance(mesh_obj, (list, tuple)):
        if len(mesh_obj) == 2 and not hasattr(mesh_obj[0], "vertices"):
            return _to_np(mesh_obj[0]).astype(np.float64), _to_np(mesh_obj[1]).astype(np.int64)
        mesh_obj = mesh_obj[0]
    verts = getattr(mesh_obj, "vertices", None)
    if verts is None:
        verts = getattr(mesh_obj, "verts", None)
    faces = getattr(mesh_obj, "faces", None)
    return _to_np(verts).astype(np.float64), _to_np(faces).astype(np.int64)


def _layout_to_transform(layout) -> np.ndarray:
    """SAM 3D layout -> (4,4) object(canonical)->scene transform.
    run_sam3d_single's `layout` carries the precomposed 'matrix' (and the raw pose
    components rotation_wxyz / translation / scale as a fallback)."""
    if isinstance(layout, dict):
        if layout.get("matrix") is not None:
            return np.asarray(layout["matrix"], dtype=float)
        return _similarity_from_quat(layout["rotation_wxyz"], layout["translation"], layout["scale"])
    return np.asarray(layout, dtype=float)
