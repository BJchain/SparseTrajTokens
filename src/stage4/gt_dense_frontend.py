"""V3 Stage-③ GT front-end: Stereo4D GROUND-TRUTH world-frame 3D tracks -> per-instance
dense_tracks_<iid>.npz (CONTRACT-IDENTICAL to the T4W version run_dense_frontend_t4w.py, so
Stage-⑦ run_link_dense and everything downstream are untouched).

Idea (user, 2026-07-31): Stereo4D ships gt_stereo4d.npz with GROUND-TRUTH world-frame 3D
trajectories (track3d_world (Npt,T,3), NaN=absent) + the LEFT-rectified world->cam extrinsics
(extrs_w2c (T,3,4)) + intrinsics_K (3,3), already sliced to the SAME T frames as %05d.jpg. These
are TRUE trajectories (static objects have world-frame std ~0.02) — far cleaner than any tracker
estimate. So skip T4W (--skip-stages t) and instead ATTRIBUTE the scene-wide GT tracks to the SAM3
instance masklets, emitting the exact dense-track contract per instance.

Method (pure CPU, needs only masklets + gt):
  1. project every GT track to each frame:  cam = R_w2c X + t_w2c ; uv = K cam / z ;
     visible_final = gt_visible & z>0 & inside-image.
  2. ATTRIBUTE: a track goes to the instance whose (tol-dilated) mask contains the highest
     fraction of the track's visible-frame projections, requiring frac >= --min-frac AND a
     margin over the 2nd-best instance (>= --ambig-margin). Background / ambiguous tracks dropped.
  3. per instance emit dense_tracks_<iid>.npz:
       xyz_world = GT track3d_world subset (0 where not visible-final)
       xyz_cam   = R_w2c X + t_w2c        (0 where not visible-final)
       uv_norm   = uv / [W,H]             (0 where not visible-final)
       vis       = visible_final ; conf = 1.0 where vis else 0.0 (GT)
       c2w       = inv([R|t]_w2c) (T,4,4) ; intrs = K repeated (T,3,3)
       seed_frame = first visible-final frame ; seed_uv_norm = its uv_norm
     Instances with a huge GT count are spatially FPS-downsampled to --cap (~T4W density);
     instances with < --min-tracks tracks emit NO file (downstream naturally drops them =
     the user's "keep only instances that have trajectories").

Coordinate note: GT world frame != SAM3D mesh frame, but ⑦'s _solve_T does a general 7-DOF
Umeyama (W_track -> W_mesh) by t* pixel pairing, which absorbs the convention difference exactly
as it does for the T4W world. No special handling here.

Run in ANY cpu env with numpy+scipy+cv2 (e.g. sam3d-objects):
  TMPDIR=/tmp <py> src/stage4/gt_dense_frontend.py --batch-root <br>
"""
from __future__ import annotations
import argparse, glob, json, os, pickle, sys, tempfile
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import SCHEMA  # noqa: F401  (register masklet/video_input classes for unpickle)
from src.geometry import static_point_residual_px  # noqa: E402  (运动可观测性判据,与 tokenizer 共用)

import cv2  # noqa: E402

GT_FILENAME = "gt_stereo4d.npz"
CAP = 800
MIN_TRACKS = 8
MIN_TRACKS_STATIC = 4  # motion=='static' instances only need ~4 representative points (finalize
                       # static branch), so don't drop them for having few attributed tracks.
TOL = 4
MIN_FRAC = 0.5
AMBIG_MARGIN = 0.15
MIN_OWN_TRACKS = 1     # 【2026-08-14 用户拍板】发射前要求实例至少有这么多条"自有"轨迹 ——
                       # 即在【未膨胀】mask 上 in-mask 比例 >= MIN_FRAC 的轨迹。为 0 则关闭本闸。
                       # 为什么需要:归属用的是 tol=4 膨胀 mask,而方形核是 L∞ 度量,沿对角能够到
                       # 4√2≈5.66px。对**又薄又亮/又小又远**的物体(镲片/栅栏/招牌/帐篷/杆子),
                       # GT 跟踪器在它身上一条轨迹都产不出,膨胀环却会把【邻近表面】的轨迹吸进来。
                       # 实测 scene_-XlKQHfYHIU inst7 'cymbal':落在自己 mask 内的 GT 轨迹 = 0 条,
                       # 被归属的 7 条全部距离 mask 4.65~5.48px(欧氏)、深度 3.7m 紧密成簇,是旁边
                       # 那个面上的点;它们的中位点被 finalize 当成镲片的 3D 代表点写进 final_repr。
                       # 这类错误【静默】:静态实例下游没有任何信号能暴露它。
                       # 判据只能是**群体级**的:逐条轨迹看不出毛病(5px 外、同深度,任何几何判据
                       # 都判不了),只有"严格口径下自有轨迹数 = 0"能识别。
                       # 实测(两批 457 个发射实例,均已用修复后前端重跑):n_own==0 的 16 个(3.5%),
                       # 且分布有天然断点(<=0 是 16 个,<=1 才 20 个),所以阈值取 1 最保守。
                       # 用户定案理由:物体本来就没有自己的 GT 轨迹,留着也给不了 MLLM 信息,
                       # 反而浪费 token。
MIN_VIS_FRAMES = 3     # reject coincidental ultra-short GT tracks (a 1-2 frame "trajectory" is
                       # useless AND biases attribution: a 1-frame track landing in a mask scores
                       # frac=1.0. Stereo4D GT is otherwise dense — most tracks span the whole clip.)
MOTION_THRESH = 0.05   # [DEPRECATED] 旧信号:GT 世界位移 std(米)。只留作审计 sidecar 的记录值。

# ============ 运动判定:【单向门】= Qwen 默认 + 只允许降级(2026-08-13 重做) ==============
# 历史与推翻理由(两代旧门都错了,别再回去):
#   第一代(位移 std >= 0.05m -> dynamic,硬覆盖 Qwen)、第二代(世界速度三段式 TAU_LO/HI)。
#   两代都建立在文件头那句"static objects have world-frame std ~0.02"上,而该前提是**假的**:
#   静态场景实测 GT 世界系抖动中位 0.35m,<5m 处 0.011m / 10-20m 处 0.525m / 20-40m 处 0.884m
#   (强深度相关,沿视线:垂直 = 5:1 = 三角化深度误差签名)。第二代注释里"斜率对深度零均值抖动
#   鲁棒"同样是假的 —— 该噪声是与相机运动相关的【平滑低频漂移】,不是零均值白噪声,斜率被它
#   直接带偏:一棵静止的树实测世界"速度"0.24~0.61 m/s,远超 TAU_HI。
# 实测(415 实例 @ stereo4d_p2 + stereo4d_dense,用不依赖深度的可观测性判据做参考):
#   升级方向(Qwen静 -> 门改动): 24 例只对 1 例(4%)。错的 23 例是 tree×10/sign/pole/fence/bench…,
#     深度中位 12.9m、世界速度中位 0.24m/s,而**图像残差只有 0.40px**。四种参考阈值下改对率
#     都是 4~8%,结论不随阈值翻转 => 这个方向是纯负收益,已删除。它正是"全静态林荫道被拟合出
#     1.2m 树轨迹"的直接原因。
#   降级方向(Qwen动 -> 门改静): 39 例对 27(69%),错 12(31%)。错的是 person×9/horse×2/cup,
#     世界速度 0.036m/s 但图像残差 5.2px = 原地往复/摆动运动,**LS 斜率对其完全盲**(旧的 std
#     口径同样盲)。这类误杀静默且不可逆:真动物体被编成 4 个静态点,下游无从察觉。
# 现在的规则:
#   final = Qwen                                  # 默认完全随 Qwen,恢复 2026-07-22 静态路由定案
#                                                 # 第①条,也与 DAVIS/MOVi/PO/TAPVid 四根一致
#   唯一例外(只降级,永不升级):Qwen 说 dynamic,但【世界速度慢】AND【图像上无可观测运动】
#                                                 -> static(纯粹为了省算力)
# 两个闸互补、缺一不可:速度闸看得见沿视线的平动(恰是可观测性判据的盲区);可观测性判据看得见
# 世界速度的盲区(原地往复/摆动),且对深度噪声免疫。要求两者同时成立才降级 = 单侧安全:
# 深度噪声只会抬高速度 -> 少降级(少省点算力),永远不会造成误降级。
# 实测该规则:准确率 92.4%(现役三段式 82.1% / Qwen 单独 84.2%);漏判真动 9 例(= Qwen 单独
# 同水平,门不再额外制造静默丢失,三段式是 20 例);降级 41 个中残差>=3px 的 0 个;升级 0 个;
# 而静态实例占比 60%,与三段式门**完全持平** —— 算力节省一分没丢。
MOTION_TAU_LO = 0.05        # m/s,世界速度闸:低于此才允许降级
RESID_TAU_ABS_PX = 1.0      # px,可观测性闸的绝对下限
RESID_TAU_MAX_PX = 1.0      # px,硬顶(2026-08-14 用户拍板由 1.5 收到 1.0 = 彻底取消自适应上浮)。
                            # 背景:自适应阈值 clip(2×floor, 1.0, MAX) 里 2×floor 通常 0.7~1.34,
                            # 会被 1.0 下限截住 —— 所以自适应**只会把阈值往上抬**,它的全部价值
                            # 就是"噪场景多降级省算力",全部代价就是"多几个贴阈值的降级"。
                            # 实测 MAX=1.5 时 44 个降级里有 4 个残差 1.04~1.41px(全是 person,
                            # 都在高噪场景);收到 1.0 后这 4 个消失,代价仅 静态率 −0.9pp。
                            # 取舍原则:静默误杀真运动不可逆且下游无从察觉,算力可以再花。
RESID_FLOOR_MULT = 2.0      # 逐场景自适应:阈值 = clip(MULT * 本场景残差地板, ABS, MAX)
RESID_FLOOR_PCT = 25        # 地板 = 本场景【背景】轨迹(eligible 且未归属)残差的 p25 —— 代表
                            # 本 clip 的 2D 跟踪+相机位姿噪声水平。必须逐场景自适应:实测 44
                            # 场景地板 p50 0.39px / p90 1.10px,但有 3.24px 的场景。
RESID_FLOOR_MIN_N = 50      # 背景轨迹少于这么多则不信地板,退回绝对下限(保守:更少降级)
RESID_MIN_FRAMES = 8        # 少于这么多可见帧不出残差(视差不足,判据无意义)


def _track_speed_mps(P, ts):
    """P (K,3) 可见帧世界坐标, ts (K,) 秒。世界速度 = 最小二乘斜率的模(m/s)。
    < 3 帧或时间无跨度 -> 0(退化为静),交由上层的 Qwen/阈值兜底。"""
    if len(P) < 3:
        return 0.0
    t = ts - ts.mean()
    denom = float((t * t).sum())
    if denom < 1e-12:
        return 0.0
    b = (t[:, None] * (P - P.mean(0))).sum(0) / denom          # (3,) 每轴 m/s
    return float(np.linalg.norm(b))


def _project_all(tw, vis, E, K, W, H):
    """Project every GT track to every frame. Returns uv (N,T,2) px, cam (N,T,3),
    fvis (N,T) bool = gt_visible & z>0 & inside-image."""
    R = E[:, :, :3].astype(np.float64)                       # (T,3,3)
    tvec = E[:, :, 3].astype(np.float64)                     # (T,3)
    X = np.nan_to_num(tw).astype(np.float64)                 # (N,T,3), 0 where absent
    cam = np.einsum("tij,ntj->nti", R, X) + tvec[None]       # (N,T,3)
    z = cam[..., 2]
    zsafe = np.where(np.abs(z) > 1e-9, z, 1.0)
    u = K[0, 0] * cam[..., 0] / zsafe + K[0, 2]
    v = K[1, 1] * cam[..., 1] / zsafe + K[1, 2]
    inim = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (z > 1e-6)
    fvis = vis.astype(bool) & inim
    uv = np.stack([u, v], -1)
    return uv, cam, fvis


def _dilate(masks_thw, tol):
    """(T,H,W) bool -> tol-dilated (T,H,W) bool (square kernel 2*tol+1)."""
    if tol <= 0:
        return masks_thw
    k = np.ones((2 * tol + 1, 2 * tol + 1), np.uint8)
    out = np.empty_like(masks_thw)
    for t in range(masks_thw.shape[0]):
        out[t] = cv2.dilate(masks_thw[t].astype(np.uint8), k).astype(bool)
    return out


def _inmask_frac(uvi_x, uvi_y, fvis, dmask):
    """For each track, fraction of its visible-final frames whose pixel falls in dmask.
    Returns frac (N,), computed only over fvis frames (0 if a track has no visible frame)."""
    N, T = fvis.shape
    hit = np.zeros(N, np.int32)
    for t in range(T):
        sel = fvis[:, t]
        if not sel.any():
            continue
        hit[sel] += dmask[t, uvi_y[sel, t], uvi_x[sel, t]]
    nvis = fvis.sum(1)
    return np.where(nvis > 0, hit / np.maximum(nvis, 1), 0.0), nvis


def _fps(feat, cap):
    """Farthest-point-sample cap indices from feat (M,3). Returns selected index array."""
    M = len(feat)
    if M <= cap:
        return np.arange(M)
    sel = np.empty(cap, np.int64)
    d = np.full(M, np.inf)
    sel[0] = 0
    for i in range(1, cap):
        d = np.minimum(d, np.sum((feat - feat[sel[i - 1]]) ** 2, axis=1))
        sel[i] = int(np.argmax(d))
    return sel


def process_scene(scene_dir, cap, min_tracks, tol, min_frac, ambig_margin, min_vis_frames,
                  min_tracks_static, motion_thresh, motion_override, min_own_tracks=MIN_OWN_TRACKS):
    vip = os.path.join(scene_dir, "video_input.pkl")
    gtp = os.path.join(scene_dir, GT_FILENAME)
    if not os.path.exists(vip):
        print(f"  [skip] {os.path.basename(scene_dir)}: no video_input.pkl", flush=True); return
    if not os.path.exists(gtp):
        print(f"  [skip] {os.path.basename(scene_dir)}: no {GT_FILENAME}", flush=True); return
    vid = pickle.load(open(vip, "rb"))
    frames = np.asarray(vid.frames)
    T, H, W = frames.shape[:3]
    g = np.load(gtp)
    tw = g["track3d_world"]                                  # (N,Tg,3)
    vis = g["visible"]                                       # (N,Tg)
    E = g["extrs_w2c"].astype(np.float64)                    # (Tg,3,4)
    K = g["intrinsics_K"].astype(np.float64)                 # (3,3)
    # 真实时间戳(秒),供速度拟合处理变步长。缺失则退回帧序号当 s(速度单位会变,但 Stereo4D
    # GT 一定带 timestamps_us —— 这个前端本就是 Stereo4D 专用)。
    ts_sec = (g["timestamps_us"].astype(np.float64) / 1e6 if "timestamps_us" in g.files
              else np.arange(tw.shape[1], dtype=np.float64))
    if tw.shape[1] != T:
        print(f"  [skip] {os.path.basename(scene_dir)}: GT T={tw.shape[1]} != frames T={T} "
              f"(stride mismatch)", flush=True); return

    uv, cam, fvis = _project_all(tw, vis, E, K, W, H)
    N = tw.shape[0]
    uvi_x = np.clip(np.round(uv[..., 0]).astype(np.int32), 0, W - 1)
    uvi_y = np.clip(np.round(uv[..., 1]).astype(np.int32), 0, H - 1)

    # per-instance in-mask fraction over each track's visible frames
    masklets = list(vid.masklets)
    if len(masklets) == 0:                            # 0-masklet clip (SAM3 found nothing to
        # ⚠️ 必须先落一份 stats 再 return。静态 clip 过滤器(run_stereo4d_batch_v3 的过滤器2)
        # 在 gt_dense_stats.json 缺失时按"未知≠静态"放行 —— 那条规则对"没跑到"是对的,但
        # 0 masklet 不是未知,是确定的"没东西可分割"。不落 stats 的后果(2026-08-18 实测):
        # 这类 clip 既产不出 final_repr、又永远不被剔除,成为每轮都被重新认领的【幽灵失败】,
        # 而重试还因 video_input.pkl 已存在而空转(实测 2 个 clip 各被空转重试 6~7 次)。
        # 写下 n_emit=0 让它直接走过滤器2 现成的 no_emitted_instance 分支。
        with open(os.path.join(scene_dir, "gt_dense_stats.json"), "w") as f:
            json.dump(dict(clip=os.path.basename(scene_dir), n_gt=int(N), n_masklets=0,
                           instances={}, n_emit=0, n_emit_static=0, n_flip_dyn2static=0,
                           n_flip_static2dyn=0, skip_reason="no_masklets"), f, indent=2)
        print(f"  [skip] {os.path.basename(scene_dir)}: 0 masklets — no instances to attribute",
              flush=True); return                     # segment) -> no dense tracks, graceful skip
    iids = [ml.instance_id for ml in masklets]
    frac = np.zeros((N, len(masklets)), np.float32)
    # 同时算一份【严格口径】(不膨胀)的 in-mask 比例:归属仍用膨胀版(见 tol),但发射前要拿严格
    # 版查一句"这个实例到底有没有【落在它自己身上】的轨迹"—— 见下面 MIN_OWN_TRACKS。
    frac_strict = np.zeros((N, len(masklets)), np.float32)
    nvis = fvis.sum(1)
    for j, ml in enumerate(masklets):
        rmask = np.asarray(ml.masks).astype(bool)
        dmask = _dilate(rmask, tol)
        frac[:, j], _ = _inmask_frac(uvi_x, uvi_y, fvis, dmask)
        frac_strict[:, j], _ = _inmask_frac(uvi_x, uvi_y, fvis, rmask)

    # attribution: best instance, require best>=min_frac AND margin over 2nd-best. Only tracks with
    # >= min_vis_frames visible frames are ELIGIBLE (short coincidental tracks are dropped up front).
    eligible = nvis >= min_vis_frames
    order = np.argsort(-frac, axis=1)
    best_j = order[:, 0]
    best_f = frac[np.arange(N), best_j]
    second_f = frac[np.arange(N), order[:, 1]] if frac.shape[1] > 1 else np.zeros(N, np.float32)
    assigned = eligible & (best_f >= min_frac) & ((best_f - second_f) >= ambig_margin)
    # audit buckets
    n_short = int(((nvis > 0) & ~eligible).sum())
    n_bg = int((eligible & (best_f < min_frac)).sum())
    n_ambig = int((eligible & (best_f >= min_frac) & ((best_f - second_f) < ambig_margin)).sum())

    # build c2w / intrs once (same for all instances)
    c2w = np.zeros((T, 4, 4), np.float32)
    for t in range(T):
        M4 = np.eye(4)
        M4[:3, :4] = E[t]
        c2w[t] = np.linalg.inv(M4).astype(np.float32)
    intrs = np.repeat(K.astype(np.float32)[None], T, axis=0)

    # ---- 运动可观测性残差(px):每条 eligible 轨迹 + 本场景地板。见文件头单向门说明。 ----
    # uv 是把(带深度噪声的)GT 3D 投影回图像得到的,这仍是忠实的 2D 观测 —— GT 的重建误差几乎
    # 全部沿视线方向,投影到图像上≈0(实测静态场景 300 条随机轨迹按静止点拟合,残差中位 0.80px)。
    resid = np.full(N, np.nan)
    el_idx = np.nonzero(eligible)[0]
    if len(el_idx):
        X0 = np.zeros((len(el_idx), 3))
        for a, kk in enumerate(el_idx):                      # 弱先验锚点:零视差时才起作用
            fv = fvis[kk]
            if fv.any():
                X0[a] = np.median(np.nan_to_num(tw[kk][fv]), axis=0)
        resid[el_idx] = static_point_residual_px(
            uv[el_idx], fvis[el_idx], E, K, X0=X0, min_frames=RESID_MIN_FRAMES)
    # 地板只用【背景】轨迹(eligible 但未归属任何实例)算。**不能用全部 eligible**:实测
    # scene_-QMYCacTyNo 有 73% 的 eligible 被归属给实例(人群场景),p25 地板于是算在"物体点"
    # 上被抬到 1.14px、阈值 2.27px,连残差 2.15px 的 person 都被降级 —— 那正是本门最该防住的
    # 静默误杀。背景轨迹才是"本 clip 的 2D 跟踪 + 相机位姿噪声"的正确代理。
    bgm = np.isfinite(resid) & eligible & ~assigned
    rv = resid[bgm]
    resid_floor = float(np.percentile(rv, RESID_FLOOR_PCT)) if len(rv) >= RESID_FLOOR_MIN_N else None
    resid_thr = (max(RESID_TAU_ABS_PX, RESID_FLOOR_MULT * resid_floor)
                 if resid_floor is not None else RESID_TAU_ABS_PX)   # 背景太少 -> 退回绝对下限
    resid_thr = min(resid_thr, RESID_TAU_MAX_PX)                     # 硬顶,防地板被污染时失控

    # Qwen motion snapshot (authoritative baseline; written once so re-runs never read an already-
    # overridden video_input as "qwen"). {str(iid): 'static'|'dynamic'|None}.
    snap_p = os.path.join(scene_dir, "qwen_motion.json")
    if os.path.exists(snap_p):
        qwen_base = json.load(open(snap_p))
    else:
        qwen_base = {str(int(ml.instance_id)): (getattr(ml, "motion", None) or None)
                     for ml in masklets}
        with open(snap_p, "w") as f:
            json.dump(qwen_base, f, indent=2)

    # per-instance emit
    stats = dict(clip=os.path.basename(scene_dir), n_gt=int(N), n_has_vis=int((nvis > 0).sum()),
                 n_eligible=int(eligible.sum()), n_assigned=int(assigned.sum()),
                 n_short=n_short, n_background=n_bg, n_ambiguous=n_ambig,
                 tol=tol, min_frac=min_frac, ambig_margin=ambig_margin, cap=cap,
                 min_tracks=min_tracks, min_tracks_static=min_tracks_static,
                 min_vis_frames=min_vis_frames, motion_thresh=motion_thresh,
                 min_own_tracks=min_own_tracks,
                 motion_gate="one_way_demote_only",         # 见文件头;旧值语义是三段式硬覆盖
                 motion_tau_lo=MOTION_TAU_LO,
                 resid_floor_px=(round(resid_floor, 3) if resid_floor is not None else None),
                 resid_thr_px=(round(resid_thr, 3) if resid_thr is not None else None),
                 instances={})
    Xw = np.nan_to_num(tw).astype(np.float32)                # 0 where absent (matches T4W)
    uv_norm_full = (uv / np.array([W, H], np.float64)).astype(np.float32)
    # ⚠️ 先清掉本场景已有的 dense_tracks_*.npz(2026-08-11 修)。
    # 本函数只为"本次归属到 >= min_tracks 条轨迹"的实例写文件,不写的 iid 上的旧文件会原地
    # 存活。而 instance_id 是 SAM 3 每次跑重新编的:lane 脚本的 STEREO4D_SLIM=1 会在 finalize
    # 后删掉 video_input.pkl,该场景再被跑到时 run_stage0 的 skip-if-exists 就失效 -> SAM 3
    # 重跑 -> masklets 与 instance_id 全变;前端重跑只覆盖它这次发射的那些 iid,**其余 iid 上
    # 留着按【旧 masklets】算的陈旧文件**,于是 dense_tracks_<iid> 里装的是别的物体的轨迹。
    # 实测后果(stereo4d_sam3d50_p1,274 实例):"轨迹落在自己 mask 内"的比例 <0.2 的占 10.2%,
    # 其中 dense_tracks_15 的 233 条轨迹 100% 属于 inst4、_23 属于 inst9、_13 属于 inst11。
    # 动态实例表现为 ⑦ 0 链接;**静态实例是静默的** —— 它的 final_repr 代表点直接落到别的物体上。
    # 走到这里说明前面所有早退检查都过了,本次一定会重新发射,清空是安全的。
    for _stale in glob.glob(os.path.join(scene_dir, "dense_tracks_*.npz")):
        os.remove(_stale)

    n_emit = 0
    n_emit_static = 0
    n_flip_d2s = 0
    n_flip_s2d = 0          # 单向门下【结构性恒为 0】;保留 key 只为老读图/审计脚本不炸
    overrides = {}                                           # j -> GT motion (emitted only)
    for j, iid in enumerate(iids):
        tk = np.nonzero(assigned & (best_j == j))[0]
        qwen_m = qwen_base.get(str(int(iid)))
        # GT world-frame SPEED (m/s) = median over attributed tracks of the least-squares slope
        # magnitude of world-position vs real time. 与时长无关。⚠️ 它**被深度噪声污染**(静止的树
        # 也能到 0.24~0.61 m/s),所以只能当"降级"的必要条件之一,绝不能单独用来判动 —— 见文件头。
        # (legacy 位移 `gt_mag` 仅留给审计 sidecar / 向后兼容。)
        speeds, mags = [], []
        for k in tk:
            v = fvis[k]
            if v.sum() >= 2:
                mags.append(float(np.linalg.norm(Xw[k, v].std(axis=0))))
            if v.sum() >= 3:
                speeds.append(_track_speed_mps(Xw[k, v], ts_sec[v]))
        gt_mag = float(np.median(mags)) if mags else None
        gt_speed = float(np.median(speeds)) if speeds else None
        rr = resid[tk]; rr = rr[np.isfinite(rr)]
        obs_resid = float(np.median(rr)) if len(rr) else None
        # 单向门:默认随 Qwen;只有【速度慢 AND 图像上无可观测运动】才把 Qwen 的 dynamic 降为
        # static(省算力)。**永不把 static 升为 dynamic**(该方向实测 24 例只对 1 例)。
        final_m = qwen_m
        demoted = False
        if (motion_override and qwen_m == "dynamic" and gt_speed is not None
                and obs_resid is not None and resid_thr is not None
                and gt_speed < MOTION_TAU_LO and obs_resid < resid_thr):
            final_m = "static"
            demoted = True
        is_static = final_m == "static"
        thr = min_tracks_static if is_static else min_tracks
        rec = dict(n_attributed=int(len(tk)), qwen_motion=qwen_m,
                   gt_motion_mag=(round(gt_mag, 4) if gt_mag is not None else None),
                   gt_speed_mps=(round(gt_speed, 4) if gt_speed is not None else None),
                   obs_resid_px=(round(obs_resid, 3) if obs_resid is not None else None),
                   resid_thr_px=(round(resid_thr, 3) if resid_thr is not None else None),
                   final_motion=final_m, demoted=bool(demoted),
                   flipped=bool(demoted), min_tracks=thr)
        # 自有轨迹数:严格(不膨胀)口径下 in-mask 比例达标的条数。见 MIN_OWN_TRACKS。
        n_own = int((frac_strict[tk, j] >= min_frac).sum()) if len(tk) else 0
        rec["n_own_tracks"] = n_own
        rec["strict_inmask_mean"] = round(float(frac_strict[tk, j].mean()), 3) if len(tk) else 0.0
        if len(tk) < thr:
            rec["emitted"] = False
            rec["reason"] = "few_tracks"
            stats["instances"][str(int(iid))] = rec
            continue
        if min_own_tracks > 0 and n_own < min_own_tracks:
            rec["emitted"] = False
            rec["reason"] = "no_own_tracks"          # 归属到的轨迹没有一条真落在它自己身上
            stats["instances"][str(int(iid))] = rec
            print(f"  inst {iid}: attributed {len(tk)} 条但自有 {n_own} 条 "
                  f"(严格 in-mask 均值 {rec['strict_inmask_mean']:.3f}) -> DROP no_own_tracks",
                  flush=True)
            continue
        # spatial FPS to cap: feature = median visible world pos per track
        if len(tk) > cap:
            feats = np.zeros((len(tk), 3), np.float64)
            for a, k in enumerate(tk):
                fv = fvis[k]
                feats[a] = np.median(Xw[k, fv], axis=0) if fv.any() else Xw[k, 0]
            sub = _fps(feats, cap)
            tk = tk[sub]
        Nt = len(tk)
        fv = fvis[tk]                                        # (Nt,T)
        xyz_world = np.where(fv[..., None], Xw[tk], 0.0).astype(np.float32)
        xyz_cam = np.where(fv[..., None], cam[tk], 0.0).astype(np.float32)
        uvn = np.where(fv[..., None], uv_norm_full[tk], 0.0).astype(np.float32)
        conf = fv.astype(np.float32)
        sf = np.array([int(np.argmax(fv[a])) for a in range(Nt)], np.int64)  # first visible frame
        suv = uvn[np.arange(Nt), sf]
        out = os.path.join(scene_dir, f"dense_tracks_{iid}.npz")
        np.savez_compressed(out, instance_id=np.int64(iid), seed_frame=sf, seed_uv_norm=suv,
                            xyz_world=xyz_world, xyz_cam=xyz_cam, uv_norm=uvn,
                            vis=fv, conf=conf, c2w=c2w, intrs=intrs,
                            Hn=np.int64(H), Wn=np.int64(W))
        vpf = fv.sum(0)
        rec["emitted"] = True
        rec["N"] = int(Nt)
        rec["vis_per_frame_med"] = int(np.median(vpf))
        rec["mean_vis"] = round(float(fv.mean()), 3)
        stats["instances"][str(int(iid))] = rec
        n_emit += 1
        n_emit_static += int(is_static)
        n_flip_d2s += int(demoted)                           # 单向门:只可能是 dynamic->static
        overrides[j] = final_m
        print(f"  inst {iid}: attributed {rec['n_attributed']} -> N={Nt} "
              f"| speed={'n/a' if gt_speed is None else f'{gt_speed:.3f}'}m/s "
              f"resid={'n/a' if obs_resid is None else f'{obs_resid:.2f}'}px "
              f"qwen={qwen_m}->{final_m}{'*DEMOTE' if demoted else ''} "
              f"| mean vis={fv.mean():.2f} -> {os.path.basename(out)}", flush=True)

    # OVERRIDE (task 1): write the GT-verdict motion back into video_input.pkl for EMITTED instances
    # (all downstream motion consumers — stage1 static-skip, run_stage2b_dense static gate,
    # stage4_finalize static branch — read masklet.motion from this file). Atomic temp+os.replace so
    # a symlinked (reused) video_input has only its LINK swapped, never its shared target.
    if motion_override and overrides:
        for j, m in overrides.items():
            masklets[j].motion = m
        fd, tmpp = tempfile.mkstemp(dir=scene_dir, suffix=".vip.tmp")
        with os.fdopen(fd, "wb") as f:
            pickle.dump(vid, f)
        os.replace(tmpp, vip)

    n_drop = len(iids) - n_emit
    stats["n_emit"] = n_emit
    stats["n_emit_static"] = n_emit_static
    stats["n_flip_dyn2static"] = n_flip_d2s
    stats["n_flip_static2dyn"] = n_flip_s2d
    print(f"  [{os.path.basename(scene_dir)}] GT tracks={N} (vis={int((nvis > 0).sum())} "
          f"elig>={min_vis_frames}f:{int(eligible.sum())}): assigned={int(assigned.sum())} "
          f"short={n_short} bg={n_bg} ambig={n_ambig} "
          f"| emit={n_emit}/{len(iids)} ({n_emit_static} static) drop={n_drop} "
          f"| resid_floor={'n/a' if resid_floor is None else f'{resid_floor:.2f}'}px "
          f"thr={'n/a' if resid_thr is None else f'{resid_thr:.2f}'}px "
          f"| DEMOTE dyn->static={n_flip_d2s} (single-direction gate: no static->dyn)", flush=True)
    with open(os.path.join(scene_dir, "gt_dense_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)


def _scene_worker(arg):
    """ProcessPoolExecutor 的入口必须是模块级函数(闭包不可 pickle)。"""
    sd, kw = arg
    print(f"[gt-dense-front] {os.path.basename(sd)}", flush=True)
    try:
        process_scene(sd, kw["cap"], kw["min_tracks"], kw["tol"], kw["min_frac"],
                      kw["ambig_margin"], kw["min_vis_frames"], kw["min_tracks_static"],
                      kw["motion_thresh"], kw["motion_override"],
                      min_own_tracks=kw["min_own_tracks"])
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"[gt-dense-front] {os.path.basename(sd)}: FAILED {type(e).__name__}: {e} "
              f"— skipped", flush=True)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--scene-dir"); g.add_argument("--batch-root")
    ap.add_argument("--cap", type=int, default=CAP,
                    help="max tracks per instance (spatial FPS downsample); ~T4W density")
    ap.add_argument("--min-tracks", type=int, default=MIN_TRACKS,
                    help="dynamic instances with fewer attributed tracks emit no dense file")
    ap.add_argument("--min-tracks-static", type=int, default=MIN_TRACKS_STATIC,
                    help="lower min-tracks for motion=='static' instances (they need only ~4 "
                         "representative points; None/missing motion is treated as dynamic)")
    ap.add_argument("--tol", type=int, default=TOL, help="mask dilation tolerance (px)")
    ap.add_argument("--min-frac", type=float, default=MIN_FRAC,
                    help="min in-mask fraction (over a track's visible frames) to attribute")
    ap.add_argument("--ambig-margin", type=float, default=AMBIG_MARGIN,
                    help="best - 2nd-best in-mask fraction must exceed this (else ambiguous->drop)")
    ap.add_argument("--min-own-tracks", type=int, default=MIN_OWN_TRACKS,
                    help="发射前要求实例至少有这么多条【自有】轨迹(严格/不膨胀 mask 上 in-mask "
                         "比例 >= --min-frac 的条数)。0 = 关闭本闸(回到 2026-08-14 前的行为)。"
                         "挡的是:GT 跟踪器在薄/亮/小/远的物体上产不出轨迹,而 tol 膨胀环把邻近"
                         "表面的轨迹吸进来,finalize 再把它们的中位点当成该物体的 3D 代表点")
    ap.add_argument("--min-vis-frames", type=int, default=MIN_VIS_FRAMES,
                    help="a GT track needs >= this many visible frames to be attribution-eligible "
                         "(drops coincidental 1-2 frame tracks that bias the mask-fraction test)")
    ap.add_argument("--motion-thresh", type=float, default=MOTION_THRESH,
                    help="[DEPRECATED, 只进审计 sidecar 不参与判定] 旧的 GT 世界位移阈值。"
                         "现在的判定见文件头『单向门』:默认随 Qwen,只在世界速度 < "
                         f"{MOTION_TAU_LO} m/s 且图像上无可观测运动时把 dynamic 降为 static")
    ap.add_argument("--no-motion-override", action="store_true",
                    help="escape hatch: 记录速度/残差但不改写 video_input.pkl —— 即完全 Qwen-only "
                         "(与 DAVIS/MOVi/PO/TAPVid 四根同款,代价是失去降级带来的算力节省)")
    ap.add_argument("--clips", default="",
                    help="只处理这些 clip(逗号分隔,或 @文件 每行一个),其余 scene_* 一律不碰。"
                         "默认空 = 整个 --batch-root。⚠️ 直接对【生产 ROOT】跑时务必带上:"
                         "2026-08-18 我为修 2 个 clip 直接 --batch-root 生产根,前端无视上游的 "
                         "--clips 把 508 个场景全部重跑(9 个被重写才发现)。批量跑没暴露这个坑, "
                         "是因为队列给每个 chunk 单独的子 root,天然把范围框住了。")
    ap.add_argument("--workers", type=int, default=1,
                    help="按 scene 并发的进程数(默认 1 = 与并发化之前逐位一致)。"
                         "本步是纯 CPU 单线程 numpy,2026-08-20 实测是全量跑的真瓶颈:"
                         "87k~93k 轨迹的 clip 单个要几分钟、GPU 全程 0%%、180 核只用 1 核。"
                         "每 worker RSS 仅 0.3~0.4 GB。")
    args = ap.parse_args()
    scenes = ([args.scene_dir] if args.scene_dir
              else sorted(glob.glob(os.path.join(args.batch_root, "scene_*"))))
    if args.clips:
        spec = args.clips
        want = ([l.strip() for l in open(spec[1:]) if l.strip()] if spec.startswith("@")
                else [c.strip() for c in spec.split(",") if c.strip()])
        want = set(want)
        scenes = [s for s in scenes if os.path.basename(s)[len("scene_"):] in want]
        missing = want - {os.path.basename(s)[len("scene_"):] for s in scenes}
        if missing:
            print(f"[gt-dense-front] ⚠️ --clips 里有 {len(missing)} 个在 batch-root 下没有 "
                  f"scene_ 目录: {sorted(missing)[:5]}{' ...' if len(missing) > 5 else ''}",
                  flush=True)
    print(f"[gt-dense-front] {len(scenes)} scene(s) cap={args.cap} min_tracks={args.min_tracks} "
          f"tol={args.tol} min_frac={args.min_frac} ambig_margin={args.ambig_margin} "
          f"min_vis_frames={args.min_vis_frames} min_tracks_static={args.min_tracks_static} "
          f"motion_thresh={args.motion_thresh} min_own_tracks={args.min_own_tracks} "
          f"motion_override={'OFF' if args.no_motion_override else 'ON'}", flush=True)
    kw = dict(cap=args.cap, min_tracks=args.min_tracks, tol=args.tol, min_frac=args.min_frac,
              ambig_margin=args.ambig_margin, min_vis_frames=args.min_vis_frames,
              min_tracks_static=args.min_tracks_static, motion_thresh=args.motion_thresh,
              motion_override=not args.no_motion_override, min_own_tracks=args.min_own_tracks)

    def _one(sd):
        """per-scene isolation:一个坏 scene(0 masklet / 退化)不能把整批带走 —— 跳过继续。"""
        print(f"[gt-dense-front] {os.path.basename(sd)}", flush=True)
        try:
            process_scene(sd, kw["cap"], kw["min_tracks"], kw["tol"], kw["min_frac"],
                          kw["ambig_margin"], kw["min_vis_frames"], kw["min_tracks_static"],
                          kw["motion_thresh"], kw["motion_override"],
                          min_own_tracks=kw["min_own_tracks"])
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[gt-dense-front] {os.path.basename(sd)}: FAILED {type(e).__name__}: {e} "
                  f"— skipped", flush=True)

    if args.workers <= 1:
        for sd in scenes:                             # workers=1:与并发化之前逐位一致
            _one(sd)
    else:
        # 每个 scene 完全独立(各读各的 gt npz + masklets,各写各的 dense_tracks_*.npz),
        # 所以按 scene 并发是安全的。这一步是【纯 CPU 单线程 numpy】:2026-08-20 实测它成了
        # 全链路瓶颈 —— 87k~93k 轨迹的 clip 单个要几分钟,而机器 180 核只跑满 1 核,
        # GPU 全程 0%。RSS 仅 0.3~0.4 GB/进程,开并发几乎没有内存代价。
        from concurrent.futures import ProcessPoolExecutor
        print(f"[gt-dense-front] 并发 {args.workers} 个 worker(每 scene 独立)", flush=True)
        with ProcessPoolExecutor(args.workers) as ex:
            list(ex.map(_scene_worker, [(sd, kw) for sd in scenes]))
    print("[gt-dense-front] done", flush=True)


if __name__ == "__main__":
    main()
