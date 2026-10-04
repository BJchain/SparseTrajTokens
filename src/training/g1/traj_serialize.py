"""SceneTokens (schema v3) → 扁平的类型化 token 记录序列。

这是"轨迹模态"进入 Qwen3-VL 的第一步:把结构化的 SceneTokens 展开成一串定长记录,
每条记录 = 一个 trajectory token,后续由 TrajectoryTokenProjector 映射到 hidden_size 向量,
再 masked_scatter 到 <traj> 占位符处(与视觉 token 注入同款机制)。

设计遵循 Tokenization.md D5 混合编码:
- 连续通道:3D 向量(norm_center / static_point / se3 ref / rotvec / trans / pl 顶点)
  以及 1D 标量(norm_scale / 顶点归一化时间 u)——走 proj,不做 binning,保精度。
- 离散通道:role(槽位角色)/ kind / object_type / knot_type / instance 身份 —— 走 embedding。

每条记录字段(见 TrajToken):
  role      : 槽位角色 id(ROLES)
  vec3      : (3,) 连续坐标;无则 0 且 has_vec3=False
  scalar    : 1 个连续标量(归一化时间 u∈[0,1] 或 norm_scale);无则 0 且 has_scalar=False
  tsec      : **绝对秒**(clip 起点=0;2026-08-22 加)。u 是相对量,3 秒和 30 秒的 clip 里
              u=0.5 是不同时刻,答"多少 m/s""第几秒"这类题不够用。拿不到时间轴的场景
              has_tsec=False(绝不用帧号冒充秒);SCENE 头的 tsec = clip 总时长
  knot      : knot_type id(KNOTS,0=n/a)
  kind      : object kind id(KINDS,0=n/a,仅 OBJ_HDR)
  otype     : object_type id(OTYPES,0=n/a,仅 OBJ_HDR)
  inst      : instance 身份桶(0=n/a,仅 OBJ_HDR;= instance_id % N_INST_BUCKET + 1)
  part      : 运动部件 id 桶(P0;0=n/a,非刚体 keypoint 携带其 part_id 桶;OBJ_HDR 的 scalar 存 n_parts)

OBJ_HDR 的 vec3 槽(2026-08-22 起):刚体放**位姿质量标量** [recon_resid_norm, 0, 0]。
上游 SE(3) 有 23% 的实例误差 >10%bbox 且该字段与真实误差秩相关 0.731 —— 与其让模型
把不可信的位姿当真,不如把质量诚实标出来。拿不到就 has_vec3=False(不发假的 0)。
该槽此前恒为空,所以**不增 token、不改编码器结构**。

相机(K/w2c)不进 trajectory token —— 它是 T1 metric 恢复的 metadata,与 world-frame
运动幻觉无关,排除(用户红线:相机不进 token,image-space 问题走 CoT 投影或 RGB 分支)。

已补(P0,2026-08-04):motion_part 结构 = part 离散通道 + OBJ_HDR n_parts + prompt 嵌套部件接地
(build_traj_block 出 [the person] [part 1] ...)。仍缺(见 ENCODER_TODO_110K.md):occlusion_gaps、
keypoint_id、visibility_summary、instance_id 真实化。
"""
from __future__ import annotations

from dataclasses import dataclass, replace as _dc_replace
from typing import List

import numpy as np

# ---- 离散词表(0 恒为 padding/NA)-----------------------------------------
ROLES = (
    "PAD", "SCENE_HDR", "NORM_CENTER", "NORM_SCALE",
    "OBJ_HDR", "STATIC_POINT", "SE3_REF", "ROT_VERT", "TRANS_VERT", "PL_VERT",
    "OCC_GAP", "VIS_SUMMARY",   # P1/P2:遮挡缺口 / 可见性摘要
)
KNOTS = ("PAD", "smooth", "reversal", "impact", "endpoint")
KINDS = ("PAD", "static", "rigid", "nonrigid")
OTYPES = ("PAD", "rigid", "non-rigid", "unknown")
N_INST_BUCKET = 64          # instance 身份桶(P2 已改 per-scene 顺序 id,不再哈希)
N_PART_BUCKET = 16          # P0:运动部件 id 桶(每物体内相对 part 身份;0=n/a 如静态/刚体单部件)
N_KP_BUCKET = 128           # P1:keypoint 身份桶(跨时间锁定同一关键点)

ROLE2ID = {r: i for i, r in enumerate(ROLES)}
KNOT2ID = {k: i for i, k in enumerate(KNOTS)}
KIND2ID = {k: i for i, k in enumerate(KINDS)}
OTYPE2ID = {o: i for i, o in enumerate(OTYPES)}


@dataclass
class TrajToken:
    role: int
    vec3: np.ndarray          # (3,) float32
    has_vec3: bool
    scalar: float
    has_scalar: bool
    knot: int = 0
    kind: int = 0
    otype: int = 0
    inst: int = 0
    part: int = 0             # P0:运动部件 id 桶(0=n/a;>=1 表示该 token 属第 part-1 个部件)
    kp: int = 0               # P1:keypoint 身份桶(0=n/a)
    group: int = -1           # 物体分组:-1=SCENE 头,>=0=第 group 个物体(实例接地用)
    tsec: float = 0.0         # 绝对秒(clip 起点=0);SCENE 头存 clip 总时长
    has_tsec: bool = False    # 该 token 有没有绝对时间(拿不到时间轴 → False)


_ZERO3 = np.zeros(3, dtype=np.float32)


def _v3(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32).reshape(3)


def _knot_id(k: str) -> int:
    return KNOT2ID.get(k, 0)


def _inst_bucket(instance_id: int) -> int:
    return int(instance_id) % N_INST_BUCKET + 1


def _part_bucket(part_id) -> int:
    """part_id -> [1, N_PART_BUCKET] 桶(物体内相对身份);None/负 -> 0(n/a)。"""
    if part_id is None:
        return 0
    return int(part_id) % N_PART_BUCKET + 1


def _kp_bucket(kp_id) -> int:
    """keypoint_id -> [1, N_KP_BUCKET] 桶;None -> 0(n/a)。"""
    if kp_id is None:
        return 0
    return int(kp_id) % N_KP_BUCKET + 1


def _frame_to_sec(times, f):
    """帧位置(可为浮点节点)→ 绝对秒。times=None 时返回 (0.0, False)。"""
    if times is None or len(times) == 0:
        return 0.0, False
    return float(np.interp(float(f), np.arange(len(times), dtype=float), times)), True


def _unwrap_rotvec(phi):
    """逐节点在 {(θ+2πk)·n} 里挑离上一节点最近的表示,使序列连续(同 tokenizer._unwrap_rotvec)。"""
    out = np.array(phi, dtype=float, copy=True)
    two_pi = 2.0 * np.pi; ks = np.arange(-4, 5)
    for i in range(1, len(out)):
        prev = out[i - 1]; nrm = float(np.linalg.norm(out[i]))
        if nrm < 1e-9:
            pn = float(np.linalg.norm(prev)); out[i] = (prev / pn) * (two_pi * round(pn / two_pi)) if pn > 1e-9 else out[i] * 0.0; continue
        axis = out[i] / nrm; cands = (nrm + two_pi * ks)[:, None] * axis[None, :]
        out[i] = cands[int(np.argmin(np.linalg.norm(cands - prev, axis=1)))]
    return out


def _abs_rot_segments(segs):
    """刚体旋转流改发**绝对**旋转向量(2026-09-14)。SplineRun.nodes 存的是相对 run 首帧 R0 的 φ,而 rot_ref=R0 不进 token,
    模型从相对量无法复原绝对姿态。这里在序列化时合成 R = exp(rot_ref)·exp(φ) → rotvec(主值)并沿节点 2π 展开保持连续;
    rot_ref=None(PL 对照臂/旧口径)原样透传。不改 SplineRun 本体(tcks/rot_ref 仍供精确重建)。"""
    from scipy.spatial.transform import Rotation
    out = []
    for seg in segs:
        rr = getattr(seg, "rot_ref", None)
        if rr is None or not hasattr(seg, "nodes"):
            out.append(seg); continue
        phi = np.asarray(seg.nodes, float).reshape(-1, 3)
        R0 = Rotation.from_rotvec(np.asarray(rr, float))
        absv = np.stack([(R0 * Rotation.from_rotvec(p)).as_rotvec() for p in phi]) if len(phi) else phi
        out.append(_dc_replace(seg, nodes=_unwrap_rotvec(absv)))
    return out


def _emit_pl_segments(records: List[TrajToken], segments, role: int, inst: int, group: int,
                      frame_lo: int = None, frame_hi: int = None, part: int = 0, kp: int = 0,
                      times=None):
    """把一组 PLSegment / SplineRun 的顶点/节点逐个 emit 为 role 类型 token。
    两族**同字段形状**(vec3=位置、scalar=归一化时间 u、knot=离散标志),v4 定案:spline 节点复用
    PL_VERT/ROT_VERT/TRANS_VERT 角色,encoder 零改动;角点=两个共位节点(位置同、knot=事件类型)。
    PLSegment:整数顶点帧,u=线性映射到 [u_start,u_end](v3 原逻辑,逐位未动);
    SplineRun:浮点节点帧 fnodes,u=同一映射的浮点版。
    frame_lo/hi 非空时只 emit 落在 [frame_lo, frame_hi] 内的顶点(区间式问句的公平窗口)。"""
    for seg in segments:
        u0, u1 = float(seg.u_start), float(seg.u_end)
        if hasattr(seg, "fnodes"):                     # ---- SplineRun(v4)
            fn = np.asarray(seg.fnodes, float).reshape(-1)
            verts = np.asarray(seg.nodes, dtype=np.float32).reshape(-1, 3)
            knots = list(seg.knot_types)
            n = len(fn)
            span = max(fn.max() - fn.min(), 1e-9) if n > 1 else 1.0
            for j in range(n):
                if frame_lo is not None and not (frame_lo - 1e-9 <= fn[j] <= frame_hi + 1e-9):
                    continue
                u = u0 + (u1 - u0) * float((fn[j] - fn.min()) / span) if n > 1 else u0
                ts, hts = _frame_to_sec(times, fn[j])
                records.append(TrajToken(
                    role=role, vec3=verts[j], has_vec3=True,
                    scalar=float(u), has_scalar=True,
                    knot=_knot_id(knots[j] if j < len(knots) else "smooth"),
                    inst=inst, part=part, kp=kp, group=group, tsec=ts, has_tsec=hts,
                ))
            continue
        vframes = np.asarray(seg.vframes).reshape(-1)   # ---- PLSegment(v3,原逻辑)
        verts = np.asarray(seg.verts, dtype=np.float32).reshape(-1, 3)
        knots = list(seg.knot_types)
        n = len(vframes)
        span = max(vframes.max() - vframes.min(), 1) if n > 0 else 1
        for j in range(n):
            if frame_lo is not None and not (frame_lo <= int(vframes[j]) <= frame_hi):
                continue          # 区间窗口:跳过窗外顶点
            # 顶点时间 u:优先用整数帧线性映射到 [u0,u1];退化时用 u0
            if n > 1:
                frac = (vframes[j] - vframes.min()) / span
                u = u0 + (u1 - u0) * float(frac)
            else:
                u = u0
            ts, hts = _frame_to_sec(times, vframes[j])
            records.append(TrajToken(
                role=role, vec3=verts[j], has_vec3=True,
                scalar=float(u), has_scalar=True,
                knot=_knot_id(knots[j] if j < len(knots) else "smooth"),
                inst=inst, part=part, kp=kp, group=group, tsec=ts, has_tsec=hts,
            ))


# 与 QA 的 describe_object 完全一致的类别归一化,保证接地文本能对齐问句指代
_CAT_NORM = [
    ("consumer goods", "item"), ("media cases", "media case"),
    ("action figures", "action figure"), ("board games", "board game"),
    ("bottles and cans and cups", "container"), ("stuffed toys", "stuffed toy"),
]


def _obj_label(o) -> str:
    """物体在 prompt 里的接地文本 = "the {category}"(与 QA 的 describe_object 对齐)。"""
    lab = (getattr(o, "label", None) or "object")
    lab = str(lab).strip().lower()
    if lab in ("", "none", "unknown"):
        lab = "object"
    for a, b in _CAT_NORM:
        lab = lab.replace(a, b)
    return f"the {lab}"


def serialize_scene(scene, frame_range=None) -> List[TrajToken]:
    """SceneTokens → List[TrajToken]。顺序:SCENE 头(group=-1) → 逐 object(group=i)。
    frame_range=(t0,t1) 非空时,动态顶点只保留落在 [t0,t1] 的帧(区间式问句的公平窗口);
    静态点/参考点/头不受影响(它们不带帧)。"""
    flo = fhi = None
    if frame_range is not None:
        flo, fhi = int(frame_range[0]), int(frame_range[1])
    recs: List[TrajToken] = []
    # 绝对时间轴(旧 pkl 无此属性 → None → 全体 has_tsec=False,行为与加它之前一致)
    times = getattr(scene, "frame_times_s", None)
    if times is not None:
        times = np.asarray(times, dtype=np.float64).reshape(-1)
        if len(times) == 0:
            times = None
    dur_s = float(times[-1]) if times is not None else 0.0
    abs_rot = getattr(scene, "rigid_param", "origin") == "centroid"   # v4.1 pkl 才发绝对旋转;旧 pkl 逐位保持旧行为
    # --- SCENE header (group=-1);tsec 存 clip 总时长(秒) ---
    recs.append(TrajToken(ROLE2ID["SCENE_HDR"], _ZERO3.copy(), False, float(scene.T), True,
                          group=-1, tsec=dur_s, has_tsec=(times is not None)))
    recs.append(TrajToken(ROLE2ID["NORM_CENTER"], _v3(scene.norm_center), True, 0.0, False, group=-1))
    recs.append(TrajToken(ROLE2ID["NORM_SCALE"], _ZERO3.copy(), False, float(scene.norm_scale), True, group=-1))

    for gi, o in enumerate(scene.objects):
        inst = min(gi, N_INST_BUCKET - 1) + 1     # P2:per-scene 顺序 id(无碰撞,替代哈希)
        n_parts = int(o.motion_part_summary.get("n_parts", 0)) if isinstance(
            getattr(o, "motion_part_summary", None), dict) else 0
        # 刚体位姿质量:塞进本来空着的 vec3 槽(见文件头说明)
        q = getattr(o, "recon_resid_norm", None) if o.kind == "rigid" else None
        qv = _ZERO3.copy()
        if q is not None:
            qv[0] = float(q)
        recs.append(TrajToken(
            role=ROLE2ID["OBJ_HDR"], vec3=qv, has_vec3=(q is not None),
            scalar=float(n_parts), has_scalar=(n_parts > 0),   # P0:OBJ_HDR 携带部件数
            kind=KIND2ID.get(o.kind, 0),
            otype=OTYPE2ID.get(getattr(o, "object_type", "unknown"), OTYPE2ID["unknown"]),
            inst=inst, group=gi,
        ))
        if o.kind == "static":
            pts = o.static_points
            if pts is not None:
                for p in np.asarray(pts, dtype=np.float32).reshape(-1, 3):
                    recs.append(TrajToken(ROLE2ID["STATIC_POINT"], p, True, 0.0, False, inst=inst, group=gi))
        elif o.kind == "rigid":
            se = o.se3_track
            for p in np.asarray(se.ref_norm, dtype=np.float32).reshape(-1, 3):
                recs.append(TrajToken(ROLE2ID["SE3_REF"], p, True, 0.0, False, inst=inst, group=gi))
            _emit_pl_segments(recs, (_abs_rot_segments(se.rot_segments) if abs_rot else se.rot_segments), ROLE2ID["ROT_VERT"],
                              inst, gi, flo, fhi, part=0, times=times)   # v4.1:绝对姿态(2026-09-14);旧 pkl:相对 run 首帧
            _emit_pl_segments(recs, se.trans_segments, ROLE2ID["TRANS_VERT"], inst, gi, flo, fhi,
                              part=0, times=times)
        elif o.kind == "nonrigid":
            # P0:按 part_id 排序使同部件 token 连续(供嵌套接地),每顶点携带 part 桶 + kp 桶。
            kts = sorted(o.keypoint_tracks or [], key=lambda kt: int(getattr(kt, "part_id", 0)))
            for kt in kts:
                pb = _part_bucket(getattr(kt, "part_id", None))
                kb = _kp_bucket(getattr(kt, "keypoint_id", None))
                # P2 可见性摘要:frac(可见占比)进 scalar,vec3=[frac, n_runs, n_visible/T]
                vs = getattr(kt, "visibility_summary", None)
                if isinstance(vs, dict):
                    frac = float(vs.get("frac", 0.0)); nruns = float(vs.get("n_runs", 0))
                    Tn = float(vs.get("n_visible", 0)) / max(1.0, float(vs.get("T", 1)))
                    recs.append(TrajToken(ROLE2ID["VIS_SUMMARY"],
                                          np.array([frac, nruns, Tn], np.float32), True,
                                          frac, True, inst=inst, part=pb, kp=kb, group=gi))
                # P1 遮挡缺口:每个 gap 一个 token,vec3=[u_a,u_b,0](该段轨迹未知,不脑补)
                for (ua, ub) in (getattr(kt, "occlusion_gaps", None) or []):
                    if flo is not None:   # 区间窗口:只留与 [t0,t1] 归一化区间相交的 gap
                        u_lo = flo / max(1.0, float(scene.T) - 1); u_hi = fhi / max(1.0, float(scene.T) - 1)
                        if ub < u_lo or ua > u_hi:
                            continue
                    recs.append(TrajToken(ROLE2ID["OCC_GAP"],
                                          np.array([float(ua), float(ub), 0.0], np.float32), True,
                                          0.0, False, inst=inst, part=pb, kp=kb, group=gi))
                _emit_pl_segments(recs, kt.pl_segments, ROLE2ID["PL_VERT"], inst, gi, flo, fhi,
                                  part=pb, kp=kb, times=times)
    return recs


def serialize_scene_labeled(scene, frame_range=None):
    """返回 (recs, labels):labels[i] = 第 i 个物体的接地文本。"""
    recs = serialize_scene(scene, frame_range=frame_range)
    labels = [_obj_label(o) for o in scene.objects]
    return recs, labels


def group_spans(recs: List[TrajToken]):
    """按出现顺序返回 [(group_id, part_id, count), ...],供 collator 嵌套接地:
    group 变化 -> 插物体 label;part 变化(part>0)-> 插 [part k]。
    group=-1 为 SCENE 头。假设同 (group,part) 记录连续(serialize 已按 part 排序)。"""
    spans = []
    for r in recs:
        if spans and spans[-1][0] == r.group and spans[-1][1] == r.part:
            spans[-1][2] += 1
        else:
            spans.append([r.group, r.part, 1])
    return [(g, p, c) for g, p, c in spans]


def records_to_arrays(recs: List[TrajToken]):
    """List[TrajToken] → 批处理用的 numpy 数组字典(供 collator/encoder 消费)。"""
    n = len(recs)
    out = dict(
        role=np.zeros(n, np.int64),
        vec3=np.zeros((n, 3), np.float32),
        has_vec3=np.zeros(n, np.float32),
        scalar=np.zeros(n, np.float32),
        has_scalar=np.zeros(n, np.float32),
        knot=np.zeros(n, np.int64),
        kind=np.zeros(n, np.int64),
        otype=np.zeros(n, np.int64),
        inst=np.zeros(n, np.int64),
        part=np.zeros(n, np.int64),
        kp=np.zeros(n, np.int64),
        tsec=np.zeros(n, np.float32),
        has_tsec=np.zeros(n, np.float32),
    )
    for i, r in enumerate(recs):
        out["role"][i] = r.role
        out["vec3"][i] = r.vec3
        out["has_vec3"][i] = 1.0 if r.has_vec3 else 0.0
        out["scalar"][i] = r.scalar
        out["has_scalar"][i] = 1.0 if r.has_scalar else 0.0
        out["knot"][i] = r.knot
        out["kind"][i] = r.kind
        out["otype"][i] = r.otype
        out["inst"][i] = r.inst
        out["part"][i] = r.part
        out["kp"][i] = r.kp
        out["tsec"][i] = r.tsec
        out["has_tsec"][i] = 1.0 if r.has_tsec else 0.0
    return out
