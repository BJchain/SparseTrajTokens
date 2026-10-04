"""Token schema v4 (split-B 样条 tokenizer 的结构化输出契约;v3=PL 保留为对照臂).

v4(2026-08-20 定案,见 Tokenization.md 文末「定案增补」):
- 轨迹曲线族 = **事件切段三次 B 样条**(splitB:逐事件帧切段、C⁰ 角点、段内三次、阶退化 B 档),
  以 `SplineRun` 表示;token = Greville 节点位置 (fnodes[i], nodes[i]) + knot_types 离散标志,
  与 v3 的 PL 顶点**同字段形状**(vec3 + 标量时间)→ serializer/encoder 结构零改动。
- `KeypointTrack.pl_segments` / `SE3Track.rot_segments/trans_segments` 字段名保留,内容为
  List[PLSegment](curve="pl" 对照臂)或 List[SplineRun](curve="bspline" 生产);两类同名接口
  n_verts()/knot_types/u_start/u_end,token 计数共用。
- SceneTokens 新增 `curve_family` 标记("pl"/"bspline";旧 pkl 无该属性 → 视为 "pl")。

以下为 v3 原头注:

方案原文 .claude/skills/4d-vlm-pipeline/references/Tokenization.md §3 (D7),按已拍板决定改造:
- 轨迹编码 = 分段线性 PL(非贝塞尔),D2 自适应分段 + D3 事件对齐 knot + D4 可见性感知。
- motion_type 简化:只用 object_type∈{rigid,non-rigid} + motion_part(2b 聚类 id)。
- 静态分支:kind=static → static_points(≤4) 直出,不拟合 PL / 不出 SE3。
- placement 改场景级归一化:SCENE header 存 (norm_center, norm_scale),多物体同一归一化系。
- 相机 (K, w2c) 进 SCENE header 供 T1 绝对 metric 恢复;无 GT 的数据集留 None(诚实)。
- 混合编码 D5:连续通道 = PL 顶点坐标 / 静态点 / se3 ref / rotvec / trans(标 continuous,
  留 embedding+投影,tokenizer 内不 binning);离散通道 = id/label/type/part/knot_type/
  可见性 flag/段数(结构 token)。tokenizer 出中间表示,不落具体词表 id。

每字段的 continuous/discrete 通道归属见 CHANNELS 常量(报告①用)。
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

KNOT_TYPES = ("smooth", "reversal", "impact", "endpoint")   # 离散(v4 沿用:角点两侧共位节点带 reversal/impact)
KIND_TYPES = ("static", "rigid", "nonrigid")                 # 离散

# 每字段通道归属(报告①)。continuous = 走 embedding+投影(保精度);discrete = 结构 token 词表。
CHANNELS = {
    # --- SceneTokens ---
    "SceneTokens.T": "discrete",
    "SceneTokens.frame_convention": "discrete(metadata)",
    "SceneTokens.norm_center": "continuous",
    "SceneTokens.norm_scale": "continuous",
    "SceneTokens.camera_K": "continuous(metadata, T1 metric 恢复)",
    "SceneTokens.camera_w2c": "continuous(metadata, T1 metric 恢复)",
    # --- ObjectTokens ---
    "ObjectTokens.instance_id": "discrete",
    "ObjectTokens.label": "discrete",
    "ObjectTokens.object_type": "discrete",
    "ObjectTokens.kind": "discrete",
    "ObjectTokens.motion_part_summary": "discrete",
    "ObjectTokens.static_points": "continuous",
    # --- SE3Track (rigid) ---
    "SE3Track.ref_norm": "continuous",
    "SE3Track.keypoint_ids": "discrete",
    "SE3Track.part_ids": "discrete",
    "SE3Track.rot_segments.verts": "continuous(rotvec so3)",
    "SE3Track.trans_segments.verts": "continuous",
    "SE3Track.*.knot_types": "discrete",
    "SE3Track.occlusion_gaps": "discrete(flag)",
    "SE3Track.visibility_summary": "discrete",
    # --- KeypointTrack (nonrigid) ---
    "KeypointTrack.keypoint_id": "discrete",
    "KeypointTrack.part_id": "discrete",
    "KeypointTrack.pl_segments.verts": "continuous",     # v4 bspline: SplineRun.nodes 同通道
    "SplineRun.nodes": "continuous",
    "SplineRun.fnodes": "continuous(标量时间 u)",
    "SplineRun.degrees": "discrete(metadata, 不进 token)",
    "SplineRun.tcks": "metadata(重建用, 不进 token)",
    "KeypointTrack.pl_segments.knot_types": "discrete",
    "KeypointTrack.occlusion_gaps": "discrete(flag)",
    "KeypointTrack.visibility_summary": "discrete",
}


@dataclass
class PLSegment:
    """一个可见 run 的分段线性折线。verts 是折线顶点(D5 连续通道);knot_types 逐顶点离散标签。
    rot_segments 复用此结构时 verts 为 rotvec(so3, 3 维),坐标含义不同但结构相同。"""
    u_start: float                    # run 归一化帧时间起点 = f0/(T-1)
    u_end: float                      # = f1/(T-1)
    vframes: np.ndarray               # (k,) int   顶点整数帧(精确重建用)
    verts: np.ndarray                 # (k,3) float  连续
    knot_types: List[str]             # (k,) 离散 ∈ KNOT_TYPES

    def n_verts(self) -> int:
        return int(len(self.vframes))


@dataclass
class SplineRun:
    """一个可见 run 的 split-B 样条(v4)。事件帧把 run 切段(相邻段共享事件帧端点=C⁰ 角点),
    每段独立三次(m<5 阶退化 B 档:k=1→k=2→插值)。节点态与控制多边形一一对应(可逆,
    Greville 配置矩阵;实测 max 1.4e-12)。
    - fnodes/nodes/knot_types/degrees:逐节点(按段拼接;角点处两个共位节点);
    - tcks:逐段 (t, c, k, f0, f1),仅供精确重建(QC/viz),不进 token;单点 run 为空表。"""
    u_start: float                    # run 起点 = f0/(T-1)
    u_end: float
    fnodes: np.ndarray                # (n,) float 节点(浮点)帧位置
    nodes: np.ndarray                 # (n,3) float 节点位置(曲线上;连续通道)
    knot_types: List[str]             # (n,) ∈ KNOT_TYPES
    degrees: np.ndarray               # (n,) int 所在段的阶 k(1/2/3;单点 run=0)
    tcks: List[tuple]                 # [(t, c, k, f0, f1)] 逐段
    budget_miss: bool = False
    max_err: float = 0.0
    # 仅刚体**旋转流**用(§2.2 联合拟合,2026-08-22):run 首帧姿态的旋转向量。
    # 此时 nodes 存的是**相对该姿态**的旋转向量 φ,真实姿态 = exp(rot_ref)·exp(φ)。
    # 相对化 + 2π 展开是为了让信号连续 —— 绝对 rotvec 在 π 附近翻轴,样条穿不过去。
    # None = 旧口径(nodes 即绝对旋转向量),反重建自动兼容。**元信息,不进 token。**
    rot_ref: Optional[np.ndarray] = None

    def n_nodes(self) -> int:
        return int(len(self.fnodes))

    def n_verts(self) -> int:         # 与 PLSegment 同名接口(token 计数共用)
        return self.n_nodes()


@dataclass
class KeypointTrack:
    keypoint_id: int
    part_id: int
    pl_segments: List                 # List[PLSegment] (pl) 或 List[SplineRun] (bspline)
    occlusion_gaps: List[Tuple[float, float]]   # [(u_a,u_b), ...] D4 遮挡缺口(不脑补)
    visibility_summary: dict

    def n_vert_tokens(self) -> int:
        return int(sum(s.n_verts() for s in self.pl_segments))


@dataclass
class SE3Track:
    """rigid 物体:全局 SE(3) 序列用 PL 在 SE(3) 上分段。
    旋转走 rotvec(log(R), so3 3 维)PL;平移走归一化 t' 的 PL(见 tokenizer 说明)。
    重建:p'(t) = exp(rotvec(t)) @ ref_norm + trans(t)。"""
    ref_norm: np.ndarray              # (M,3) 归一化参考点。v4.1(2026-09-14)= (ref_positions − 质心)/scale,物体自身形状;
                                      #        旧口径 = ref_positions/scale(世界系,SceneTokens 无 rigid_param 属性时)
    keypoint_ids: List[int]
    part_ids: List[int]
    rot_segments: List                # verts/nodes=rotvec(3,);PLSegment 或 SplineRun
    trans_segments: List              # verts/nodes=归一化平移(3,);v4.1 = 归一化**质心轨迹**(旧口径 = 归一化 se3_t)
    occlusion_gaps: List[Tuple[float, float]]
    visibility_summary: dict

    def n_vert_tokens(self) -> int:
        r = sum(s.n_verts() for s in self.rot_segments)
        t = sum(s.n_verts() for s in self.trans_segments)
        return int(r + t)


@dataclass
class ObjectTokens:
    instance_id: int
    label: str
    object_type: str                  # {rigid, non-rigid} —— Qwen 判定
    kind: str                         # {static, rigid, nonrigid} —— final_repr 实际形态
    motion_part_summary: dict         # {n_parts, part_ids}
    static_points: Optional[np.ndarray] = None       # (<=4,3) 归一化 —— kind=static
    se3_track: Optional[SE3Track] = None             # kind=rigid
    keypoint_tracks: Optional[List[KeypointTrack]] = None  # kind=nonrigid
    # 刚体位姿质量标量(2026-08-22):上游 stage4 的 SE(3) 重建残差。
    # 为什么要带上:上游 SE(3) 本身有 23% 的实例误差 >10%bbox,而这个字段与真实 keypoint
    # 误差秩相关 0.731 —— 诚实标注出来,好过让模型/评测把不可信的位姿当真。
    # None = 拿不到(MOVi 恒 0 是**正确值**:位姿来自 Kubric GT,残差本就是 0)。
    recon_resid_norm: Optional[float] = None


@dataclass
class SceneTokens:
    scene_id: str
    T: int
    frame_convention: str
    norm_center: np.ndarray           # (3,) 逆归一化用:p_metric = p_norm*scale + center
    norm_scale: float
    camera_K: Optional[np.ndarray]    # (3,3) or None
    camera_w2c: Optional[np.ndarray]  # (T,3,4) or None
    objects: List[ObjectTokens] = field(default_factory=list)
    curve_family: str = "pl"          # "pl" | "bspline"(v4;旧 pkl 无此属性=pl)
    # ---- 绝对时间轴(2026-08-22 加)。clip 起点 = 0 秒。 ----
    # None = 该场景拿不到时间轴 → 进隔离名单;**禁止用帧号冒充秒**(v0.3 §1.2-5)。
    # 旧 pkl 无这两个属性,读取方一律用 getattr(scene, "frame_times_s", None) 取。
    frame_times_s: Optional[np.ndarray] = None    # (T,) float32,每帧的绝对秒
    fps_effective: Optional[float] = None         # 有效帧率(Stereo4D 抽帧后 / MOVi 固定 12)
    rigid_param: str = "origin"                   # 刚体参数化:"centroid"(v4.1,2026-09-14)/ "origin"(旧;旧 pkl 无此属性 → getattr 默认)

    # ---------------------------------------------------------- token 预算(探针 B)
    def token_counts(self) -> dict:
        """混合编码 token 计数。
        continuous = 走 embedding+投影的 3D 向量槽数(PL 顶点/静态点/se3 ref/rotvec/trans);
        discrete   = 结构 token 数(id/label/type/part/knot_type/段数/遮挡 flag/头分隔);
        camera     = 单列(metadata, 不算进'替代 RGB'的正文预算)。
        分层:static_* / dynamic_* 贡献。
        """
        c = dict(
            cont_scene=1 + 1,                    # norm_center(1 3-vec 槽) + norm_scale(1)
            disc_scene=3,                        # [SCENE] + T + n_objects
            cont_static=0, disc_static=0,
            cont_dyn=0, disc_dyn=0,
            camera=0, n_static=0, n_rigid=0, n_nonrigid=0,
        )
        if self.camera_K is not None:
            c["camera"] += 1                     # K
        if self.camera_w2c is not None:
            c["camera"] += int(self.camera_w2c.shape[0])   # 逐帧 w2c

        for o in self.objects:
            disc_hdr = 4 + int(o.motion_part_summary.get("n_parts", 0))  # id,label,otype,kind + parts
            if o.kind == "static":
                c["n_static"] += 1
                n = int(len(o.static_points)) if o.static_points is not None else 0
                c["cont_static"] += n
                c["disc_static"] += disc_hdr + 1     # +1: n_points flag
            elif o.kind == "rigid":
                c["n_rigid"] += 1
                se = o.se3_track
                nref = int(len(se.ref_norm))
                nseg = len(se.rot_segments) + len(se.trans_segments)
                nvert = se.n_vert_tokens()
                c["cont_dyn"] += nref + nvert
                # 离散:头 + keypoint_ids + part_ids + 每顶点 knot_type + 段数 + 遮挡 flag
                c["disc_dyn"] += (disc_hdr + nref * 2 + nvert
                                  + nseg + len(se.occlusion_gaps))
            elif o.kind == "nonrigid":
                c["n_nonrigid"] += 1
                c["disc_dyn"] += disc_hdr
                for kt in o.keypoint_tracks:
                    nvert = kt.n_vert_tokens()
                    nseg = len(kt.pl_segments)
                    c["cont_dyn"] += nvert
                    c["disc_dyn"] += (2 + nvert + nseg + len(kt.occlusion_gaps))  # kpid,partid + knots + segs + gaps
        c["continuous_total"] = c["cont_scene"] + c["cont_static"] + c["cont_dyn"]
        c["discrete_total"] = c["disc_scene"] + c["disc_static"] + c["disc_dyn"]
        c["total_no_camera"] = c["continuous_total"] + c["discrete_total"]
        c["total_with_camera"] = c["total_no_camera"] + c["camera"]
        return c


# ------------------------------------------------------------------ 序列化(pickle, numpy 友好)
def save_scene_tokens(scene: SceneTokens, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(scene, f)


def load_scene_tokens(path: str) -> SceneTokens:
    with open(path, "rb") as f:
        return pickle.load(f)
