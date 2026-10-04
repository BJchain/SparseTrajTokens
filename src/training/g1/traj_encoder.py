"""TrajectoryTokenProjector:类型化 token 记录 → hidden_size 向量序列。

**命名(2026-08-22 定案)**:叫 *Projector* 不叫 *Encoder*。本模块 forward 里每条记录
各算各的 —— 没有 attention、没有卷积、没有池化,**零跨 token 运算**,做的是
`结构化 token → Qwen hidden 空间` 的逐元素映射。轨迹之间的上下文由 Qwen 自身的自注意力
建立,不在这里。写论文时不要叫它 "trajectory (transformer) encoder",那会让审稿人以为
我们对整条轨迹做了 contextualized representation;真要升级成带自注意力的版本再改口。
`TrajectoryEncoder` 保留为向后兼容别名。文件名 `traj_encoder.py` 与检查点名
`traj_encoder.pt` **不改** —— 改了会断路径与命令行参数,而 state_dict 不含类名,换名零收益。

对应 Tokenization.md D5 混合编码的"embedding+投影"落地:
- 离散通道 role/knot/kind/otype/inst/part/kp → nn.Embedding。
- 连续通道 vec3(坐标/rotvec/trans)→ Linear(3→d);scalar(时间 u / norm_scale)→ Linear(1→d);
  用 has_* 门控(缺失置 0),避免"无坐标"槽位注入噪声。
- 融合 → LayerNorm → 2 层 MLP(残差)→ LayerNorm → [N, hidden]。

融合方式(2026-08-21 起可切,消融 #6「Additive vs Concatenation」):
- fusion="add"(默认,与 08-04 起的现役实现**逐位一致**):9 路(7 离散 + vec3 + scalar)
  全部在 hidden_size 维上相加。属性叠在同一个子空间里,交叉信息(如"reversal 且属于 part 3")
  只能靠后面的 MLP 解耦。
- fusion="concat":每路各占 ch_dim 维的**独立子空间**,拼成 9*ch_dim 后一层 Linear 回 hidden。
  ch_dim=456 时两臂输入总宽度基本相当(9*456=4104 ≈ 4096)。
- fusion="add_proj":**参数对齐的对照臂**。相加与 add 完全一样,只在其后多一层 Linear(h→h),
  把 concat 臂多出来的那一层参数补给 add 臂。用途:堵住"concat 只是因为多了 1670 万参数才更好"
  这个漏洞 —— 若 add_proj ≈ concat,收益来自多一层而非独立子空间。
三臂的下游(LayerNorm → MLP 残差 → LayerNorm)完全相同,只隔离"融合方式"这一个变量。

绝对时间通道(2026-08-22,use_tsec,默认关):打开后多一路连续通道 tsec(clip 起点=0 的秒数),
走 proj_tsec 并由 has_tsec 门控。关着时不创建该层 → 与之前逐位一致,所以"有没有绝对秒"
本身就是一条干净的 A/B。

全参训练(视觉编码器冻结,LLM 走 LoRA,本模块和 <traj> 相关权重全参)。
输出直接喂给包装后的 embed_tokens 做 masked_scatter。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .traj_serialize import (ROLES, KNOTS, KINDS, OTYPES, N_INST_BUCKET,
                             N_PART_BUCKET, N_KP_BUCKET)

FUSIONS = ("add", "concat", "add_proj")
N_CHANNELS = 9        # 7 路离散 + vec3 + scalar(打开 use_tsec 再 +1 路 tsec)


class TrajectoryTokenProjector(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.0,
                 fusion: str = "add", ch_dim: int = 456, use_tsec: bool = False,
                 out_scale: float = 1.0):
        super().__init__()
        if fusion not in FUSIONS:
            raise ValueError(f"fusion 必须是 {FUSIONS},收到 {fusion!r}")
        self.hidden_size = hidden_size
        self.fusion = fusion
        self.use_tsec = bool(use_tsec)
        self.n_channels = N_CHANNELS + (1 if use_tsec else 0)
        # add / add_proj 臂:每路都是 hidden 宽(现役行为);只有 concat 臂每路 ch_dim 宽
        d = int(ch_dim) if fusion == "concat" else hidden_size
        self.ch_dim = d
        self.role_emb = nn.Embedding(len(ROLES), d, padding_idx=0)
        self.knot_emb = nn.Embedding(len(KNOTS), d, padding_idx=0)
        self.kind_emb = nn.Embedding(len(KINDS), d, padding_idx=0)
        self.otype_emb = nn.Embedding(len(OTYPES), d, padding_idx=0)
        self.inst_emb = nn.Embedding(N_INST_BUCKET + 1, d, padding_idx=0)
        self.part_emb = nn.Embedding(N_PART_BUCKET + 1, d, padding_idx=0)  # P0
        self.kp_emb = nn.Embedding(N_KP_BUCKET + 1, d, padding_idx=0)      # P1
        self.proj_vec3 = nn.Linear(3, d)
        self.proj_scalar = nn.Linear(1, d)
        # use_tsec=False 时不创建 proj_tsec —— 保证关着时随机数消耗与加它之前完全一致
        self.proj_tsec = nn.Linear(1, d) if use_tsec else None
        # add 臂不创建 fuse —— 保证 add 臂的模块创建顺序/随机数消耗与改动前完全一致
        if fusion == "concat":
            self.fuse = nn.Linear(self.n_channels * d, hidden_size)   # 拼接 → hidden
        elif fusion == "add_proj":
            self.fuse = nn.Linear(hidden_size, hidden_size)      # 参数对齐用的额外一层
        else:
            self.fuse = None
        self.norm_in = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.norm_out = nn.LayerNorm(hidden_size)
        # 末尾 LayerNorm 的 weight 决定注入向量的尺度:weight≡w 时输出范数 = w·sqrt(hidden)。
        # 默认 1.0 → 范数恒 64,而 Qwen3-VL 词嵌入行范数只有 ~1.59(**差 40 倍**)。
        # 2026-08-25 加这个旋钮是为了能把它对齐;默认 1.0 = 现役行为,逐位不变。
        self.out_scale = float(out_scale)
        self._init()

    def _init(self):
        # 初始化顺序与改动前逐字保持一致;fuse 放最后,add 臂上不存在故不消耗随机数
        for m in (self.proj_vec3, self.proj_scalar):
            nn.init.normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)
        for emb in (self.role_emb, self.knot_emb, self.kind_emb, self.otype_emb,
                    self.inst_emb, self.part_emb, self.kp_emb):
            nn.init.normal_(emb.weight, std=0.02)
            with torch.no_grad():
                emb.weight[0].zero_()
        for lin in self.mlp:
            if isinstance(lin, nn.Linear):
                nn.init.normal_(lin.weight, std=0.02)
                nn.init.zeros_(lin.bias)
        for m in (self.fuse, self.proj_tsec):     # 都放最后:关着时不存在,不消耗随机数
            if m is not None:
                nn.init.normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)
        if self.out_scale != 1.0:                # 不消耗随机数;==1.0 时整段跳过,保逐位一致
            nn.init.constant_(self.norm_out.weight, self.out_scale)

    @classmethod
    def from_state_dict(cls, sd: dict, hidden_size: int, dropout: float = 0.0):
        """按 checkpoint 自身还原融合方式与通道宽度(旧 checkpoint 无 fuse.* → add,完全兼容)。"""
        d = int(sd["role_emb.weight"].shape[1])
        use_tsec = "proj_tsec.weight" in sd
        if "fuse.weight" not in sd:
            fusion = "add"
        else:   # fuse 的输入宽度区分两臂:hidden=在相加之后,9*ch_dim=在拼接之后
            fusion = "add_proj" if int(sd["fuse.weight"].shape[1]) == hidden_size else "concat"
        enc = cls(hidden_size, dropout=dropout, fusion=fusion, ch_dim=d, use_tsec=use_tsec)
        enc.load_state_dict(sd)
        return enc

    def forward(self, batch: dict) -> torch.Tensor:
        """batch: 已转 tensor 且在目标 device 上的字段(见 traj_serialize.records_to_arrays)。
        role/knot/kind/otype/inst: (N,) long;vec3:(N,3);has_vec3/has_scalar/scalar:(N,).
        返回 (N, hidden) —— 与 <traj> 占位符总数一致(跨样本已拼接)。"""
        dev = self.role_emb.weight.device
        dt = self.role_emb.weight.dtype
        role = batch["role"].to(dev)
        e_role = self.role_emb(role)
        e_knot = self.knot_emb(batch["knot"].to(dev))
        e_kind = self.kind_emb(batch["kind"].to(dev))
        e_otype = self.otype_emb(batch["otype"].to(dev))
        e_inst = self.inst_emb(batch["inst"].to(dev))
        e_part = self.part_emb(batch["part"].to(dev))
        e_kp = self.kp_emb(batch["kp"].to(dev))
        vec3 = batch["vec3"].to(dev, dt)
        scal = batch["scalar"].to(dev, dt).unsqueeze(-1)
        has3 = batch["has_vec3"].to(dev, dt).unsqueeze(-1)
        has1 = batch["has_scalar"].to(dev, dt).unsqueeze(-1)
        e_vec3 = self.proj_vec3(vec3) * has3
        e_scal = self.proj_scalar(scal) * has1
        chans = [e_role, e_knot, e_kind, e_otype, e_inst, e_part, e_kp, e_vec3, e_scal]
        if self.proj_tsec is not None:
            # 绝对秒:缺时间轴的场景 has_tsec=0,该路被门控掉(不注入假时间)
            tsec = batch["tsec"].to(dev, dt).unsqueeze(-1)
            hts = batch["has_tsec"].to(dev, dt).unsqueeze(-1)
            chans.append(self.proj_tsec(tsec) * hts)
        if self.fusion == "concat":
            base = self.fuse(torch.cat(chans, dim=-1))
        else:
            # 加法顺序与改动前逐字一致(浮点加不满足结合律,顺序即数值)
            base = (e_role + e_knot + e_kind + e_otype + e_inst + e_part + e_kp)
            base = base + e_vec3 + e_scal
            if self.proj_tsec is not None:
                base = base + chans[-1]
            if self.fusion == "add_proj":
                base = self.fuse(base)
        h = self.norm_in(base)
        h = h + self.mlp(h)
        return self.norm_out(h)


# 向后兼容别名(2026-08-22 改名前的旧名字;新代码请用 TrajectoryTokenProjector)
TrajectoryEncoder = TrajectoryTokenProjector
