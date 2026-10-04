"""把 TrajectoryTokenProjector 接进 Qwen3-VL:轨迹 embedding 在 embed_tokens 查表处
masked_scatter 到 <traj> 占位符,不改 forward、不碰 deepstack。

**位置编码(2026-08-22 已核实,论文里别写成"不碰 3D-RoPE")**:轨迹 token 在位置编码上
被当作**普通文本**处理,三根 RoPE 轴取同一位置。依据两处源码 + 一次实测:
  · processing_qwen3_vl.py 只把 image/video token id 标成模态 1/2,其余一律 0=文本;
    `<|traj_pad|>` 是新加的 special token,不等于那两个 → 恒为 0。
  · get_rope_index 对模态 0 的连续段做 `arange(len)+cur` 再 `.expand(3,-1)` = 三轴同值的 1D 文本位置。
  · 实测(8 帧图 + 15 个 traj 占位):traj_token_id=151669 ≠ image 151655 ≠ video 151656;
    traj 位置模态标记全 0;position id 组内 +1 连续,组间跳 6/7/11 = 中间插的接地文本占位。
理由讲得通:轨迹 token 不是像素网格,没有高/宽两个空间轴可言;时间由 token 自带的时间标量
携带,不靠位置编码。⚠️副作用:视觉一张图消耗的位置数远少于其 token 数(按 max(h,w)//merge
推进),**视觉与轨迹的位置刻度不同** —— 想靠位置对齐"第3帧的图 ↔ t=3 的节点"走不通。

原理:Qwen3VLModel.forward 里 `inputs_embeds = self.get_input_embeddings()(input_ids)`
是唯一的文本查表入口,且发生在图像 masked_scatter **之前**。我们把 embed_tokens 包一层
TrajEmbedding —— 查表后就地把 <traj> 位置替换为轨迹 embedding。之后模型原生做图像
scatter(deepstack 保留)、用 input_ids 算 3D-RoPE(视觉 token 位置保留),全都不受影响。

轨迹 embedding 每步动态注入:训练循环先 encoder 前向得到 [ΣN, h],挂到 wrapper 上,
再 model(**batch);梯度经 masked_scatter 回流到 encoder(可微)。generate 解码阶段
input_ids 长度=1 不含 <traj>,scatter 自然跳过。
"""
from __future__ import annotations

import torch
import torch.nn as nn

TRAJ_TOKEN = "<|traj_pad|>"


class TrajEmbedding(nn.Module):
    """包装原 embed_tokens。查表后把 <traj> 位置替换为待注入的轨迹 embedding。"""

    def __init__(self, base: nn.Embedding, traj_token_id: int):
        super().__init__()
        self.base = base
        self.traj_token_id = traj_token_id
        self._pending = None       # (ΣN, h),训练/推理每步由外部设置

    # 让 tie_weights / 依赖 .weight 的代码继续可用(权重张量与 base 同一份)
    @property
    def weight(self):
        return self.base.weight

    @property
    def num_embeddings(self):
        return self.base.num_embeddings

    @property
    def embedding_dim(self):
        return self.base.embedding_dim

    def set_pending(self, embeds):
        self._pending = embeds

    def clear_pending(self):
        self._pending = None

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        emb = self.base(input_ids)
        pend = self._pending
        if pend is not None and input_ids.dim() >= 1:
            mask = (input_ids == self.traj_token_id)
            if mask.any():
                n_slot = int(mask.sum())
                if n_slot != pend.shape[0]:
                    raise ValueError(
                        f"<traj> 占位符数 {n_slot} 与轨迹 embedding 数 {pend.shape[0]} 不一致")
                m = mask.unsqueeze(-1).expand_as(emb)
                emb = emb.masked_scatter(m, pend.to(emb.dtype))
        return emb


def add_traj_token(processor, model):
    """给 tokenizer 增加 <traj> 特殊 token 并 resize embedding。返回 traj_token_id。"""
    tok = processor.tokenizer
    if TRAJ_TOKEN not in tok.get_vocab():
        tok.add_special_tokens({"additional_special_tokens": [TRAJ_TOKEN]})
        model.resize_token_embeddings(len(tok))
    return tok.convert_tokens_to_ids(TRAJ_TOKEN)


def wrap_traj_embedding(model, traj_token_id: int) -> TrajEmbedding:
    """把 model 的输入 embedding 换成 TrajEmbedding 包装器,返回该包装器(留句柄)。
    必须在 resize_token_embeddings 之后、get_peft_model 之前调用。"""
    base = model.get_input_embeddings()
    if isinstance(base, TrajEmbedding):
        return base
    wrapper = TrajEmbedding(base, traj_token_id)
    model.set_input_embeddings(wrapper)
    return wrapper
