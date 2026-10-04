"""构建 G1 模型:Qwen3-VL-8B + <traj> token + 轨迹 embedding 包装 + LoRA。

冻结策略(v1.1 §2):视觉编码器冻结;LLM 走 LoRA(r=128/α=256);
轨迹 token 投影器(TrajectoryTokenProjector)全参训练;base embed_tokens/lm_head 冻结
(<traj> 行永远被 scatter 覆盖,不需训练)。
"""
from __future__ import annotations

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import LoraConfig, get_peft_model

from .modeling_traj import add_traj_token, wrap_traj_embedding
from .traj_encoder import TrajectoryTokenProjector

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def build(model_path: str, arm: str, lora_r: int = 128, lora_alpha: int = 256,
          lora_dropout: float = 0.05, dtype=torch.bfloat16, grad_ckpt: bool = True,
          stage: str = "1b", fusion: str = "add", ch_dim: int = 456,
          use_tsec: bool = False, fp32_encoder: bool = False,
          enc_out_scale: float = 1.0):
    """stage="1b"(默认)= LoRA + 编码器全参(G1 原行为,逐位不变)。
    stage="1a" = **冻结预热**(v1.1 增补,硬门):LLM/视觉全冻结、不挂 LoRA,只训编码器+投影;
    梯度经 masked_scatter 回流(enable_input_require_grads 保证 embed 输出带 grad)。
    fusion="add"(默认)= 现役 9 路相加,逐位不变;"concat" = 消融 #6 的独立子空间拼接臂。
    use_tsec=False(默认)= 编码器不吃绝对秒,逐位不变;True = 多一路 tsec 连续通道。

    fp32_encoder(2026-08-25 加,默认 False = 现役行为):把可训练的投影器保持 fp32。
      为什么需要:纯 bf16 训练时,**初值为 1.0 的参数完全动不了** —— bf16 在 1.0 处的步长是
      2⁻⁷=0.0078,而 AdamW(lr=2e-4)每步更新约 2e-4,每次都被舍回 1.0。实测 21/21 个 run 的
      norm_in.weight / norm_out.weight 训练后**精确等于 1.0、偏离 0.00e+00**。
      能动的参数(std 0.02)也有损:460 步后位移只有 fp32 的 87%、方向余弦 0.924。
      冻结的 LLM 仍是 bf16 —— 这就是标准混合精度;注入处已有 pend.to(emb.dtype) 自动降精度。
    enc_out_scale:末尾 LayerNorm weight 的初值,决定注入向量的范数(= scale·sqrt(hidden))。"""
    proc = AutoProcessor.from_pretrained(model_path)
    proc.tokenizer.padding_side = "left"   # 生成/左 pad 一致
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, dtype=dtype, attn_implementation="sdpa")

    tid = add_traj_token(proc, model)
    wrapper = wrap_traj_embedding(model, tid)

    # 冻结全部 base 权重
    for p in model.parameters():
        p.requires_grad_(False)

    if stage == "1b":
        # LoRA 只作用于 LLM 的线性层名(视觉块用别的名字,不会被命中);
        # 但为绝对安全,显式把 vision tower 排除在 target 之外由 name 天然保证。
        lcfg = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            target_modules=LORA_TARGETS, bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lcfg)
    elif stage != "1a":
        raise ValueError(f"unknown stage {stage!r} (expect '1a' or '1b')")

    # 轨迹 token 投影器(全参),放到与 model 同 device/dtype
    encoder = None
    if arm in ("traj", "traj_only"):
        H = model.config.text_config.hidden_size
        edt = torch.float32 if fp32_encoder else dtype
        encoder = TrajectoryTokenProjector(H, fusion=fusion, ch_dim=ch_dim,
                                           use_tsec=use_tsec,
                                           out_scale=enc_out_scale).to(edt)
        # 顺手把"该对齐到多少"量出来打印:注入向量范数 vs 模型自己的词嵌入范数
        with torch.no_grad():
            W = model.get_input_embeddings().weight
            emb_norm = float(W[:20000].float().norm(dim=-1).median())
        inj_norm = enc_out_scale * (H ** 0.5)
        print(f"[scale] 注入向量范数 {inj_norm:.2f} | 该模型词嵌入行范数中位 {emb_norm:.3f} "
              f"| 比值 {inj_norm/max(emb_norm,1e-9):.1f}x | 对齐所需 enc_out_scale="
              f"{emb_norm/(H ** 0.5):.5f} | encoder dtype={edt}", flush=True)
    if stage == "1a" and encoder is None:
        raise ValueError("stage=1a 只训编码器,arm 必须是 traj/traj_only")

    if grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()   # 输入 embed 需 grad 以支撑 ckpt
    model.config.use_cache = False

    return model, encoder, proc, tid, wrapper


def trainable_report(model, encoder):
    def count(mod):
        t = sum(p.numel() for p in mod.parameters() if p.requires_grad)
        a = sum(p.numel() for p in mod.parameters())
        return t, a
    mt, ma = count(model)
    et, ea = (count(encoder) if encoder is not None else (0, 0))
    return dict(lora_trainable=mt, base_total=ma,
                enc_trainable=et, enc_total=ea,
                total_trainable=mt + et)


def load_adapter_weights(model, ckpt_dir: str) -> int:
    """把 checkpoint 目录里的 LoRA 权重(adapter_model.safetensors)载入**当前已挂好 LoRA 的**模型(2026-09-15,两阶段课程用)。
    先核对 adapter_config 的 r / alpha / target_modules 与当前模型一致,再按键名逐张量 copy_ 并核对:缺键、多键、形状不符直接报错。
    ⚠️ 09-16 修:不用 peft.set_peft_model_state_dict —— 在 torch.distributed 已初始化时它会走张量并行分支,
    导入本环境 transformers 4.57.3 里不存在的 EmbeddingParallel 而崩(CPU 单进程验证时不触发)。返回载入张量数。"""
    import json, os
    from safetensors.torch import load_file
    cfg = json.load(open(os.path.join(ckpt_dir, "adapter_config.json")))
    cur = model.peft_config["default"]
    for k in ("r", "lora_alpha"):
        if cfg.get(k) != getattr(cur, k):
            raise ValueError(f"--init-adapter 配置不符:{k} checkpoint={cfg.get(k)} 当前={getattr(cur, k)}")
    if set(cfg.get("target_modules") or []) != set(cur.target_modules or []):
        raise ValueError(f"--init-adapter target_modules 不符:{cfg.get('target_modules')} vs {cur.target_modules}")
    sd = load_file(os.path.join(ckpt_dir, "adapter_model.safetensors"))
    params = {n: p for n, p in model.named_parameters() if "lora_" in n}
    def to_model_key(k):
        for ab in (".lora_A.", ".lora_B."):
            if ab + "weight" in k:
                return k.replace(ab + "weight", ab + "default.weight")
        return k
    missing = [n for n in params if n not in {to_model_key(k) for k in sd}]
    unknown = [k for k in sd if to_model_key(k) not in params]
    if missing or unknown:
        raise ValueError(f"--init-adapter 键不齐:模型缺 {len(unknown)}(例 {unknown[:2]}),checkpoint 缺 {len(missing)}(例 {missing[:2]})")
    n_ok = 0
    with torch.no_grad():
        for k, v in sd.items():
            p = params[to_model_key(k)]
            if tuple(p.shape) != tuple(v.shape):
                raise ValueError(f"--init-adapter 形状不符 {k}: {tuple(v.shape)} vs {tuple(p.shape)}")
            p.copy_(v.to(device=p.device, dtype=p.dtype))
            if not torch.equal(p.detach().to(v.dtype).cpu(), v):
                raise ValueError(f"--init-adapter 键 {k} 载入后数值不等")
            n_ok += 1
    return n_ok
