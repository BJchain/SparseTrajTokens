"""G1 模态对齐 SFT(单臂)。手写循环以完全控制轨迹注入 + 梯度回流。

用法(相对 stage0_4 运行):
  python -m g1.train --arm traj --out runs/g1_traj --epochs 1 ...
两臂各跑一次(rgb / traj),同一 scene 划分、同一超参,只差轨迹通道。

v1.1 增补(2026-08-20):
  1a 冻结预热(硬门): --stage 1a --arm traj --qa data/captions/t8_train.jsonl --allow-open
      LLM 不挂 LoRA、全冻结,只训编码器+投影;数据 = T8 开放式描述;产物只存 encoder(+proc)。
  1b 接续:            --stage 1b --init-encoder runs/g1_1a/final/traj_encoder.pt ...
  模态 dropout(非对称,只丢 RGB): --rgb-dropout 0.15(仅样本 ≥ 万级启用)。
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

sys.path.insert(0, "src/tokenization")
sys.path.insert(0, "src")
import schema  # noqa: F401  (让 pickle 解析 SceneTokens 类)
sys.path.insert(0, "src/training")

from g1.data import G1Dataset, G1Collator, VIDEO_MODES
from g1.split import split3_items
from g1.model_build import build, trainable_report


def set_seed(s):
    import random
    import numpy as np
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def move(batch, device):
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["rgb", "traj", "traj_only"], required=True)
    ap.add_argument("--stage", choices=["1a", "1b"], default="1b",
                    help="1a=冻结预热(只训 encoder,不挂 LoRA);1b=LoRA SFT(默认,原行为)")
    ap.add_argument("--init-encoder", default="",
                    help="1b 用:加载 1a 产出的 traj_encoder.pt 作为编码器初始化")
    ap.add_argument("--allow-open", action="store_true",
                    help="接受开放式条目(T8 描述;默认只收 MCQ)")
    ap.add_argument("--model", default=os.environ.get("QWEN3VL_PATH", "checkpoints/Qwen3-VL-8B-Instruct"))
    ap.add_argument("--qa", nargs="+", default=["data/trajqa_b/final/train.jsonl"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--enc-lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--n-frames", type=int, default=8)
    ap.add_argument("--max-traj", type=int, default=512)
    ap.add_argument("--max-side", type=int, default=384)
    ap.add_argument("--rgb-dropout", type=float, default=0.0)
    ap.add_argument("--fusion", choices=["add", "concat", "add_proj"], default="add",
                    help="编码器融合方式(消融 #6):add=9 路相加(默认,现役);concat=各属性独立子空间拼接;add_proj=参数对齐对照臂(相加后多一层 Linear)")
    ap.add_argument("--ch-dim", type=int, default=456,
                    help="concat 臂每路通道宽度;456 时输入总宽 9*456=4104 与 add 臂的 4096 基本相当")
    ap.add_argument("--video-mode", choices=list(VIDEO_MODES), default="images",
                    help="视觉通道给不给时间:images=不给(默认,现役);tstamp=每帧前插字面秒数"
                         "(视觉 token 数不变,推荐);video=原生视频通道(视觉 token 512→196,"
                         "RGB 基线口径会变,不能与旧结果直接比)")
    ap.add_argument("--fp32-encoder", action="store_true",
                    help="投影器保持 fp32(默认 bf16)。bf16 下初值 1.0 的参数会被舍入完全冻结")
    ap.add_argument("--enc-out-scale", type=float, default=1.0,
                    help="投影器末尾 LayerNorm weight 初值;注入范数=scale*sqrt(4096)。"
                         "1.0=现役(范数 64,是 Qwen 词嵌入的 40 倍);0.0249=对齐词嵌入尺度")
    ap.add_argument("--use-tsec", action="store_true",
                    help="轨迹 token 多喂一路绝对秒(clip 起点=0)。默认关=只有归一化时间 u")
    ap.add_argument("--structured", action="store_true",
                    help="结构化中间监督:目标 = <evidence>GT 轨迹事实</evidence><answer>L</answer>(条目需带 evidence 字段)")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1,
                    help="test 占比;0=不切 test(旧行为)")
    ap.add_argument("--split-mode", choices=["scene", "video"], default="video",
                    help="划分粒度:video=按来源 YouTube 视频分组(110k 默认,防同视频跨集);"
                         "scene=按片段(G1 旧行为,只为复算旧 run)")
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--save-every", type=int, default=0)
    ap.add_argument("--val-qa", nargs="+", default=[],
                    help="显式验证集文件:给了就不再按哈希三分(--qa 全部当训练、这些文件当 val)。B 集(test=u<0.2)必须用它,"
                         "否则其 val(u∈[0.2,0.25))会被并入训练、且 val 为空(2026-09-14 修)")
    ap.add_argument("--dry-run", action="store_true", help="只打印数据划分与配置,不建模型")
    ap.add_argument("--save-every-epochs", type=int, default=0,
                    help="每 N 个 epoch 存一次 checkpoint(按 steps/epoch 换算,与卡数无关;2026-09-15 饱和点搜索用)")
    ap.add_argument("--init-adapter", default="",
                    help="从已有 checkpoint 目录加载 LoRA 权重再训练(两阶段课程的第二阶段;须与本次 LoRA 配置一致)")
    # ---- 2026-09-20:1a 拟合步数诊断(方法见 src/training/1a调查.md)。以下全部默认关 = 原行为逐位不变 ----
    ap.add_argument("--sched", choices=["cosine", "constant"], default="cosine",
                    help="cosine=warmup+余弦(默认,原行为);constant=warmup 后恒定 LR(WSD 的 W+S 段)。"
                         "事先不知道要多少步时用它:中途 checkpoint 之间只差步数,没有 cosine horizon 这个混杂")
    ap.add_argument("--decay-from", type=int, default=0,
                    help="WSD 的 D 段:从这一步起 LR 线性降到 0(配 --decay-steps;通常与 --resume 某个恒定 LR 存档连用)")
    ap.add_argument("--decay-steps", type=int, default=0, help="decay 段长度;须满足 --max-steps = decay_from + decay_steps")
    ap.add_argument("--max-steps", type=int, default=0,
                    help=">0:直接给总优化步数(覆盖 --epochs;数据自动多轮)")
    ap.add_argument("--warmup-steps", type=int, default=0, help=">0:绝对 warmup 步数(覆盖 --warmup 比例)")
    ap.add_argument("--val-every", type=int, default=0,
                    help=">0:每 N 步做一次内联验证诊断(g1.val_diag:按题型拆 loss + 数字 token + donor 对照;"
                         "各卡分片 + all_reduce),逐行写 <out>/val_curve.jsonl;第 0 步也评一次(随机投影器底线 + 尽早暴露验证路径的错)")
    ap.add_argument("--no-val-donor", action="store_true", help="内联验证不算 donor 口径(省一半验证时间)")
    ap.add_argument("--val-donor-subs", nargs="*", default=[],
                    help="donor 与 bf16-whatif 两遍只评这些题型(空=全部)。T8 传 object_motion:inventory 可盲答,评它是浪费")
    ap.add_argument("--val-bf16-whatif", action="store_true",
                    help="--fp32-encoder 时:内联验证额外算一遍「投影器权重舍入到 bf16」的 clean loss(1b 载入时正是这么舍入的)")
    ap.add_argument("--no-grad-ckpt", action="store_true",
                    help="关梯度检查点。1a(LLM 冻结)下重算那遍前向约占 1/3 算力,关掉数值等价、显存换速度")
    ap.add_argument("--answer-logits-only", action="store_true",
                    help="训练前向只在答案段算 logits(logits_to_keep;左 pad 下答案都在序列末尾)。被丢掉的位置标签全是 -100,"
                         "loss 与梯度数学上不变;省掉 (序列长×15 万词表) 的 logits 及其 fp32 副本,约 4–7G 显存")
    ap.add_argument("--stop-at-step", type=int, default=0,
                    help=">0:到这一步就存 final 退出,**调度仍按 total_steps 算**(短程探针用:与某条长 run 的第 N 步同口径)")
    ap.add_argument("--val-limit", type=int, default=0,
                    help=">0:内联验证只用等间隔抽出的 N 条(控制开销;结尾那次旧式 [val] 仍是全量)")
    ap.add_argument("--val-batch", type=int, default=2,
                    help="内联验证的批大小(默认 2 = 历史 [val] 的口径,L_legacy 才能和旧 meta.json 直接比;与训练微批无关)")
    ap.add_argument("--save-optim", action="store_true",
                    help="每次 --save-every 存档时滚动保存续训状态到 <out>/resume/(投影器 + 优化器 + 步数;仅 1a)")
    ap.add_argument("--resume", default="", help="从 <run>/resume 目录续训(仅 1a;LR 按 --sched 从该步继续)")
    args = ap.parse_args()
    # --save-optim / --resume:1a = 滚动存 <out>/resume/(投影器 + 优化器);1b(2026-09-22)= 优化器与步数直接存进每个 step 目录
    # (那里本来就有 LoRA 与投影器),--resume 指向某个 step 目录即可。
    # ---- 多卡数据并行(2026-09-14):torchrun 起就走 DDP,否则单卡路径逐位不变 --------------------------
    # 语义:--batch 是**每卡**微批,全局批 = batch × grad_accum × 卡数(P1 配方 16 → 4 卡时 --batch 2 --grad-accum 2)。
    # 1a(LLM 冻结、只训编码器):只把编码器包进 DDP;1b:LoRA 的 LLM 与编码器各包一个 DDP。
    world = int(os.environ.get("WORLD_SIZE", "1")); rank = int(os.environ.get("RANK", "0")); local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    ddp = world > 1
    if ddp:
        from datetime import timedelta
        dist.init_process_group("nccl", timeout=timedelta(hours=2)); torch.cuda.set_device(local_rank)
    is_main = (rank == 0)
    P = print if is_main else (lambda *a, **k: None)
    if is_main:
        os.makedirs(args.out, exist_ok=True)
    set_seed(args.seed)
    device = f"cuda:{local_rank}" if ddp else "cuda"

    ds = G1Dataset(args.qa, allow_open=args.allow_open)
    if args.val_qa:                                    # 显式文件划分(2026-09-14):不再哈希三分
        train_items = list(ds.items)
        val_items = G1Dataset(args.val_qa, allow_open=args.allow_open).items
        test_items = []
        from g1.split import group_key
        ov = {group_key(d["video_id"]) for d in train_items} & {group_key(d["video_id"]) for d in val_items}
        if ov:
            raise SystemExit(f"--val-qa 与 --qa 有 {len(ov)} 个同视频重叠,拒绝开跑(划分泄漏)")
        split_desc = "files"
    else:
        train_items, val_items, test_items = split3_items(
            ds.items, args.val_frac, args.test_frac, args.split_mode)
        split_desc = args.split_mode
    if args.limit:
        train_items = train_items[:args.limit]
    ds_train = type(ds).__new__(type(ds)); ds_train.items = train_items
    P(f"[data] train={len(train_items)} val={len(val_items)} test={len(test_items)} "
          f"(split={split_desc})", flush=True)
    if args.dry_run:
        P("[dry-run] 数据划分如上,退出", flush=True); return

    # 开跑前护栏:--use-tsec / video-mode 依赖 token pkl 里的时间轴。若 token 目录是没带时间的
    # 旧批,tsec 会恒为 0(has_tsec=0)、时间戳会全体消失,而且**不会报错** —— 在这里挡掉。
    if args.use_tsec or args.video_mode != "images":
        from g1.data import scene_time_axis, STEREO_TOKENS, MOVI_TOKENS
        probe = [it["video_id"] for it in train_items[:50]]
        n_ok = sum(1 for v in probe if scene_time_axis(v)[0] is not None)
        P(f"[time] 抽查 {len(probe)} 个样本:{n_ok} 个有绝对时间轴 "
              f"(token 目录 {STEREO_TOKENS} / {MOVI_TOKENS})", flush=True)
        if n_ok == 0:
            raise SystemExit(
                "--use-tsec / --video-mode 需要带时间轴的 token(SceneTokens.frame_times_s),"
                "但抽查的样本一个都没有。多半是 token 目录是旧批 —— 用 "
                "src/tokenization/build_tokens.py 重建,或设 G1_STEREO_TOKENS 指向新目录。")

    model, encoder, proc, tid, wrapper = build(args.model, args.arm, stage=args.stage,
                                               fusion=args.fusion, ch_dim=args.ch_dim,
                                               use_tsec=args.use_tsec,
                                               fp32_encoder=args.fp32_encoder,
                                               enc_out_scale=args.enc_out_scale,
                                               grad_ckpt=not args.no_grad_ckpt)
    if args.init_encoder:
        sd_enc = torch.load(args.init_encoder, map_location="cpu")
        ck_fusion = "concat" if any(k.startswith("fuse.") for k in sd_enc) else "add"
        ck_dim = int(sd_enc["role_emb.weight"].shape[1])
        if (ck_fusion, ck_dim) != (encoder.fusion, encoder.ch_dim):
            raise ValueError(
                f"--init-encoder 的 checkpoint 是 {ck_fusion}/ch_dim={ck_dim},"
                f"本次 build 的是 {encoder.fusion}/ch_dim={encoder.ch_dim} —— 两臂不能混用。")
        encoder.load_state_dict(sd_enc)
        P(f"[init] encoder <- {args.init_encoder}", flush=True)
    if args.init_adapter:                               # 两阶段课程:第一阶段的 LoRA 权重接着训(2026-09-15)
        from g1.model_build import load_adapter_weights
        n_loaded = load_adapter_weights(model, args.init_adapter)
        P(f"[init] adapter <- {args.init_adapter}({n_loaded} 个张量)", flush=True)
    model.to(device)
    if encoder is not None:
        encoder.to(device)
    rep = trainable_report(model, encoder)
    P(f"[params] LoRA={rep['lora_trainable']/1e6:.2f}M enc={rep['enc_trainable']/1e6:.2f}M "
          f"total_trainable={rep['total_trainable']/1e6:.2f}M base={rep['base_total']/1e9:.2f}B", flush=True)

    col = G1Collator(proc, tid, args.arm, n_frames=args.n_frames,
                     max_traj=args.max_traj, max_side=args.max_side,
                     rgb_dropout=args.rgb_dropout, seed=args.seed + rank,
                     video_mode=args.video_mode, structured=args.structured)
    sampler = (DistributedSampler(ds_train, num_replicas=world, rank=rank, shuffle=True, seed=args.seed, drop_last=True)
               if ddp else None)
    dl = DataLoader(ds_train, batch_size=args.batch, shuffle=(sampler is None), sampler=sampler,
                    collate_fn=col, num_workers=4, drop_last=True)

    model_params = [p for p in model.parameters() if p.requires_grad]
    params = []
    if model_params:                                   # 1a 时 model 无可训参数
        params.append({"params": model_params, "lr": args.lr})
    if encoder is not None:
        params.append({"params": list(encoder.parameters()), "lr": args.enc_lr})
    opt = torch.optim.AdamW(params, weight_decay=0.0, betas=(0.9, 0.95))
    raw_model, raw_encoder = model, encoder                 # 存盘/验证用未包装的模块
    if ddp:
        if model_params:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False, find_unused_parameters=True)
        if encoder is not None:
            encoder = DDP(encoder, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False, find_unused_parameters=True)
        P(f"[ddp] world={world} 每卡批={args.batch} accum={args.grad_accum} 全局批={args.batch * args.grad_accum * world}", flush=True)

    steps_per_epoch = math.ceil(len(dl) / args.grad_accum)
    total_steps = args.max_steps if args.max_steps > 0 else int(steps_per_epoch * args.epochs)
    warmup_steps = args.warmup_steps if args.warmup_steps > 0 else max(1, int(total_steps * args.warmup))

    if args.decay_steps and (args.sched != "constant" or total_steps != args.decay_from + args.decay_steps):
        raise SystemExit("--decay-steps 需要 --sched constant 且 --max-steps = --decay-from + --decay-steps")

    def lr_at(step):
        if step < warmup_steps:
            return step / warmup_steps
        if args.decay_steps and step >= args.decay_from:   # WSD 的 D 段:线性降到 0
            return max(0.0, 1.0 - (step - args.decay_from) / args.decay_steps)
        if args.sched == "constant":                       # WSD 的稳定段:warmup 后恒定
            return 1.0
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    P(f"[sched] {args.sched} steps/epoch={steps_per_epoch} total_steps={total_steps} warmup={warmup_steps} "
      f"(world={world}, grad_ckpt={not args.no_grad_ckpt})", flush=True)
    save_every = args.save_every
    if args.save_every_epochs:
        save_every = steps_per_epoch * args.save_every_epochs
        P(f"[save] 每 {args.save_every_epochs} epoch = {save_every} 步存一次", flush=True)

    model.train()
    if encoder is not None:
        encoder.train()
    t0 = time.time()
    opt_step = 0
    micro = 0
    running = 0.0
    max_epochs_iter = float("inf") if args.max_steps > 0 else args.epochs
    epoch = 0
    stop = False

    # ---- 续训(2026-09-20,仅 1a):投影器 + 优化器 + 步数;数据顺序按 (seed, epoch) 重放并跳过已消耗的样本 ----
    skip_rank_samples = 0
    if args.resume:
        st = json.load(open(os.path.join(args.resume, "trainer_state.json")))
        if args.stage == "1b":                              # LoRA 权重(按键名逐张量核对,见 model_build.load_adapter_weights)
            from g1.model_build import load_adapter_weights
            P(f"[resume] adapter <- {args.resume}({load_adapter_weights(raw_model, args.resume)} 个张量)", flush=True)
        if raw_encoder is not None:
            raw_encoder.load_state_dict(torch.load(os.path.join(args.resume, "traj_encoder.pt"), map_location="cpu"))
        opt.load_state_dict(torch.load(os.path.join(args.resume, "optim.pt"), map_location="cpu"))
        opt_step, epoch = int(st["opt_step"]), int(st["epoch"])
        micro = opt_step * args.grad_accum
        skip_rank_samples = (int(st["samples_in_epoch_global"]) // world) // args.batch * args.batch
        P(f"[resume] <- {args.resume}:opt_step={opt_step} epoch={epoch} 本 epoch 每卡跳过 {skip_rank_samples} 样本", flush=True)
    log = open(os.path.join(args.out, "train_log.jsonl"), "a" if args.resume else "w") if is_main else open(os.devnull, "w")

    # ---- 内联验证诊断(2026-09-20):每 --val-every 步,全部进程一起评(各评一片),主进程落盘 ----
    vd = None
    if args.val_every and val_items:
        from g1.val_diag import ValDiag
        vd_items = val_items
        if args.val_limit and len(val_items) > args.val_limit:
            vd_items = [val_items[int(i * len(val_items) / args.val_limit)] for i in range(args.val_limit)]
        vd = ValDiag(proc, tid, args.arm, vd_items, n_frames=args.n_frames, max_traj=args.max_traj,
                     max_side=args.max_side, video_mode=args.video_mode, structured=args.structured,
                     batch=args.val_batch, rank=rank, world=world, donor=not args.no_val_donor,
                     donor_subs=args.val_donor_subs or None)
        P(f"[valdiag] 每 {args.val_every} 步;val={len(vd_items)}/{len(val_items)} 题型={vd.subs[1:]} donor 池={vd.donor_stats}", flush=True)
    snap = {}                                              # 主进程:投影器参数快照(CPU),量"这 N 步参数到底动了多少"

    def do_val(step):
        if vd is None:
            return
        t_v = time.time()
        rec = dict(step=step, sec=round(time.time() - t0, 1), **vd.run(raw_model, raw_encoder, wrapper, device))
        if args.val_bf16_whatif and args.fp32_encoder and raw_encoder is not None:
            import copy
            e16 = copy.deepcopy(raw_encoder).to(torch.bfloat16)
            rec["clean_bf16cast"] = vd.run(raw_model, e16, wrapper, device, conds=("clean",),
                                           subs=(set(args.val_donor_subs) or None))["clean"]
            del e16
        if is_main and raw_encoder is not None:
            cur = torch.cat([p.detach().float().flatten().cpu() for p in raw_encoder.parameters()])
            snap.setdefault("init", cur)
            if "prev" in snap:
                d = cur - snap["prev"]
                rec["param"] = dict(d_rms_since_prev=float(d.pow(2).mean().sqrt()),
                                    frac_changed_since_prev=round(float((d != 0).float().mean()), 5),
                                    d_rms_since_init=float((cur - snap["init"]).pow(2).mean().sqrt()),
                                    w_rms=float(cur.pow(2).mean().sqrt()))
            snap["prev"] = cur
        rec["val_sec"] = round(time.time() - t_v, 1)
        if is_main:
            try:
                with open(os.path.join(args.out, "val_curve.jsonl"), "a") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except OSError as e:                            # 写不进盘也要把数字留在 stdout 里
                print(f"[valdiag] ❌ val_curve.jsonl 写失败({e});本次记录:{json.dumps(rec, ensure_ascii=False)}", flush=True)
            c, dn, g = rec["clean"], rec.get("donor", {}), rec.get("gap", {})
            P(f"[valdiag] step={step} L_legacy={c['L_legacy']} "
              + " ".join(f"L_tok/{s}={c.get('L_tok/' + s)}(gap {g.get('L_tok/' + s)})" for s in vd.subs[1:])
              + " | 首位命中 clean/donor: "
              + " ".join(f"{t}={c.get(f'num_{t}/first_acc')}/{dn.get(f'num_{t}/first_acc')}" for t in ("disp", "speed", "frame"))
              + f" | {rec['val_sec']}s", flush=True)
        torch.cuda.empty_cache()

    if vd is not None and not args.resume:
        do_val(0)                                          # 第 0 步:随机投影器底线;也让验证路径的错在 2 分钟内暴露而不是第 1000 步
    stop_file = os.path.join(args.out, "STOP")             # touch 它 → 在下一个 50 步边界存档 + 验证后干净退出(恒定 LR 下任意步都是合法终点)
    stopped_early = False
    micro_in_epoch = 0
    while not stop:
        if sampler is not None:
            sampler.set_epoch(epoch)
        cur_dl, micro_in_epoch = dl, 0
        if skip_rank_samples and sampler is not None:      # 续训的第一个 epoch:跳过本卡已消耗的样本
            idx = list(iter(sampler))[skip_rank_samples:]
            cur_dl = DataLoader(ds_train, batch_size=args.batch, sampler=idx, collate_fn=col, num_workers=4, drop_last=True)
            micro_in_epoch, skip_rank_samples = skip_rank_samples // args.batch, 0
        for batch in cur_dl:
            traj_batch = batch.pop("_traj_batch", None)
            batch.pop("_n_traj_total", None)
            batch = move(batch, device)
            sync = ((micro + 1) % args.grad_accum == 0)      # 非同步微步不做 all-reduce(no_sync)
            with contextlib.ExitStack() as stack:
                if ddp and not sync:
                    for m in (model, encoder):
                        if isinstance(m, DDP):
                            stack.enter_context(m.no_sync())
                if args.arm in ("traj", "traj_only"):
                    tb = {k: v.to(device) for k, v in traj_batch.items()}
                    traj_emb = encoder(tb)              # (ΣN, H) 需 grad
                    wrapper.set_pending(traj_emb)
                lab, kw = batch["labels"], {}
                if args.answer_logits_only:                 # 末 k 位 = 最长答案 + 1(多的那 1 位是预测首个答案 token 的位置)
                    k = min(int((lab != -100).sum(1).max()) + 1, lab.shape[1])
                    lab, kw = lab[:, -k:], dict(logits_to_keep=k)
                out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                            pixel_values=batch.get("pixel_values"),
                            image_grid_thw=batch.get("image_grid_thw"),
                            labels=lab, **kw)
                loss = out.loss / args.grad_accum
                loss.backward()
            if args.arm in ("traj", "traj_only"):
                wrapper.clear_pending()
            running += float(out.loss)
            micro += 1
            micro_in_epoch += 1
            if micro % args.grad_accum == 0:
                cur_scale = lr_at(opt_step)
                base_lrs = ([args.lr] if model_params else []) + \
                           ([args.enc_lr] if encoder is not None else [])
                for g, b in zip(opt.param_groups, base_lrs):
                    g["lr"] = b * cur_scale
                if model_params:
                    torch.nn.utils.clip_grad_norm_(model_params, 1.0)
                if encoder is not None:
                    torch.nn.utils.clip_grad_norm_(raw_encoder.parameters(), 1.0)
                opt.step(); opt.zero_grad(set_to_none=True)
                opt_step += 1
                if opt_step % args.log_every == 0:
                    avg = running / (args.grad_accum * args.log_every)
                    mem = torch.cuda.max_memory_allocated() / 1e9
                    if ddp and args.no_grad_ckpt:          # 关检查点后显存是硬约束:记所有卡里的最大峰值,不只主卡
                        mt = torch.tensor([mem], device=device); dist.all_reduce(mt, op=dist.ReduceOp.MAX); mem = float(mt)
                    rec = dict(step=opt_step, loss=round(avg, 4),
                               lr=round(opt.param_groups[0]["lr"], 8), mem_gb=round(mem, 1),
                               sec=round(time.time() - t0, 1))
                    try:                                    # 盘满时写日志不该杀掉训练(2026-09-20:/data 被他人以 ~12G/h 写满中)
                        P(f"[train] {rec}", flush=True)
                        log.write(json.dumps(rec) + "\n"); log.flush()
                    except OSError:
                        pass
                    running = 0.0
                do_save = bool(save_every and opt_step % save_every == 0 and opt_step < total_steps)
                do_valnow = bool(vd is not None and (opt_step % args.val_every == 0 or opt_step == args.stop_at_step))
                if opt_step % 50 == 0 and (args.val_every or args.save_optim):      # STOP 文件:主进程看,广播给所有进程
                    flag = torch.tensor([1.0 if (is_main and os.path.exists(stop_file)) else 0.0], device=device)
                    if ddp:
                        dist.broadcast(flag, 0)
                    if flag.item() > 0 and opt_step < total_steps:
                        stopped_early, do_save, do_valnow = True, True, (vd is not None)
                        P(f"[stop] 发现 {stop_file},在第 {opt_step} 步存档 + 验证后退出", flush=True)
                if do_save and is_main:
                    try:                                    # 中途存档失败(共享盘写满等)不该杀掉长跑:报错、跳过、继续训
                        _save(args.out, raw_model, raw_encoder, proc, f"step{opt_step}", stage=args.stage)
                        if args.save_optim and args.stage == "1b":
                            sd = os.path.join(args.out, f"step{opt_step}")
                            torch.save(opt.state_dict(), os.path.join(sd, "optim.pt"))
                            with open(os.path.join(sd, "trainer_state.json"), "w") as f:
                                json.dump(dict(opt_step=opt_step, epoch=epoch, world=world, batch=args.batch,
                                               grad_accum=args.grad_accum, seed=args.seed, sched=args.sched,
                                               samples_in_epoch_global=micro_in_epoch * args.batch * world), f, indent=1)
                        elif args.save_optim:
                            _save_resume(args.out, raw_encoder, opt, dict(
                                opt_step=opt_step, epoch=epoch, world=world, batch=args.batch, grad_accum=args.grad_accum,
                                samples_in_epoch_global=micro_in_epoch * args.batch * world, seed=args.seed, sched=args.sched))
                    except Exception as e:                  # noqa: BLE001
                        print(f"[save] ❌ 第 {opt_step} 步存档失败,继续训练:{type(e).__name__}: {e}", flush=True)
                if do_valnow:
                    do_val(opt_step)
                if opt_step >= total_steps or stopped_early or (args.stop_at_step and opt_step >= args.stop_at_step):
                    stop = True; break
        epoch += 1
        if epoch >= max_epochs_iter:
            stop = True
    if is_main:
        _save(args.out, raw_model, raw_encoder, proc, "final", stage=args.stage)
    log.close()
    model, encoder = raw_model, raw_encoder                # 以下验证/存盘只用未包装模块(主进程单独做,不含集合通信)

    # ---- 验证集 loss(2026-08-22 加):消融的可比数字 ----------------------------------
    # 原来只记训练 loss,两条臂之间没有任何可比量。1a 是开放式生成(T8 描述),MCQ 那套
    # 准确率用不了;同数据同任务下的 val loss 是最直接、最灵敏的对照量。
    # 逐位提醒:val 划分由 --seed 决定,**对照的几条臂必须用同一个 seed**,否则比的是划分不是臂。
    val_loss = None
    if val_items:
        ds_val = type(ds).__new__(type(ds)); ds_val.items = val_items[rank::world] if ddp else val_items   # 多卡:各评一片,末尾 all_reduce
        dl_val = DataLoader(ds_val, batch_size=(args.val_batch if args.val_every else args.batch), shuffle=False,   # 开了内联验证就与它同口径(批 2 = 历史口径)
                            collate_fn=col, num_workers=2, drop_last=False)
        model.eval()
        if encoder is not None:
            encoder.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for batch in dl_val:
                tb = batch.pop("_traj_batch", None)
                batch.pop("_n_traj_total", None)
                batch = move(batch, device)
                if args.arm in ("traj", "traj_only"):
                    wrapper.set_pending(encoder({k: v.to(device) for k, v in tb.items()}))
                out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                            pixel_values=batch.get("pixel_values"),
                            image_grid_thw=batch.get("image_grid_thw"),
                            labels=batch["labels"])
                if args.arm in ("traj", "traj_only"):
                    wrapper.clear_pending()
                bs = batch["input_ids"].shape[0]
                tot += float(out.loss) * bs; n += bs
        if ddp:
            agg = torch.tensor([tot, float(n)], dtype=torch.float64, device=device)
            dist.all_reduce(agg); tot, n = float(agg[0]), int(agg[1])
        val_loss = round(tot / max(n, 1), 5)
        P(f"[val] loss={val_loss} on {n} samples (seed={args.seed}, world={world})", flush=True)

    # 存划分与元信息供 eval 复用
    if is_main:
      with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(dict(arm=args.arm, stage=args.stage, tid=tid, val_qa=list(args.val_qa), val_frac=args.val_frac, world_size=world,
                       global_batch=args.batch * args.grad_accum * world,
                       test_frac=args.test_frac, split_mode=args.split_mode,
                       n_frames=args.n_frames, max_traj=args.max_traj,
                       max_side=args.max_side, seed=args.seed,
                       fusion=args.fusion, ch_dim=args.ch_dim,
                       video_mode=args.video_mode, use_tsec=bool(args.use_tsec), structured=bool(args.structured),
                       fp32_encoder=bool(args.fp32_encoder), enc_out_scale=args.enc_out_scale,
                       train=len(train_items), val=len(val_items), test=len(test_items),
                       sched=args.sched, warmup_steps=warmup_steps, grad_ckpt=not args.no_grad_ckpt,
                       enc_lr=args.enc_lr, lr=args.lr, limit=args.limit,
                       opt_steps_done=opt_step, stopped_early=stopped_early, resumed_from=args.resume,
                       stop_at_step=args.stop_at_step, init_encoder=args.init_encoder,
                       decay_from=args.decay_from, decay_steps=args.decay_steps,
                       total_steps=total_steps, val_loss=val_loss), f, indent=1)
    P(f"[done] {args.arm} out={args.out} val_loss={val_loss} "
          f"sec={round(time.time()-t0,1)}", flush=True)
    if ddp:
        dist.barrier(); dist.destroy_process_group()


def _save_resume(out, encoder, opt, state):
    """滚动保存续训状态(只留最新一份):先写 resume.tmp,再原子换名,写到一半被杀也不会毁掉上一份。"""
    import shutil
    tmp, dst, old = (os.path.join(out, n) for n in ("resume.tmp", "resume", "resume.old"))
    shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
    torch.save(encoder.state_dict(), os.path.join(tmp, "traj_encoder.pt"))
    torch.save(opt.state_dict(), os.path.join(tmp, "optim.pt"))
    with open(os.path.join(tmp, "trainer_state.json"), "w") as f:
        json.dump(state, f, indent=1)
    shutil.rmtree(old, ignore_errors=True)
    if os.path.isdir(dst):
        os.rename(dst, old)
    os.rename(tmp, dst)
    shutil.rmtree(old, ignore_errors=True)


def _save(out, model, encoder, proc, tag, stage="1b"):
    d = os.path.join(out, tag)
    os.makedirs(d, exist_ok=True)
    if stage == "1b":
        model.save_pretrained(d)                # LoRA adapter(1a 无 adapter,base 17G 不落盘)
    if encoder is not None:
        torch.save(encoder.state_dict(), os.path.join(d, "traj_encoder.pt"))
    proc.save_pretrained(d)


if __name__ == "__main__":
    main()
