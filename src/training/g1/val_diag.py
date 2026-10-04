"""训练中的内联验证诊断(2026-09-20,1a 拟合步数调查用;方法见 src/training/1a调查.md)。

为什么不只看一个 val loss:
  · T8 的 scene_inventory(验证集 649/1033)只看 prompt 的文本骨架就能 100% 盲答(见 eval_val_loss.py 头注),
    总 val loss 有一半落在"不读 token 也能答"的题上,会把"读没读 token"摊薄;
  · 09-13 已证实 1a 的 t8_probe(MCQ)量的是格式/偏好而不是内容(换成别的场景的轨迹,分数纹丝不动)。
⇒ 这里按题型拆开,并对答案里的**数字**(object_motion:位移 m / 速度 m/s / 反转帧号)单独记 teacher-forced 的
  NLL、逐 token top-1、首位数字命中率、整个数字全对率;每个量都同时算 **donor 口径**
  (eval_permute.Corruptor:轨迹内容换成别的场景、格式逐位不变)。clean 与 donor 之差 = 投影器 + 冻结 LLM
  真正从 token **内容**里读到的量;它随步数的曲线就是"1a 拟合到哪了"。
口径:
  · L_legacy 与 train.py 末尾那段 [val] 同构(逐批 token 平均、按批大小加权),可与历史 meta.json 的 val_loss 直接比;
  · L_tok/* 是逐 token 平均(不受批内配对影响,更稳)。
  · teacher forcing 下答案模板是确定的,所以**第一个数字的首位**与自由生成等价;后续位以 GT 前缀为条件。
多卡:各进程评 val_items[rank::world],末尾一次 all_reduce(sum);单卡路径无集合通信。
"""
from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import G1Dataset, G1Collator

NUM_TYPES_OM = ("disp", "speed", "frame")       # object_motion 答案里数字的固定出现顺序(模板决定)


def _number_spans(ids, is_digit, dot_id):
    """标签 id 序列 → [(start, end_exclusive)]:极大的「数字 token 或夹在数字间的 '.'」连续段(句末句号不算)。"""
    spans, i, n = [], 0, len(ids)
    while i < n:
        if is_digit[ids[i]]:
            j = i + 1
            while j < n and (is_digit[ids[j]] or (ids[j] == dot_id and j + 1 < n and is_digit[ids[j + 1]])):
                j += 1
            spans.append((i, j)); i = j
        else:
            i += 1
    return spans


class ValDiag:
    def __init__(self, proc, tid, arm, val_items, *, n_frames, max_traj, max_side, video_mode, structured,
                 batch=2, rank=0, world=1, donor=True, donor_seed=0, num_workers=2, donor_subs=None):
        """donor_subs:donor 口径只评这些题型(None=全部)。T8 上传 ["object_motion"]:scene_inventory 可盲答、
        与 token 内容无关,donor 那一遍评它纯属浪费(占验证集 63%)。供体池始终在**整份** val 上建。"""
        self.arm, self.rank, self.world, self.batch, self.num_workers = arm, rank, world, batch, num_workers
        self.inject = arm in ("traj", "traj_only")
        tok = proc.tokenizer
        self.is_digit = [False] * len(tok)
        for d in "0123456789":
            self.is_digit[tok.convert_tokens_to_ids(d)] = True
        self.dot_id = tok.convert_tokens_to_ids(".")
        self.subs = ["all"] + sorted({d.get("event_subtype") or "?" for d in val_items})
        self.numtypes = list(NUM_TYPES_OM) + ["other"]
        self.donor_subs = set(donor_subs) if donor_subs else None
        self.sets = {"clean": list(val_items)}
        ck = dict(n_frames=n_frames, max_traj=max_traj, max_side=max_side, rgb_dropout=0.0, seed=0,
                  video_mode=video_mode, structured=structured)
        self.cols = {"clean": G1Collator(proc, tid, arm, **ck)}
        self.donor_stats = None
        if donor and self.inject:
            from .eval_permute import Corruptor            # 供体池在**整份** val 上建(各进程相同 → 供体确定、与卡数无关)
            cor = Corruptor("donor", donor_seed)
            cor.build_pool(val_items, max_traj, whole=False)
            self.cols["donor"] = G1Collator(proc, tid, arm, traj_fn=cor, **ck)
            self.sets["donor"] = [d for d in val_items
                                  if self.donor_subs is None or (d.get("event_subtype") or "?") in self.donor_subs]
            self.donor_stats = dict(pool_scenes=cor.stats["pool_scenes"], pool_keys=cor.stats["pool_keys"])
        self.keys = []
        for c in self.cols:
            self.keys += [f"{c}|legacy_x_bs", f"{c}|n"]
            for s in self.subs:
                self.keys += [f"{c}|{s}|{f}" for f in ("nll", "ntok", "dig_nll", "dig_n", "dig_ok")]
            for t in self.numtypes:
                self.keys += [f"{c}|num_{t}|{f}" for f in ("n", "first_ok", "em", "nll", "ntok")]

    @torch.no_grad()
    def run(self, model, encoder, wrapper, device, conds=None, subs=None):
        """model / encoder 传**未包 DDP** 的模块(验证内不含逐批集合通信)。返回 {clean:{}, donor:{}, gap:{}}。
        conds:只算其中几种口径(如 ("clean",));subs:只评这些题型。键全集不变,所以多卡 all_reduce 的形状不受影响。"""
        was_m = model.training
        was_e = encoder.training if encoder is not None else False
        model.eval()
        if encoder is not None:
            encoder.eval()
        acc = {k: 0.0 for k in self.keys}
        for cond, col in self.cols.items():
            if conds is not None and cond not in conds:
                continue
            its = [d for d in self.sets[cond] if subs is None or (d.get("event_subtype") or "?") in subs]
            ds = G1Dataset.__new__(G1Dataset)
            ds.items = its[self.rank::self.world] if self.world > 1 else its     # 过滤后再分片 → 各卡负载均匀
            if not ds.items:
                continue
            dl = DataLoader(ds, batch_size=self.batch, shuffle=False, collate_fn=col,
                            num_workers=self.num_workers, drop_last=False)
            pos = 0                                          # 不打乱、不丢尾 → 第 pos 个样本就是 ds.items[pos]
            for batch in dl:
                tb = batch.pop("_traj_batch", None); batch.pop("_n_traj_total", None)
                batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
                if self.inject and tb is not None:
                    wrapper.set_pending(encoder({k: v.to(device) for k, v in tb.items()}))
                out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                            pixel_values=batch.get("pixel_values"), image_grid_thw=batch.get("image_grid_thw"))
                if self.inject:
                    wrapper.clear_pending()
                labels = batch["labels"]
                sl, tl = out.logits[:, :-1], labels[:, 1:]
                mask = tl != -100
                sel, tgt = sl[mask].float(), tl[mask]
                nll = F.cross_entropy(sel, tgt, reduction="none")
                ok = (sel.argmax(-1) == tgt)
                bs = labels.shape[0]
                acc[f"{cond}|legacy_x_bs"] += float(nll.mean()) * bs
                acc[f"{cond}|n"] += bs
                cnt = mask.sum(1).tolist()
                nll_c, ok_c, tgt_c = nll.cpu().tolist(), ok.cpu().tolist(), tgt.cpu().tolist()
                o = 0
                for bi, n_i in enumerate(cnt):
                    ids, nl, okk = tgt_c[o:o + n_i], nll_c[o:o + n_i], ok_c[o:o + n_i]
                    o += n_i
                    sub = ds.items[pos + bi].get("event_subtype") or "?"
                    dig = [k for k, t in enumerate(ids) if self.is_digit[t]]
                    for s in ("all", sub):
                        acc[f"{cond}|{s}|nll"] += sum(nl); acc[f"{cond}|{s}|ntok"] += n_i
                        acc[f"{cond}|{s}|dig_nll"] += sum(nl[k] for k in dig)
                        acc[f"{cond}|{s}|dig_n"] += len(dig)
                        acc[f"{cond}|{s}|dig_ok"] += sum(1 for k in dig if okk[k])
                    for si, (a, b) in enumerate(_number_spans(ids, self.is_digit, self.dot_id)):
                        t = NUM_TYPES_OM[si] if (sub == "object_motion" and si < len(NUM_TYPES_OM)) else "other"
                        acc[f"{cond}|num_{t}|n"] += 1
                        acc[f"{cond}|num_{t}|first_ok"] += int(okk[a])
                        acc[f"{cond}|num_{t}|em"] += int(all(okk[a:b]))
                        acc[f"{cond}|num_{t}|nll"] += sum(nl[a:b]); acc[f"{cond}|num_{t}|ntok"] += (b - a)
                pos += bs
        if self.world > 1:
            v = torch.tensor([acc[k] for k in self.keys], dtype=torch.float64, device=device)
            dist.all_reduce(v)
            acc = {k: float(x) for k, x in zip(self.keys, v.tolist())}
        model.train(was_m)
        if encoder is not None:
            encoder.train(was_e)
        return self._summarize(acc, restricted=(subs is not None or self.donor_subs is not None))

    def _summarize(self, a, restricted=False):
        r5 = lambda x: round(x, 5)
        out = {}
        for c in self.cols:
            if not a[f"{c}|n"]:
                continue
            d = {"n": int(a[f"{c}|n"]), "L_legacy": r5(a[f"{c}|legacy_x_bs"] / max(a[f"{c}|n"], 1))}
            for s in self.subs:
                if a[f"{c}|{s}|ntok"]:
                    d[f"L_tok/{s}"] = r5(a[f"{c}|{s}|nll"] / a[f"{c}|{s}|ntok"])
                if a[f"{c}|{s}|dig_n"]:
                    d[f"dig_nll/{s}"] = r5(a[f"{c}|{s}|dig_nll"] / a[f"{c}|{s}|dig_n"])
                    d[f"dig_acc/{s}"] = r5(a[f"{c}|{s}|dig_ok"] / a[f"{c}|{s}|dig_n"])
            for t in self.numtypes:
                n = a[f"{c}|num_{t}|n"]
                if n:
                    d[f"num_{t}/n"] = int(n)
                    d[f"num_{t}/first_acc"] = r5(a[f"{c}|num_{t}|first_ok"] / n)
                    d[f"num_{t}/em"] = r5(a[f"{c}|num_{t}|em"] / n)
                    d[f"num_{t}/nll_per_tok"] = r5(a[f"{c}|num_{t}|nll"] / max(a[f"{c}|num_{t}|ntok"], 1))
            out[c] = d
        if "donor" in out:                   # 读内容的量:loss 类 = donor − clean;准确率类 = clean − donor(都是越大越"在读")
            g = {}
            for k, v in out["clean"].items():
                if k == "n" or k.endswith("/n") or k not in out["donor"]:
                    continue
                if restricted and (k == "L_legacy" or k.endswith("/all") or k.startswith("num_other")):
                    continue                                 # donor 只评了部分题型:合计类的键两边不是同一批题,不求差
                dv = out["donor"][k]
                g[k] = r5((dv - v) if (k.startswith("L_") or "nll" in k) else (v - dv))
            out["gap"] = g
        return out
