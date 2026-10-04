"""G1 评测:加载某臂 checkpoint,在 val 集上生成选项字母算准确率。

支持三种模式(--mode):
  rgb        : RGB-only 模型(traj checkpoint 里若无 encoder 亦可)。
  traj       : 轨迹模型,正常注入轨迹 token。
  traj_drop  : 轨迹模型,但**不注入轨迹**(把 <traj> 从 prompt 去掉)——
               若准确率相对 traj 明显下降,证明模型真在用轨迹 token(gate 条件之一)。

输出:overall acc + 逐 hallucination_type acc + 逐 source(S4D/MOVi)。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "src/tokenization")
sys.path.insert(0, "src")
import schema  # noqa: F401
sys.path.insert(0, "src/training")

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel

from g1.data import (G1Dataset, G1Collator, LETTERS, build_prompt_text, build_traj_block,
                     sample_frames, traj_arrays_for, build_visual_content, VIDEO_MODES)
from g1.split import split3_items
from g1.modeling_traj import add_traj_token, wrap_traj_embedding, TRAJ_TOKEN
from g1.traj_encoder import TrajectoryTokenProjector


def load_model(base, ckpt, want_encoder, no_adapter=False):
    proc = AutoProcessor.from_pretrained(ckpt)
    proc.tokenizer.padding_side = "left"
    model = Qwen3VLForConditionalGeneration.from_pretrained(base, dtype=torch.bfloat16, attn_implementation="sdpa")
    # 基座 ckpt(零样本基线)的 tokenizer 没有 <traj>:按训练时同样的方式加进去,否则 tid=None
    # 会让 bad_words_ids=[[None]] 直接崩(09-09 实测)。已含该 token 的 ckpt 走原路径,逐位不变。
    tid = add_traj_token(proc, model)
    model.resize_token_embeddings(len(proc.tokenizer))
    wrapper = wrap_traj_embedding(model, tid)
    if not no_adapter:                       # 1a 检查点无 LoRA:base + encoder 直评(硬门)
        model = PeftModel.from_pretrained(model, ckpt)
    model.eval().cuda()
    encoder = None
    if want_encoder and os.path.exists(os.path.join(ckpt, "traj_encoder.pt")):
        H = model.config.text_config.hidden_size
        sd = torch.load(os.path.join(ckpt, "traj_encoder.pt"), map_location="cpu")
        # 融合方式/通道宽度由 checkpoint 自身决定(旧 checkpoint 无 fuse.* → add,完全兼容)
        encoder = TrajectoryTokenProjector.from_state_dict(sd, H).to(torch.bfloat16).cuda().eval()
    return model, encoder, proc, tid, wrapper


_NO_TIME = set()          # 无绝对时间轴的 clip(不静默降级,首次会打印)


@torch.no_grad()
def run(args):
    inject = args.mode in ("traj", "traj_only")
    use_frames = args.mode not in ("traj_only", "blind")
    model, encoder, proc, tid, wrapper = load_model(args.base, args.ckpt, want_encoder=inject,
                                                    no_adapter=getattr(args, "no_adapter", False))

    ds = G1Dataset(args.qa)
    if args.on == "all":                      # 整份文件当评测集(如 t8_probe 这种独立评测集)
        val = list(ds.items)
    else:
        _, val, test = split3_items(ds.items, args.val_frac, args.test_frac, args.split_mode)
        if args.on == "test":
            val = test
    if args.limit:
        val = val[:args.limit]
    print(f"[eval] mode={args.mode} val={len(val)}", flush=True)
    per_item = []

    bad = [[tid]]  # 禁止生成 <traj>
    correct = defaultdict(int); total = defaultdict(int)
    src_correct = defaultdict(int); src_total = defaultdict(int)
    sub_correct = defaultdict(int); sub_total = defaultdict(int)
    t0 = time.time()

    for s in range(0, len(val), args.batch):
        chunk = val[s:s + args.batch]
        convs, metas = [], []
        traj_list, vmeta = [], []
        for item in chunk:
            vid = item["video_id"]
            imgs, fidx = ((sample_frames(vid, args.n_frames, args.max_side,
                                         frame_range=(None if args.frames_whole_clip else item.get("frame_range")), return_idx=True))
                          if use_frames else ([], []))
            n_traj = 0
            traj_block = ""
            if inject:
                arr, n_traj, _, spans, labels = traj_arrays_for(
                    vid, args.max_traj, frame_range=(None if args.frames_whole_clip else item.get("frame_range")))
                traj_list.append(arr)
                traj_block = build_traj_block(spans, labels)
            content = build_visual_content(vid, imgs, fidx, args.video_mode, vmeta, _NO_TIME)
            arm_txt = args.mode if inject else "rgb"
            content.append({"type": "text", "text": build_prompt_text(item, arm_txt, traj_block, args.structured)})
            convs.append([{"role": "user", "content": content}])
            metas.append(item)
        vkw = (dict(video_metadata=vmeta, do_sample_frames=False)
               if (args.video_mode == "video" and vmeta) else {})
        enc = proc.apply_chat_template(convs, tokenize=True, add_generation_prompt=True,
                                       return_dict=True, return_tensors="pt", padding=True,
                                       **vkw).to("cuda")
        if inject and traj_list:
            import numpy as np
            cat = {k: np.concatenate([a[k] for a in traj_list], 0) for k in traj_list[0]}
            tb = {k: torch.from_numpy(v).cuda() for k, v in cat.items()}
            wrapper.set_pending(encoder(tb))
        gen = model.generate(**enc, max_new_tokens=(args.max_new if args.structured else 4), do_sample=False, bad_words_ids=bad)
        wrapper.clear_pending()
        out_txt = proc.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for item, txt in zip(metas, out_txt):
            if args.structured:
                import re as _re
                m = _re.search(r"<answer>\s*([A-Da-d])", txt)
                pred = m.group(1).upper() if m else next((ch for ch in txt.strip()[::-1].upper() if ch in LETTERS), "?")
            else:
                pred = next((ch for ch in txt.strip().upper() if ch in LETTERS), "?")
            ht = item["hallucination_type"]
            st = item.get("event_subtype", "?")
            src = "MOVi" if item["video_id"].startswith("val_") else "S4D"
            ok = int(pred == item["answer"])
            # 逐题记录(09-09 加):配对检验 / strict 子集 / McNemar 都要逐题结果,只有聚合数做不了
            per_item.append(dict(id=item.get("id"), video_id=item["video_id"], event_subtype=st,
                                 hallucination_type=ht, answer=item["answer"], pred=pred, ok=ok,
                                 eval_strict=bool(item.get("eval_strict", False)),
                                 **({"gen": txt.strip()[:600], "evidence_gt": item.get("evidence")} if args.structured else {})))
            correct[ht] += ok; total[ht] += 1
            correct["ALL"] += ok; total["ALL"] += 1
            sub_correct[st] += ok; sub_total[st] += 1
            src_correct[src] += ok; src_total[src] += 1
        if (s // args.batch) % 20 == 0:
            acc = correct["ALL"] / max(1, total["ALL"])
            print(f"  {total['ALL']}/{len(val)} acc={acc:.3f} {time.time()-t0:.0f}s", flush=True)

    res = dict(mode=args.mode, ckpt=args.ckpt, n=total["ALL"],
               acc_overall=round(correct["ALL"] / max(1, total["ALL"]), 4),
               by_type={k: [correct[k], total[k], round(correct[k] / max(1, total[k]), 4)]
                        for k in sorted(total) if k != "ALL"},
               by_subtype={k: [sub_correct[k], sub_total[k], round(sub_correct[k] / max(1, sub_total[k]), 4)]
                           for k in sorted(sub_total)},
               by_source={k: [src_correct[k], src_total[k], round(src_correct[k] / max(1, src_total[k]), 4)]
                          for k in sorted(src_total)})
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    with open(args.out + ".items.jsonl", "w") as f:        # 逐题结果
        for r in per_item:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if any(r["eval_strict"] for r in per_item):             # 时间 QA 的严格分层子集另报一行
        st_items = [r for r in per_item if r["eval_strict"]]
        print(f"[eval] eval_strict 子集 n={len(st_items)} acc={sum(r['ok'] for r in st_items)/len(st_items):.4f}", flush=True)
    print(json.dumps(res, indent=1, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.environ.get("QWEN3VL_PATH", "checkpoints/Qwen3-VL-8B-Instruct"))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--mode", choices=["rgb", "traj", "traj_drop", "traj_only", "blind"], required=True)
    ap.add_argument("--no-adapter", action="store_true", help="1a 检查点:不加载 LoRA adapter(base+encoder 直评)")
    ap.add_argument("--qa", nargs="+", default=["data/trajqa_b/final/heldout_test.jsonl"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.0,
                    help="0=不切 test(旧行为);跑 110k 的 run 要与其 meta.json 一致")
    ap.add_argument("--split-mode", choices=["scene", "video"], default="scene",
                    help="划分粒度,须与被评 run 的 meta.json 一致(旧 run=scene)")
    ap.add_argument("--on", choices=["val", "test", "all"], default="val",
                    help="all=不划分,整份 --qa 当评测集(独立评测集用)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--n-frames", type=int, default=8)
    ap.add_argument("--max-traj", type=int, default=512)
    ap.add_argument("--max-side", type=int, default=384)
    ap.add_argument("--video-mode", choices=list(VIDEO_MODES), default="images",
                    help="视觉通道时间戳:images=无(默认);tstamp=每帧前插字面秒数;video=原生视频通道(视觉 token 会变少)。**必须与训练时一致**")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--structured", action="store_true", help="结构化目标:生成 evidence+answer,解析 <answer>L</answer>")
    ap.add_argument("--max-new", type=int, default=160)
    ap.add_argument("--frames-whole-clip", action="store_true",
                    help="评测协议:帧与轨迹都从整段 clip 采,不用题目的 frame_range 窗口(默认用窗口 = 把答案区间告诉模型)")
    run(ap.parse_args())
