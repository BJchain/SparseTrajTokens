#!/usr/bin/env python3
"""Qwen 系「喂法」三路 A/B:裸图 vs 真实帧号 vs 原生 video 通道。

    python tools/ab_qwen.py --model $BASELINE_ROOT/models/qwen3vl-32b --n 300

三臂(同一批题、同一批帧,只改喂法):
  base      现行协议:8 张裸图,无帧号、无时间   —— 对照组
  frameidx  8 张图,每张前面加真实帧号 + 时间戳:"Frame 6 of 34 (t=1.00s):"
  video     Qwen 原生 video 通道 + 正确的 video_metadata
            (Qwen3-VL 会自动把 <0.5 seconds> 这类时间戳插进 prompt)

⚠️ 原生 video 通道的坑(实测):不传 video_metadata / 不关 do_sample_frames,
   processor 会按默认 fps=2(并假定源视频 24fps)**重采样丢帧** ——
   8 帧会被压成 2 个时间 patch(288 token)。必须 do_sample_frames=False + video_metadata。
   即便如此,temporal_patch_size=2 会把 8 帧并成 4 个时间 patch → 576 视觉 token,
   只有 image 口径(1,152)的一半。这是 Qwen 原生 video 通路的固有代价。

用户 2026-09-21 决定:时间信息可以给,目的是诚实测出基模推理能力的上限。
抽帧与 prompt 仍然走 tools/common.py,不改。
"""
import argparse, collections, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from common import NUMERIC3, build_prompt, extract_letter, load_split, sample_frames

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARMS = ("base", "frameidx", "video")


def clip_fps(man, video_id, default=5.0):
    m = man.get(video_id) or man.get(video_id.replace("scene_", ""))
    f = (m or {}).get("fps")
    return (f, False) if f else (default, True)


def stratified(items, n, gold):
    """三个数值族各取 n/3,保证能看出 speed_grid(题面不给秒)上时间信息的效果。"""
    per = n // len(NUMERIC3)
    buckets = collections.defaultdict(list)
    for it in items:
        g = gold.get(it["id"])
        if g and g["family"] in NUMERIC3:
            buckets[g["family"]].append(it)
    out = []
    for f in NUMERIC3:
        out += buckets[f][:per]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pkg", default=PKG)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--split", default="heldout_test")
    ap.add_argument("--n-frames", type=int, default=8)
    ap.add_argument("--max-side", type=int, default=384)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=4)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--out", default=os.path.join(PKG, "preds", "ab_qwen"))
    a = ap.parse_args()
    arms = [x.strip() for x in a.arms.split(",")]

    gold = {d["id"]: d for d in map(json.loads,
            open(os.path.join(a.pkg, "qa", "answers.jsonl")))}
    man = json.load(open(os.path.join(a.pkg, "qa", "clips_manifest.json")))
    items = stratified(load_split(a.pkg, a.split), a.n, gold)
    print(f"[ab] {len(items)} 题 ("
          + ", ".join(f"{f}={sum(1 for i in items if gold[i['id']]['family']==f)}" for f in NUMERIC3)
          + f");模型 {a.model}", flush=True)

    from transformers import AutoProcessor, AutoModelForImageTextToText
    proc = AutoProcessor.from_pretrained(a.model, trust_remote_code=True)
    proc.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map="auto",
        attn_implementation="sdpa", trust_remote_code=True).eval()

    os.makedirs(a.out, exist_ok=True)
    results = {}
    n_fallback = 0

    for arm in arms:
        recs, t0 = [], time.time()
        for s in range(0, len(items), a.batch):
            batch = items[s:s + a.batch]
            convs, metas = [], []
            for it in batch:
                imgs, idx = sample_frames(a.pkg, it["video_id"], a.n_frames, a.max_side)
                q = build_prompt(it, "plain")
                T = it["n_frames_total"] or len(imgs)
                fps, fb = clip_fps(man, it["video_id"])
                if fb:
                    n_fallback += 1
                if arm == "base":
                    content = [{"type": "image", "image": im} for im in imgs]
                    content.append({"type": "text", "text": q})
                elif arm == "frameidx":
                    content = []
                    for im, i in zip(imgs, idx):
                        content.append({"type": "text",
                                        "text": f"Frame {i+1} of {T} (t={i/fps:.2f}s):"})
                        content.append({"type": "image", "image": im})
                    content.append({"type": "text", "text": q})
                else:                                   # video
                    content = [{"type": "video", "video": [np.asarray(im) for im in imgs]},
                               {"type": "text", "text": q}]
                    metas.append({"fps": fps, "duration": T / fps, "total_num_frames": T,
                                  "frames_indices": list(idx), "video_backend": "custom",
                                  "width": imgs[0].size[0], "height": imgs[0].size[1]})
                convs.append([{"role": "user", "content": content}])

            kw = dict(do_sample_frames=False, video_metadata=metas) if arm == "video" else {}
            enc = proc.apply_chat_template(convs, tokenize=True, add_generation_prompt=True,
                                           return_dict=True, return_tensors="pt",
                                           padding=True, **kw).to(model.device)
            if s == 0:
                vt = {k: tuple(v.shape) for k, v in enc.items() if "pixel" in k or "grid" in k}
                print(f"  [{arm}] ctx={enc['input_ids'].shape[1]} {vt}", flush=True)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=a.max_new, do_sample=False)
            txts = proc.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for it, txt in zip(batch, txts):
                recs.append({"id": it["id"], "pred": extract_letter(txt, "plain"),
                             "raw": txt.strip()[:120]})
        el = time.time() - t0
        with open(os.path.join(a.out, f"{arm}.jsonl"), "w") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        results[arm] = (recs, el)
        print(f"  [{arm}] 完成 {len(recs)} 条 {el:.0f}s", flush=True)

    # ---------------- 汇总 ----------------
    def acc(recs, fam=None):
        c = n = 0
        for r in recs:
            g = gold[r["id"]]
            if fam and g["family"] != fam:
                continue
            n += 1; c += int(r["pred"] == g["answer"])
        return (100.0 * c / n if n else float("nan")), n

    print(f"\n{'='*74}\n三路对比(chance=25.0;{len(items)} 题 @ {a.split})\n{'='*74}")
    print(f"{'喂法':22s} {'数值三族':>9s} {'cam_disp':>10s} {'disp':>10s} {'speed':>10s} {'耗时':>8s}")
    base_all = None
    for arm in arms:
        recs, el = results[arm]
        all_a, n = acc(recs)
        if arm == "base":
            base_all = all_a
        row = f"{arm:22s} {all_a:9.1f}"
        for f in NUMERIC3:
            fa, fn = acc(recs, f)
            row += f" {fa:10.1f}"
        row += f" {el:7.0f}s"
        if base_all is not None and arm != "base":
            row += f"   ({all_a-base_all:+.1f} vs base)"
        print(row)
    if n_fallback:
        print(f"\n注:{n_fallback} 次用了 fps 缺失时的兜底值 5.0"
              f"(全集 110/3103 个 clip 的 manifest 里 fps 为 null)")
    print(f"\n预测已存 {a.out}/<arm>.jsonl")


if __name__ == "__main__":
    sys.exit(main())
