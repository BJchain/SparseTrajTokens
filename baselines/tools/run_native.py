#!/usr/bin/env python3
"""TrajQA-B Q1「能力上限」推理:每个模型走**自己的官方原生视频配置**。

  python tools/run_native.py --model <dir> --family qwen --mode video \
      --split heldout_test,test --out preds/Q1_xxx.jsonl

与 Q2(tools/run_hf.py)的区别只在**怎么把视频喂进去**,题面/取字母/评分完全一致:
  - 抽帧:整段 clip **全帧**(20~38 帧),不是 8 帧
  - 分辨率:源图 **512** 不降采样,不是 384
  - 通道:各家族官方视频通路
  - 时间信息:**给**(Qwen video 自带时间戳;InternVL/LLaVA 补一行 fps/时长/真实帧号)

--family / --mode:
  qwen  video   官方 video 通道(temporal_patch_size=2,两帧并一个时间 patch)
  qwen  images  逐帧 image(帧数/分辨率同样放开)—— 作为"官方通道 vs 密集图"的对照
  internvl      官方 load_video 口径:每帧 1 tile、448,Frame 前缀用**真实帧号**
  llava         官方 video 通道

断点续跑:--out 已有的 id 自动跳过。
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from common import build_prompt, extract_letter, load_split
from common_native import clip_meta, load_manifest, sample_frames_native, temporal_header

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _dmap(a):
    """device_map。38B 这种大模型用 'auto' 时 accelerate 可能整个丢到 CPU,必须能强制指定。"""
    return {"": 0} if getattr(a, "device_map", "auto") == "cuda0" else "auto"


def assert_on_gpu(model):
    """加载后立刻自检:只要有参数留在 CPU 就直接报错退出,不要再白跑 5 小时。"""
    devs = {str(p.device) for p in model.parameters()}
    on_gpu = sum(p.numel() for p in model.parameters() if p.device.type == "cuda")
    total = sum(p.numel() for p in model.parameters())
    pct = 100.0 * on_gpu / max(total, 1)
    print(f"  [设备自检] 参数分布={sorted(devs)} 在 GPU 上的占比={pct:.1f}%", flush=True)
    if pct < 99.0:
        raise RuntimeError(
            f"❌ 只有 {pct:.1f}% 的参数在 GPU 上(其余在 {sorted(devs)})。"
            f"38B 用 device_map='auto' 会被 accelerate 丢到 CPU,请加 --device-map cuda0")


class QwenNative:
    def __init__(self, a):
        from transformers import AutoProcessor, AutoModelForImageTextToText
        self.proc = AutoProcessor.from_pretrained(a.model, trust_remote_code=True)
        self.proc.tokenizer.padding_side = "left"
        self.model = AutoModelForImageTextToText.from_pretrained(
            a.model, dtype=torch.bfloat16, device_map=_dmap(a),
            attn_implementation=a.attn, trust_remote_code=True).eval()

    def __call__(self, batch, a):
        convs, metas = [], []
        for it, imgs, idx, fps in batch:
            q = build_prompt(it, a.prompt)
            T = it["n_frames_total"] or len(imgs)
            if a.mode == "video":
                convs.append([{"role": "user", "content": [
                    {"type": "video", "video": [np.asarray(im) for im in imgs]},
                    {"type": "text", "text": q}]}])
                metas.append({"fps": fps, "duration": T / fps, "total_num_frames": T,
                              "frames_indices": list(idx), "video_backend": "custom",
                              "width": imgs[0].size[0], "height": imgs[0].size[1]})
            else:                                   # images:帧数/分辨率放开的逐帧图
                content = [{"type": "text", "text": temporal_header(idx, T, fps)}]
                content += [{"type": "image", "image": im} for im in imgs]
                content.append({"type": "text", "text": q})
                convs.append([{"role": "user", "content": content}])
        kw = dict(do_sample_frames=False, video_metadata=metas) if a.mode == "video" else {}
        enc = self.proc.apply_chat_template(
            convs, tokenize=True, add_generation_prompt=True, return_dict=True,
            return_tensors="pt", padding=True, **kw).to(self.model.device)
        self.last_ctx = enc["input_ids"].shape[1]
        with torch.no_grad():
            gen = self.model.generate(**enc, max_new_tokens=a.max_new, do_sample=False)
        return self.proc.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)


class InternVLNative:
    """官方 load_video 口径:每帧 1 tile、448 直采;Frame 前缀改用**真实帧号**。"""
    def __init__(self, a):
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True, use_fast=False)
        self.model = AutoModel.from_pretrained(
            a.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
            trust_remote_code=True, device_map=_dmap(a)).eval()
        # 2026-09-24:InternVL 的 remote code 在没装 flash_attn 时把 LLM 硬设成 eager。
        # 38B × 32 帧(ctx≈8.5k)时 eager 要在 fp32 里物化 8.5k×8.5k×40 头 ≈ 11.3 GiB 注意力矩阵,
        # 71.5 GiB 权重之外放不下 → 实测 OOM。改成 PyTorch SDPA(与官方 flash_attention_2 一样是
        # **精确**注意力,只是不物化矩阵,数值等价),显存 O(n)。tf 4.51 的 Qwen2 是前向时按
        # config._attn_implementation 动态分发,所以加载后改这一项即可生效。8B 档当年用 eager 跑完,
        # 需要复现旧口径时传 --llm-attn eager。
        llm = getattr(self.model, "language_model", None)
        if llm is not None and a.llm_attn != "keep":
            llm.config._attn_implementation = a.llm_attn
        print(f"  [LLM attn] {getattr(getattr(llm, 'config', None), '_attn_implementation', '?')}"
              f"  vision flash_attn={getattr(self.model.config.vision_config, 'use_flash_attn', '?')}", flush=True)
        self.gen_cfg = dict(max_new_tokens=a.max_new, do_sample=False)
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode
        self.tf = T.Compose([T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
                             T.ToTensor(),
                             T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    def __call__(self, batch, a):
        outs = []
        for it, imgs, idx, fps in batch:
            T_ = it["n_frames_total"] or len(imgs)
            pv = torch.stack([self.tf(im) for im in imgs]).to(torch.bfloat16).to(self.model.device)
            prefix = "".join(f"Frame{i+1}: <image>\n" for i in idx)   # 真实帧号
            q = temporal_header(idx, T_, fps) + "\n" + build_prompt(it, a.prompt)
            r = self.model.chat(self.tok, pv, prefix + q, self.gen_cfg,
                                num_patches_list=[1] * len(imgs))     # 每帧 1 tile,官方口径
            outs.append(r)
        self.last_ctx = None
        return outs


class LlavaNative:
    def __init__(self, a):
        from transformers import AutoProcessor
        try:
            from transformers import AutoModelForVision2Seq as _AutoVis
        except ImportError:
            from transformers import AutoModelForImageTextToText as _AutoVis
        self.proc = AutoProcessor.from_pretrained(a.model, trust_remote_code=True)
        self.proc.tokenizer.padding_side = "left"
        self.model = _AutoVis.from_pretrained(
            a.model, dtype=torch.bfloat16, device_map=_dmap(a), trust_remote_code=True).eval()

    def __call__(self, batch, a):
        outs = []
        for it, imgs, idx, fps in batch:
            T_ = it["n_frames_total"] or len(imgs)
            q = temporal_header(idx, T_, fps) + "\n" + build_prompt(it, a.prompt)
            prompt = self.proc.apply_chat_template(
                [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": q}]}],
                add_generation_prompt=True)
            enc = self.proc(text=prompt, return_tensors="pt",
                            videos=np.stack([np.asarray(im) for im in imgs])).to(self.model.device)
            self.last_ctx = enc["input_ids"].shape[1]
            with torch.no_grad():
                gen = self.model.generate(**enc, max_new_tokens=a.max_new, do_sample=False)
            outs.append(self.proc.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=True))
        return outs


FAMILIES = {"qwen": QwenNative, "internvl": InternVLNative, "llava": LlavaNative}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--family", required=True, choices=list(FAMILIES))
    ap.add_argument("--mode", default="video", choices=("video", "images"),
                    help="仅 qwen 用:官方 video 通道,或帧数放开的逐帧图(对照)")
    ap.add_argument("--split", default="heldout_test,test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pkg", default=PKG)
    ap.add_argument("--n-frames", type=int, default=0, help="0=整段 clip 全帧(默认,最强)")
    ap.add_argument("--max-side", type=int, default=0, help="0=源分辨率 512 不降采样")
    ap.add_argument("--prompt", default="plain", choices=("plain", "cot"))
    ap.add_argument("--max-new", type=int, default=4)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--llm-attn", default="sdpa", choices=("sdpa", "eager", "keep"),
                    help="仅 internvl 用:加载后把 LLM 的注意力实现改成 sdpa(默认;38B+32帧 eager 会 OOM)"
                         " / eager(8B 档旧口径)/ keep(不动 remote code 的设定)")
    ap.add_argument("--device-map", default="auto",
                    help="auto | cuda0。⚠️ 38B(71.5GiB)用 auto 时 accelerate 会整模型丢到 CPU "
                         "(实测 RSS 81.5GB / GPU 0MiB / 0 条每分钟),必须用 cuda0 强制上卡")
    a = ap.parse_args()
    if a.prompt == "cot" and a.max_new < 64:
        a.max_new = 256

    man = load_manifest(a.pkg)
    items = load_split(a.pkg, a.split, a.limit or None)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    if os.path.exists(a.out):
        done = {json.loads(l)["id"] for l in open(a.out) if l.strip()}
        items = [d for d in items if d["id"] not in done]
        print(f"[resume] 已有 {len(done)} 条,剩 {len(items)}", flush=True)
    print(f"[Q1] {a.model} family={a.family} mode={a.mode} n={len(items)} "
          f"frames={'ALL' if not a.n_frames else a.n_frames}@"
          f"{'native512' if not a.max_side else a.max_side} 时间信息=给", flush=True)

    runner = FAMILIES[a.family](a)
    assert_on_gpu(runner.model)
    t0 = time.time(); n = 0; nfb = 0
    with open(a.out, "a") as f:
        for s in range(0, len(items), a.batch):
            raw = items[s:s + a.batch]
            batch = []
            for it in raw:
                imgs, idx = sample_frames_native(a.pkg, it["video_id"],
                                                 a.n_frames or None, a.max_side or None)
                fps, fb = clip_meta(man, it["video_id"])
                nfb += int(fb)
                batch.append((it, imgs, idx, fps))
            t1 = time.time()
            txts = runner(batch, a)
            dt = (time.time() - t1) / max(len(batch), 1)
            for (it, imgs, idx, _), txt in zip(batch, txts):
                f.write(json.dumps({"id": it["id"], "pred": extract_letter(txt, a.prompt),
                                    "raw": txt.strip()[:300], "t": round(dt, 4),
                                    "nf": len(imgs)}, ensure_ascii=False) + "\n")
            f.flush(); n += len(batch)
            if s == 0:
                ctx = getattr(runner, "last_ctx", None)
                print(f"  [首批] 帧数={len(batch[0][1])} ctx={ctx}", flush=True)
            if s % (a.batch * 50) == 0 and s:
                el = time.time() - t0
                print(f"  {n}/{len(items)}  {n/el:.2f} it/s  "
                      f"ETA {(len(items)-n)/max(n/el,1e-9)/60:.0f} min", flush=True)
    print(f"[done] {n} 条 {time.time()-t0:.0f}s -> {a.out}"
          + (f"(其中 {nfb} 次用了 fps 兜底 5.0)" if nfb else ""))


if __name__ == "__main__":
    main()
