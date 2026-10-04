#!/usr/bin/env python3
"""TrajQA-B 本地 baseline 推理(HuggingFace transformers)。

  python tools/run_hf.py --model Qwen/Qwen2.5-VL-7B-Instruct --family qwen \
      --split heldout_test --out preds/qwen25vl7b_heldout.jsonl
  # 盲测地板(同 prompt 不给帧)—— 每个模型都必须跑,否则数字不可比
  python tools/run_hf.py --model ... --family qwen --split heldout_test --no-frames \
      --out preds/qwen25vl7b_heldout_blind.jsonl

--family:
  qwen     Qwen2.5-VL / Qwen3-VL / VG-LLM(Qwen2.5-VL 底座)  —— 走 AutoProcessor + chat template
  internvl InternVL2.5 / InternVL3.5 / InternVideo2.5        —— 走模型自带 .chat(),trust_remote_code
  llava    LLaVA-NeXT-Video / VLM-3R(LLaVA-Video 底座)      —— 走 LlavaNextVideo* 或自带 chat
断点续跑:--out 已存在的 id 自动跳过。
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from common import build_prompt, extract_letter, load_split, sample_frames

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ------------------------------------------------------------------ qwen
class QwenRunner:
    def __init__(self, a):
        from transformers import AutoProcessor, AutoModelForImageTextToText
        self.proc = AutoProcessor.from_pretrained(a.model, trust_remote_code=True)
        self.proc.tokenizer.padding_side = "left"
        self.model = AutoModelForImageTextToText.from_pretrained(
            a.model, dtype=torch.bfloat16, device_map="auto",
            attn_implementation=a.attn, trust_remote_code=True).eval()

    def __call__(self, batch, imgs_batch, a):
        convs = []
        for item, imgs in zip(batch, imgs_batch):
            content = [{"type": "image", "image": im} for im in imgs]
            content.append({"type": "text", "text": build_prompt(item, a.prompt)})
            convs.append([{"role": "user", "content": content}])
        enc = self.proc.apply_chat_template(
            convs, tokenize=True, add_generation_prompt=True, return_dict=True,
            return_tensors="pt", padding=True).to(self.model.device)
        with torch.no_grad():
            gen = self.model.generate(**enc, max_new_tokens=a.max_new, do_sample=False)
        return self.proc.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)


# ------------------------------------------------------------------ internvl
class InternVLRunner:
    """InternVL 家族用自带 .chat(),不吃 chat template。多图 = 拼 tile + num_patches_list。"""
    def __init__(self, a):
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True, use_fast=False)
        self.model = AutoModel.from_pretrained(
            a.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
            trust_remote_code=True, device_map="auto").eval()
        self.gen_cfg = dict(max_new_tokens=a.max_new, do_sample=False)

    @staticmethod
    def _tf(size=448):
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode
        return T.Compose([T.Resize((size, size), interpolation=InterpolationMode.BICUBIC),
                          T.ToTensor(),
                          T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    def __call__(self, batch, imgs_batch, a):
        tf = self._tf()
        outs = []
        for item, imgs in zip(batch, imgs_batch):              # InternVL .chat 单条
            q = build_prompt(item, a.prompt)
            if imgs:
                pv = torch.stack([tf(im) for im in imgs]).to(torch.bfloat16).to(self.model.device)
                prefix = "".join(f"Frame{i+1}: <image>\n" for i in range(len(imgs)))
                r = self.model.chat(self.tok, pv, prefix + q, self.gen_cfg,
                                    num_patches_list=[1] * len(imgs))
            else:
                r = self.model.chat(self.tok, None, q, self.gen_cfg)
            outs.append(r)
        return outs


# ------------------------------------------------------------------ llava
class LlavaRunner:
    def __init__(self, a):
        from transformers import AutoProcessor
        try:                       # transformers v4
            from transformers import AutoModelForVision2Seq as _AutoVis
        except ImportError:        # v5 把 AutoModelForVision2Seq 改名成 AutoModelForImageTextToText
            from transformers import AutoModelForImageTextToText as _AutoVis
        self.proc = AutoProcessor.from_pretrained(a.model, trust_remote_code=True)
        self.proc.tokenizer.padding_side = "left"
        self.model = _AutoVis.from_pretrained(
            a.model, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True).eval()

    def __call__(self, batch, imgs_batch, a):
        import numpy as np
        outs = []
        for item, imgs in zip(batch, imgs_batch):
            q = build_prompt(item, a.prompt)
            content = ([{"type": "video"}] if imgs else []) + [{"type": "text", "text": q}]
            prompt = self.proc.apply_chat_template([{"role": "user", "content": content}],
                                                   add_generation_prompt=True)
            kw = dict(videos=np.stack([np.asarray(im) for im in imgs])) if imgs else {}
            enc = self.proc(text=prompt, return_tensors="pt", **kw).to(self.model.device)
            with torch.no_grad():
                gen = self.model.generate(**enc, max_new_tokens=a.max_new, do_sample=False)
            outs.append(self.proc.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=True))
        return outs


FAMILIES = {"qwen": QwenRunner, "internvl": InternVLRunner, "llava": LlavaRunner}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--family", required=True, choices=list(FAMILIES))
    ap.add_argument("--split", default="heldout_test",
                    help="heldout_test / test / val / train / all,逗号可多选")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pkg", default=PKG)
    ap.add_argument("--n-frames", type=int, default=8, help="我方 RGB 臂用 8;给 baseline 更多帧只会更强")
    ap.add_argument("--max-side", type=int, default=384, help="我方 RGB 臂用 384;原始帧 512")
    ap.add_argument("--no-frames", action="store_true", help="盲测地板:不给帧,只给题")
    ap.add_argument("--prompt", default="plain", choices=("plain", "cot"))
    ap.add_argument("--max-new", type=int, default=4)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--attn", default="sdpa")
    a = ap.parse_args()
    if a.prompt == "cot" and a.max_new < 64:
        a.max_new = 256

    items = load_split(a.pkg, a.split, a.limit or None)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["id"] for l in open(a.out) if l.strip()}
        items = [d for d in items if d["id"] not in done]
        print(f"[resume] 已有 {len(done)} 条,剩 {len(items)}", flush=True)
    print(f"[run] {a.model} family={a.family} split={a.split} n={len(items)} "
          f"frames={0 if a.no_frames else a.n_frames}@{a.max_side} prompt={a.prompt}", flush=True)

    runner = FAMILIES[a.family](a)
    t0 = time.time(); n = 0
    with open(a.out, "a") as f:
        for s in range(0, len(items), a.batch):
            batch = items[s:s + a.batch]
            imgs_batch = [[] if a.no_frames
                          else sample_frames(a.pkg, it["video_id"], a.n_frames, a.max_side)[0]
                          for it in batch]
            _t0 = time.time()
            txts = runner(batch, imgs_batch, a)
            _dt = (time.time() - _t0) / max(len(batch), 1)
            for it, txt in zip(batch, txts):
                f.write(json.dumps({"id": it["id"], "pred": extract_letter(txt, a.prompt),
                                    "raw": txt.strip()[:300], "t": round(_dt, 4)},
                                   ensure_ascii=False) + "\n")
            f.flush(); n += len(batch)
            if s % (a.batch * 25) == 0:
                el = time.time() - t0
                print(f"  {n}/{len(items)}  {n/max(el,1e-9):.2f} it/s  "
                      f"ETA {(len(items)-n)/max(n/max(el,1e-9),1e-9)/60:.1f} min", flush=True)
    print(f"[done] {n} 条, {time.time()-t0:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
