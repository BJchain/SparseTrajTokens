#!/usr/bin/env python3
"""TrajQA-B Q1(官方原生口径)推理 —— MLLM-4D-SFT(ICML'26,Qwen3-VL-8B + LoRA SFT,已合并权重)。

  $BASELINE_ROOT/mllm4d_env/bin/python tools/run_mllm4d.py \
      --model $BASELINE_ROOT/models/MLLM-4D/MLLM-4D-SFT-1.0 --out preds/Q1_MLLM-4D-SFT-8B.jsonl

官方用法($BASELINE_ROOT/scratchpad/MLLM-4D:scripts/inference.py 的 MLLM-4D-SFT 分支,
与 evaluation/model_inference/qwen3_vl.py 逐行相同),这里照抄:
  - Qwen3VLForConditionalGeneration.from_pretrained(path, torch_dtype=bf16,
        attn_implementation="flash_attention_2", device_map="auto");AutoProcessor.from_pretrained(path)
  - 不加 system;user = [{"type":"video"}, {"type":"text"}]
  - processor.apply_chat_template(..., tokenize=True, add_generation_prompt=True, return_dict=True,
        return_tensors="pt", num_frames=32, fps=None)
    → 处理器自己按 linspace(0,T-1,32).round() 从**整段 clip** 取 32 帧(不足 32 帧的 clip 会重复帧,
      官方就是这样),并按 fps 给每个时间 patch 插 "<x.x seconds>" 时间戳 —— 时间信息由官方通路给出。
  - 权重/环境:transformers==4.57.0(官方钉)、torch 2.8.0+cu128、flash-attn 2.8.3(官方 README 版本;
    官方 wheel 含 sm_120 内核,本卡可用)。
官方是给视频文件路径,由 transformers 解码后再抽帧;TrajQA-B 只有 JPG 帧,所以这里把整段 clip 的全部帧
按原顺序堆成 (T,H,W,3) 数组、连同 video_metadata(total_num_frames=T, fps=clip fps)交给同一个处理器,
抽帧/缩放/时间戳走的是同一段官方代码。与"写成无损视频再按路径喂"逐比特等价,见 tools/verify_mllm4d.py。

与 TrajQA-B 协议相同的部分(同 run_native.py / run_3d.py):题面 build_prompt(item,'plain')、断点续跑、
输出字段(id/pred/raw/t/nf)、加载后 assert_on_gpu、解码 greedy。
两处为这个模型必须改的(见 RESULTS.md,冒烟实测决定):
  - max_new_tokens:SFT 训练目标是 "<answer>X</answer>"(6 个 token),协议的 4 会把标签截成 "<answer>X</";
    改 16(greedy 碰到 <|im_end|> 就停,不影响答案本身)。
  - 取字母:先用官方 evaluation/model_inference/qwen3_vl.py 的 extract_option_key(认 <answer> 标签),
    抽不到再退回 common.extract_letter。直接用 extract_letter 会从 "<ANSWER>" 里抓到 A,且不触发报警。
"""
import argparse, json, os, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from common import build_prompt, extract_letter, load_split
from common_native import clip_meta, load_manifest, sample_frames_native
from run_native import assert_on_gpu

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def extract_option_key(answer_text: str) -> str:
    """逐字抄自 MLLM-4D/evaluation/model_inference/qwen3_vl.py(官方 SFT 评测的取答案函数)。"""
    if not answer_text or not isinstance(answer_text, str): return ""
    match_tag = re.search(r"<answer>(.*?)</answer>", answer_text, re.DOTALL | re.IGNORECASE)
    if match_tag:
        content_in_tag = match_tag.group(1).strip()
        match_option_in_tag = re.search(r"\b([A-D])\b", content_in_tag, re.IGNORECASE)
        if match_option_in_tag: return match_option_in_tag.group(1).upper()
        if content_in_tag.upper() in ["A", "B", "C", "D"]: return content_in_tag.upper()
    match_declaration = re.search(
        r"(?:answer|option|最终答案|答案)\s*[:：]\s*([A-D])",
        answer_text, re.DOTALL | re.IGNORECASE | re.UNICODE
    )
    if match_declaration: return match_declaration.group(1).upper()
    tail_text = answer_text[-100:].strip()
    match_end = re.search(r"\b([A-D])\s*$", tail_text, re.IGNORECASE)
    if match_end: return match_end.group(1).upper()
    clean_text = answer_text.strip().upper()
    if clean_text in ["A", "B", "C", "D"]: return clean_text
    return ""


def parse_letter(txt: str):
    """-> (字母, 用了哪条规则)。官方规则优先,抽不到再用协议的 extract_letter。"""
    k = extract_option_key(txt)
    if k:
        return k, "official"
    return extract_letter(txt, "plain"), "protocol"


class MLLM4D:
    def __init__(self, a):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            a.model, torch_dtype=torch.bfloat16, attn_implementation=a.attn, device_map="auto").eval()
        self.proc = AutoProcessor.from_pretrained(a.model)
        cfg = self.model.config
        print(f"  [attn] top={cfg._attn_implementation} text={cfg.text_config._attn_implementation} "
              f"vision={cfg.vision_config._attn_implementation}  dtype={self.model.dtype}  "
              f"hf_device_map={sorted(set(getattr(self.model, 'hf_device_map', {}).values()))}", flush=True)
        vp = self.proc.video_processor
        print(f"  [video_processor] fps={vp.fps} min/max_frames={vp.min_frames}/{vp.max_frames} "
              f"size={dict(vp.size)} patch={vp.patch_size} merge={vp.merge_size} "
              f"temporal_patch={vp.temporal_patch_size}", flush=True)

    def inputs(self, it, imgs, fps, video=None, metadata=None):
        """official 喂法。video/metadata 缺省 = 整段 clip 全部帧 + clip 的 fps。"""
        T = len(imgs)
        if video is None:
            video = np.stack([np.asarray(im) for im in imgs])          # (T,H,W,3) uint8,整段不裁
            metadata = {"total_num_frames": T, "fps": fps, "duration": T / fps,
                        "height": video.shape[1], "width": video.shape[2], "video_backend": "jpg-frames"}
        conversation = [{"role": "user", "content": [
            {"type": "video", "video": video},
            {"type": "text", "text": build_prompt(it, "plain")}]}]
        kw = {} if metadata is None else {"video_metadata": [metadata]}
        return self.proc.apply_chat_template(
            conversation, tokenize=True, add_generation_prompt=True, return_dict=True,
            return_tensors="pt", num_frames=32, fps=None, **kw)

    def sampled_indices(self, T, fps):
        """处理器内部实际取的帧下标(调用的就是它自己的 sample_frames,只为记日志)。"""
        from transformers.video_utils import VideoMetadata
        return [int(i) for i in self.proc.video_processor.sample_frames(
            VideoMetadata(total_num_frames=T, fps=fps), num_frames=32, fps=None)]

    def __call__(self, it, imgs, fps, a):
        enc = self.inputs(it, imgs, fps).to(self.model.device)
        self.last_ctx = enc["input_ids"].shape[1]
        self.last_idx = self.sampled_indices(len(imgs), fps)
        with torch.no_grad():
            gen = self.model.generate(**enc, max_new_tokens=a.max_new, do_sample=False,
                                      temperature=None, top_p=None, top_k=None)
        return self.proc.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True,
                                      clean_up_tokenization_spaces=False)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default="heldout_test,test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pkg", default=PKG)
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", default="", help="只跑这些 id(逗号分隔或 @文件),冒烟用")
    ap.add_argument("--attn", default="flash_attention_2")
    a = ap.parse_args()

    man = load_manifest(a.pkg)
    items = load_split(a.pkg, a.split, None)
    if a.ids:
        want = (open(a.ids[1:]).read().split() if a.ids.startswith("@") else a.ids.split(","))
        want = [w.strip() for w in want if w.strip()]
        pos = {w: i for i, w in enumerate(want)}
        items = sorted([d for d in items if d["id"] in pos], key=lambda d: pos[d["id"]])
    if a.limit:
        items = items[:a.limit]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    if os.path.exists(a.out):
        done = {json.loads(l)["id"] for l in open(a.out) if l.strip()}
        items = [d for d in items if d["id"] not in done]
        print(f"[resume] 已有 {len(done)} 条,剩 {len(items)}", flush=True)
    print(f"[Q1-MLLM4D] {a.model} n={len(items)} 官方 num_frames=32(整段 clip)@native512 "
          f"时间戳=官方 attn={a.attn} max_new={a.max_new}", flush=True)

    runner = MLLM4D(a)
    assert_on_gpu(runner.model)
    t0 = time.time(); n = 0; nfb = 0; rule = {"official": 0, "protocol": 0}
    with open(a.out, "a") as f:
        for it in items:
            imgs, idx = sample_frames_native(a.pkg, it["video_id"], None, None)   # 整段 clip 全部帧,512 不缩
            T = it["n_frames_total"]
            assert len(imgs) == T == len(idx), f"{it['video_id']}: 帧目录 {len(imgs)} 张 != n_frames_total {T}"
            fps, fb = clip_meta(man, it["video_id"])
            nfb += int(fb)
            t1 = time.time()
            txt = runner(it, imgs, fps, a)
            dt = time.time() - t1
            pred, how = parse_letter(txt)
            rule[how] += 1
            f.write(json.dumps({"id": it["id"], "pred": pred, "raw": txt.strip()[:300], "t": round(dt, 4),
                                "nf": len(runner.last_idx), "T": T, "ctx": runner.last_ctx,
                                "rule": how}, ensure_ascii=False) + "\n")
            f.flush(); n += 1
            if n == 1:
                print(f"  [首条] clip {T} 帧 → 官方抽 {len(runner.last_idx)} 帧 {runner.last_idx} ctx={runner.last_ctx} "
                      f"raw={txt.strip()[:40]!r} pred={pred} 峰值显存="
                      f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
            if n % 200 == 0:
                el = time.time() - t0
                print(f"  {n}/{len(items)}  {n / el:.2f} it/s  ETA {(len(items) - n) / max(n / el, 1e-9) / 60:.0f} min"
                      f"  取字母规则={rule}  峰值显存={torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
    print(f"[done] {n} 条 {time.time() - t0:.0f}s -> {a.out}  取字母规则={rule}  峰值显存="
          f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB"
          + (f"(其中 {nfb} 次用了 fps 兜底 5.0)" if nfb else ""), flush=True)


if __name__ == "__main__":
    main()
