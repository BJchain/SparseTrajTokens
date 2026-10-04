#!/usr/bin/env python3
"""TrajQA-B Q1(官方原生口径)推理 —— 3D 组:VG-LLM / VLM-3R。

  $BASELINE_ROOT/vgllm_env/bin/python tools/run_3d.py --family vgllm \
      --model $BASELINE_ROOT/models/vgllm-qa-vggt-8b --out preds/Q1_VG-LLM-8B.jsonl
  $BASELINE_ROOT/vlm3r_env/bin/python tools/run_3d.py --family vlm3r \
      --model $BASELINE_ROOT/models/vlm-3r-llava-qwen2-lora \
      --model-base $BASELINE_ROOT/models/LLaVA-Video-7B-Qwen2 --out preds/Q1_VLM-3R-7B.jsonl

与 run_native.py 同一套协议:题面 build_prompt(item,'plain')、取字母 extract_letter、断点续跑、
输出字段(id/pred/raw/t/nf)、加载后 assert_on_gpu 全部一致;只有"怎么把视频喂进去"照各自官方评测代码:
  vgllm  VG-LLM/src/lmms_eval/models/vgllm.py:帧逐张当 image;每帧 load_and_preprocess_images(VGGT 口径
         518)得到的同一张量既进几何编码器,又裁到 28 的倍数(504)以 do_rescale=False 进 Qwen2.5-VL;
         system="You are a helpful assistant.";min/max_pixels、max_length=12800 同官方。bf16。
  vlm3r  VLM-3R/thinking-in-space/lmms_eval/models/vlm_3r.py:SigLIP 预处理 384 → fp16,
         "<image>\n"+题面,conv_template=qwen_1_5,overwrite_config 同官方默认;
         CUT3R 在模型内部对同一批帧取 camera/patch token。
帧:两者官方都是 max 32 帧 → sample_frames_native(n_frames=32):clip ≤32 帧全用,多于 32 均匀取 32
    (与 38B 的 32 帧同算法;源图 512 交给各自官方预处理,不先缩)。
时间信息:两者官方都不带时间戳 → 与 InternVL/LLaVA 的 Q1 一样补一行 temporal_header(fps/时长/真实帧号)。
注意力:官方默认 flash_attention_2,本卡 sm_120 没有轮子 → sdpa(精确注意力,数值等价)。
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn.functional as F
from common import build_prompt, extract_letter, load_split
from common_native import clip_meta, load_manifest, sample_frames_native, temporal_header
from run_native import assert_on_gpu
BASELINE_ROOT = os.environ.get("BASELINE_ROOT", os.path.expanduser("~/baseline_workspace"))  # 基模权重/代码/数据所在工作目录

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE = {"vgllm": f"{BASELINE_ROOT}/code/VG-LLM", "vlm3r": f"{BASELINE_ROOT}/code/VLM-3R"}


def _vision_sdpa_blockwise(self, hidden_states, cu_seqlens, rotary_pos_emb=None, position_embeddings=None):
    """替换 VG-LLM(照 transformers 4.50 抄的)Qwen2_5_VLVisionSdpaAttention.forward。

    原实现把 32 帧 × 1296 patch ≈ 4.1 万个 token 拼成一条,建 41k×41k 的块对角 mask 再做一次 SDPA,
    实测要一次申请 102.5 GiB → OOM。官方默认的 flash_attention_2 走 varlen,本来就是每个块
    (窗口层 = 每个窗口,全注意力层 = 每帧)各算各的;新版 transformers 的 sdpa 分支也改成了逐块算。
    这里同样逐块做**精确**注意力(同长度的块叠成 batch 一次算),数学上与块对角 mask / flash varlen 等价。
    2026-09-25 对 float64 手写逐块 softmax 核对:bf16 下与原版同精度(1296 块 5.7e-4 vs 5.5e-4,窗口块均 3.5e-3)。
    ⚠️ fp32 下本卡(sm_120, torch 2.7)SDPA 默认选的内核对这种 4D 批输入给出错值(误差 0.03,强制 MATH 则 1e-6),
    所以只许 bf16/fp16 用 —— 正式运行全程 bf16。
    """
    from qwen_vl.model.modeling_qwen2_5_vl import apply_rotary_pos_emb_vision
    assert hidden_states.dtype != torch.float32, "逐块 SDPA 在本卡 fp32 默认内核下数值不对,只在 bf16 下验证过"
    seq_length = hidden_states.shape[0]
    q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    if position_embeddings is None:
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        cos, sin = emb.cos(), emb.sin()
    else:
        cos, sin = position_embeddings
    q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
    lens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    qs, ks, vs = (t.transpose(0, 1).split(lens, dim=1) for t in (q, k, v))   # 各块 (heads, len, dim)
    out = [None] * len(lens)
    for L in set(lens):
        ids = [i for i, l in enumerate(lens) if l == L]
        o = F.scaled_dot_product_attention(torch.stack([qs[i] for i in ids]), torch.stack([ks[i] for i in ids]),
                                           torch.stack([vs[i] for i in ids]), dropout_p=0.0)
        for j, i in enumerate(ids):
            out[i] = o[j]
    return self.proj(torch.cat(out, dim=1).transpose(0, 1).reshape(seq_length, -1))


class VGLLM:
    def __init__(self, a):
        sys.path.insert(0, os.path.join(CODE["vgllm"], "src"))
        from transformers import AutoConfig, AutoProcessor, AutoTokenizer
        import qwen_vl.model.modeling_qwen2_5_vl as M
        from qwen_vl.model.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGenerationWithVGGT
        from qwen_vl.data.utils import load_and_preprocess_images
        M.Qwen2_5_VLVisionSdpaAttention.forward = _vision_sdpa_blockwise
        self.prep = load_and_preprocess_images
        cfg = AutoConfig.from_pretrained(a.model)
        assert getattr(cfg, "use_geometry_encoder", False), "config 里没开 geometry encoder,不是 VG-LLM 权重?"
        self.model = Qwen2_5_VLForConditionalGenerationWithVGGT.from_pretrained(
            a.model, config=cfg, torch_dtype=torch.bfloat16, device_map={"": 0},
            attn_implementation=a.attn).eval()
        self.proc = AutoProcessor.from_pretrained(a.model, max_pixels=1605632, min_pixels=256 * 28 * 28,
                                                  padding_side="left")
        self.tok = AutoTokenizer.from_pretrained(a.model, padding_side="left")
        self.proc.tokenizer.model_max_length = self.tok.model_max_length = 12800   # 官方 max_length=12800
        print(f"  [attn] {self.model.config._attn_implementation}", flush=True)

    def __call__(self, it, imgs, idx, fps, a):
        T = it["n_frames_total"] or len(imgs)
        q = temporal_header(idx, T, fps) + "\n" + build_prompt(it, a.prompt)
        msgs = [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": [{"type": "image", "image": im} for im in imgs]
                                            + [{"type": "text", "text": q}]}]
        text = self.proc.apply_chat_template([msgs], tokenize=False, add_generation_prompt=True)
        ps, ms = self.proc.image_processor.patch_size, self.proc.image_processor.merge_size
        geo, pix = [], []
        for im in imgs:                         # 与官方逐行同构:同一张量两用,Qwen 那份裁到 28 的倍数
            x = self.prep([im])[0]
            geo.append(x)
            _, h, w = x.shape
            w -= (w // ps) % ms * ps
            h -= (h // ps) % ms * ps
            pix.append(x[:, :h, :w])
        enc = self.proc(text=text, images=pix, videos=None, padding=True, return_tensors="pt",
                        do_rescale=False)
        enc["geometry_encoder_inputs"] = [torch.stack(geo).to(self.model.device)]
        enc = enc.to(self.model.device)
        self.last_ctx = enc["input_ids"].shape[1]
        with torch.no_grad():
            gen = self.model.generate(**enc, eos_token_id=self.tok.eos_token_id,
                                      pad_token_id=self.tok.pad_token_id, do_sample=False,
                                      num_beams=1, max_new_tokens=a.max_new, use_cache=True)
        return self.proc.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True,
                                      clean_up_tokenization_spaces=False)[0]


class VLM3R:
    def __init__(self, a):
        # cut3r_spatial_encoder.py 用 sys.path.append('CUT3R')(相对 cwd)找 src.dust3r,这里给绝对路径
        sys.path[:0] = [CODE["vlm3r"], os.path.join(CODE["vlm3r"], "CUT3R")]
        # CUT3R 的 .pth 里 pickle 了 omegaconf.DictConfig,torch ≥2.6 默认 weights_only=True 拒载。
        # 官方环境 torch 2.1.1 的默认就是 False;该文件已与 Google Drive 官方原件逐字节核对(HANDOFF §2.1)。
        os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path
        name = get_model_name_from_path(a.model)
        assert all(k in name.lower() for k in ("llava", "qwen", "lora")), \
            f"目录名 {name} 必须含 llava/qwen/lora,builder 靠它选 LoRA+Qwen 分支"
        # 官方 vlm_3r.py 的 overwrite_config(eval 脚本只改了 max_frames_num=32 / conv_template)
        ow = dict(mm_resampler_type="spatial_pool", mm_spatial_pool_stride=2, mm_spatial_pool_out_channels=1024,
                  mm_spatial_pool_mode="bilinear", mm_pooling_position="after", mm_newline_position="grid",
                  add_faster_video=False, delay_load=False)
        # 官方 vlm_3r.py:171 没把 attn_implementation 传下去 → builder 默认 flash_attention_2;这里显式传 sdpa
        self.tok, self.model, self.ip, _ = load_pretrained_model(
            a.model, a.model_base, name, device_map="cuda:0", overwrite_config=ow,
            attn_implementation=a.attn)
        self.model.eval()
        self.model.tie_weights()
        self.model.to("cuda:0")
        print(f"  [attn] {self.model.config._attn_implementation}  dtype={self.model.dtype}", flush=True)
        self._check_non_lora(self.model, a.model)

    @staticmethod
    def _check_non_lora(model, path):
        """builder 用 load_state_dict(strict=False) 灌 non_lora_trainables(融合块 + 投影层),
        键对不上会**静默**留着随机初始化,模型照样吐字母 → 这里按 builder 同样的改名规则逐个核对。"""
        nl = torch.load(os.path.join(path, "non_lora_trainables.bin"), map_location="cpu")
        nl = {(k[11:] if k.startswith("base_model.") else k): v for k, v in nl.items()}
        if any(k.startswith("model.model.") for k in nl):
            nl = {(k[6:] if k.startswith("model.") else k): v for k, v in nl.items()}
        sd = model.state_dict()
        bad = [k for k, v in nl.items() if k not in sd or not torch.equal(sd[k].cpu(), v.to(sd[k].dtype))]
        print(f"  [权重自检] non_lora_trainables {len(nl) - len(bad)}/{len(nl)} 个张量已落到模型里", flush=True)
        if bad:
            raise RuntimeError(f"❌ non_lora_trainables 有 {len(bad)} 个张量没落到模型里: {bad[:5]}")

    def __call__(self, it, imgs, idx, fps, a):
        from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        from llava.conversation import SeparatorStyle, conv_templates
        from llava.mm_utils import KeywordsStoppingCriteria, tokenizer_image_token
        T = it["n_frames_total"] or len(imgs)
        video = self.ip.preprocess(np.stack([np.asarray(im) for im in imgs]),
                                   return_tensors="pt")["pixel_values"].half().cuda()
        qs = DEFAULT_IMAGE_TOKEN + "\n" + temporal_header(idx, T, fps) + "\n" + build_prompt(it, a.prompt)
        conv = conv_templates["qwen_1_5"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        ids = tokenizer_image_token(conv.get_prompt(), self.tok, IMAGE_TOKEN_INDEX,
                                    return_tensors="pt").unsqueeze(0).cuda()
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        stop = KeywordsStoppingCriteria([conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2],
                                        self.tok, ids)
        self.last_ctx = ids.shape[1]
        with torch.inference_mode():
            out = self.model.generate(inputs=ids, images=[video], attention_mask=ids.ne(pad).long().cuda(),
                                      modalities=["video"], use_cache=True, stopping_criteria=[stop],
                                      do_sample=False, num_beams=1, max_new_tokens=a.max_new)
        return self.tok.batch_decode(out, skip_special_tokens=True)[0].strip()


FAMILIES = {"vgllm": VGLLM, "vlm3r": VLM3R}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=list(FAMILIES))
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-base", default=None, help="仅 vlm3r:LoRA 的底座目录")
    ap.add_argument("--split", default="heldout_test,test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pkg", default=PKG)
    ap.add_argument("--n-frames", type=int, default=32, help="两者官方都是 max 32 帧")
    ap.add_argument("--prompt", default="plain", choices=("plain", "cot"))
    ap.add_argument("--max-new", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--attn", default="sdpa")
    a = ap.parse_args()
    if a.family == "vlm3r" and not a.model_base:
        ap.error("vlm3r 需要 --model-base")

    man = load_manifest(a.pkg)
    items = load_split(a.pkg, a.split, a.limit or None)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    if os.path.exists(a.out):
        done = {json.loads(l)["id"] for l in open(a.out) if l.strip()}
        items = [d for d in items if d["id"] not in done]
        print(f"[resume] 已有 {len(done)} 条,剩 {len(items)}", flush=True)
    print(f"[Q1-3D] {a.model} family={a.family} n={len(items)} frames≤{a.n_frames}@native512 "
          f"时间信息=给 attn={a.attn}", flush=True)

    runner = FAMILIES[a.family](a)
    assert_on_gpu(runner.model)
    t0 = time.time(); n = 0; nfb = 0
    with open(a.out, "a") as f:
        for it in items:
            imgs, idx = sample_frames_native(a.pkg, it["video_id"], a.n_frames, None)
            fps, fb = clip_meta(man, it["video_id"])
            nfb += int(fb)
            t1 = time.time()
            txt = runner(it, imgs, idx, fps, a)
            dt = time.time() - t1
            f.write(json.dumps({"id": it["id"], "pred": extract_letter(txt, a.prompt), "raw": txt.strip()[:300],
                                "t": round(dt, 4), "nf": len(imgs)}, ensure_ascii=False) + "\n")
            f.flush(); n += 1
            if n == 1:
                print(f"  [首条] 帧数={len(imgs)} ctx={runner.last_ctx} raw={txt.strip()[:40]!r} "
                      f"峰值显存={torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
            if n % 100 == 0:
                el = time.time() - t0
                print(f"  {n}/{len(items)}  {n / el:.2f} it/s  ETA {(len(items) - n) / max(n / el, 1e-9) / 60:.0f} min"
                      f"  峰值显存={torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
    print(f"[done] {n} 条 {time.time() - t0:.0f}s -> {a.out}  峰值显存="
          f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB"
          + (f"(其中 {nfb} 次用了 fps 兜底 5.0)" if nfb else ""), flush=True)


if __name__ == "__main__":
    main()
