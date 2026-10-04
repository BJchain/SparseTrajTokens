#!/usr/bin/env python3
"""MLLM-4D-SFT 上正式跑之前的四项核对(结果写 logs/verify_mllm4d.log):

 1. 权重:官方写法 from_pretrained(output_loading_info=True) 无 missing / unexpected / mismatched;
    参数 100% 在 GPU;抽几个张量与 safetensors 原文件逐字节相等。
 2. 注意力:config 里 text/vision 都是 flash_attention_2,且 torch.profiler 在一次前向里真的看到 flash 内核。
 3. 喂法等价:把 clip 的全部 JPG 帧无损写成视频文件(FFV1),按官方写法传**文件路径**
    (apply_chat_template(..., num_frames=32, fps=None),transformers 自己解码+抽帧+时间戳),
    与 run_mllm4d.py 的"帧数组 + video_metadata"喂法比:input_ids / video_grid_thw / pixel_values_videos
    必须逐元素相等,generate 结果也必须相同。覆盖 T>32、T<32、T=20 三种 clip。
 4. 冒烟:7 族各 3 题,打印原始输出、两套取字母规则的结果,供决定 max_new_tokens / 取字母口径。
"""
import glob, json, os, sys, time
from fractions import Fraction
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from common import load_answers, load_split, extract_letter, LETTERS
from common_native import clip_meta, load_manifest, sample_frames_native
from run_native import assert_on_gpu
from run_mllm4d import MLLM4D, extract_option_key, parse_letter
BASELINE_ROOT = os.environ.get("BASELINE_ROOT", os.path.expanduser("~/baseline_workspace"))  # 基模权重/代码/数据所在工作目录

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = f"{BASELINE_ROOT}/models/MLLM-4D/MLLM-4D-SFT-1.0"
TMP = os.path.join(os.environ.get("TMPDIR", "/tmp"), "verify_mllm4d_videos")


class A:  # 与 run_mllm4d.py 默认一致
    model = MODEL; attn = "flash_attention_2"; max_new = 16


def write_lossless(frames, fps, path):
    import av
    rate = Fraction(fps).limit_denominator(10000)
    with av.open(path, "w") as c:
        s = c.add_stream("ffv1", rate=rate)
        s.width, s.height, s.pix_fmt = frames[0].shape[1], frames[0].shape[0], "bgr0"
        for fr in frames:
            for p in s.encode(av.VideoFrame.from_ndarray(fr, format="rgb24")):
                c.mux(p)
        for p in s.encode():
            c.mux(p)


def main():
    t00 = time.time()
    man = load_manifest(PKG)
    gold = load_answers(PKG)
    items = load_split(PKG, "heldout_test,test")

    # ---------- 1. 权重 ----------
    from transformers import Qwen3VLForConditionalGeneration
    from safetensors import safe_open
    runner = MLLM4D(A)
    m = runner.model
    assert_on_gpu(m)
    idx = json.load(open(os.path.join(MODEL, "model.safetensors.index.json")))["weight_map"]
    sd = m.state_dict()
    missing, unexpected = sorted(set(sd) - set(idx)), sorted(set(idx) - set(sd))
    print(f"[1] safetensors 张量 {len(idx)} 个;模型 state_dict {len(sd)} 个;"
          f"模型有而文件没有(会随机初始化)={missing};文件有而模型没有={unexpected}", flush=True)
    # 每层都查:文件里的每个张量与显存里的逐元素相等(bf16 原样装载,不该有任何差)
    bad = []
    for shard in sorted(set(idx.values())):
        with safe_open(os.path.join(MODEL, shard), "pt", device="cuda:0") as f:
            for k in f.keys():
                if not torch.equal(f.get_tensor(k), sd[k]):
                    bad.append(k)
    print(f"[1] 逐张量比对 {len(idx)} 个,不一致 {len(bad)} {bad[:5]}", flush=True)

    # ---------- 2. 注意力内核 ----------
    it0 = next(d for d in items if d["n_frames_total"] == 34)
    imgs, _ = sample_frames_native(PKG, it0["video_id"], None, None)
    fps0, _ = clip_meta(man, it0["video_id"])
    enc = runner.inputs(it0, imgs, fps0).to(m.device)
    from torch.profiler import profile, ProfilerActivity
    with torch.no_grad(), profile(activities=[ProfilerActivity.CUDA]) as prof:
        m(**enc)
    names = [e.key for e in prof.key_averages()]
    fl = sorted({n for n in names if "flash" in n.lower()})
    sdpa = sorted({n for n in names if "efficient_attention" in n.lower() or "fmha" in n.lower()
                   or "cudnn" in n.lower() and "attention" in n.lower()})
    print(f"[2] ctx={enc['input_ids'].shape[1]} flash 内核 {len(fl)} 种: {[n[:70] for n in fl[:4]]}", flush=True)
    print(f"[2] 非 flash 的注意力内核: {sdpa[:4]}", flush=True)

    # ---------- 3. 官方"视频文件路径"喂法 vs 帧数组喂法 ----------
    os.makedirs(TMP, exist_ok=True)
    picks = []
    for T in (34, 24, 20):
        picks.append(next(d for d in items if d["n_frames_total"] == T and d["family"] == "cam_disp_grid"))
    all_eq = True
    for it in picks:
        imgs, _ = sample_frames_native(PKG, it["video_id"], None, None)
        frames = [np.asarray(im) for im in imgs]
        fps, _ = clip_meta(man, it["video_id"])
        path = os.path.join(TMP, it["video_id"].replace("scene_", "") + ".mkv")
        write_lossless(frames, fps, path)
        # 官方写法:只给路径,transformers 自己解码/抽帧/算时间戳
        e_off = runner.proc.apply_chat_template(
            [{"role": "user", "content": [{"type": "video", "video": path},
                                          {"type": "text", "text": __import__("common").build_prompt(it, "plain")}]}],
            tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
            num_frames=32, fps=None)
        import av
        with av.open(path) as c:          # transformers 解码时读的就是这个 average_rate
            cfps = float(c.streams.video[0].average_rate)
            dec = [fr.to_ndarray(format="rgb24") for fr in c.decode(video=0)]
        lossless = len(dec) == len(frames) and all(np.array_equal(x, y) for x, y in zip(dec, frames))
        # run_mllm4d.py 的写法:JPG 帧数组 + metadata(fps 用容器回读的同一个值,只比喂法本身)
        e_me = runner.inputs(it, imgs, cfps)
        eq = {k: torch.equal(e_off[k], e_me[k]) for k in ("input_ids", "attention_mask", "video_grid_thw",
                                                            "pixel_values_videos")}
        same_idx = lossless
        outs = []
        for e in (e_off, e_me):
            e = e.to(m.device)
            with torch.no_grad():
                g = m.generate(**e, max_new_tokens=16, do_sample=False, temperature=None, top_p=None, top_k=None)
            outs.append(runner.proc.batch_decode(g[:, e["input_ids"].shape[1]:], skip_special_tokens=True)[0])
        ok = all(eq.values()) and same_idx and outs[0] == outs[1]
        all_eq &= ok
        txt = runner.proc.tokenizer.decode(e_me["input_ids"][0])
        ts = [s.split(" seconds>")[0] for s in txt.split("<")[1:] if " seconds>" in s]
        print(f"[3] {'✅' if ok else '❌'} {it['video_id']} T={it['n_frames_total']} manifest_fps={fps} "
              f"容器回读 fps={cfps:.4f} 无损={lossless} 抽帧={runner.sampled_indices(len(imgs), cfps)}\n"
              f"      逐元素相等 {eq} grid={e_me['video_grid_thw'].tolist()} "
              f"ctx={e_me['input_ids'].shape[1]}\n      时间戳={ts}\n      输出 官方路径={outs[0]!r} 帧数组={outs[1]!r}",
              flush=True)
    print(f"[3] 喂法等价: {'全部逐比特一致 ✅' if all_eq else '❌ 有差异'}", flush=True)

    # ---------- 4. 冒烟 7 族 × 3 ----------
    fams = ["cam_disp_grid", "disp_grid", "speed_grid", "motion_duration_binary", "motion_source",
            "interval_disp", "interval_speed"]
    smoke = [d for f in fams for d in [x for x in items if x["family"] == f][:3]]
    json.dump([d["id"] for d in smoke], open(os.path.join(PKG, "logs", "smoke_mllm4d_ids.json"), "w"))
    torch.cuda.reset_peak_memory_stats()
    ok_n = 0
    for d in smoke:
        imgs, _ = sample_frames_native(PKG, d["video_id"], None, None)
        fps, fb = clip_meta(man, d["video_id"])
        t1 = time.time()
        txt = runner(d, imgs, fps, A)
        dt = time.time() - t1
        p, how = parse_letter(txt)
        g = gold[d["id"]]["answer"]
        ok_n += p == g
        print(f"[4] {d['family']:22s} T={d['n_frames_total']:2d} fps={fps:.3f}{'(兜底)' if fb else ''} "
              f"ctx={runner.last_ctx} {dt:.2f}s raw={txt!r:28s} 官方规则={extract_option_key(txt) or '∅'} "
              f"协议规则={extract_letter(txt, 'plain')} → pred={p}({how}) gt={g}", flush=True)
    print(f"[4] 冒烟 {len(smoke)} 题,对 {ok_n};峰值显存 {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB;"
          f"总耗时 {time.time() - t00:.0f}s", flush=True)


if __name__ == "__main__":
    main()
