#!/usr/bin/env python3
"""TrajQA-B 走 OpenAI 兼容接口的 baseline 推理。

两种用法同一个脚本:
  1) 闭源 baseline(GPT-4o / Gemini 走 OpenAI 兼容网关):
     OPENAI_API_KEY=... python tools/run_openai_api.py --model gpt-4o --split heldout_test --out preds/gpt4o.jsonl
  2) 开源 baseline 用 vLLM 起服务后当本地 API 跑(65k 题时比 transformers 快 5-10 倍):
     vllm serve Qwen/Qwen2.5-VL-7B-Instruct --limit-mm-per-prompt image=8 --port 8000
     python tools/run_openai_api.py --base-url http://localhost:8000/v1 --api-key EMPTY \
        --model Qwen/Qwen2.5-VL-7B-Instruct --split heldout_test --out preds/qwen25vl7b.jsonl

⚠️ 图片按 base64 内联。8 帧 384px ≈ 每题 2.5k-9k input token(按各家计价规则不同),
   跑前先用 --limit 50 估一次单价再放开(65,360 题 × 8 图不是小钱)。
断点续跑:--out 已存在的 id 自动跳过。
"""
import argparse, base64, io, json, os, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import build_prompt, extract_letter, load_split, sample_frames

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_lock = threading.Lock()


def b64(im, quality=90):
    b = io.BytesIO(); im.save(b, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default=None, help="vLLM: http://localhost:8000/v1")
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    ap.add_argument("--split", default="heldout_test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pkg", default=PKG)
    ap.add_argument("--n-frames", type=int, default=8)
    ap.add_argument("--max-side", type=int, default=384)
    ap.add_argument("--no-frames", action="store_true", help="盲测地板")
    ap.add_argument("--prompt", default="plain", choices=("plain", "cot"))
    ap.add_argument("--max-new", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.prompt == "cot" and a.max_new < 64:
        a.max_new = 256

    from openai import OpenAI
    client = OpenAI(api_key=a.api_key, base_url=a.base_url)

    items = load_split(a.pkg, a.split, a.limit or None)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    if os.path.exists(a.out):
        done = {json.loads(l)["id"] for l in open(a.out) if l.strip()}
        items = [d for d in items if d["id"] not in done]
        print(f"[resume] 已有 {len(done)} 条,剩 {len(items)}", flush=True)
    print(f"[run] {a.model} split={a.split} n={len(items)} "
          f"frames={0 if a.no_frames else a.n_frames}@{a.max_side}", flush=True)

    fout = open(a.out, "a")
    t0 = time.time(); cnt = [0]; usage = [0, 0]

    def work(it):
        content = []
        if not a.no_frames:
            imgs, _ = sample_frames(a.pkg, it["video_id"], a.n_frames, a.max_side)
            content += [{"type": "image_url", "image_url": {"url": b64(im)}} for im in imgs]
        content.append({"type": "text", "text": build_prompt(it, a.prompt)})
        for attempt in range(5):
            try:
                _t0 = time.time()
                r = client.chat.completions.create(
                    model=a.model, messages=[{"role": "user", "content": content}],
                    max_tokens=a.max_new, temperature=0)
                _dt = time.time() - _t0
                txt = r.choices[0].message.content or ""
                u = getattr(r, "usage", None)
                rec = {"id": it["id"], "pred": extract_letter(txt, a.prompt), "raw": txt.strip()[:300],
                       "t": round(_dt, 4)}
                with _lock:
                    if u:
                        usage[0] += u.prompt_tokens or 0; usage[1] += u.completion_tokens or 0
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n"); fout.flush()
                    cnt[0] += 1
                    if cnt[0] % 100 == 0:
                        el = time.time() - t0
                        print(f"  {cnt[0]}/{len(items)} {cnt[0]/el:.1f} it/s "
                              f"in_tok={usage[0]:,} out_tok={usage[1]:,} "
                              f"ETA {(len(items)-cnt[0])/max(cnt[0]/el,1e-9)/60:.0f} min", flush=True)
                return
            except Exception as e:                       # 限流/瞬时错误退避重试
                if attempt == 4:
                    with _lock:
                        fout.write(json.dumps({"id": it["id"], "pred": "?",
                                               "error": str(e)[:200]}) + "\n"); fout.flush()
                    return
                time.sleep(2 ** attempt)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(work, items))
    fout.close()
    print(f"[done] {cnt[0]} 条 {time.time()-t0:.0f}s  input_tokens={usage[0]:,} "
          f"output_tokens={usage[1]:,}  -> {a.out}")


if __name__ == "__main__":
    main()
