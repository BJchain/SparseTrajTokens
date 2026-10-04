#!/usr/bin/env python3
"""TrajQA-B Q1 口径的闭源 API 推理,均走官方 SDK:
  OpenAI    GPT-6 Astra       chat.completions
  Anthropic Claude Fable 5.1  messages

用户 2026-09-25 定的口径(两家相同):
  - 帧:整段 clip 均匀 8 帧(与 Q2 / common.py 同算法 linspace(0,T-1,8).round()),**512 原图**,
        原 JPEG 字节直接 base64 发送(不解码、不重压)。
  - 时间信息:给(Q1)。写法照 InternVL 的 Q1:每帧前一段文本 "Frame{真实帧号}:",
        题面前一行 temporal_header(fps / 时长 / 所见帧号)。
  - 题面 build_prompt(item,'plain')、取字母 extract_letter(common.py 不改)。
  - 解码:两家都是推理模型,做不到 greedy;推理强度都取最低档 low;--max-completion-tokens 只作输出上限(含推理)。
        OpenAI:temperature 只收默认 1、不收 max_tokens → max_completion_tokens;reasoning_effort=low
                (none / minimal 返回 400;上限=4 时推理把额度吃光,输出为空)。
        Anthropic:temperature / top_k 返回 400,thinking 常开(不传 thinking 参数 = adaptive)→ max_tokens;
                output_config.effort=low。拒答(stop_reason=refusal)照实记为该题输出(取不到字母 → "?",算错),
                不开 server-side fallback(那会换别的模型作答)。
  - 子集:每个 split 的 id 排序后按 (seed, split) 打乱,取前 --per-split 题。
        试点 50 是正式 500 的前缀,配置不变时续跑直接复用。
  - 取答案 parse_answer():Fable 在 effort=low 时不走 thinking,而是先在正文里写推理、最后一行给字母;
        extract_letter 的 plain 规则取"全文第一个 A–D",会从 "a car" / "Between" 里误抓(冒烟 2/2 抓错)。
        所以:整段就是一个字母(bare,Astra 全是这种,与 plain 结果相同)→ 最后一个非空行去掉 markdown / 标点 /
        "Answer:" 后是单个字母(last_line)→ common.extract_letter(txt,'cot')(cot,需抽查)→ 取不到记 "?"(none)。
        pred 里记 rule,raw 存完整原文,规则再改可以离线重算,不必重新调用。

与旧 run_openai_api.py 的区别(那个脚本对这类模型每题 400,且失败会写 "?" 并被续跑当成已完成):
  - 失败**不写进 pred**,记到 <out>.errors.jsonl,续跑时自动重试。
    401/403/404、同一种 4xx 连续 3 次 = 配置错误,余额耗尽 → 立即中止整轮;错误过多也中止。
  - 预算护栏 --budget-usd:按每题 usage 累计花费(含 out 里已有记录),将超预算时停止派发新题。
  - 自动去掉 socks5 的 all_proxy(SDK 缺 socksio 时建 client 就 ImportError);
    保留 http(s)_proxy(本机直连 api.openai.com 超时、api.anthropic.com 403)。
  - <out>.meta.json 记口径;口径不同的续跑直接拒绝,防止两套配置混在一个文件里。

    python tools/run_api_native.py --model gpt-6-astra      --per-split 50 --budget-usd 100 --out preds/Q1_GPT-6-Astra.jsonl
    python tools/run_api_native.py --model claude-fable-5-1 --per-split 50 --budget-usd 100 --out preds/Q1_Claude-Fable-5.1.jsonl
"""
import argparse, base64, collections, glob, json, os, random, re, sys, threading, time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import LETTERS, build_prompt, extract_letter, frames_dir, load_split
from common_native import clip_meta, load_manifest, sample_frames_native, temporal_header

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# USD / 1M tokens:(input, cached input, output)。gpt-6-astra:OpenAI 官方模型页 2026-09-25 查得;
# claude-fable-5-1:Anthropic 官方价目(缓存读 $0.25;本脚本不设 cache_control,不会产生缓存写)。
PRICES = {"gpt-6-astra": (10.0, 1.0, 50.0), "claude-fable-5-1": (10.0, 0.25, 50.0)}
PROVIDER = {"gpt-6-astra": "openai", "claude-fable-5-1": "anthropic"}
KEY_ENV = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}


def subset(items, per_split, seed):
    """每个 split:按 id 排序 → 按 (seed, split) 打乱 → 取前 per_split 题。小 N 是大 N 的前缀。"""
    by = collections.defaultdict(list)
    for d in items:
        by[d["split"]].append(d)
    out = []
    for sp in sorted(by):
        ds = sorted(by[sp], key=lambda d: d["id"])
        random.Random(f"{seed}:{sp}").shuffle(ds)
        out += ds[:per_split] if per_split else ds
    return out


def load_frames(pkg, it, n_frames):
    """与 sample_frames_native 同一套下标;返回 (0-based 下标, 原 JPEG 字节, clip 帧数)。"""
    fs = sorted(glob.glob(os.path.join(frames_dir(pkg, it["video_id"]), "*.jpg")))
    _, idx = sample_frames_native(pkg, it["video_id"], n_frames, None)
    return idx, [open(fs[i], "rb").read() for i in idx], len(fs)


def build_content(it, idx, jpgs, T, fps, provider="openai"):
    """两家内容结构相同(文本 / 图片交替,最后是时间行 + 题面),只有图片块的写法不同。"""
    content = []
    for i, b in zip(idx, jpgs):
        b64 = base64.b64encode(b).decode()
        content.append({"type": "text", "text": f"Frame{i+1}:"})                 # 真实帧号,同 InternVL Q1
        content.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}}
                       if provider == "openai" else
                       {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}})
    content.append({"type": "text", "text": temporal_header(idx, T, fps) + "\n" + build_prompt(it, "plain")})
    return content


_LABEL = re.compile(r"^\W*(?:the\s+)?(?:final\s+)?(?:answer|option|choice)(?:\s+is)?\W*", re.I)
_STRIP = re.compile(r"[\s*_`#.:：()\[\]\"']+")


def parse_answer(txt):
    """-> (字母, 规则)。规则见文件头;bare / last_line 是明确作答,cot 是兜底(需抽查)。"""
    s = txt.strip()
    if s.rstrip(".").upper() in LETTERS:
        return s.rstrip(".").upper(), "bare"
    lines = [l for l in s.splitlines() if l.strip()]
    if lines:
        c = _STRIP.sub("", _LABEL.sub("", lines[-1].strip()))
        if len(c) == 1 and c.upper() in LETTERS:
            return c.upper(), "last_line"
    L = extract_letter(txt, "cot")
    return L, ("cot" if L != "?" else "none")


def cost_of(u, price):
    pin, pcached, pout = price
    c = u.get("cached", 0)
    return ((u["in"] - c) * pin + c * pcached + u["out"] * pout) / 1e6


def out_of_credit(err):
    """OpenAI:429 insufficient_quota;Anthropic:400 "Your credit balance is too low ..."。"""
    return (getattr(err, "code", None) in ("insufficient_quota", "credit_balance_exhausted")
            or "credit balance" in str(err).lower())


def err_type(err):
    """4xx 的错误类别:OpenAI 的 body 是 error 对象(有 code);Anthropic 在 body['error']['type']。"""
    body = getattr(err, "body", None)
    if isinstance(body, dict):
        e = body.get("error", body)
        if isinstance(e, dict):
            return e.get("code") or e.get("type")
    return getattr(err, "code", None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-6-astra", choices=sorted(PRICES))
    ap.add_argument("--split", default="heldout_test,test")
    ap.add_argument("--per-split", type=int, default=0, help="每个 split 取前 N 题(按 seed 打乱);0=全部")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="冒烟用:本次最多派发几题")
    ap.add_argument("--n-frames", type=int, default=8)
    ap.add_argument("--reasoning-effort", default="low", help="OpenAI reasoning_effort / Anthropic output_config.effort")
    ap.add_argument("--max-completion-tokens", type=int, default=2048,
                    help="输出上限(含推理 / thinking);Anthropic 即 max_tokens")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--budget-usd", type=float, required=True, help="含 out 里已有记录的累计花费上限")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pkg", default=PKG)
    a = ap.parse_args()
    prov = PROVIDER[a.model]

    for k in ("all_proxy", "ALL_PROXY"):
        os.environ.pop(k, None)
    if not (os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY")):
        print("⚠️ 没有 https_proxy:本机直连 OpenAI 会超时、Anthropic 会 403", flush=True)
    if not os.environ.get(KEY_ENV[prov]):
        sys.exit(f"❌ 环境里没有 {KEY_ENV[prov]}")

    meta = {"model": a.model, "n_frames": a.n_frames, "max_side": "native512", "time_info": True,
            "frame_labels": "Frame{i+1}: + temporal_header", "prompt": "plain",
            "reasoning_effort": a.reasoning_effort, "max_completion_tokens": a.max_completion_tokens,
            "temperature": "default(1)" if prov == "openai" else "not settable (API rejects)",
            "seed": a.seed, "endpoint": "chat.completions" if prov == "openai" else "messages"}
    mpath = a.out.replace(".jsonl", "") + ".meta.json"
    if os.path.exists(mpath):
        old = json.load(open(mpath))
        diff = {k: (old.get(k), v) for k, v in meta.items() if old.get(k) != v}
        if diff:
            sys.exit(f"❌ 口径与已有 {mpath} 不同,拒绝续跑(换 --out):{diff}")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
        json.dump(dict(meta, created=time.strftime("%Y-%m-%d %H:%M:%S")), open(mpath, "w"),
                  ensure_ascii=False, indent=1)

    price = PRICES[a.model]
    man = load_manifest(a.pkg)
    items = subset(load_split(a.pkg, a.split), a.per_split, a.seed)
    n_sel = len(items)
    done, spent0 = set(), 0.0
    if os.path.exists(a.out):
        for l in open(a.out):
            if l.strip():
                d = json.loads(l); done.add(d["id"]); spent0 += cost_of(d["usage"], price)
    items = [d for d in items if d["id"] not in done]
    if a.limit:
        items = items[:a.limit]
    print(f"[Q1-API] {a.model} split={a.split} per_split={a.per_split or 'ALL'} seed={a.seed} → 选中 {n_sel},"
          f"已有 {len(done)},本次派发 {len(items)} | {a.n_frames}帧@512原图 给时间 effort={a.reasoning_effort} "
          f"max_completion={a.max_completion_tokens} workers={a.workers} | 已花 ${spent0:.4f} / 预算 ${a.budget_usd}",
          flush=True)
    if not items:
        return 0

    # 首题的文本部分原样打出来,便于核对 prompt(不打 base64)
    idx, jpgs, T = load_frames(a.pkg, items[0], a.n_frames)
    fps, _ = clip_meta(man, items[0]["video_id"])
    c0 = build_content(items[0], idx, jpgs, items[0]["n_frames_total"] or T, fps, prov)
    print("  [首题文本] " + " | ".join(p["text"] if p["type"] == "text" else "<jpeg>" for p in c0), flush=True)

    if prov == "openai":
        import openai as sdk
        client = sdk.OpenAI(max_retries=2, timeout=180)
        TRANSIENT = (sdk.RateLimitError, sdk.InternalServerError, sdk.APIConnectionError, sdk.APITimeoutError)

        def call(content):
            r = client.chat.completions.create(
                model=a.model, messages=[{"role": "user", "content": content}],
                max_completion_tokens=a.max_completion_tokens, reasoning_effort=a.reasoning_effort)
            ch = r.choices[0]
            u, pd = r.usage, getattr(r.usage, "prompt_tokens_details", None)
            cd = getattr(r.usage, "completion_tokens_details", None)
            usage = {"in": u.prompt_tokens, "cached": getattr(pd, "cached_tokens", 0) or 0,
                     "out": u.completion_tokens, "reason": getattr(cd, "reasoning_tokens", 0) or 0}
            return ch.message.content or "", ch.finish_reason, usage, r.model, getattr(ch.message, "refusal", None)
    else:
        import anthropic as sdk
        client = sdk.Anthropic(max_retries=2, timeout=300)
        TRANSIENT = (sdk.RateLimitError, sdk.OverloadedError, sdk.ServiceUnavailableError,
                     sdk.DeadlineExceededError, sdk.InternalServerError, sdk.APIConnectionError)

        def call(content):
            # thinking 常开:不传 thinking 参数(= adaptive);不传 temperature / top_k(该模型返回 400)
            r = client.messages.create(
                model=a.model, max_tokens=a.max_completion_tokens,
                messages=[{"role": "user", "content": content}],
                output_config={"effort": a.reasoning_effort})
            u = r.usage
            cached, created = u.cache_read_input_tokens or 0, u.cache_creation_input_tokens or 0
            usage = {"in": u.input_tokens + cached + created, "cached": cached, "out": u.output_tokens,
                     "reason": getattr(u.output_tokens_details, "thinking_tokens", 0) or 0}
            txt = "".join(b.text for b in r.content if b.type == "text")
            refusal = str(r.stop_details) if r.stop_reason == "refusal" else None
            return txt, r.stop_reason, usage, r.model, refusal

    worst = ((a.n_frames * 400 + 600) * price[0] + a.max_completion_tokens * price[2]) / 1e6
    lock = threading.Lock()
    st = {"spent": spent0, "n": 0, "err": 0, "abort": None, "sig": None, "rep": 0, "tmis": 0}
    finish, letters, rules, nonbare = collections.Counter(), collections.Counter(), collections.Counter(), []
    tok = collections.Counter()
    fout = open(a.out, "a")
    ferr = open(a.out.replace(".jsonl", "") + ".errors.jsonl", "a")
    t_start = time.time()

    def work(it):
        idx, jpgs, T = load_frames(a.pkg, it, a.n_frames)
        if it["n_frames_total"] and it["n_frames_total"] != T:
            with lock:
                st["tmis"] += 1
        fps, _ = clip_meta(man, it["video_id"])
        content = build_content(it, idx, jpgs, it["n_frames_total"] or T, fps, prov)
        res = err = None
        for attempt in range(4):
            if st["abort"]:
                return
            t0 = time.time()
            try:
                res = call(content)
                break
            except TRANSIENT as e:                      # 限流 / 5xx / 过载 / 网络:退避重试
                err = e
                if out_of_credit(e):
                    st["abort"] = st["abort"] or f"账户余额耗尽({type(e).__name__}),充值后重跑同一条命令续跑"
                    break                               # 余额耗尽不是限流,重试无用
                time.sleep(5 * 2 ** attempt)
            except sdk.APIStatusError as e:             # 其余 4xx:不重试
                err = e
                if out_of_credit(e):
                    st["abort"] = st["abort"] or f"账户余额耗尽({type(e).__name__}),充值后重跑同一条命令续跑"
                break
        dt = time.time() - t0
        if res is None:
            code = getattr(err, "status_code", None)
            with lock:
                ferr.write(json.dumps({"id": it["id"], "status": code, "time": time.strftime("%H:%M:%S"),
                                       "error": f"{type(err).__name__}: {str(err)[:400]}"},
                                      ensure_ascii=False) + "\n"); ferr.flush()
                st["err"] += 1
                if code is not None and 400 <= code < 500 and code != 429:
                    sig = (code, err_type(err))
                    st["rep"] = st["rep"] + 1 if st["sig"] == sig else 1
                    st["sig"] = sig
                    if code in (401, 403, 404) or st["rep"] >= 3:
                        st["abort"] = st["abort"] or f"HTTP {code} ×{st['rep']}: {str(err)[:300]}"
                if st["err"] > max(5, 0.02 * (st["n"] + st["err"])):
                    st["abort"] = st["abort"] or f"错误过多:{st['err']} 条(成功 {st['n']})"
            return
        txt, fin, usage, model, refusal = res
        pred, rule = parse_answer(txt)
        rec = {"id": it["id"], "pred": pred, "rule": rule, "raw": txt.strip(),
               "t": round(dt, 4), "nf": len(idx), "fidx": [i + 1 for i in idx],
               "finish": fin, "usage": usage, "model": model}
        if refusal:
            rec["refusal"] = str(refusal)[:300]
        with lock:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n"); fout.flush()
            st["spent"] += cost_of(usage, price); st["n"] += 1; st["rep"] = 0
            finish[fin] += 1; letters[rec["pred"]] += 1; rules[rule] += 1
            tok.update({k: v for k, v in usage.items()})
            if rule in ("cot", "none"):
                nonbare.append((it["id"], txt.strip()[-80:]))
            if st["n"] % 50 == 0:
                el = time.time() - t_start
                print(f"  {st['n']}/{len(items)}  {st['n']/el:.2f} 题/s  累计 ${st['spent']:.3f}  "
                      f"错误 {st['err']}  ETA {(len(items)-st['n'])/max(st['n']/el,1e-9)/60:.1f} min", flush=True)

    stop = None
    with ThreadPoolExecutor(a.workers) as ex:
        pending = set()
        for it in items:
            if st["abort"]:
                break
            if st["spent"] + (len(pending) + 1) * worst > a.budget_usd:
                stop = f"预算护栏:已花 ${st['spent']:.3f} + 在途上限 {len(pending)+1}×${worst:.3f} > ${a.budget_usd}"
                break
            pending.add(ex.submit(work, it))
            if len(pending) >= a.workers * 2:
                fin_, pending = wait(pending, return_when=FIRST_COMPLETED)
                for f in fin_:
                    f.result()                      # work 里的意外异常(缺帧等)直接抛出,不静默
        for f in wait(pending)[0]:
            f.result()
    fout.close(); ferr.close()

    el = time.time() - t_start
    n = max(st["n"], 1)
    print(f"[done] 成功 {st['n']} / 派发 {len(items)},错误 {st['err']},墙钟 {el:.0f}s({st['n']/max(el,1e-9):.2f} 题/s)")
    print(f"  本次花费 ${st['spent']-spent0:.4f},累计 ${st['spent']:.4f} / 预算 ${a.budget_usd}")
    print(f"  平均 token/题:in {tok['in']/n:.0f}(cached {tok['cached']/n:.0f})  out {tok['out']/n:.1f}"
          f"(reasoning {tok['reason']/n:.1f})")
    print(f"  finish_reason {dict(finish)}  字母 {dict(letters)}  取答案规则 {dict(rules)}"
          + (f"\n  ⚠️ 兜底/取不到 {len(nonbare)} 条(需抽查),结尾例:{nonbare[:5]}" if nonbare else ""))
    if st["tmis"]:
        print(f"  ⚠️ {st['tmis']} 题 n_frames_total 与帧目录里的 jpg 数不一致")
    if stop:
        print(f"⛔ {stop}")
    if st["abort"]:
        print(f"❌ 中止:{st['abort']}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
