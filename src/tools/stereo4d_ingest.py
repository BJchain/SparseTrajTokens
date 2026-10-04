"""Stereo4D 全量取数:从本地 tar 抽 mp4 + 下官方 npz -> 复用 stereo4d_adapter 建帧 -> 喂队列。

为什么要"按波次"而不是一次性备齐:官方 npz 均 9.245 MiB/clip,108k 全量就是 **884 GiB**,
而它的大字段 `track3d_world` 只在 GT 稠密前端跑那一次被读。所以取数必须和处理交错:
每波只落 WAVE 个 clip 的 npz,跑完就由 compress_gt_npz.py 压成 3.95 KiB 的相机切片。
峰值 = WAVE ×(npz 9.245 + jpg 2.39 + mp4 0.988)MiB,2000 个 clip 只要 ~25 GiB。

mp4 是**用完即删**的中间物:tar 里永远还有,而抽好的 jpg 才是训练要读的。

抽帧口径不自己造:直接调 `src/stage0/stereo4d_adapter.py::prepare_clip`,也就是
2026-07-31 用户拍板的**自适应抽帧**(每 clip 选最大 stride 使帧数落在 20~34)。

用法:
  python3 src/tools/stereo4d_ingest.py --queue-list <文件> --wave 2000 --min-free-g 60
"""
from __future__ import annotations

import argparse, base64, collections, hashlib, os, pickle, subprocess, sys, tarfile, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.stage0.stereo4d_adapter import prepare_clip                      # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TARS = {"train": "data/stereo4d/raw_mp4/train_mp4s.tar",
        "test":  "data/stereo4d/raw_mp4/test_mp4s.tar"}
IDX_PKL = "data/stereo4d/tar_index.pkl"
FRAMES = "data/stereo4d/frames"
STAGE_NPZ = "data/stereo4d/_ingest_npz"
STAGE_MP4 = "data/stereo4d/_ingest_mp4"
NO_NPZ_LIST = "data/stereo4d/no_npz_clips.txt"   # 官方没发布 npz 的 clip,别反复重试
URL = "https://storage.googleapis.com/stereo4d/{split}/{clip}.npz"
SUFFIX = "-left_rectified.mp4"


def build_index() -> dict:
    """clip -> (split, offset, size)。扫一遍 tar 头(~2 分钟),存盘复用 —— 之后就能 seek 直取,
    不用每波都线性扫 95 GiB。"""
    if os.path.exists(IDX_PKL):
        return pickle.load(open(IDX_PKL, "rb"))
    idx = {}
    for split, tp in TARS.items():
        print(f"[index] 扫 {tp} ...", flush=True)
        with tarfile.open(tp) as tf:
            for m in tf:
                if m.name.endswith(SUFFIX):
                    idx[m.name[:-len(SUFFIX)]] = (split, m.offset_data, m.size)
    os.makedirs(os.path.dirname(IDX_PKL), exist_ok=True)
    pickle.dump(idx, open(IDX_PKL, "wb"))
    print(f"[index] {len(idx)} 个 clip", flush=True)
    return idx


def extract_mp4(clip, split, off, size) -> str:
    d = os.path.join(STAGE_MP4, clip); os.makedirs(d, exist_ok=True)
    out = os.path.join(d, clip + SUFFIX)
    if os.path.exists(out) and os.path.getsize(out) == size:
        return out
    with open(TARS[split], "rb") as f:
        f.seek(off)
        with open(out + ".part", "wb") as g:
            left = size
            while left:
                b = f.read(min(1 << 20, left))
                if not b: raise IOError(f"{clip}: tar 读断")
                g.write(b); left -= len(b)
    os.replace(out + ".part", out)
    return out


class NoNpz(Exception):
    """官方没发布这个 clip 的 npz(tar 里有 mp4 不代表 bucket 里有 npz —— 实测 40 个视频里 6 个没有)。"""


# ═══════════════════════════════════════════════════════════════════════════════
# 取数线路 + 续传(2026-09-04 重写)
#
# 为什么重写(当天实测,见 memory project_ingest_proxy_collapse_20260904):
#  * 生产原来把【别人的】代理 127.0.0.1:17890 写死进环境,只在 curl rc=7(代理拒连)时才改直连。
#    09-04 代理上游塌到单流 93 KB/s、12 路并发合计只剩 38 KB/s,而同一时刻直连 IPv4 单流
#    0.5~9 MB/s、12 路合计 5.1 MB/s —— 「代理活着但极慢」这种状态旧逻辑永远不切,取数 4 小时只成 37 个。
#    08-29 选代理的依据(代理中位 4.84 vs 直连 0.11 MB/s)在 09-04 完全反过来了:
#    **哪条快是随天变的,不能写死,要量了再走**。
#  * curl 的 --retry 是把输出截断后【从 0 字节重下】,配上每次尝试的 --max-time,大于
#    「上限 × 速度」的文件永远下不完,还烧 3 倍 worker 时间。现在改成 Python 自己循环续传:
#    每次只请求还缺的字节 `-r a+got-b` 追加到文件尾,超时/断连都不清零。
#  * 有了续传,分段并发不再是 p^4(任一段失败整体作废)—— 段各自续传到齐。
#
# 线路:direct(--noproxy '*')/ proxy(-x $S4D_FETCH_PROXY,默认 127.0.0.1:17890;设为空串=禁用)。
# 探测:开跑 / 每波开始 / 每 S4D_PROBE_EVERY 秒 / 当前线路近 12 次一半失败 时,两条线路各下同一
#       对象的前 2 MB(并行,同负载下公平),另一条快 S4D_PROBE_SWITCH(1.5)倍以上才切(防抖)。
# 兜底:单次尝试失败后,下一次尝试先换另一条线路试一把(不等探测)。
# 完整性:拼好后 长度==Content-Length + md5==x-goog-hash + 魔数 PK,任一不符即丢弃重来。
# ═══════════════════════════════════════════════════════════════════════════════
_PROXY = os.environ.get("S4D_FETCH_PROXY", "").strip()   # 可选 HTTP 代理(空 = 只走直连)
ROUTES = {"direct": ["--noproxy", "*"]}
if _PROXY:
    ROUTES["proxy"] = ["-x", _PROXY]
ROUTE = {"name": os.environ.get("S4D_ROUTE", "direct"),      # 当前线路(S4D_ROUTE 只定起步)
         "fixed": os.environ.get("S4D_ROUTE_FIXED", "0") == "1",   # 1=不探测不切换(调试/对照用)
         "t_probe": 0.0, "probing": False}
if ROUTE["name"] not in ROUTES:
    ROUTE["name"] = "direct"
ROUTE_LOCK = threading.Lock()
_STATS = {rt: collections.deque(maxlen=12) for rt in ROUTES}   # 各线路最近尝试 成/败
RECENT_URLS = collections.deque(maxlen=8)                      # 最近 HEAD 成功的对象,给探测用

PROBE_BYTES  = int(os.environ.get("S4D_PROBE_MB", "2")) << 20
PROBE_EVERY  = int(os.environ.get("S4D_PROBE_EVERY", "900"))      # 秒
PROBE_SWITCH = float(os.environ.get("S4D_PROBE_SWITCH", "1.5"))   # 另一条快这么多倍才切
ATTEMPT_MAX  = int(os.environ.get("S4D_ATTEMPT_MAX_TIME", "600"))  # 单次 curl 上限(秒)。续传后超时=免费重连,
                                                                   # 短一点反而让线路切换更快生效
STALL_BPS    = int(os.environ.get("S4D_STALL_BPS", "20000"))      # 30s 均速低于此=卡死,断开续传
FLOOR_BPS    = int(os.environ.get("S4D_FLOOR_BPS", "20000"))      # 整个对象的总时限按这个地板速率算
MAX_ATTEMPTS = int(os.environ.get("S4D_MAX_ATTEMPTS", "60"))      # 一段最多尝试次数(每次都有进展也不该这么多)
NPZ_RANGES   = int(os.environ.get("S4D_NPZ_RANGES", "4"))         # 走代理时每个 npz 拆几段(代理按连接限速)
NPZ_RANGES_DIRECT = int(os.environ.get("S4D_NPZ_RANGES_DIRECT", "1"))   # 直连单流已够(09-04 实测 12 路合计 5 MB/s)
RANGE_MIN    = int(os.environ.get("S4D_RANGE_MIN", str(4 << 20)))  # 小于这个就不拆
PROXY_SEM    = threading.BoundedSemaphore(int(os.environ.get("S4D_PROXY_MAX_CONN", "8")))  # 代理并发连接上限
                                                                   # (09-04 实测代理 4 连接 292 KB/s、12 连接反而只剩 38 KB/s)
_CURL_IP = ["-4"]   # 本机 IPv6 不通(解析得到 2607:f8b0:: 但连不出去),省掉每次先撞 v6 的等待


def _ts():
    return time.strftime("%H:%M:%S")


def _other(route):
    for rt in ROUTES:
        if rt != route:
            return rt
    return route


def _route_order():
    cur = ROUTE["name"]
    return [cur] + [rt for rt in ROUTES if rt != cur]


def _curl_base(route, extra_time=None):
    return (["curl", "-sS", *_CURL_IP, *ROUTES[route],
             "--connect-timeout", "20", "--max-time", str(extra_time or ATTEMPT_MAX),
             "--speed-limit", str(STALL_BPS), "--speed-time", "30"])


def _note(route, ok):
    with ROUTE_LOCK:
        _STATS[route].append(bool(ok))
    if not ok:
        _maybe_reprobe("当前线路失败偏多")


def _maybe_reprobe(why):
    with ROUTE_LOCK:
        d = _STATS[ROUTE["name"]]
        if len(d) < 6 or sum(1 for x in d if not x) * 2 < len(d):
            return
        if ROUTE["probing"] or time.time() - ROUTE["t_probe"] < 120:
            return
        ROUTE["probing"] = True
    try:
        probe_routes(list(RECENT_URLS), why)
    finally:
        ROUTE["probing"] = False


def _probe_one(route, url):
    """下 url 的前 PROBE_BYTES 字节,回 (MB/s, http码);失败回 (0, 码)。"""
    r = subprocess.run(["curl", "-sS", *_CURL_IP, *ROUTES[route], "-r", f"0-{PROBE_BYTES - 1}",
                        "-o", "/dev/null", "--connect-timeout", "15", "--max-time", "30",
                        "-w", "%{http_code} %{size_download} %{time_total}", url],
                       capture_output=True, text=True)
    f = (r.stdout or "").split()
    code = f[0] if f else ""
    try:
        got, t = int(f[1]), float(f[2])
    except (IndexError, ValueError):
        return 0.0, code
    if code != "206" or got <= 0:
        return 0.0, code
    return got / max(t, 1e-3) / 2**20, code


def probe_routes(urls, why):
    """两条线路各下同一对象前 2 MB 测速(并行,同负载下公平),快的那条成为当前线路。"""
    if ROUTE["fixed"] or len(ROUTES) < 2:
        return
    ROUTE["t_probe"] = time.time()
    res = None
    for url in list(urls)[:4]:
        with ThreadPoolExecutor(len(ROUTES)) as ex:
            futs = {rt: ex.submit(_probe_one, rt, url) for rt in ROUTES}
            r = {rt: f.result() for rt, f in futs.items()}
        if all(code in ("404", "416") for _, code in r.values()):
            continue                                   # 这个对象不存在,换一个探
        res = r
        break
    if res is None:
        print(f"[route {_ts()}] 探测({why}):候选对象都不可用,维持 {ROUTE['name']}", flush=True)
        return
    speeds = {rt: s for rt, (s, _) in res.items()}
    with ROUTE_LOCK:
        cur = ROUTE["name"]
        best = max(speeds, key=speeds.get)
        if speeds[best] <= 0:
            decision = f"两条都不通,维持 {cur}"
        elif best == cur:
            decision = f"维持 {cur}"
        elif speeds[best] >= PROBE_SWITCH * speeds.get(cur, 0.0):
            ROUTE["name"] = best
            for d in _STATS.values():
                d.clear()
            decision = f"切换 {cur} -> {best}"
        else:
            decision = f"维持 {cur}(差距不到 {PROBE_SWITCH}x)"
    print(f"[route {_ts()}] 探测({why}): "
          + " / ".join(f"{rt} {s:.2f} MB/s" for rt, s in speeds.items())
          + f" => {decision}", flush=True)


def _head(url):
    """回 (总长, 支持 Range, http 码, md5_base64)。先走当前线路,拿不到响应就换另一条。"""
    last = ""
    for route in _route_order():
        r = subprocess.run(["curl", "-sS", *_CURL_IP, *ROUTES[route], "-I", "--connect-timeout", "15",
                            "--max-time", "40", "-o", "/dev/null", "-D", "-", url],
                           capture_output=True, text=True)
        total, rng, code, md5 = 0, False, "", ""
        for line in (r.stdout or "").splitlines():
            low = line.lower()
            if low.startswith("http/"):
                f = line.split()
                if len(f) > 1:
                    code = f[1]                       # 走代理时先有一行 CONNECT 的 200,取最后一个状态行
            elif low.startswith("content-length:"):
                try: total = int(low.split(":", 1)[1].strip())
                except ValueError: pass
            elif low.startswith("accept-ranges:") and "bytes" in low:
                rng = True
            elif low.startswith("x-goog-hash:") and "md5=" in low:
                md5 = line.split("md5=", 1)[1].strip()
        if code == "404" or (code.startswith("2") and total > 0):
            _note(route, True)
            if code != "404":
                RECENT_URLS.append(url)
            return total, rng, code, md5
        _note(route, False)
        last = f"{route}: rc={r.returncode} http={code or '?'} {(r.stderr or '').strip()[:60]}"
    raise IOError(f"HEAD 失败 {last}")


def _last_status(hdr):
    code = ""
    try:
        for line in open(hdr, errors="replace"):
            if line.lower().startswith("http/"):
                f = line.split()
                if len(f) > 1:
                    code = f[1]
    except OSError:
        pass
    return code


def _download_segment(url, a, b, pf, deadline):
    """把 url 的字节 [a, b] 下到 pf。可续传:pf 已有的字节数就是已下好的部分,每次只请求还缺的
    尾段追加到文件末尾;超时/断连不清零。非 206 的响应体不是数据(错误页/整文件),截回去。
    回 None=成功,否则错误串(部分文件保留,下次回到队列时接着下)。"""
    need = b - a + 1
    attempts = zero = 0
    last_route = None
    while True:
        got = os.path.getsize(pf) if os.path.exists(pf) else 0
        if got >= need:
            if got > need:
                os.truncate(pf, need)
            return None
        if time.time() > deadline:
            return f"总时限到 {got}/{need}"
        attempts += 1
        if attempts > MAX_ATTEMPTS:
            return f"尝试 {MAX_ATTEMPTS} 次仍未完成 {got}/{need}"
        # 上一次失败就先换另一条线路试一把(不等探测);其余时候跟着全局线路走
        route = _other(last_route) if (last_route and zero) else ROUTE["name"]
        hdr = pf + ".hdr"
        sem = PROXY_SEM if route == "proxy" else None
        if sem: sem.acquire()
        try:
            with open(pf, "ab") as fh:
                r = subprocess.run(_curl_base(route) + ["-r", f"{a + got}-{b}", "-D", hdr, url],
                                   stdout=fh, stderr=subprocess.PIPE)
        finally:
            if sem: sem.release()
        code = _last_status(hdr)
        try: os.remove(hdr)
        except OSError: pass
        now = os.path.getsize(pf)
        err = (r.stderr or b"").decode(errors="replace").strip()[:70]
        if code == "404":
            raise NoNpz(url)
        if code != "206":
            if now > got:
                os.truncate(pf, got)               # 错误页/整文件不是我们要的字节
            now = got
            if code == "416":
                return f"416 范围错 {a + got}-{b}"
        if now >= need:
            _note(route, True)
            return None
        progressed = now > got
        _note(route, progressed)
        zero = 0 if progressed else zero + 1
        last_route = route
        if zero >= 3:
            return f"连续 3 次零进展 {route} http={code or '?'} rc={r.returncode} {err}"
        time.sleep(min(2 * zero, 10) if not progressed else 0.5)


def fetch_npz(clip, split) -> str:
    """下一个 npz(线路自适应 + 续传 + 完整性三闸)。"""
    os.makedirs(STAGE_NPZ, exist_ok=True)
    out = os.path.join(STAGE_NPZ, clip + ".npz")
    if os.path.exists(out) and os.path.getsize(out) > 1024:
        return out
    url = URL.format(split=split, clip=clip)
    total, rng, hcode, md5_b64 = _head(url)
    if hcode == "404":
        raise NoNpz(clip)
    if not total:
        raise IOError(f"HEAD 无长度 http={hcode}")
    tmp = out + ".part"                     # 不带 pid:ingest 重启后能接着续传(单实例锁保证不会两个人写)
    route0 = ROUTE["name"]
    n = 1
    if rng and total > RANGE_MIN:
        n = max(1, NPZ_RANGES if route0 == "proxy" else NPZ_RANGES_DIRECT)
    chunk = (total + n - 1) // n
    segs = []
    for i in range(n):
        a = i * chunk
        b = min(a + chunk - 1, total - 1)
        if a > b:
            break
        segs.append((a, b, tmp if n == 1 else f"{tmp}.r{i}"))
    deadline = time.time() + total / FLOOR_BPS + 120
    try:
        if len(segs) == 1:
            errs = [_download_segment(url, *segs[0], deadline)]
        else:
            with ThreadPoolExecutor(len(segs)) as ex:
                errs = list(ex.map(lambda s: _download_segment(url, *s, deadline), segs))
        errs = [f"seg{i} {e}" for i, e in enumerate(errs) if e]
        if errs:
            raise IOError("; ".join(errs)[:200])          # 部分文件保留,回队列后续传
        if len(segs) > 1:
            with open(tmp, "wb") as g:
                for _, _, pf in segs:
                    with open(pf, "rb") as f:
                        while True:
                            blk = f.read(1 << 20)
                            if not blk:
                                break
                            g.write(blk)
            for _, _, pf in segs:
                try: os.remove(pf)
                except OSError: pass
        got = os.path.getsize(tmp)
        if got != total:
            os.remove(tmp)
            raise IOError(f"拼接长度不符 {got}!={total}")
        h = hashlib.md5()
        with open(tmp, "rb") as f:
            head4 = f.read(4)
            h.update(head4)
            for blk in iter(lambda: f.read(1 << 20), b""):
                h.update(blk)
        if md5_b64 and base64.b64encode(h.digest()).decode() != md5_b64:
            os.remove(tmp)
            raise IOError("md5 与 x-goog-hash 不符,已丢弃重来")
        if head4 != b"PK\x03\x04":
            # 404 现在按状态码判、非 206 的响应体根本进不了文件,所以走到这里只可能是损坏 ——
            # 只能当损坏丢弃重来,绝不能判成 NoNpz(那会把 clip 永久拉进 no_npz 黑名单)。
            os.remove(tmp)
            raise IOError("不是 zip(魔数不对),已丢弃重来")
        os.replace(tmp, out)
        return out
    except NoNpz:
        for p in [tmp] + [pf for _, _, pf in segs]:
            try: os.remove(p)
            except OSError: pass
        raise


def sweep_stale_parts(max_age_s=2 * 86400):
    """清掉超过两天没动的 .part(永远下不完的对象不该无限占盘)。"""
    n = 0
    if not os.path.isdir(STAGE_NPZ):
        return 0
    now = time.time()
    for e in os.scandir(STAGE_NPZ):
        if ".part" in e.name and now - e.stat().st_mtime > max_age_s:
            try: os.remove(e.path); n += 1
            except OSError: pass
    return n


def ingest_one(clip, split, off, size) -> tuple[str, str]:
    """返回 (clip, "" 成功 | 错误串)。做完就把 mp4 和暂存 npz 删掉。"""
    try:
        extract_mp4(clip, split, off, size)
        fetch_npz(clip, split)
        prepare_clip(clip, STAGE_NPZ, STAGE_MP4, FRAMES,
                     os.path.join(FRAMES, clip), None)      # stride=None -> 自适应抽帧
        return clip, ""
    except NoNpz:
        return clip, "NO_NPZ"
    except Exception as e:
        return clip, f"{type(e).__name__}: {e}"
    finally:
        for p in (os.path.join(STAGE_MP4, clip, clip + SUFFIX), os.path.join(STAGE_NPZ, clip + ".npz")):
            try: os.remove(p)
            except OSError: pass
        try: os.rmdir(os.path.join(STAGE_MP4, clip))
        except OSError: pass


def cleanup_skipped_frames(root) -> tuple[int, int]:
    """把【已剔除】clip 的帧目录删掉腾空间。回 (删了几个, 释放字节)。

    硬证据(为什么这么删是安全的):`stereo4d_queue.sh` 认领循环里有
        grep -qxF -e "$c" "$Q/skipped.txt" && continue   # 静态过滤器已剔除 = 视为完成
    所以只要 clip 在 skipped.txt 里,lane 就【永远领不到它】⇒ 它的帧不会再被任何人读。
    ⚠️ 成品(有 final_repr.pkl)的帧是【训练真输入】(data.py 里 require_frames),绝不能删;
       所以这里只删 skipped 且不在成品集里的。
    ⚠️ 用 Python 删而不是 shell:Stereo4D 的 clip 名大量以 `-` 开头,shell 里已经咬过三次
       (`rm` 当成选项 / `grep` 当成 flag)。os 层没有这个问题。"""
    import shutil
    if not root:
        return 0, 0
    sk = os.path.join(root, "_q", "skipped.txt")
    if not os.path.exists(sk):
        return 0, 0
    skipped = {l.strip() for l in open(sk) if l.strip()}
    if not skipped:
        return 0, 0
    done = set()
    try:
        for e in os.scandir(root):
            if e.is_dir() and e.name.startswith("scene_") and \
               os.path.exists(os.path.join(e.path, "final_repr.pkl")):
                done.add(e.name[len("scene_"):])
    except OSError:
        pass
    n = freed = 0
    for c in skipped - done:
        d = os.path.join(FRAMES, c)
        if not os.path.isdir(d):
            continue
        try:
            sz = sum(e.stat().st_size for e in os.scandir(d) if e.is_file())
        except OSError:
            sz = 0
        try:
            shutil.rmtree(d)
            n += 1; freed += sz
        except OSError:
            pass
    return n, freed


def pending_count(args) -> int:
    """真实待跑 = 队列清单 - 已完成(有 scene_manifest.json)- 已剔除。
    ⚠️ 不能数 --queue-list 的行数:那是【累计已取】的追加清单,只增不减,超过阈值后取数就永久停摆。"""
    if not args.root:
        return sum(1 for _ in open(args.queue_list)) if os.path.exists(args.queue_list) else 0
    ready = os.path.join(args.root, "clips_ready.txt")
    n = sum(1 for l in open(ready) if l.strip()) if os.path.exists(ready) else 0
    done = 0
    try:
        done = sum(1 for e in os.scandir(args.root)
                   if e.is_dir() and e.name.startswith("scene_")
                   and os.path.exists(os.path.join(e.path, "scene_manifest.json")))
    except OSError:
        pass
    sk = os.path.join(args.root, "_q", "skipped.txt")
    nsk = len({l.strip() for l in open(sk) if l.strip()}) if os.path.exists(sk) else 0
    return max(0, n - done - nsk)


def free_g() -> float:
    s = os.statvfs("/data"); return s.f_bavail * s.f_frsize / 2**30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue-list", required=True, help="把建好的 clip 追加进这个文件(队列清单)")
    ap.add_argument("--wave", type=int, default=2000, help="每波取多少 clip")
    ap.add_argument("--backlog", type=int, default=3000,
                    help="【恢复线】真实待跑降到此值以下才恢复取数(配合 --backlog-hi 构成滞回带)")
    ap.add_argument("--backlog-hi", type=int, default=6000,
                    help="【暂停线】真实待跑超过此值就暂停取数。2026-09-06 用户拍板加的滞回带:"
                         "原来只有单阈值 3000,取数会在阈值上反复抖动;而且磁盘被外部反复撑爆时,"
                         "囤太多待跑 clip 的帧只是白占盘。暂停后待跑降到 --backlog 以下,"
                         "先清掉已剔除 clip 的帧腾出空间,再恢复取数。")
    ap.add_argument("--root", default=None,
                    help="批次根:用它算真实待跑量 = 队列清单 - 已完成 - 已剔除。"
                         "不给的话退回数 --queue-list 的行数 —— ⚠️ 那个文件是累计追加的,"
                         "一旦超过 backlog 就【永远】不会再取数(2026-08-20 实测卡了 12 小时)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--min-free-g", type=float, default=60.0, help="磁盘低于此值就停止取数")
    ap.add_argument("--done-roots", nargs="*", default=[], help="这些批次里已处理过的 clip 不再取")
    args = ap.parse_args()
    os.chdir(ROOT)

    # 单实例锁:两个 ingest 同时跑会抢同一个暂存目录 —— 一个的 os.replace 会把另一个正在写的
    # .part 抽走(FileNotFoundError),半截文件还会被当成 npz 读(BadZipFile)。实测踩过。
    lock = "data/stereo4d/.ingest.lock"
    if os.path.exists(lock):
        try:
            pid = int(open(lock).read().strip()); os.kill(pid, 0)
            print(f"[ingest] 已有实例在跑(pid {pid})— 退出"); return
        except (ValueError, ProcessLookupError, PermissionError):
            pass                                        # 陈旧锁,覆盖
    open(lock, "w").write(str(os.getpid()))
    try:
        run(args)
    finally:
        # ⚠️ 只删【自己写的】那把锁。原来是无条件 os.remove:旧实例退出时会把新实例刚写的锁
        #    删掉,supervisor 下一轮看不到锁就再起一个 ⇒ 两个 ingest 同时跑,互相抽走对方的
        #    .part(FileNotFoundError / 半截文件被当 npz 读成 BadZipFile)——这个坑 2026-08-20
        #    踩过一次,当时靠加单实例锁堵住,但锁本身的这个竞态一直在。
        try:
            if open(lock).read().strip() == str(os.getpid()):
                os.remove(lock)
        except (OSError, ValueError):
            pass


def run(args):
    idx = build_index()
    done = set()
    for r in args.done_roots:
        for n in os.listdir(r) if os.path.isdir(r) else []:
            if n.startswith("scene_"): done.add(n[6:])
        sk = os.path.join(r, "_skipped_static")
        for n in os.listdir(sk) if os.path.isdir(sk) else []:
            if n.startswith("scene_"): done.add(n[6:])
    have = set(os.listdir(FRAMES)) if os.path.isdir(FRAMES) else set()
    nonp = set()
    if os.path.exists(NO_NPZ_LIST):
        nonp = {l.strip() for l in open(NO_NPZ_LIST) if l.strip()}
    todo = [c for c in sorted(idx) if c not in done and c not in have and c not in nonp]
    print(f"[ingest] tar 里 {len(idx)} 个 clip;已处理 {len(done)};已建帧 {len(have)};待取 {len(todo)}", flush=True)
    ns = sweep_stale_parts()
    print(f"[ingest {_ts()}] 线路候选 {list(ROUTES)},起步 {ROUTE['name']}"
          + (",固定不切换" if ROUTE["fixed"] else f",探测间隔 {PROBE_EVERY}s")
          + (f";清掉 {ns} 个两天没动的 .part" if ns else ""), flush=True)
    probe_routes([URL.format(split=idx[c][0], clip=c) for c in todo[:6]], "开跑")

    got = fail = 0
    paused = False              # 滞回带状态:True=已暂停,等待跑降到 --backlog 以下
    while todo:
        if free_g() < args.min_free_g:
            print(f"[ingest] 🛑 磁盘只剩 {free_g():.0f}G < {args.min_free_g}G — 停止取数", flush=True)
            break
        pend = pending_count(args)
        # 滞回带:超过 backlog_hi 暂停,降到 backlog 以下才恢复(恢复前先清理腾空间)
        if pend > args.backlog_hi:
            if not paused:
                print(f"[ingest] ⏸ 真实待跑 {pend} > {args.backlog_hi} — 暂停取数,"
                      f"等降到 {args.backlog} 以下", flush=True)
                paused = True
            time.sleep(120); continue
        if paused:
            if pend >= args.backlog:
                time.sleep(120); continue          # 还在滞回带里,继续等
            nfr, fb = cleanup_skipped_frames(args.root)
            print(f"[ingest] ▶️ 真实待跑 {pend} < {args.backlog} — 恢复取数"
                  + (f";顺手清掉 {nfr} 个已剔除 clip 的帧,释放 {fb/2**30:.1f}G" if nfr else "")
                  + f",磁盘 {free_g():.0f}G", flush=True)
            paused = False
        print(f"[ingest] 真实待跑 {pend} <= {args.backlog_hi},继续取数", flush=True)
        wave, todo = todo[:args.wave], todo[args.wave:]
        t0 = time.time(); ok = []; retry = []
        if time.time() - ROUTE["t_probe"] > 60:
            probe_routes([URL.format(split=idx[c][0], clip=c) for c in wave[:6]], "波首")
        # 🔴 2026-08-29 改成【流式】:每取到一个就立刻写 queue-list 并 flush。
        #    原来是整波(500 个)跑完才一次性写 —— 按实测 496 clip/h,一波要 ~60 分钟,
        #    于是下游(Qwen 预取→clips_ready→autoscale 起 lane)整整一小时看不到新活,
        #    8 张卡就那么空着,而取数其实一直在正常出货。
        # 🔴 同时把 ex.map 换成 as_completed:map 按【提交顺序】产出,队首一个慢 clip
        #    会把后面已经下完的全堵住 —— 这是第二层、同样白等的延迟。
        with open(args.queue_list, "a") as qf:
            with ThreadPoolExecutor(args.workers) as ex:
                futs = {ex.submit(ingest_one, c, *idx[c]): c for c in wave}
                for fu in as_completed(futs):
                    if time.time() - ROUTE["t_probe"] > PROBE_EVERY and RECENT_URLS:
                        probe_routes(list(RECENT_URLS), "定期")
                    try:
                        clip, err = fu.result()
                    except Exception as e:               # 兜底:线程里没被 ingest_one 接住的异常
                        clip, err = futs[fu], f"{type(e).__name__}: {e}"
                    if err == "NO_NPZ":
                        nonp.add(clip)
                        with open(NO_NPZ_LIST, "a") as nf: nf.write(clip + "\n")
                    elif err:
                        fail += 1; retry.append(clip)     # 放回队尾,别把它从待取里烧掉
                        print(f"  [fail] {clip}: {err}", flush=True)
                    else:
                        ok.append(clip); got += 1
                        qf.write(clip + "\n"); qf.flush()   # 立刻可见,下游 5 分钟内就能领走
        todo.extend(retry)                            # 失败的排到最后重来
        rate = len(ok) / max(len(wave) - (len(wave) - len(ok) - len(retry)), 1)
        if retry and len(ok) < 0.3 * (len(ok) + len(retry)):
            cool = int(os.environ.get("INGEST_COOLDOWN", "600"))
            print(f"[ingest] ⚠️ 本波成功率仅 {len(ok)}/{len(ok)+len(retry)} — 网络在抖,"
                  f"退避 {cool}s 再继续(失败的 {len(retry)} 个已放回队尾)", flush=True)
            time.sleep(cool)
        print(f"[ingest] 一波完成 {len(ok)}/{len(wave)},累计 {got} 成功 {fail} 失败,"
              f"{time.time()-t0:.0f}s,无 npz 累计 {len(nonp)},磁盘 {free_g():.0f}G,剩 {len(todo)} 待取", flush=True)
    print(f"[ingest] 结束:成功 {got} 失败 {fail}", flush=True)


if __name__ == "__main__":
    main()
