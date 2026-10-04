"""Qwen (stage0-pre) 预取 pass —— 把 VLM 判类整体挪出 GPU 流水线。

为什么
------
stage0-pre 是一次 DashScope 往返(实测 ~17 s/clip),期间 GPU 完全空转。原来它串在每个
sub-batch 的最前面、逐 clip 串行,后果在 2026-08-01 那次 1k 跑里看得很清楚:起跑 49 分钟后
六卡均值只有 32%、三张卡读 0%,因为各 lane 还在挨个调 Qwen(见 GPU利用率.md §3)。

这个工具把它做成**独立的预处理 pass**:高并发跑完全量 clip,结果落成 scene_<clip>/stage0_pre.json,
之后 GPU 流水线用 --skip-stages p 启动,只读磁盘上已有的 json。110k 尺度上,串行是
110000 x 17 s ≈ 519 小时;48 路并发约 11 小时,而且**一秒 GPU 都不占**。

⚠️ 下游必须先有 has_pre 从磁盘识别 stage0_pre.json 的修复(2026-08-12),否则 --skip-stages p
会静默退回 --concepts person,把这一步的产出整个丢掉。

用法
----
  python src/tools/qwen_prefetch.py --clips @/tmp/clips.txt \
      --frames-root data/stereo4d/frames --batch-root data/scenes --workers 48
  (DASHSCOPE_API_KEY 必须已 export;非登录 shell 不会自动 source ~/.bashrc)
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from src.stage0.run_stage0_pre import run as run_pre  # noqa: E402

_LOCK = threading.Lock()


def _parse_clips(spec: str, frames_root: Path) -> list[str]:
    spec = spec.strip()
    if spec.upper() == "ALL":
        return sorted(d.name for d in frames_root.iterdir() if d.is_dir())
    if spec.startswith("@"):
        return [l.strip() for l in Path(spec[1:]).read_text().splitlines() if l.strip()]
    return [s.strip() for s in spec.split(",") if s.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True, help="'ALL' | 逗号列表 | @文件(每行一个 clip)")
    ap.add_argument("--frames-root", default=str(_ROOT / "data" / "stereo4d" / "frames"))
    ap.add_argument("--batch-root", required=True,
                    help="输出到 <batch-root>/scene_<clip>/stage0_pre.json")
    ap.add_argument("--workers", type=int, default=32,
                    help="并发请求数。纯网络等待,给大点;受 DashScope 限流约束")
    ap.add_argument("--model", default=None)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--legacy-calls", action="store_true")
    ap.add_argument("--retries", type=int, default=2,
                    help="每个 clip 整体重试次数(qwen_client 内部已有单请求级退避重试)")
    args = ap.parse_args()

    if not os.environ.get("DASHSCOPE_API_KEY"):
        sys.exit("FATAL: DASHSCOPE_API_KEY 没设 —— 先 export(非登录 shell 不读 ~/.bashrc)")

    frames_root = Path(args.frames_root)
    br = Path(args.batch_root)
    if not br.is_absolute():
        br = _ROOT / br
    clips = _parse_clips(args.clips, frames_root)

    todo = []
    for c in clips:
        out = br / f"scene_{c}" / "stage0_pre.json"
        if out.exists():
            continue
        if not (frames_root / c).is_dir():
            print(f"[prefetch] {c}: 帧目录不存在 — 跳过", flush=True)
            continue
        todo.append((c, out))

    print(f"[prefetch] {len(clips)} clips, {len(todo)} 待跑, {args.workers} 路并发 -> {br}",
          flush=True)
    if not todo:
        return

    done = {"ok": 0, "fail": 0, "n": 0}
    failures: list[str] = []
    t0 = time.time()

    def _one(item):
        clip, out = item
        kw = dict(model=args.model, stride=args.stride, max_frames=args.max_frames,
                  legacy_calls=args.legacy_calls)
        kw = {k: v for k, v in kw.items() if v is not None}
        last = None
        for attempt in range(args.retries + 1):
            try:
                buf = io.StringIO()
                # run_pre 打得很啰嗦且线程间会串行交错 —— 收进 buffer,只在失败时吐出来
                with contextlib.redirect_stdout(buf):
                    run_pre(frames_root / clip, out, **kw)
                if out.exists():
                    return clip, True, ""
                last = "run 完成但没写出文件"
            except Exception as e:
                last = f"{type(e).__name__}: {e}"
                if attempt < args.retries:
                    time.sleep(2.0 * (attempt + 1))
        return clip, False, last or "未知"

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = [ex.submit(_one, it) for it in todo]
        for f in as_completed(futs):
            clip, ok, err = f.result()
            with _LOCK:
                done["n"] += 1
                done["ok" if ok else "fail"] += 1
                if not ok:
                    failures.append(f"{clip}: {err}")
                    print(f"[prefetch] ❌ {clip} — {err}", flush=True)
                n = done["n"]
                if n % 25 == 0 or n == len(todo):
                    el = time.time() - t0
                    rate = n / max(el, 1e-9)
                    eta = (len(todo) - n) / max(rate, 1e-9)
                    print(f"[prefetch] {n}/{len(todo)} ok={done['ok']} fail={done['fail']} "
                          f"{rate * 3600:.0f} clip/h ETA {eta / 3600:.1f} h", flush=True)

    el = time.time() - t0
    print(f"\n[prefetch] DONE ok={done['ok']} fail={done['fail']} 用时 {el / 60:.1f} min "
          f"({done['n'] / max(el / 3600, 1e-9):.0f} clip/h)", flush=True)
    if failures:
        fp = br / "prefetch_failures.txt"
        fp.write_text("\n".join(failures) + "\n")
        print(f"[prefetch] 失败清单 -> {fp}(重跑本命令即可,已成功的会 skip)", flush=True)


if __name__ == "__main__":
    main()
