#!/usr/bin/env python3
"""逐族推理时间统计(README §5 的第 2 项)。

    python tools/timing.py --pred preds/xxx.jsonl [--wall 1234.5] [--workers 32]

读 pred jsonl 里的 "t" 字段(每题推理耗时,由 run_hf.py / run_openai_api.py 记录):
  - run_openai_api.py: 每题真实请求往返耗时。**注意 --workers>1 时该值含排队等待**,
    因此并发下 sum(t) 会远大于墙钟;横向比较请看 "均值" 和 "占比",绝对吞吐看 it/s。
  - run_hf.py: 每批耗时摊到每题(不含抽帧 IO)。

给 --wall(整轮墙钟秒)时,额外按各族 t 的占比把墙钟分摊下去,得到"等效墙钟"。
"""
import argparse, collections, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import NUMERIC3

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--pkg", default=PKG)
    ap.add_argument("--wall", type=float, default=None, help="整轮墙钟秒数,用于分摊出等效墙钟")
    ap.add_argument("--workers", type=int, default=None, help="并发数,仅用于打印说明")
    a = ap.parse_args()

    gold = {d["id"]: d for d in map(json.loads,
            open(os.path.join(a.pkg, "qa", "answers.jsonl")))}

    per = collections.defaultdict(list)
    n_missing = 0
    for line in open(a.pred):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        g = gold.get(d["id"])
        if g is None:
            continue
        if "t" not in d:
            n_missing += 1
            continue
        per[g["family"]].append(d["t"])
        if g["family"] in NUMERIC3:
            per["_NUM3"].append(d["t"])
        per["_ALL"].append(d["t"])

    if not per:
        print(f"# {a.pred}\n没有 t 字段,无法统计耗时"
              f"{'(该文件由未打时间补丁的旧脚本产出)' if n_missing else ''}")
        return
    if n_missing:
        print(f"⚠️ {n_missing} 条缺 t 字段(旧脚本产出),已跳过\n")

    total_t = sum(per["_ALL"])
    print(f"# {a.pred}")
    if a.workers:
        print(f"并发 workers={a.workers}(每题 t 含排队等待,sum(t) 会大于墙钟)")
    if a.wall:
        print(f"整轮墙钟 {a.wall:.0f}s = {a.wall/60:.1f} min,"
              f"吞吐 {len(per['_ALL'])/a.wall:.2f} it/s")
    print(f"\n{'族':28s} {'n':>6s} {'均值s':>8s} {'中位s':>8s} {'合计s':>10s} {'占比%':>7s}"
          + (f" {'等效墙钟s':>10s}" if a.wall else ""))

    def row(key, label):
        ts = per.get(key) or []
        if not ts:
            return
        ts_sorted = sorted(ts)
        s = sum(ts)
        share = 100.0 * s / total_t if total_t else 0.0
        line = (f"{label:28s} {len(ts):6d} {s/len(ts):8.3f} "
                f"{ts_sorted[len(ts)//2]:8.3f} {s:10.1f} {share:7.1f}")
        if a.wall:
            line += f" {a.wall*share/100:10.1f}"
        print(line)

    row("_ALL", "总体")
    print("-" * 72)
    row("_NUM3", "数值三族合并(主指标)")
    for f in NUMERIC3:
        row(f, f"  {f}")
    print("-" * 72)
    for f in sorted(k for k in per if not k.startswith("_") and k not in NUMERIC3):
        row(f, f"  {f}")


if __name__ == "__main__":
    main()
