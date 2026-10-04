"""把扩量后的训练集按**族占比**对齐到基线(2026-09-17):扩量时相机族占比从 36% 涨到 49%,与基线训练集不可比。
按"窗口内逐名次等量丢弃"抽样,保持 P(名次|窗口) 均匀这一条不被破坏(数值族);小族与二分族原样保留。
用法:python src/tools/trajqa_b_mixmatch.py <src_dir> <out_dir> [--ref 基线目录]"""
import argparse, json, os, random, shutil
from collections import Counter, defaultdict

NUM = ("cam_disp_grid", "disp_grid", "speed_grid")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("out")
    ap.add_argument("--ref", default="data/trajqa_b/build_m1"); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); rng = random.Random(a.seed)
    ref = Counter(json.loads(l)["event_subtype"] for l in open(os.path.join(a.ref, "train.jsonl")))
    src = [json.loads(l) for l in open(os.path.join(a.src, "train.jsonl"))]
    have = Counter(d["event_subtype"] for d in src)
    tot_ref = sum(ref[f] for f in NUM); share = {f: ref[f] / tot_ref for f in NUM}      # 数值三族内部比例(基线)
    T = min(have[f] / share[f] for f in NUM)                                            # 能达到的最大数值族总量
    target = {f: int(T * share[f]) for f in NUM}
    out = [d for d in src if d["event_subtype"] not in NUM]                             # 小族全留
    for f in NUM:
        F = [d for d in src if d["event_subtype"] == f]
        bywin = defaultdict(lambda: defaultdict(list))
        for d in F: bywin[d["gt_payload"]["window"]][d["gt_payload"]["rank"]].append(d)
        ratio = target[f] / len(F); kept = []
        for w, byrank in sorted(bywin.items()):
            m = min(len(v) for v in byrank.values()); k = int(round(m * ratio))
            if k <= 0: continue
            for r, v in byrank.items():
                rng.shuffle(v); kept += v[:k]
        out += kept
        print(f"{f}: {len(F)} → {len(kept)}(目标 {target[f]},基线占比 {100*share[f]:.1f}%)")
    rng.shuffle(out)
    with open(os.path.join(a.out, "train.jsonl"), "w") as fh:
        for d in out: fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    for n in ("val", "test"):
        shutil.copyfile(os.path.join(a.src, f"{n}.jsonl"), os.path.join(a.out, f"{n}.jsonl"))
    c = Counter(d["event_subtype"] for d in out); n = len(out)
    print(f"合计 {n};族占比 " + " ".join(f"{k} {100*v/n:.0f}%" for k, v in c.most_common()))


if __name__ == "__main__":
    main()
