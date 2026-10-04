#!/usr/bin/env python3
"""TrajQA-B 评测 clip 与 MLLM-4D 训练数据的重叠核对(2026-09-25)。

MLLM-4D 的数据全部来自 Stereo4D(scene 名 = <YouTube id>_<时间戳>),TrajQA-B 的 clip 名是同一格式。
  python3 tools/overlap_mllm4d.py [--pred preds/Q1_MLLM-4D-SFT-8B.jsonl]
写 logs/overlap_mllm4d.json(每个评测 clip 属于哪一档),并按档拆分准确率(给了 --pred 时)。

档位(按 SFT 用的 MLLM4D-2M 判定;MLLM-4D-SFT-1.0 只见过 2M,RFT/Cold-Start 另列供参考):
  seen_clip    同一个 clip(scene 名完全相同)出现在 MLLM4D-2M 训练集
  seen_video   clip 本身不在,但同一个 YouTube 视频的其他片段在
  unseen       YouTube 视频都没出现过
"""
import argparse, collections, json, math, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import NUMERIC3
BASELINE_ROOT = os.environ.get("BASELINE_ROOT", os.path.expanduser("~/baseline_workspace"))  # 基模权重/代码/数据所在工作目录

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DS = os.environ.get("MLLM4D_DATASETS", os.path.join(PKG, "..", "data", "external", "MLLM-4D-Datasets"))   # HF flow666/MLLM-4D-Datasets 的标注(tools/dl_mllm4d_ann.sh)


def yt(c):
    return c.rsplit("_", 1)[0]


def scene_of(path):
    return re.sub(r"-left_rectified\.mp4$", "", os.path.basename(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=None)
    ap.add_argument("--out", default=os.path.join(PKG, "logs", "overlap_mllm4d.json"))
    a = ap.parse_args()

    qs = [json.loads(l) for l in open(os.path.join(PKG, "qa", "questions.jsonl"))]
    qs = [q for q in qs if q["split"] in ("heldout_test", "test")]
    clip = lambda q: q["video_id"][len("scene_"):] if q["video_id"].startswith("scene_") else q["video_id"]
    ev = sorted({clip(q) for q in qs})

    print("读 MLLM4D-2M.json(2.28 GB)…", flush=True)
    sft = json.load(open(os.path.join(DS, "MLLM4D-2M.json")))
    sft_scene = collections.Counter()
    sft_types = collections.defaultdict(collections.Counter)
    of_n = collections.Counter()
    for x in sft:
        s = x.get("scene_name") or scene_of(x["video"])
        sft_scene[s] += 1
        sft_types[s][x.get("question_type")] += 1
        m = re.search(r"of (\d+)\?", x["conversations"][1]["value"]) if len(x.get("conversations", [])) > 1 else None
        if m:
            of_n[int(m.group(1))] += 1
    srcs = collections.Counter(x.get("data_source") for x in sft)
    sft_yt = {yt(s) for s in sft_scene}
    print(f"MLLM4D-2M: {len(sft):,} 条 QA,{len(sft_scene):,} 个 scene,{len(sft_yt):,} 个 YouTube 视频;"
          f"data_source={dict(srcs)};题面 'of N' 的 N 分布(前 5)={of_n.most_common(5)}", flush=True)

    r1 = {scene_of(json.loads(l)["video"]) for l in open(os.path.join(DS, "MLLM4D-R1-30k.jsonl"))}
    cs = {x.get("scene_name") or scene_of(x["video"]) for x in json.load(open(os.path.join(DS, "MLLM4D-Cold-Start-Data.json")))}

    tier = {}
    for c in ev:
        tier[c] = "seen_clip" if c in sft_scene else ("seen_video" if yt(c) in sft_yt else "unseen")
    by_split = collections.defaultdict(collections.Counter)
    for q in qs:
        by_split[q["split"]][tier[clip(q)]] += 0   # 占位,下面按 clip 数
    clips_by_split = collections.defaultdict(set)
    for q in qs:
        clips_by_split[q["split"]].add(clip(q))
    print("\n== clip 级重叠(按 SFT 训练集 MLLM4D-2M)==")
    for sp in ("heldout_test", "test"):
        C = clips_by_split[sp]
        t = collections.Counter(tier[c] for c in C)
        print(f"  {sp:13s} {len(C):4d} clip: 同 clip 在训练集 {t['seen_clip']:4d} ({100*t['seen_clip']/len(C):.1f}%)  "
              f"仅同 YouTube 视频 {t['seen_video']:4d}  完全没见过 {t['unseen']:4d} ({100*t['unseen']/len(C):.1f}%)")
    tq = collections.Counter(tier[clip(q)] for q in qs)
    tq3 = collections.Counter(tier[clip(q)] for q in qs if q["family"] in NUMERIC3)
    print(f"  题级(7,922):{dict(tq)};数值三族(7,500):{dict(tq3)}")
    seen = [c for c in ev if tier[c] == "seen_clip"]
    if seen:
        per = [sft_scene[c] for c in seen]
        agg = collections.Counter()
        for c in seen:
            agg.update(sft_types[c])
        print(f"  训练集里这些 clip 平均各有 {sum(per)/len(per):.1f} 道 SFT 题(中位 {sorted(per)[len(per)//2]}),"
              f"题型合计 {dict(agg)}")
    print(f"  参考:R1-30k(RFT)含同 clip {sum(c in r1 for c in ev)} 个;Cold-Start 含 {sum(c in cs for c in ev)} 个"
          f"(MLLM-4D-SFT-1.0 不用这两份)")

    json.dump({"source": "MLLM4D-2M.json sha256 5a47adb7…", "tier": tier,
               "counts": {sp: dict(collections.Counter(tier[c] for c in clips_by_split[sp])) for sp in clips_by_split}},
              open(a.out, "w"), ensure_ascii=False, indent=0)
    print(f"\n写入 {a.out}")

    if a.pred and os.path.exists(a.pred):
        gold = {d["id"]: d for d in map(json.loads, open(os.path.join(PKG, "qa", "answers.jsonl")))}
        preds = {d["id"]: d for d in map(json.loads, open(a.pred))}
        qmap = {q["id"]: q for q in qs}
        rows = collections.defaultdict(lambda: [0, 0])
        for qid, d in preds.items():
            q, g = qmap.get(qid), gold.get(qid)
            if not q or not g or q["family"] not in NUMERIC3:
                continue
            ok = int((d.get("pred") or "?")[:1] == g["answer"])
            t = tier[clip(q)]
            for k in (t, f"{t}/{q['split']}", f"{t}/{q['family']}", "ALL"):
                rows[k][0] += ok; rows[k][1] += 1
        print(f"\n== 数值三族准确率按重叠档拆分({os.path.basename(a.pred)},{len(preds)} 条)==")
        for k in ("ALL", "seen_clip", "seen_video", "unseen"):
            c, n = rows[k]
            if n:
                p = c / n
                print(f"  {k:11s} n={n:5d}  acc={100*p:5.1f} ±{100*math.sqrt(p*(1-p)/n):.1f}   " + "  ".join(
                    f"{s}={100*rows[f'{k}/{s}'][0]/rows[f'{k}/{s}'][1]:.1f}(n={rows[f'{k}/{s}'][1]})"
                    for s in ("heldout_test", "test") + NUMERIC3 if rows.get(f"{k}/{s}", [0, 0])[1]))


if __name__ == "__main__":
    main()
