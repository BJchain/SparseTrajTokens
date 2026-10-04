#!/usr/bin/env python3
"""解包后自检:题量/划分零交集/帧齐不齐/抽帧能不能跑通。

    python tools/verify.py [--pkg .] [--splits val,test,heldout_test]
只解了 frames_eval.tar 的话就别带 train(--splits 默认已排除 train)。
"""
import argparse, collections, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import LETTERS, NUMERIC3, frames_dir, sample_frames

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPECT = {"train": 56120, "val": 1318, "test": 4492, "heldout_test": 3430}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", default=PKG)
    ap.add_argument("--splits", default="val,test,heldout_test")
    a = ap.parse_args()
    want = [s.strip() for s in a.splits.split(",")]
    ok = True

    Q = [json.loads(l) for l in open(os.path.join(a.pkg, "qa", "questions.jsonl"))]
    A = {d["id"]: d for d in map(json.loads, open(os.path.join(a.pkg, "qa", "answers.jsonl")))}
    n = collections.Counter(q["split"] for q in Q)
    print(f"题目 {len(Q)} 条 / 答案 {len(A)} 条")
    for s, e in EXPECT.items():
        if n[s] == 0:                      # 本包没含这个 split(如 eval-only 包),不算失败
            print(f"  --  {s:13s}      - (本包未含)")
            continue
        m = "OK " if n[s] == e else "❌ "
        ok &= n[s] == e
        print(f"  {m}{s:13s} {n[s]:6d} (应 {e})")

    if {q["id"] for q in Q} != set(A):
        print("❌ question/answer 的 id 集合不一致"); ok = False
    if any(A[q["id"]]["answer"] not in LETTERS for q in Q):
        print("❌ 有答案不是 A-D"); ok = False
    if any(q["frame_range"][0] != 0 for q in Q):
        print("❌ 有题 frame_range 不从 0 开始(整段 clip 协议被破坏)"); ok = False

    vids = collections.defaultdict(set)
    for q in Q:
        vids[q["split"]].add(q["video_id"])
    for i, s1 in enumerate(EXPECT):
        for s2 in list(EXPECT)[i + 1:]:
            if vids[s1] & vids[s2]:
                print(f"❌ 视频泄漏 {s1}∩{s2} = {len(vids[s1] & vids[s2])}"); ok = False
    print("OK  四个 split 的视频两两零交集")

    need = set().union(*(vids[s] for s in want))
    miss = [v for v in need if not os.path.isdir(frames_dir(a.pkg, v))]
    print(f"帧目录 {len(need) - len(miss)}/{len(need)} 齐({','.join(want)})")
    if miss:
        print(f"  ❌ 缺 {len(miss)} 个,例:{miss[:3]}"); ok = False
    else:
        probe = [q for q in Q if q["split"] in want][:5]
        for q in probe:
            imgs, idx = sample_frames(a.pkg, q["video_id"], 8, 384)
            assert len(imgs) == min(8, q["n_frames_total"] or 8), (q["id"], len(imgs))
        print(f"OK  抽帧跑通(抽样 {len(probe)} 条,8 帧 @384,尺寸 {imgs[0].size})")

    num3 = sum(1 for q in Q if q["family"] in NUMERIC3 and q["split"] in want)
    print(f"主指标口径:数值三族在 {','.join(want)} 上共 {num3} 题")
    print("\n" + ("✅ 自检全过" if ok else "❌ 自检有失败项,见上"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
