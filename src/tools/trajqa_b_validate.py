"""B 集结构验证(2026-09-16 扩量用):把"生成 B 集"时定下的判据做成每步必跑的检查。
不跑模型,只查结构与分布 —— 模型侧的捷径审计仍由 src/tools/trajqa_b_audit.py 负责(train 拟合 → test 作答)。
检查项:
  ① 划分泄漏:train/val/test 之间的 **视频(ytid)** 交集、题 id 交集、(视频,实例,窗口,族) 键交集 —— 必须全为 0
  ② Gate 0:每道保留题在 gate0 文件里都判一致(数值族同格 / 二分族同答案)
  ③ 配平:数值族 P(名次|窗口) 最大偏差、字母分布;二分族逐上下文格的答案两侧是否等量;类别分布
  ④ 分布漂移:train 与 test 的真值(log)、帧距、深度、类别、帧率档分位数对比(扩量不该改变题的性质)
  ⑤ 冗余:训练集内同 (视频,实例,族) 的窗口重叠率(近重复题占比)、每视频题数上尾
用法:python src/tools/trajqa_b_validate.py <build_dir> [--gate0 <gate0.jsonl>] [--ref <参照 build_dir>]"""
import argparse, json, math, os, sys
from collections import Counter, defaultdict
import numpy as np
sys.path.insert(0, "src/training")
from g1.split import group_key

NUM = {"cam_disp_grid", "disp_grid", "speed_grid"}
BIN = {"interval_disp", "interval_speed", "motion_source"}
TIME = {"motion_duration_binary"}


def load(d, name):
    p = os.path.join(d, f"{name}.jsonl")
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def key_window(d):
    g = d["gt_payload"]
    return (d["video_id"], tuple(d.get("involved_inst_ids") or []), g.get("i"), g.get("j"), g.get("k"), d["event_subtype"])


def quant(v, ps=(10, 50, 90)):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    return {p: round(float(np.percentile(v, p)), 3) for p in ps} if len(v) else {}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("--gate0", default=""); ap.add_argument("--ref", default="")
    a = ap.parse_args()
    S = {n: load(a.dir, n) for n in ("train", "val", "test")}
    print(f"== {a.dir}  题数 " + " / ".join(f"{n} {len(S[n])}" for n in S))
    ok = True

    # ① 划分泄漏
    vids = {n: {group_key(d["video_id"]) for d in L} for n, L in S.items()}
    ids = {n: {d["id"] for d in L} for n, L in S.items()}
    wins = {n: {key_window(d) for d in L} for n, L in S.items()}
    for x, y in (("train", "test"), ("train", "val"), ("val", "test")):
        nv, ni, nw = len(vids[x] & vids[y]), len(ids[x] & ids[y]), len(wins[x] & wins[y])
        bad = nv or ni or nw; ok &= not bad
        print(f"① 泄漏 {x}∩{y}: 视频 {nv} / 题 id {ni} / (视频,实例,窗口,族) {nw} {'❌' if bad else '✓'}")
    dup = len(S["train"]) - len(ids["train"]); print(f"① train 内重复 id {dup} {'❌' if dup else '✓'}"); ok &= not dup
    print(f"   视频数 train {len(vids['train'])} / val {len(vids['val'])} / test {len(vids['test'])}")

    # ② Gate 0
    if a.gate0:
        G = {}
        for l in open(a.gate0):
            r = json.loads(l); G[r["id"]] = r
        bad = 0; miss = 0
        for d in S["train"]:
            g = G.get(d["id"])
            if g is None: miss += 1; continue
            fam = d["event_subtype"]; gp = d["gt_payload"]
            if fam in NUM:
                tv = g.get("tok_value")
                if tv is None or int(np.floor(np.log(max(tv, 1e-9) / gp["grid"]["v0"]) / np.log(gp["grid"]["r"]))) != gp["cell"]: bad += 1
            elif fam in BIN:
                truth = gp["answer_interval"] if fam != "motion_source" else gp["source"]
                if g.get("tok_answer") != truth: bad += 1
            elif fam in TIME:
                if g.get("tok_answer") != d["options"][d["answer"]]: bad += 1
        print(f"② Gate 0(train):token 侧复算不一致 {bad},缺记录 {miss} {'❌' if (bad or miss) else '✓'}"); ok &= not (bad or miss)

    # ③ 配平
    for name in ("train", "test"):
        L = S[name]
        if not L: continue
        line = []
        for fam in sorted({d["event_subtype"] for d in L}):
            F = [d for d in L if d["event_subtype"] == fam]
            let = Counter(d["answer"] for d in F); k = len(let)
            dev_let = max(abs(c - len(F) / k) / (len(F) / k) for c in let.values()) if k else 0
            extra = ""
            if fam in NUM:
                cw = Counter((d["gt_payload"]["window"], d["gt_payload"]["rank"]) for d in F); ws = Counter(d["gt_payload"]["window"] for d in F)
                dev = max(abs(cw[(w, r)] - ws[w] / 4) / (ws[w] / 4) for w in ws for r in (1, 2, 3, 4))
                extra = f" 名次|窗口最大偏差 {dev:.2f}"; ok &= dev < 1e-9
            if fam in BIN:
                ansk = "answer_interval" if fam != "motion_source" else "source"
                c = Counter(d["gt_payload"][ansk] for d in F); dev2 = max(abs(v - len(F) / len(c)) / (len(F) / len(c)) for v in c.values())
                extra = f" 答案两侧最大偏差 {dev2:.2f}"; ok &= dev2 < 1e-9
            line.append(f"{fam} n={len(F)} 字母偏差 {dev_let:.2f}{extra}")
        print(f"③ 配平({name}):\n   " + "\n   ".join(line))

    # ④ 分布漂移
    def prof(L, fam):
        F = [d for d in L if d["event_subtype"] == fam]
        if not F: return {}
        g = [d["gt_payload"] for d in F]
        out = dict(n=len(F))
        if fam in NUM: out["真值"] = quant([x["value"] for x in g])
        if "gap" in g[0]: out["帧距"] = quant([x.get("gap") for x in g])
        if "obj_dist_m" in g[0]: out["深度"] = quant([x.get("obj_dist_m") for x in g])
        out["类别"] = {k: round(v / len(F), 2) for k, v in Counter(x.get("cls") for x in g).most_common(4)}
        out["帧率档"] = {k: round(v / len(F), 2) for k, v in Counter(x.get("fps_bin") for x in g).most_common(3)}
        return out
    print("④ 分布(train vs test):")
    for fam in sorted(NUM | BIN | TIME):
        a_, b_ = prof(S["train"], fam), prof(S["test"], fam)
        if a_ and b_: print(f"   {fam}\n     train {a_}\n     test  {b_}")

    # ⑤ 冗余
    byk = defaultdict(list)
    for d in S["train"]:
        g = d["gt_payload"]
        if g.get("i") is not None: byk[(d["video_id"], tuple(d.get("involved_inst_ids") or []), d["event_subtype"])].append((g["i"], g.get("k") or g["j"]))
    dupw = 0; tot = 0
    for k, W in byk.items():
        for x in range(len(W)):
            for y in range(x + 1, len(W)):
                a1, b1 = W[x]; a2, b2 = W[y]; tot += 1
                inter = max(0, min(b1, b2) - max(a1, a2)); union = max(b1, b2) - min(a1, a2)
                if union > 0 and inter / union > 0.8: dupw += 1
    perv = Counter(group_key(d["video_id"]) for d in S["train"])
    print(f"⑤ 冗余(train):同(视频,实例,族)窗口 IoU>0.8 的题对 {dupw}/{tot}({100*dupw/max(tot,1):.1f}%);每视频题数 中位 {np.median(list(perv.values())):.0f} p95 {np.percentile(list(perv.values()),95):.0f} 最大 {max(perv.values())}")
    print("== 结构检查:" + ("全部通过 ✓" if ok else "有未通过项 ❌"))


if __name__ == "__main__":
    main()
