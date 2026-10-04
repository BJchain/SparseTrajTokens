"""B 集离线审计(纯 CPU,不跑模型;2026-09-14)。train 拟合 → test(另一批视频)作答,给出"盲读者能学到的下界"。
  ① 文字/选项捷径:数值族 = 岭回归(题面词袋 + 帧号特征 + 窗口位置)预测 log 真值→选最近选项,以及 多分类逻辑回归直接预测名次;
     规则族:恒选第 k 小、选离族中位数最近、选离类别中位数最近。二分族 = 逻辑回归(题面 + 选项 A−B 词袋 + 区间长度差)+ 规则(选长区间/选第一区间/恒选相机)。
     时间族 = 逻辑回归(题面 + 阈值)+ 恒 Yes/No。
  ② 粗估读者模拟(数值族):真值 × exp(N(0,σ)),σ∈{0.3,0.5}(09-13 估的 RGB 米制精度)→ 最近选项;给 RGB 臂的预期上界。
  ③ 配平表:字母、类别×答案、帧率档;④ 跨族同题重复;⑤ 来源一致性。
用法:python src/tools/trajqa_b_audit.py <build_dir> [out.json]"""
import json, re, sys
from collections import Counter, defaultdict
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Ridge, LogisticRegression

D = sys.argv[1].rstrip("/") + "/"; OUT = sys.argv[2] if len(sys.argv) > 2 else D + "_audit.json"
rx_fr = re.compile(r"frame (\d+) and frame (\d+)"); rx_N = re.compile(r"of (\d+)"); rx_iv = re.compile(r"between frame (\d+) and frame (\d+)")
NUMERIC = {"disp_grid", "speed_grid", "start_time_grid", "cam_disp_grid"}
BINARY = {"interval_disp", "interval_speed", "motion_source"}
TIME = {"motion_duration_binary", "motion_duration_mcq"}


def load(fn):
    out = defaultdict(list)
    for l in open(D + fn):
        d = json.loads(l); out[d["event_subtype"]].append(d)
    return out


def numfeats(d):
    q = d["question"]; f = np.zeros(7)
    m = rx_fr.search(q); n = rx_N.search(q)
    N = int(n.group(1)) if n else 0
    if m:
        i, j = int(m.group(1)), int(m.group(2)); f[0], f[1], f[2], f[3] = i / max(N, 1), j / max(N, 1), np.log1p(j - i), 1
    f[4] = np.log(max(N, 1))
    ivs = rx_iv.findall(q)
    if len(ivs) == 2:
        a, b = [int(y) - int(x) for x, y in ivs]; f[5] = np.log1p(a) - np.log1p(b); f[6] = 1
        # 2026-09-14:补非线性可见特征(|log 时长比|、分割点位置、总跨度)——训练后盲模型曾在 interval_speed 上靠时长比绝对值拿 67%,线性差值特征漏掉
        f = np.concatenate([f, [abs(np.log((max(a, b) + 1e-6) / (min(a, b) + 1e-6))), int(ivs[0][1]) / max(N, 1), (int(ivs[1][1]) - int(ivs[0][0])) / max(N, 1)]])
    else:
        f = np.concatenate([f, [0.0, 0.0, 0.0]])
    th = re.search(r"more than ([\d.]+) seconds", q)
    if th: f[5] = float(th.group(1)); f[6] = 1
    return f


def optvals(d):
    ks = list(d["options"]); return ks, np.array([float(re.findall(r"[\d.]+", d["options"][k])[0]) for k in ks])


def feats(vec, L, fit=False, with_window=True):
    txt = [re.sub(r"[\d.]+", " ", d["question"]) for d in L]
    X = vec.fit_transform(txt) if fit else vec.transform(txt)
    nf = np.stack([numfeats(d) for d in L])
    if with_window:
        w = np.stack([[np.log(optvals(d)[1].min()), np.log(optvals(d)[1].max())] for d in L]); nf = np.hstack([nf, w])
    return sparse.hstack([X, sparse.csr_matrix(nf)]).tocsr()


def nearest_ok(d, logpred):
    ks, v = optvals(d); return ks[int(np.argmin(np.abs(np.log(v) - logpred)))] == d["answer"]


def rank_of(d):
    ks, v = optvals(d); return int(np.argsort(np.argsort(v))[ks.index(d["answer"])]) + 1


def cls_of_text(d):
    return d["gt_payload"].get("cls", "?")


tr, te = load("train.jsonl"), load("test.jsonl")
res = {}
print(f"{'族':24s} {'n_te':>5s} {'chance':>6s} {'①线性':>6s} {'①名次':>6s} {'①规则':>6s} {'②σ.3':>6s} {'②σ.5':>6s}  字母 / 备注")
for fam in sorted(set(tr) | set(te)):
    A, B = tr.get(fam, []), te.get(fam, [])
    r = dict(n_train=len(A), n_test=len(B), letters=dict(sorted(Counter(d["answer"] for d in B).items())))
    if len(A) < 40 or len(B) < 20:
        print(f"{fam:24s} {len(B):5d}   (题太少,跳过) {r['letters']}"); res[fam] = r; continue
    vec = CountVectorizer(token_pattern=r"[A-Za-z]+", min_df=2, binary=True)
    if fam in NUMERIC or fam == "motion_duration_mcq":
        key = "value" if fam in NUMERIC else "duration_s"
        y = np.log([d["gt_payload"][key] for d in A])
        Xa, Xb = feats(vec, A, True), feats(vec, B)
        mu = Ridge(alpha=3.0).fit(Xa, y).predict(Xb); r["lin"] = float(np.mean([nearest_ok(d, m) for d, m in zip(B, mu)]))
        ya = [rank_of(d) for d in A]
        r["rank_lr"] = float(np.mean([p == rank_of(d) for p, d in zip(LogisticRegression(max_iter=3000).fit(Xa, ya).predict(Xb), B)])) if len(set(ya)) > 1 else None
        rules = {f"rank{k}": float(np.mean([rank_of(d) == k for d in B])) for k in (1, 2, 3, 4)}
        rules["fam_median"] = float(np.mean([nearest_ok(d, np.median(y)) for d in B]))
        cm = defaultdict(list)
        for d, v in zip(A, y): cm[cls_of_text(d)].append(v)
        rules["cls_median"] = float(np.mean([nearest_ok(d, np.median(cm.get(cls_of_text(d), y))) for d in B]))
        r["rules"] = rules; r["rule_max"] = max(rules.values())
        rng = np.random.RandomState(0)
        for s in (0.3, 0.5):
            acc = []
            for _ in range(10):
                acc.append(np.mean([nearest_ok(d, np.log(d["gt_payload"][key]) + rng.normal(0, s)) for d in B]))
            r[f"coarse_{s}"] = float(np.mean(acc))
        chance = 0.25
        print(f"{fam:24s} {len(B):5d} {chance:6.2f} {r['lin']:6.3f} {(r['rank_lr'] or 0):6.3f} {r['rule_max']:6.3f} {r['coarse_0.3']:6.3f} {r['coarse_0.5']:6.3f}  {r['letters']} rules={ {k: round(v, 2) for k, v in rules.items()} }")
    else:
        vec.fit([re.sub(r"[\d.]+", " ", d["question"]) for d in A] + [o for d in A for o in d["options"].values()])
        def M(L):
            q = vec.transform([re.sub(r"[\d.]+", " ", d["question"]) for d in L])
            a = vec.transform([list(d["options"].values())[0] for d in L]); b = vec.transform([list(d["options"].values())[1] for d in L])
            nf = np.stack([numfeats(d) for d in L])
            ivd = np.zeros((len(L), 1))
            for i, d in enumerate(L):
                oa, ob = list(d["options"].values())
                ia, ib = rx_iv.search(oa), rx_iv.search(ob)
                if ia and ib: ivd[i, 0] = np.log1p(int(ia.group(2)) - int(ia.group(1))) - np.log1p(int(ib.group(2)) - int(ib.group(1)))
            return sparse.hstack([q, a - b, sparse.csr_matrix(nf), sparse.csr_matrix(ivd)]).tocsr()
        first_ok = lambda d: list(d["options"])[0] == d["answer"]
        p = LogisticRegression(max_iter=3000).fit(M(A), [int(first_ok(d)) for d in A]).predict(M(B))
        r["lin"] = float(np.mean([(pi == 1) == first_ok(d) for pi, d in zip(p, B)]))
        rules = {"always_A": float(np.mean([d["answer"] == "A" for d in B]))}
        if fam in ("interval_disp", "interval_speed"):
            def longer(d):
                oa, ob = list(d["options"].values()); ia, ib = rx_iv.search(oa), rx_iv.search(ob)
                la, lb = int(ia.group(2)) - int(ia.group(1)), int(ib.group(2)) - int(ib.group(1))
                return "A" if la > lb else "B" if lb > la else None
            rules["pick_longer_window"] = float(np.mean([longer(d) == d["answer"] for d in B if longer(d)]))
            rules["pick_first_interval"] = float(np.mean([(d["options"][d["answer"]] == d["gt_payload"]["intervals"][0]) for d in B]))
        if fam == "motion_source":
            rules["always_camera"] = float(np.mean([d["options"][d["answer"]] == "the camera moved" for d in B]))
        if fam == "motion_duration_binary":
            rules["always_yes"] = float(np.mean([d["options"][d["answer"]] == "Yes" for d in B]))
        r["rules"] = rules; r["rule_max"] = max(rules.values()); chance = 0.5
        r["cls_x_answer"] = {c: dict(Counter(d["options"][d["answer"]] if fam != "motion_duration_binary" else d["options"][d["answer"]] for d in B if cls_of_text(d) == c)) for c in sorted({cls_of_text(d) for d in B})}
        print(f"{fam:24s} {len(B):5d} {chance:6.2f} {r['lin']:6.3f} {'-':>6s} {r['rule_max']:6.3f} {'-':>6s} {'-':>6s}  {r['letters']} rules={ {k: round(v, 2) for k, v in rules.items()} }")
    res[fam] = r
# ④ 跨族重复:同 (video, inst, i, j) 出现在多族
key = Counter()
for L in list(tr.values()) + list(te.values()):
    for d in L:
        g = d["gt_payload"]
        if "i" in g and "j" in g: key[(d["video_id"], tuple(d["involved_inst_ids"]), g["i"], g["j"], "k" in g)] += 1
dup = sum(1 for v in key.values() if v > 1)
srcs = Counter(json.dumps(d.get("src")) for L in list(tr.values()) + list(te.values()) for d in L)
res["_dup_window_keys"] = dup; res["_src"] = dict(srcs)
print(f"跨族同窗口重复 {dup};来源 {dict(srcs)}")
json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
