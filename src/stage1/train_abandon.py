"""训练"提前放弃"模型(src/stage1/abandon.py 用的那个 npz),并做零 GPU 的策略回放。

数据来源是生产跑落的 `stage1_candidates.jsonl` 插桩(每候选一行:rank / 4 特征 / on_mask /
是否达标),**不需要任何额外 GPU**。

两件事必须显式给定,否则训出来的模型是错的:

* `--engine`:标签是"这个实例达不达得到 accept_onmask",而**达标率是引擎的函数** ——
  官方 SAM3D 52%、Fast-SAM3D 25%(同一 29 场景 A/B 实测)。特征(mask 几何)与引擎无关,
  所以两边共用同一套特征,但权重/截距/阈值表必须分开训。
* `--k`:候选帧上限。标签是"在**前 k 个**候选里达标",k 变了标签就变了。

⚠️ 截尾(censoring):采数那一跑本身开着提前放弃,被砍到 cap 的实例只观测到前 cap 个候选,
"它在第 cap+1 个候选会不会达标"是未知的。这类实例**从训练和回放里剔除**(不能当负样本 ——
那等于把老模型的判断当真值,自我强化)。剔除数会打印出来。

用法:
    python3 src/stage1/train_abandon.py --logs outputs/ab_fsam3d --engine fastsam3d --k 3 \
        --out src/stage1/abandon_model_fastsam3d.npz
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os

import numpy as np

FRACS = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60])


# --------------------------------------------------------------------------- 数据
def load_instances(log_roots, k: int):
    """把候选日志聚成"每实例一条":特征取 rank-1(决策时唯一拿得到的),标签 = 前 k 个候选里
    有没有达标。返回 (feats, label, group, clip, calls_k) 四元组的列表 + 被截尾剔除的条数。"""
    rows = collections.defaultdict(list)
    for root in log_roots:
        for f in glob.glob(os.path.join(root, "scene_*", "stage1_candidates.jsonl")):
            for line in open(f):
                line = line.strip()
                if line:
                    d = json.loads(line)
                    rows[d["group"]].append(d)
    out, censored = [], 0
    for g, v in rows.items():
        v.sort(key=lambda d: d["rank"])
        n_cand = int(v[0].get("n_cand", len(v)))
        pool_k = min(k, n_cand)                     # k=3 时实际能试几个
        obs = v[:k]
        acc_rank = next((d["rank"] for d in obs if d["on_mask"] >= 0.9), None)
        if acc_rank is None and len(v) < pool_k:
            censored += 1                           # 观测被老 abandon 砍断,标签未知 -> 弃
            continue
        out.append(dict(
            feats=np.asarray(v[0]["feats"], float),
            fail=int(acc_rank is None),             # 1 = 注定失败(模型要预测的)
            group=g, clip=g.split(":")[0],
            acc_rank=acc_rank, on1=float(v[0]["on_mask"]),                      # None = 前 k 个都没达标
            calls_k=(acc_rank if acc_rank is not None else pool_k),   # 无 abandon 时的调用数
            pool_k=pool_k,
        ))
    return out, censored


# --------------------------------------------------------------------------- 模型
def fit_logistic(X, y, l2=1.0, iters=200):
    """标准化 + L2 逻辑回归(IRLS)。纯 numpy,不引入 sklearn 依赖。"""
    mean, std = X.mean(0), X.std(0)
    std = np.where(std < 1e-8, 1.0, std)
    Z = np.column_stack([(X - mean) / std, np.ones(len(X))])
    w = np.zeros(Z.shape[1])
    R = np.eye(Z.shape[1]) * l2
    R[-1, -1] = 0.0                                  # 不正则截距
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(Z @ w)))
        W = np.clip(p * (1 - p), 1e-6, None)
        H = Z.T @ (Z * W[:, None]) + R
        gnorm = Z.T @ (y - p) - R @ w
        step = np.linalg.solve(H, gnorm)
        w = w + step
        if np.abs(step).max() < 1e-9:
            break
    return mean, std, w[:-1], float(w[-1])


def predict(mean, std, weights, bias, X):
    return 1.0 / (1.0 + np.exp(-(((X - mean) / std) @ weights + bias)))


def auc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def grouped_oof(data, X, y, clips, n_folds=5, l2=1.0):
    """按 clip 分组的交叉验证 —— 必须分组:同一 clip 的实例特征高度相关,随机分折 AUC 虚高。"""
    uniq = sorted(set(clips))
    fold_of = {c: i % n_folds for i, c in enumerate(uniq)}
    oof = np.zeros(len(y)); aucs = []
    for f in range(n_folds):
        te = np.array([fold_of[c] == f for c in clips])
        tr = ~te
        if te.sum() == 0 or len(set(y[tr])) < 2:
            continue
        m, s, w, b = fit_logistic(X[tr], y[tr], l2=l2)
        oof[te] = predict(m, s, w, b, X[te])
        if len(set(y[te])) == 2:
            aucs.append(auc(y[te], oof[te]))
    return oof, (float(np.mean(aucs)) if aucs else float("nan")), \
        (float(np.std(aucs)) if aucs else float("nan"))


# --------------------------------------------------------------------------- 回放
def replay(data, scores, thresholds, fracs, cap):
    """给定每实例的(样本外)分数,枚举 abandon_frac -> (省的调用数, 真丢的达标实例数)。
    完全可复算:调用数 = min(达标 rank, 上限);上限 = 命中就 cap,否则 pool_k。"""
    base = sum(d["calls_k"] for d in data)
    rows = []
    for fr, thr in zip(fracs, thresholds):
        calls = lost = flagged = 0
        for d, s in zip(data, scores):
            hit = s >= thr
            lim = min(cap, d["pool_k"]) if hit else d["pool_k"]
            flagged += int(hit)
            if d["acc_rank"] is not None and d["acc_rank"] <= lim:
                calls += d["acc_rank"]
            else:
                calls += lim
                lost += int(d["acc_rank"] is not None)     # 本来能达标,被砍掉了
        rows.append(dict(frac=float(fr), thr=float(thr), calls=calls, base=base,
                         saved=1 - calls / base, flagged=flagged / len(data),
                         lost=lost, lost_frac=lost / len(data)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="+", required=True, help="批次根目录(含 scene_*/stage1_candidates.jsonl)")
    ap.add_argument("--engine", required=True, choices=("sam3d", "fastsam3d"))
    ap.add_argument("--k", type=int, required=True, help="候选帧上限(标签口径)")
    ap.add_argument("--cap", type=int, default=2, help="回放用的 abandon_cap")
    ap.add_argument("--out", help="不给就只分析不落盘")
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--compare", help="另一个 npz:同样回放一遍做对照臂")
    ap.add_argument("--onmask-gate", type=float, default=0.0,
                    help="第 1 个候选的 on_mask 低于此值就不再试后面的候选(0=关)。"
                         "这是【调用后】的门,信号比 4 个 mask 特征强得多,但阈值是引擎特异的 ——"
                         "见 __doc__ 与 abandon.py 的表。")
    ap.add_argument("--rec-frac", type=float, default=None,
                    help="写进模型的建议 abandon_frac(仅提示,不强制)")
    args = ap.parse_args()

    data, censored = load_instances(args.logs, args.k)
    X = np.array([d["feats"] for d in data])
    y = np.array([d["fail"] for d in data])
    clips = [d["clip"] for d in data]
    print(f"[train] engine={args.engine} k={args.k} 实例={len(data)}(截尾剔除 {censored}) "
          f"clip={len(set(clips))} 注定失败率={y.mean():.3f}")

    oof, m_auc, s_auc = grouped_oof(data, X, y, clips, l2=args.l2)
    print(f"[train] 分组 5 折 CV AUC = {m_auc:.3f} ± {s_auc:.3f}(样本内 AUC 见下)")

    mean, std, w, b = fit_logistic(X, y, l2=args.l2)
    ins = predict(mean, std, w, b, X)
    print(f"[train] 样本内 AUC = {auc(y, ins):.3f}  weights={np.round(w, 3)} bias={b:.3f}")

    thr = np.quantile(ins, 1.0 - FRACS)          # 分数越高越该放弃 -> 上分位
    print("\n[回放:本模型(样本外分数)] cap=%d" % args.cap)
    for r in replay(data, oof, np.quantile(oof, 1.0 - FRACS), FRACS, args.cap):
        print(f"  frac={r['frac']:.2f}  省调用 {r['saved']:6.1%}  判死 {r['flagged']:5.1%}  "
              f"真丢达标 {r['lost']:3d} ({r['lost_frac']:.1%})")

    if args.compare:
        from src.stage1.abandon import AbandonModel
        old = AbandonModel.load(args.compare)
        so = np.array([old.score(x) for x in X])
        print(f"\n[回放:对照臂 {os.path.basename(args.compare)}] cap={args.cap}")
        for r in replay(data, so, np.array([old.threshold_for(f) for f in FRACS]), FRACS, args.cap):
            print(f"  frac={r['frac']:.2f}  省调用 {r['saved']:6.1%}  判死 {r['flagged']:5.1%}  "
                  f"真丢达标 {r['lost']:3d} ({r['lost_frac']:.1%})")
        print(f"  (该模型在本数据上的 AUC = {auc(y, so):.3f})")

    if args.onmask_gate > 0:
        print(f"\n[回放:on-mask 门 {args.onmask_gate}] (逐实例模拟 k={args.k},与上表独立)")
        base = sum(d["calls_k"] for d in data)
        calls = lost = 0
        for d in data:
            used = d["calls_k"]
            if d["acc_rank"] != 1 and d["on1"] < args.onmask_gate:
                used = 1
                lost += int(d["acc_rank"] is not None)
            calls += used
        print(f"  省调用 {1 - calls / base:.1%}  真丢达标 {lost} ({lost / len(data):.1%})")

    if args.out:
        np.savez(args.out, mean=mean, std=std, weights=w, bias=b,
                 fracs=FRACS, thresholds=thr,
                 n_train=len(data), base_rate=float(y.mean()), auc_cv=m_auc, auc_cv_std=s_auc,
                 n_unique=len(data), n_clips=len(set(clips)),
                 engine=args.engine, k=args.k, n_censored=censored,
                 onmask_gate=float(args.onmask_gate),
                 rec_frac=float(args.rec_frac if args.rec_frac is not None else 0.2))
        print(f"\n[train] 已写 {args.out}")


if __name__ == "__main__":
    main()
