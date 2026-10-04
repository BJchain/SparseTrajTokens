"""B 集构建(2026-09-14):候选 + Gate 0 → 固定网格选项 + 名次配平(LP)/ 二分族配平 / 时间族配平 / 按视频三分 / evidence / 报告。

选项设计(踩坑对策):
  · 数值族用 **固定全局 log 网格**(每族 v0·r^k),每题展示 4 个**相邻**格的几何中心;真值所在格在 4 格中的名次
    用 LP 配平到 **P(名次 | 窗口) 均匀**(按层:类别 × 帧距档 / 帧率档)。这样:
      - 盲读者看到选项(=窗口)也只能 25%(旧阶梯:窗口随真值移动 → 选"离典型值最近"31–39%);
      - 分辨率保住 r=1.35(09-13 实测 RGB 米制精度 log-SD 0.3–0.5,1.36× 刚好卡住;先验采样放到 1.7× 后 RGB +11–20);
      - 选项不随真值生成 → 无"真值在帧网格上而干扰项不在"这类泄漏。
    真值落在格边界 ±EDGE_MARGIN(格宽的 8%)内的题剔除(标签对噪声不稳),不影响盲=chance(选项集与真值位置无关)。
  · 二分族按 (类别 × 答案 × 混淆变量) 等量:区间比较配平"长区间是否答案",速率比较再配平"与位移答案是否一致",
    运动来源按类别配平"物体动/相机动";字母在每格内严格各半。
  · 时间族复用 time_qa_gen._balance(阈值桶内 yes/no 配对、帧率档封顶、严格子集)。
  · Gate 0(token 侧复算)不一致的题剔除(硬:位移/速率/相机/区间/来源;软:开始时刻/时长只记账,因其标签规则依赖逐帧速度门,
    节点折线不能逐位复现,而 rd0.2 实测 token-only 时长题 88% 可学)。
  · 按视频封顶(每视频每族 ≤ CAP_VF、每视频 ≤ CAP_V)后再配平,配平后不再删题。
用法:PYTHONPATH=src/training python -m g1.trajqa_b_build --cand <candidates.jsonl> --gate0 <gate0.jsonl> --out <dir>"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict

import numpy as np
from scipy.optimize import linprog

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))
from g1.split import split3_items, group_key, _u  # noqa: E402
from g1.time_qa_gen import _balance as time_balance, _waterfill  # noqa: E402

LETTERS = "ABCD"
GRID = {"disp_grid": (0.5, 1.35, 12), "speed_grid": (0.1, 1.35, 14), "start_time_grid": (0.5, 1.35, 10), "cam_disp_grid": (0.5, 1.35, 13)}
NUMERIC = set(GRID)
BINARY = {"interval_disp", "interval_speed", "motion_source"}
TIME = {"motion_duration_binary", "motion_duration_mcq"}
G0_SOFT = {"start_time_grid"}   # (已撤族;保留字段兼容)
FPS_CAPPED = {"speed_grid", "start_time_grid"}          # 题面无秒 → 帧率档先验要压(~5 档 ≤ cap)
EDGE_MARGIN = 0.06
GAP_BINS = ((0, 7, "g≤7"), (8, 15, "g8-15"), (16, 10 ** 6, "g≥16"))
GAP_BINS_FINE = ((0, 5, "g≤5"), (6, 8, "g6-8"), (9, 12, "g9-12"), (13, 18, "g13-18"), (19, 10 ** 6, "g≥19"))


def gap_bin_fine(g):
    for lo, hi, name in GAP_BINS_FINE:
        if lo <= g <= hi: return name
    return GAP_BINS_FINE[-1][2]
CAP_VF, CAP_V = 10, 40          # 默认值(与 09-14 定稿的 sft_trajqa_b_v4td 一致);扩量时只对 train 放开,见 --cap-vf/--cap-v


def fmt_num(v):
    return f"{v:.2f}" if v < 1 else (f"{v:.1f}" if v < 10 else f"{v:.0f}")


def gap_bin(g):
    for lo, hi, name in GAP_BINS:
        if lo <= g <= hi: return name
    return GAP_BINS[-1][2]


def cell_of(v, fam):
    v0, r, n = GRID[fam]
    x = np.log(max(v, 1e-9) / v0) / np.log(r); c = int(np.floor(x)); return c, float(x - c)


def stratum(d, fam, split):
    """train 按 (类别 × 帧距档) 配平(盲模型从 train 学不到 帧距/类别→名次 的相关);val/test 只按类别(小切分保供给,
    窗口内名次均匀即可保证盲=chance)。"""
    g = d["gt_payload"]; fine = (split == "train")
    if fam in ("disp_grid", "speed_grid"): return (g.get("cls"), gap_bin_fine(g.get("gap", 0))) if fine else (g.get("cls"),)
    if fam == "start_time_grid": return (g.get("cls"), g.get("fps_bin")) if fine else (g.get("cls"),)
    if fam == "cam_disp_grid": return (gap_bin_fine(g.get("gap", 0)),) if fine else ("all",)
    return ("all",)


def lp_windows(counts, n_cells):
    """counts: cell→题数。求每个 4 格窗口的出题数 m_w(整数),使 4 格各出 m_w 题、总量最大。"""
    nwin = n_cells - 3
    if nwin <= 0: return {}
    A = np.zeros((n_cells, nwin)); b = np.array([counts.get(c, 0) for c in range(n_cells)], float)
    for w in range(nwin):
        A[w:w + 4, w] = 1
    res = linprog(-np.ones(nwin), A_ub=A, b_ub=b, bounds=[(0, None)] * nwin, method="highs")
    if not res.success: return {}
    m = np.floor(res.x + 1e-9).astype(int)
    for w in np.argsort(-(res.x - m)):
        m[w] += 1
        if (A @ m > b + 1e-9).any(): m[w] -= 1
    return {int(w): int(m[w]) for w in range(nwin) if m[w] > 0}


def finalize_numeric(d, fam, w, k, rng):
    v0, r, n = GRID[fam]
    vals = [v0 * r ** (w + kk + 0.5) for kk in range(4)]
    txt = [fmt_num(x) for x in vals]
    assert len(set(txt)) == 4, txt
    order = list(range(4)); rng.shuffle(order)
    d = dict(d); d["options"] = {LETTERS[p]: txt[order[p]] for p in range(4)}; d["answer"] = LETTERS[order.index(k)]
    gp = dict(d["gt_payload"]); gp.update(window=w, rank=k + 1, grid=dict(v0=v0, r=r, n=n)); d["gt_payload"] = gp
    return d


def assign_numeric(items, fam, rng):
    bycell = defaultdict(list)
    for d in items: bycell[d["gt_payload"]["cell"]].append(d)
    for L in bycell.values(): rng.shuffle(L)
    m = lp_windows({c: len(L) for c, L in bycell.items()}, GRID[fam][2])
    out = []; ptr = defaultdict(int)
    for w, mw in sorted(m.items()):
        for k in range(4):
            c = w + k; L = bycell[c]; take = L[ptr[c]:ptr[c] + mw]; ptr[c] += mw
            assert len(take) == mw
            out += [finalize_numeric(d, fam, w, k, rng) for d in take]
    return out


ANS_VALUES = {"answer_interval": (1, 2), "longer_window_is_answer": (True, False), "speed_eq_disp": (True, False), "source": ("object", "camera"),
              "iv_type": ("longer_farther", "shorter_nearer")}
# interval_speed 的逻辑结构:更快的区间若更长则必然位移也更大;可行格只有 (长&位移大)、(短&位移大)、(短&位移小)。
# 同时压住"选短窗口"文字捷径与"选位移大"RGB 捷径 ⇒ 只留 (长&位移大) 与 (短&位移小) 各半("短&位移大"不出)。
BIN_SPEC = {   # ctx = 必须在答案两侧匹配的上下文(题面可见的特征);ans = 答案侧要等量的键
    "interval_disp": dict(ctx=("cls",), ans=("answer_interval", "longer_window_is_answer")),
    # 2026-09-14 pB_rgb_s0 训练后盲测 interval_speed 67%:两区间时长比越大越可能"短的更快"(比 1.5–2 → 长者为答案 0.62–0.75,3–5 → 0.30–0.36),
    # 线性审计漏掉(线索在比值的绝对值,非线性)。上下文再按 时长比档 × 分割点位置档 匹配。
    "interval_speed": dict(ctx=("cls", "ratiobin", "jpos"), ans=("answer_interval", "iv_type")),
    "motion_source": dict(ctx=("cls", "head", "Tbin", "gapbin", "fps2", "has_lr", "posbin"), ans=("source",)),
}


def ctx_key(d, ctx):
    g = d["gt_payload"]; out = []
    for k in ctx:
        if k == "gapbin": out.append(gap_bin(g.get("gap", 0)))
        elif k == "fps2": out.append("~5" if g.get("fps_bin") == "~5" else "fast")
        elif k == "has_lr": out.append(str(g.get("obj", "")).startswith(("the left", "the right")))
        elif k == "posbin": out.append(int(3 * g.get("i", 0) / max(g.get("T", 1), 1)))
        elif k == "head": out.append(str(g.get("obj", "")).split()[-1])
        elif k == "ratiobin":
            r = max(g.get("dt1", 1), g.get("dt2", 1)) / max(min(g.get("dt1", 1), g.get("dt2", 1)), 1e-6)
            out.append("r<2" if r < 2 else "r<3" if r < 3 else "r<5" if r < 5 else "r≥5")
        elif k == "jpos": out.append(int(3 * g.get("j", 0) / max(g.get("T", 1), 1)))
        elif k == "Tbin": out.append("T34" if g.get("T") == 34 else ("T<34" if g.get("T", 0) < 34 else "T>34"))
        else: out.append(g.get(k))
    return tuple(out)


def balance_cells(items, keys, rng):
    """答案侧全部组合(ANS_VALUES 的笛卡尔积)必须都在场,取最小数各抽等量;缺任一格 → 0(整层丢)。
    (旧实现只对"在场的格"取 min:某类别只有单一答案时会整层原样保留 → motion_source 的 other 类 281 题全是 object,线性盲 75%。)"""
    import itertools
    cells = defaultdict(list)
    for d in items: cells[tuple(d["gt_payload"].get(k) for k in keys)].append(d)
    expected = list(itertools.product(*[ANS_VALUES[k] for k in keys]))
    k = min(len(cells.get(c, [])) for c in expected)
    out = []
    for c in expected:
        v = cells.get(c, []); rng.shuffle(v); out += v[:k]
    return out, {str(c): len(v) for c, v in cells.items()}, k


def finalize_binary(items, fam, rng):
    """字母严格各半(每个配平格内交替)。"""
    out = []
    for idx, d in enumerate(items):
        g = d["gt_payload"]
        if fam in ("interval_disp", "interval_speed"):
            texts = list(g["intervals"]); correct = texts[g["answer_interval"] - 1]
        else:
            texts = list(g["choices"]); correct = texts[0] if g["source"] == "object" else texts[1]
        other = [t for t in texts if t != correct][0]
        d = dict(d)
        if idx % 2 == 0: d["options"] = {"A": correct, "B": other}; d["answer"] = "A"
        else: d["options"] = {"A": other, "B": correct}; d["answer"] = "B"
        out.append(d)
    return out


def cap_videos(items, rng, cap_vf=None, cap_v=None):
    cap_vf = CAP_VF if cap_vf is None else cap_vf; cap_v = CAP_V if cap_v is None else cap_v
    by = defaultdict(list)
    for d in items: by[(group_key(d["video_id"]), d["event_subtype"])].append(d)
    kept = []
    for L in by.values():
        rng.shuffle(L); kept += L[:cap_vf]
    byv = defaultdict(list)
    for d in kept: byv[group_key(d["video_id"])].append(d)
    out = []
    for L in byv.values():
        rng.shuffle(L); out += L[:cap_v]
    return out


def fps_cap(items, cap, rng):
    groups = defaultdict(list)
    for i, d in enumerate(items): groups[d["gt_payload"].get("fps_bin", "?")].append(i)
    k = _waterfill({b: len(v) for b, v in groups.items()}, cap)
    keep = []
    for b, v in groups.items():
        rng.shuffle(v); keep += v[:k[b]]
    return [items[i] for i in sorted(keep)]


def f1(x): return f"{x:.1f}"
def f2(x): return f"{x:.2f}"
def pos(p): return "(" + ", ".join(f1(v) for v in p) + ")"


def evidence(d):
    st = d["event_subtype"]; g = d["gt_payload"]
    if st == "disp_grid":
        return "\n".join([f"object: {g['obj']}", f"position_frame_{g['i']}_m: {pos(g['pos_i'])}", f"position_frame_{g['j']}_m: {pos(g['pos_j'])}", f"displacement_m: {f1(g['disp_m'])}"])
    if st == "speed_grid":
        return "\n".join([f"object: {g['obj']}", f"position_frame_{g['i']}_m: {pos(g['pos_i'])}", f"position_frame_{g['j']}_m: {pos(g['pos_j'])}", f"displacement_m: {f1(g['disp_m'])}", f"time_s: {f2(g['dt_s'])}", f"speed_mps: {f2(g['value'])}"])
    if st == "interval_disp":
        return "\n".join([f"object: {g['obj']}", f"displacement_m {g['intervals'][0]}: {f1(g['d1'])}", f"displacement_m {g['intervals'][1]}: {f1(g['d2'])}", f"longer_distance: {g['intervals'][g['answer_interval'] - 1]}"])
    if st == "interval_speed":
        return "\n".join([f"object: {g['obj']}", f"{g['intervals'][0]}: displacement_m {f1(g['d1'])}, time_s {f2(g['dt1'])}, speed_mps {f2(g['v1'])}", f"{g['intervals'][1]}: displacement_m {f1(g['d2'])}, time_s {f2(g['dt2'])}, speed_mps {f2(g['v2'])}", f"faster: {g['intervals'][g['answer_interval'] - 1]}"])
    if st == "start_time_grid":
        return "\n".join([f"object: {g['obj']}", f"motion_start_frame: {g['start_frame']}", f"motion_start_s: {f2(g['value'])}"])
    if st == "cam_disp_grid":
        return "\n".join([f"camera_frame_{g['i']}_m: {pos(g['cam_i'])}", f"camera_frame_{g['j']}_m: {pos(g['cam_j'])}", f"camera_displacement_m: {f1(g['value'])}"])
    if st == "motion_source":
        return "\n".join([f"object: {g['obj']}", f"object_world_displacement_m: {f1(g['obj_disp_m'])}", f"camera_displacement_m: {f1(g['cam_disp_m'])}", f"source_of_screen_motion: {'object' if g['source'] == 'object' else 'camera'}"])
    if st == "motion_duration_binary":
        return "\n".join([f"object: {g['obj']}", f"motion_frames: {g['motion_frames'][0]}-{g['motion_frames'][1]}", f"motion_duration_s: {f2(g['duration_s'])}", f"threshold_s: {g['threshold_s']}", f"exceeds_threshold: {'yes' if g['duration_s'] > g['threshold_s'] else 'no'}"])
    if st == "motion_duration_mcq":
        return "\n".join([f"object: {g['obj']}", f"motion_frames: {g['motion_frames'][0]}-{g['motion_frames'][1]}", f"motion_duration_s: {f2(g['duration_s'])}"])
    return None


FAM_CAP = dict(train=9000, val=1500, test=2500)   # 每族候选封顶(LP/配平前随机抽),防相机族独大


def build_split(items, name, rng, rep, caps=None):
    """caps = dict(cap_vf, cap_v, fam_cap):该划分的封顶参数(None = 用默认)。扩量只改 train 的 caps,val/test 不动。"""
    out = []
    byfam = defaultdict(list)
    for d in items: byfam[d["event_subtype"]].append(d)
    caps = caps or {}
    fam_cap_n = caps.get("fam_cap") or FAM_CAP[name]
    for fam, L in sorted(byfam.items()):
        R = dict(n_cand=len(L)); L = cap_videos(L, rng, caps.get("cap_vf"), caps.get("cap_v")); R["n_after_video_cap"] = len(L)
        if len(L) > fam_cap_n and fam not in TIME:
            rng.shuffle(L); L = L[:fam_cap_n]; R["n_after_family_cap"] = len(L)
        # Gate 0
        if fam in NUMERIC:
            keep, ng0, nedge, nrange = [], 0, 0, 0
            for d in L:
                g = d["gt_payload"]; c, frac = cell_of(g["value"], fam)
                if c < 0 or c >= GRID[fam][2]: nrange += 1; continue
                if frac < EDGE_MARGIN or frac > 1 - EDGE_MARGIN: nedge += 1; continue
                tv = g.get("g0_tok_value")
                agree = tv is not None and cell_of(tv, fam)[0] == c
                g = dict(g); g["cell"] = c; g["g0_agree"] = bool(agree); d = dict(d); d["gt_payload"] = g
                if not agree and fam not in G0_SOFT: ng0 += 1; continue
                keep.append(d)
            R.update(drop_out_of_grid=nrange, drop_edge=nedge, drop_gate0=ng0, g0_soft_agree=(round(np.mean([d["gt_payload"]["g0_agree"] for d in keep]), 3) if keep and fam in G0_SOFT else None))
            pass   # (历史遗留:帧率档封顶只在 test/val 标 eval_strict,训练集不压,见下)
            bystr = defaultdict(list)
            for d in keep: bystr[stratum(d, fam, name)].append(d)
            fin = []
            for s, LL in sorted(bystr.items()): fin += assign_numeric(LL, fam, rng)
            R["n_final"] = len(fin); R["strata"] = {str(s): len(v) for s, v in bystr.items()}
            R["cells"] = dict(Counter(d["gt_payload"]["cell"] for d in fin))
            R["rank_by_window_max_dev"] = 0.0
            if fin:
                cw = Counter((d["gt_payload"]["window"], d["gt_payload"]["rank"]) for d in fin); ws = Counter(d["gt_payload"]["window"] for d in fin)
                R["rank_by_window_max_dev"] = round(max(abs(cw[(w, r)] - ws[w] / 4) / (ws[w] / 4) for w in ws for r in (1, 2, 3, 4)), 3)
            if fam in FPS_CAPPED and fin and name != "train":      # 严格子集(每档 ≤ 1/3)标 eval_strict(报数用;题面无秒 → 帧率档先验)
                strict = {d["id"] for d in fps_cap(fin, 1 / 3, rng)}
                for d in fin: d["eval_strict"] = d["id"] in strict
        elif fam in BINARY:
            keep, ng0 = [], 0
            for d in L:
                g = d["gt_payload"]; ta = g.get("g0_tok_answer")
                truth = g["answer_interval"] if fam != "motion_source" else g["source"]
                if ta != truth: ng0 += 1; continue
                if fam == "interval_speed":
                    g["iv_type"] = ("longer_farther" if (g["longer_window_is_answer"] and g["speed_eq_disp"]) else
                                    "shorter_nearer" if (not g["longer_window_is_answer"] and not g["speed_eq_disp"]) else "mixed")
                keep.append(d)
            R["drop_gate0"] = ng0
            spec = BIN_SPEC[fam]
            if fam == "motion_source": keep = [d for d in keep if d["gt_payload"]["cls"] in ("human", "vehicle", "animal")]   # 两侧都只用可动类别
            fin = []
            byctx = defaultdict(list)
            for d in keep: byctx[ctx_key(d, spec["ctx"])].append(d)
            cells_rep = {}
            for c, LL in sorted(byctx.items(), key=str):
                b, cells, k = balance_cells(LL, spec["ans"], rng); cells_rep[str(c)] = dict(cells=cells, per_cell=k); fin += finalize_binary(b, fam, rng)
            R["n_final"] = len(fin); R["cells"] = cells_rep
            if fam == "interval_speed": R["longer_window_is_answer"] = dict(Counter(d["gt_payload"]["longer_window_is_answer"] for d in fin))
        elif fam in TIME:
            for d in L:
                g = d["gt_payload"]
                if fam == "motion_duration_binary": g["g0_agree"] = (g.get("g0_tok_answer") == d["options"][d["answer"]])
                else:
                    tv = g.get("g0_tok_value")
                    if tv is not None:
                        ks = list(d["options"]); vals = [float(d["options"][k].split()[1]) for k in ks]
                        g["g0_agree"] = ks[int(np.argmin([abs(np.log(tv / x)) for x in vals]))] == d["answer"]
                    else: g["g0_agree"] = False
            keep = [dict(d) for d in L if d["gt_payload"].get("g0_agree")]; R["drop_gate0"] = len(L) - len(keep)
            fin = time_balance(keep, rng, "all", 0.5, 1.0 / 3.0)
            rng.shuffle(fin)
            for idx, d in enumerate(fin):          # 字母严格各半:按序交替定正确字母(与内容无关),再摆选项
                yes = d["options"][d["answer"]] == "Yes"; ans = "A" if idx % 2 == 0 else "B"; oth = "B" if ans == "A" else "A"
                d["options"] = {ans: ("Yes" if yes else "No"), oth: ("No" if yes else "Yes")}; d["options"] = dict(sorted(d["options"].items())); d["answer"] = ans
            R["n_final"] = len(fin)
            R["fps"] = dict(Counter(d["gt_payload"]["fps_bin"] for d in fin)); R["n_strict"] = sum(1 for d in fin if d.get("eval_strict"))
            if fam == "motion_duration_binary": R["answers"] = dict(Counter(d["options"][d["answer"]] for d in fin))
        else:
            continue
        for d in fin:
            ev = evidence(d)
            if ev: d["evidence"] = ev
        R["letters"] = dict(Counter(d["answer"] for d in fin)); R["cls"] = dict(Counter(d["gt_payload"].get("cls") for d in fin))
        R["n_videos"] = len({group_key(d["video_id"]) for d in fin})
        rep[name][fam] = R; out += fin
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True); ap.add_argument("--gate0", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--splits", default="train,val,test", help="只构建这些划分(扩量时用 train,val/test 从 --copy-from 原样复制,保证与已有评测可比)")
    ap.add_argument("--copy-from", default="", help="未构建的划分从这个目录复制(逐字节)")
    ap.add_argument("--cap-vf", type=int, default=0, help="train 的每视频每族上限(0=默认 10)")
    ap.add_argument("--cap-v", type=int, default=0, help="train 的每视频上限(0=默认 40)")
    ap.add_argument("--fam-cap", type=int, default=0, help="train 的每族候选上限(0=默认 9000)")
    ap.add_argument("--band", default="", help="lo,hi:只取视频哈希 u∈[lo,hi) 的视频,按 test 口径单独建一份(2026-09-17 切全新 held-out test 用)")
    ap.add_argument("--exclude-band", default="", help="lo,hi:先把这个视频哈希区间整体剔除再划分(held-out 波段不得进训练集;必须在配平之前剔,否则名次均匀会被破坏)")
    ap.add_argument("--band-name", default="test", help="--band 产出的文件名(默认 test.jsonl)")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    G0 = {}
    for l in open(a.gate0):
        r = json.loads(l); G0[r["id"]] = r
    items = []
    for l in open(a.cand):
        d = json.loads(l); r = G0.get(d["id"], {})
        d["gt_payload"]["g0_tok_value"] = r.get("tok_value"); d["gt_payload"]["g0_tok_answer"] = r.get("tok_answer"); d["gt_payload"]["g0_note"] = r.get("note", "")
        items.append(d)
    if a.exclude_band:
        lo, hi = [float(x) for x in a.exclude_band.split(",")]
        n0 = len(items); items = [d for d in items if not (lo <= _u(group_key(d["video_id"])) < hi)]
        print(f"[exclude-band {lo}-{hi}] 候选 {n0} → {len(items)}(剔除 held-out 波段的视频)", flush=True)
    if a.band:                                             # 单独建一个视频哈希区间的集合(全新 held-out test)
        lo, hi = [float(x) for x in a.band.split(",")]
        sel = [d for d in items if lo <= _u(group_key(d["video_id"])) < hi]
        rep_ = defaultdict(dict); rng_ = random.Random(a.seed)
        out = build_split(sel, "test", rng_, rep_, caps=dict(cap_vf=a.cap_vf or None, cap_v=a.cap_v or None, fam_cap=a.fam_cap or None))
        with open(os.path.join(a.out, f"{a.band_name}.jsonl"), "w") as fh:
            for d in out: fh.write(json.dumps(d, ensure_ascii=False) + "\n")
        json.dump(dict(band=[lo, hi], n=len(out), n_videos=len({group_key(d["video_id"]) for d in out}), report=rep_["test"]),
                  open(os.path.join(a.out, f"_{a.band_name}_report.json"), "w"), indent=1, ensure_ascii=False)
        print(f"[band {lo}-{hi}] {len(out)} 题 / {len({group_key(d['video_id']) for d in out})} 视频 → {a.out}/{a.band_name}.jsonl")
        for f, v in sorted(rep_["test"].items()): print(f"  {f:24s} cand={v['n_cand']:6d} final={v['n_final']:6d} letters={v['letters']}")
        return
    tr, va, te = split3_items(items, 0.05, 0.2, "video")   # B: test=u<0.2(= A 的 test∪val 视频,A 的训练视频不会进 B-test),val 5%
    rep = defaultdict(dict); rng = random.Random(a.seed)
    want = [x.strip() for x in a.splits.split(",") if x.strip()]
    caps = dict(cap_vf=a.cap_vf or None, cap_v=a.cap_v or None, fam_cap=a.fam_cap or None)
    outs = {}
    for name, L in (("train", tr), ("val", va), ("test", te)):
        if name not in want:                                  # 不构建的划分:从 --copy-from 逐字节复制
            src = os.path.join(a.copy_from, f"{name}.jsonl")
            if not os.path.exists(src): raise SystemExit(f"--splits 未含 {name} 且 --copy-from 里没有 {src}")
            import shutil; shutil.copyfile(src, os.path.join(a.out, f"{name}.jsonl"))
            outs[name] = [json.loads(l) for l in open(src)]
            print(f"[{name}] 从 {src} 原样复制 {len(outs[name])} 题(不重建,保证与已有评测可比)", flush=True)
            continue
        outs[name] = build_split(L, name, rng, rep, caps=(caps if name == "train" else None))
        with open(os.path.join(a.out, f"{name}.jsonl"), "w") as fh:
            for d in outs[name]: fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    meta = dict(cand=a.cand, gate0=a.gate0, grid=GRID, edge_margin=EDGE_MARGIN, gap_bins=GAP_BINS, seed=a.seed, splits_built=want, copy_from=a.copy_from,
                caps=dict(video_family=a.cap_vf or CAP_VF, video=a.cap_v or CAP_V, fam_cap=a.fam_cap or FAM_CAP["train"]),
                n=dict((k, len(v)) for k, v in outs.items()), n_videos=dict((k, len({group_key(d["video_id"]) for d in v})) for k, v in outs.items()),
                overlap_videos=len({group_key(d["video_id"]) for d in outs["train"]} & {group_key(d["video_id"]) for d in outs["test"]}))
    json.dump(dict(meta=meta, report=rep), open(os.path.join(a.out, "_build_report.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(meta, ensure_ascii=False, indent=1))
    for name in ("train", "test"):
        print(f"--- {name}")
        for fam, R in rep[name].items():
            print(f"  {fam:24s} cand={R['n_cand']:6d} final={R['n_final']:6d} g0drop={R.get('drop_gate0', '-')} edge={R.get('drop_edge', '-')} rankdev={R.get('rank_by_window_max_dev', '-')} letters={R['letters']} cls={R['cls']} soft_g0={R.get('g0_soft_agree', '-')}")


if __name__ == "__main__":
    main()
