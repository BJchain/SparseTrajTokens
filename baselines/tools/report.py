#!/usr/bin/env python3
"""跑完一个模型 -> 立刻把结果写进 RESULTS.md(§1 主表 / §2 逐族 Acc / §3 逐族耗时 / 运行日志)。

    python tools/report.py --model Qwen2.5-VL-7B --weights 15.4 --config "8帧@384 plain b=8 vLLM" \\
        --pred preds/qwen25vl7b.jsonl --blind preds/qwen25vl7b_blind.jsonl \\
        [--wall 1234] [--engine vLLM] [--note "..."]

口径全部复用 tools/common.py 的 NUMERIC3 与 answers.jsonl,不另发明。
同一 (模型, 配置) 重复跑会**替换**已有行,不会重复追加。
"""
import argparse, collections, json, math, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import LETTERS, NUMERIC3

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAM_ORDER = list(NUMERIC3) + ["motion_duration_binary", "motion_source",
                              "interval_disp", "interval_speed"]
SANITY_LO, SANITY_HI = 25.0, 50.0     # README:基模零样本数值三族的正常区间
LEAK_ALARM = 60.0                     # 超过就当泄漏报警,不报喜


def load_preds(p):
    out = {}
    if not p or not os.path.exists(p):
        return out
    for line in open(p):
        line = line.strip()
        if line:
            d = json.loads(line)
            out[d["id"]] = d
    return out


def stats(preds, gold):
    """-> acc_rows[key]=[correct,n], times[family]=[t...], letters"""
    rows = collections.defaultdict(lambda: [0, 0])
    times = collections.defaultdict(list)
    letters = collections.Counter()
    for qid, d in preds.items():
        g = gold.get(qid)
        if g is None:
            continue
        p = (d.get("pred") or "?").strip().upper()[:1]
        ok = int(p == g["answer"])
        letters[p if p in LETTERS else "?"] += 1
        fam, sp = g["family"], g["split"]
        for k in ("ALL", f"split:{sp}", f"fam:{fam}", f"{sp}/{fam}"):
            rows[k][0] += ok; rows[k][1] += 1
        if fam in NUMERIC3:
            for k in ("NUM3", f"NUM3:{sp}"):
                rows[k][0] += ok; rows[k][1] += 1
        if "t" in d:
            times[fam].append(d["t"])
            if fam in NUMERIC3:
                times["_NUM3"].append(d["t"])
            times["_ALL"].append(d["t"])
    return rows, times, letters


def acc(r):
    return 100.0 * r[0] / r[1] if r[1] else float("nan")


def se(r):
    a = r[0] / r[1] if r[1] else 0.0
    return 100.0 * math.sqrt(max(a * (1 - a), 1e-12) / r[1]) if r[1] else float("nan")


def fmt(x, nd=1):
    return "—" if x != x else f"{x:.{nd}f}"


def upsert_row(text, section_head, key_cell, new_row):
    """在 markdown 小节的表格末尾插入/替换一行(按第一列匹配)。"""
    i = text.index(section_head)
    j = text.find("\n## ", i + 1)
    if j == -1:
        j = len(text)
    block = text[i:j]
    lines = block.split("\n")
    # 丢掉占位行
    lines = [l for l in lines if not l.startswith("| _(")]
    hit = None
    for n, l in enumerate(lines):
        if l.startswith("|") and l.split("|")[1].strip() == key_cell:
            hit = n
            break
    if hit is not None:
        lines[hit] = new_row
    else:
        last = max((n for n, l in enumerate(lines) if l.startswith("|")), default=None)
        if last is None:
            lines.append(new_row)
        else:
            lines.insert(last + 1, new_row)
    return text[:i] + "\n".join(lines) + text[j:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="表里显示的模型名")
    ap.add_argument("--weights", default="", help="权重 GiB,照抄 GPU_BUDGET")
    ap.add_argument("--config", required=True, help="如 '8帧@384 plain b=8 vLLM'")
    ap.add_argument("--pred", required=True)
    ap.add_argument("--blind", default=None)
    ap.add_argument("--pkg", default=PKG)
    ap.add_argument("--results", default=os.path.join(PKG, "RESULTS.md"))
    ap.add_argument("--wall", type=float, default=None, help="看帧臂整轮墙钟秒")
    ap.add_argument("--engine", default="")
    ap.add_argument("--note", default="")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    gold = {d["id"]: d for d in map(json.loads,
            open(os.path.join(a.pkg, "qa", "answers.jsonl")))}
    rows, times, letters = stats(load_preds(a.pred), gold)
    brows, _, bletters = stats(load_preds(a.blind), gold) if a.blind else (
        collections.defaultdict(lambda: [0, 0]), {}, collections.Counter())

    if not rows["ALL"][1]:
        print("❌ pred 文件里没有任何能对上 id 的记录,不写表"); return 1

    num3, bnum3 = acc(rows["NUM3"]), acc(brows["NUM3"])
    net = num3 - bnum3 if brows["NUM3"][1] else float("nan")
    n_done, n_total = rows["ALL"][1], 7922

    # ---------- 合理性检查 ----------
    flags = []
    if num3 == num3:
        if num3 > LEAK_ALARM:
            flags.append(f"🔴 数值三族 {num3:.1f}% > {LEAK_ALARM}%,**疑似泄漏**,先查 prompt/抽帧再报数")
        elif num3 > SANITY_HI:
            flags.append(f"⚠️ 数值三族 {num3:.1f}% 高于预期区间 {SANITY_LO}–{SANITY_HI}%,需复核")
        elif num3 < SANITY_LO - 3:
            flags.append(f"⚠️ 数值三族 {num3:.1f}% 明显低于 chance(25%),可能是字母抽取/模板出问题")
    if n_done < n_total:
        flags.append(f"⚠️ 只评了 {n_done}/{n_total} 题(未跑完或有失败条目)")
    bad = letters.get("?", 0)
    if bad > 0.02 * n_done:
        flags.append(f"⚠️ {bad} 条({100*bad/n_done:.1f}%)输出里没抽到 A–D")
    if a.blind and not brows["NUM3"][1]:
        flags.append("⚠️ 盲测臂为空,净增无法计算")

    status = "OK" if not flags else ("🔴 存疑" if any(f.startswith("🔴") for f in flags) else "⚠️")

    # ---------- §1 主表 ----------
    r1 = (f"| {a.model} | {a.weights} | {a.config} | **{fmt(num3)}** | {fmt(bnum3)} | "
          f"{fmt(net, 1) if net == net else '—'} | {fmt(acc(rows['NUM3:heldout_test']))} | {status} |")
    # ---------- §2 逐族 ----------
    cells = [f"{fmt(acc(rows[f'fam:{f}']))} <sub>n={rows[f'fam:{f}'][1]}</sub>"
             if rows[f"fam:{f}"][1] else "—" for f in FAM_ORDER]
    r2 = f"| {a.model} | " + " | ".join(cells) + " |"
    # ---------- §3 逐族耗时 ----------
    def tcell(key):
        ts = times.get(key) or []
        return f"{sum(ts)/len(ts):.3f}s" if ts else "—"
    tail4 = [t for f in FAM_ORDER[3:] for t in (times.get(f) or [])]
    wall = f"{a.wall/60:.1f} min" if a.wall else "—"
    itps = f"{n_done/a.wall:.2f}" if a.wall else "—"
    r3 = (f"| {a.model} | {wall} | {itps} | {tcell('cam_disp_grid')} | {tcell('disp_grid')} | "
          f"{tcell('speed_grid')} | "
          f"{f'{sum(tail4)/len(tail4):.3f}s' if tail4 else '—'} |")

    # ---------- 运行日志 ----------
    log = [f"\n### {a.model} — {a.config}" + (f" — {a.engine}" if a.engine else ""),
           f"- pred: `{os.path.relpath(a.pred, PKG)}`"
           + (f" / blind: `{os.path.relpath(a.blind, PKG)}`" if a.blind else " / **无盲测臂**"),
           f"- 评了 {n_done}/{n_total} 题;字母分布 {dict(letters)}"
           + (f";盲测字母分布 {dict(bletters)}" if a.blind else ""),
           f"- 数值三族合并(7,500 题口径,实评 {rows['NUM3'][1]}):"
           f"**看帧 {fmt(num3)} ±{fmt(se(rows['NUM3']),2)}** / "
           f"盲测 {fmt(bnum3)} / 净增 {fmt(net)}",
           f"  - heldout_test: 看帧 {fmt(acc(rows['NUM3:heldout_test']))} "
           f"(n={rows['NUM3:heldout_test'][1]}) / 盲 {fmt(acc(brows['NUM3:heldout_test']))}",
           f"  - test:         看帧 {fmt(acc(rows['NUM3:test']))} "
           f"(n={rows['NUM3:test'][1]}) / 盲 {fmt(acc(brows['NUM3:test']))}"]
    if a.note:
        log.append(f"- 备注:{a.note}")
    for f in flags:
        log.append(f"- {f}")
    log.append("")

    if a.dry_run:
        print("\n".join([r1, r2, r3] + log)); return 0

    t = open(a.results).read()
    t = upsert_row(t, "## 1. 主表", a.model, r1)
    t = upsert_row(t, "## 2. 逐族 Acc", a.model, r2)
    t = upsert_row(t, "## 3. 逐族推理时间", a.model, r3)
    t = t.rstrip("\n") + "\n" + "\n".join(log)
    open(a.results, "w").write(t)

    print(f"✅ 已写入 {a.results}")
    print("\n".join([r1, "", *log]))
    for f in flags:
        print(f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
