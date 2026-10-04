#!/usr/bin/env python3
"""把表格图(make_table.py)里的全部数字导出成 CSV(可选再出一份 Markdown 表)—— 直接复用 make_table 的
模型清单、compute() 和取整函数 f(),与 PNG 同源。

    python tools/export_table_csv.py --out results_table.csv [--md results_table.md]

CSV 保留 4 位小数(百分数);Markdown 与图一样由全精度直接取 1 位小数 ——
别拿 CSV 先截到 2 位再取 1 位,那样会有几格与图差 0.1(两次取整)。
每行一个模型(MLLM-4D 另有两行子集行,我方两个微调臂作参考行)。
n_items / n_numeric3 是实际参与计算的题数:两个闭源 API 模型只跑了子集(Astra 998、Fable 1,000 题),其余 7,922 题。
"""
import argparse, csv, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_table import GROUP_TITLE, MODELS, NOTES_ITEMS, OURS, PKG, SUBROWS, SUBSET_OK, TIER_FILE, compute, f, load_preds

KEYS = (("numeric3", "num3"), ("heldout", "held"), ("test", "test"), ("cam_disp", "cam"),
        ("obj_disp", "disp"), ("speed", "speed"), ("non_num", "back4"))
COLS = ["group", "model", "row_type", "pred_file", "protocol", "weights_GiB", "n_items", "n_numeric3"] + [k for k, _ in KEYS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", default=PKG)
    ap.add_argument("--out", default=os.path.join(PKG, "results_table.csv"))
    ap.add_argument("--md", default=None, help="另存一份 Markdown 表(1 位小数,与图一致)")
    ap.add_argument("--notes", default=None, help="把表格脚注(2026-09-25 起不再画进 PNG)写成一个 md 备用")
    a = ap.parse_args()
    gold = {d["id"]: d for d in map(json.loads, open(os.path.join(a.pkg, "qa", "answers.jsonl")))}
    rows = []

    def add(grp, disp, rtype, base, w, st):
        ok = st and "partial" not in st
        rows.append({"group": GROUP_TITLE[grp], "model": disp, "row_type": rtype,
                     "pred_file": f"preds/{base}.jsonl" if base else "",
                     "protocol": ("OURS" if grp == "ours" else ("Q1" if base.startswith("Q1_") else "Q2")) if base else "",
                     "weights_GiB": w,
                     "n_items": len(load_preds(os.path.join(a.pkg, "preds", f"{base}.jsonl"))) if base and rtype == "model" else "",
                     "n_numeric3": st["n_num3"] if ok else "",
                     **{k: (st.get(src) if ok else None) for k, src in KEYS}})

    tier = json.load(open(TIER_FILE))["tier"] if os.path.exists(TIER_FILE) else {}
    qclip = {}
    for q in map(json.loads, open(os.path.join(a.pkg, "qa", "questions.jsonl"))):
        v = q["video_id"]
        qclip[q["id"]] = v[len("scene_"):] if v.startswith("scene_") else v
    for slug, disp, w, grp in MODELS:                 # 与 make_table.main() 同一套取文件规则
        base = next((b for b in (f"Q1_{slug}", slug)
                     if os.path.exists(os.path.join(a.pkg, "preds", f"{b}.jsonl"))), None)
        st = compute(base, gold, a.pkg, full=grp not in SUBSET_OK) if base else None
        add(grp, disp, "model", base, w, st)
        if slug in SUBROWS and st and "partial" not in st and tier:
            for t, label in SUBROWS[slug]:
                keep = {qid for qid, c in qclip.items() if tier.get(c) == t}
                add(grp, label.strip(), "subrow", base, "", compute(base, gold, a.pkg, keep=keep))
    for nm, w, sg, _ in OURS:                          # 用户给的锚点:heldout_test 数值三族
        rows.append({"group": GROUP_TITLE["ours"], "model": nm, "row_type": "reference", "pred_file": "",
                     "protocol": "", "weights_GiB": w, "n_items": "", "n_numeric3": "",
                     **{k: None for k, _ in KEYS}, "numeric3": sg, "heldout": sg})

    num = lambda x: "" if x is None or x != x else f"{x:.4f}"
    with open(a.out, "w", newline="") as fo:
        wr = csv.DictWriter(fo, fieldnames=COLS)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (num(r[k]) if k in dict(KEYS) else r[k]) for k in COLS})
    print(f"✅ {a.out}  ({len(rows)} 行)")

    if a.md:
        md = ["| 组 | 模型 | 口径 | 题数 | **数值三族** | Heldout | Test | Cam.Disp | Obj.Disp | Speed | Non-num. |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            name = r["model"] if r["row_type"] != "subrow" else "　└ " + r["model"]
            n = r["n_items"] or (f"数值 {r['n_numeric3']:,}" if r["n_numeric3"] else "heldout")
            n = f"{n:,}" if isinstance(n, int) else n
            v = [f(r[k]) for k, _ in KEYS]
            md.append(f"| {r['group'].split('  (')[0]} | {name} | {r['protocol'] or '—'} | {n} | **{v[0]}** | "
                      + " | ".join(v[1:]) + " |")
        open(a.md, "w").write("\n".join(md) + "\n")
        print(f"✅ {a.md}")
    if a.notes:
        open(a.notes, "w").write("# Notes for figure/trajqa_b_baseline.png (moved out of the figure on 2026-09-25)\n\n"
                                 + "\n\n".join("- " + " ".join(x.strip() for x in it.split("\n")) for it in NOTES_ITEMS) + "\n")
        print(f"✅ {a.notes}")


if __name__ == "__main__":
    sys.exit(main())
