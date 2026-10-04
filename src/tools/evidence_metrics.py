"""结构化评测的 evidence 指标(2026-09-15):从 g1.eval --structured 的 *.items.jsonl 计算
  逐族 MCQ、数值三族合并 MCQ、生成数值 |log(pred/GT)| 中位与 ±20% 内比例、位置误差中位(物体/相机)。
用法:python src/tools/evidence_metrics.py <items.jsonl> [--json out.json] [--tag 名称]"""
import argparse, json, re, collections
import numpy as np
KEY = {"disp_grid": "displacement_m", "speed_grid": "speed_mps", "cam_disp_grid": "camera_displacement_m"}
NUM = r"-?\d+(?:\.\d+)?"          # 严格数字(生成文本里会出现 "-0..0" 这类坏数,09-15 实测;不匹配即视为未解析)
RX_POS = re.compile(r"(position_frame_\d+_m|camera_frame_\d+_m):\s*\((" + NUM + r"),\s*(" + NUM + r"),\s*(" + NUM + r")\)")


def metrics(path):
    R = [json.loads(l) for l in open(path)]
    fam = collections.defaultdict(list)
    for r in R: fam[r["event_subtype"]].append(r["ok"])
    out = dict(n=len(R), acc_all=round(float(np.mean([r["ok"] for r in R])), 4),
               acc=dict((k, [len(v), round(float(np.mean(v)), 4)]) for k, v in sorted(fam.items())))
    num = [r["ok"] for r in R if r["event_subtype"] in KEY]
    out["acc_numeric_pooled"] = [len(num), round(float(np.mean(num)), 4)] if num else None
    for f, key in KEY.items():
        d = []
        for r in R:
            if r["event_subtype"] != f: continue
            mg = re.search(key + r":\s*(" + NUM + r")\b", r.get("evidence_gt") or ""); mp = re.search(key + r":\s*(" + NUM + r")(?![\d.])", r.get("gen") or "")
            if mg and mp and float(mg.group(1)) > 0 and float(mp.group(1)) > 0:
                d.append(abs(np.log(float(mp.group(1)) / float(mg.group(1)))))
        if d: out[f"{f}_abslogratio_med"] = round(float(np.median(d)), 4); out[f"{f}_within20"] = round(float(np.mean(np.array(d) < np.log(1.2))), 4); out[f"{f}_parsed"] = len(d)
    pe = collections.defaultdict(list)
    for r in R:
        if r["event_subtype"] not in KEY: continue
        G = {m.group(1): np.array([float(m.group(k)) for k in (2, 3, 4)]) for m in RX_POS.finditer(r.get("evidence_gt") or "")}
        P = {m.group(1): np.array([float(m.group(k)) for k in (2, 3, 4)]) for m in RX_POS.finditer(r.get("gen") or "")}
        for k in G:
            if k in P: pe["cam" if k.startswith("camera") else "obj"].append(float(np.linalg.norm(P[k] - G[k])))
    for k, v in pe.items(): out[f"pos_err_med_{k}"] = round(float(np.median(v)), 4); out[f"pos_n_{k}"] = len(v)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("items"); ap.add_argument("--json", default=""); ap.add_argument("--tag", default="")
    a = ap.parse_args(); m = metrics(a.items); m["tag"] = a.tag or a.items
    if a.json:
        with open(a.json, "a") as f: f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(json.dumps(m, ensure_ascii=False))
