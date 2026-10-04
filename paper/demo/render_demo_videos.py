"""Demo 网页用的视频渲染:图 4 / 图 4b 用的 9 个 clip,把数据管线的结果逐帧画到整段视频上
(SAM 3 掩码轮廓 + 降采样关键点的世界系轨迹,用 GT 相机逐帧投影;彩虹色 = 时间,紫 = 开始,红 = 结束)。

绘制函数直接复用 paper/figure_scripts/render_figs.py(与论文图 4 的帧图同一套代码、同一份数据),运行时逐像素核对:
本脚本画出的第 t 帧 == 论文图 4 用的 <clip>/<prefix>_fXX.png(t 为图里选的帧)。不一致就报错退出。

输出(paper/demo/):
  videos/<key>_tiles.mp4   2×2 拼图:左上 输入帧 | 右上 掩码 | 左下 轨迹 | 右下 掩码+轨迹(网页用,一个文件四个图层天然同步)
  videos/<key>_both.mp4    掩码+轨迹(512×512,给剪辑软件)
  videos/<key>_traj.mp4    只画轨迹
  videos/<key>_masks.mp4   只画掩码
  videos/<key>_input.mp4   原始输入帧
  videos/<key>_sbs.mp4     左右并排:输入 | 掩码+轨迹(1024×512)
  posters/<key>.jpg        末帧(完整轨迹)
  data.js                  网页用的数据:物体清单 / 配色 / T8 caption / TrajQA-B 题目 / 两个模型的原样输出
帧率 = 该 clip 的 fps_effective(stereo4d_meta.json),即真实时间轴;多数 clip 5 fps。
用法:<stereo4d 或 sam3d-objects python> paper/demo/render_demo_videos.py [--keys k1,k2] [--tmp DIR]
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import shutil
import subprocess
import sys

import cv2
import numpy as np

R = str(__import__("pathlib").Path(__file__).resolve().parents[2])          # repo root(本文件在 paper/demo/)
S = R
M = R + "/data"
FIG = R + "/paper/figure_assets"
QA = M + "/trajqa_b/final"
PREDS = R + "/baselines/preds"
OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, R + "/paper/figure_scripts")   # render_figs.py
from render_figs import FR, OK, PAL, RERUN, TOK, draw, match_masks, obj_points, project  # noqa: E402

# 与 paper/figure_scripts/render_fig4.py、render_fig4b.py 里的 EX 一致(顺序 = 图 4 四例 + 图 4b 五例)
EX = [
    dict(key="complex_interaction", name="Complex interaction", clip="sbcUC3xaXfE_475542209", fig="fig4",
         prefix="fig4_row1_complex_interaction", qid="scene_sbcUC3xaXfE_475542209_disp_grid_5df62b6a", split="heldout_test"),
    dict(key="multiple_people", name="Multiple people", clip="2ekr1IfOn5c_101835169", fig="fig4",
         prefix="fig4_row3_multiple_people", qid="scene_2ekr1IfOn5c_101835169_speed_grid_75a03fb6", split="test"),
    dict(key="swapping_positions", name="Objects swapping positions", clip="Kh42LoZOS2Y_102502503", fig="fig4",
         prefix="fig4_row4_swapping_positions", qid="scene_Kh42LoZOS2Y_102502503_speed_grid_dda38c97", split="test"),
    dict(key="complex_video", name="Complex video", clip="yRsUHBgOt8k_61795128", fig="fig4",
         prefix="fig4_row5_complex_video", qid="scene_yRsUHBgOt8k_61795128_speed_grid_7dfb88c4", split="test"),
    dict(key="distant_animal", name="Distant animal", clip="N6EH3Lk0eRo_255522189", fig="fig4b",
         prefix="fig4b_N6EH3Lk0eRo_255522189", qid="scene_N6EH3Lk0eRo_255522189_speed_grid_f27c768c", split="test"),
    dict(key="crossing_truck", name="Crossing truck", clip="WWdorMyLg3M_426693360", fig="fig4b",
         prefix="fig4b_WWdorMyLg3M_426693360", qid="scene_WWdorMyLg3M_426693360_disp_grid_1b56523e", split="test"),
    dict(key="camera_motion", name="Camera motion", clip="7uTgv66Q29U_36169503", fig="fig4b",
         prefix="fig4b_7uTgv66Q29U_36169503", qid="scene_7uTgv66Q29U_36169503_cam_disp_grid_84fb6571", split="test"),
    dict(key="carried_object", name="Carried object", clip="uDWH1bRUvYQ_217984651", fig="fig4b",
         prefix="fig4b_uDWH1bRUvYQ_217984651", qid="scene_uDWH1bRUvYQ_217984651_disp_grid_fddedbbd", split="heldout_test"),
    dict(key="approaching_person", name="Approaching person", clip="q-B2xu8B3Js_29963297", fig="fig4b",
         prefix="fig4b_q-B2xu8B3Js_29963297", qid="scene_q-B2xu8B3Js_29963297_speed_grid_c8681538", split="heldout_test"),
]
MODELS = [("rgb", "Ours_RGB-arm", "Qwen3-VL-8B · frames only (same SFT data)"),
          ("ours", "Ours_TrajRGB-final", "Ours · frames + trajectory tokens")]
VARIANTS = ["tiles", "both", "traj", "masks", "input", "sbs"]


def hex_of(bgr):
    b, g, r = bgr
    return f"#{r:02X}{g:02X}{b:02X}"


def load_clip(e):
    """与 render_figs.render() 的非 fig1 分支完全一致地构造 objs(顺序 / 配色 / 掩码匹配都相同)。"""
    clip = e["clip"]
    sd_rerun, sd = f"{RERUN}/scene_{clip}", f"{OK}/scene_{clip}"
    vi = pickle.load(open(f"{sd_rerun}/video_input.pkl", "rb"))
    fr = pickle.load(open(f"{sd}/final_repr.pkl", "rb"))
    g = np.load(f"{sd}/gt_stereo4d.npz", allow_pickle=True)
    E = np.asarray(g["extrs_w2c"], float); K = np.asarray(g["intrinsics_K"], float)
    ts = np.asarray(g["timestamps_us"], float) / 1e6
    meta = json.load(open(f"{sd}/stereo4d_meta.json"))
    fps = float(meta["fps_effective"])
    assert abs((len(ts) - 1) / (ts[-1] - ts[0]) / fps - 1) < 0.1, (clip, "fps 与时间戳不一致")
    sc = pickle.load(open(f"{TOK}/scene_{clip}.pkl", "rb"))
    frames = [cv2.imread(f"{FR}/{clip}/{i:05d}.jpg") for i in range(E.shape[0])]
    assert all(f is not None for f in frames), clip
    T = len(frames); H, W = frames[0].shape[:2]
    assert vi.masklets and vi.masklets[0].masks.shape == (T, H, W), (clip, vi.masklets[0].masks.shape, T, H, W)
    moving = {o.instance_id for o in sc.objects if o.kind in ("nonrigid", "rigid") and o.instance_id != 1_000_000}
    iids = [int(o["instance_id"]) for o in fr]
    mm = match_masks(vi, sd, iids, T, H, W)
    objs, report = [], []
    for o in fr:
        iid = int(o["instance_id"])
        if iid not in moving:
            continue
        P, V = obj_points(o)
        if P is None:
            continue
        uv, okp = project(P, E, K, W, H)
        m = mm.get(iid)
        objs.append(dict(iid=iid, label=str(o.get("label")), P=P, V=V, uv=uv, ok=okp & V,
                         mask=(m[0].masks if m else None), color=None))
        report.append(dict(iid=iid, label=str(o.get("label")), kind=("rigid" if "se3_R" in o else "nonrigid"),
                           n_keypoints=int(P.shape[0]), mask_matched=bool(m), mask_overlap=(round(m[1], 3) if m else None)))
    for i, o in enumerate(objs):
        o["color"] = PAL[i % len(PAL)]
        report[i]["color"] = hex_of(o["color"])
    return dict(frames=frames, objs=objs, report=report, T=T, fps=fps, H=H, W=W, n_masklets=len(vi.masklets))


def paper_check(e, d):
    """本脚本的帧 == 论文图 4 帧图(逐像素)。"""
    summ_p = f"{FIG}/render_summary.json" if e["fig"] == "fig4" else f"{FIG}/render_summary_extra.json"
    s = {r["clip"]: r for r in json.load(open(summ_p))}[e["clip"]]
    assert s["prefix"] == e["prefix"], (e["key"], s["prefix"])
    assert [o["iid"] for o in s["objects"] if "iid" in o] == [o["iid"] for o in d["report"]], (e["key"], "物体清单顺序不同")
    assert s["T"] == d["T"]
    for t in s["frames"]:
        ref = cv2.imread(f"{FIG}/{e['clip']}/{e['prefix']}_f{t:02d}.png")
        mine = draw(d["frames"][t], t, d["T"], d["objs"])
        assert ref is not None and ref.shape == mine.shape, (e["key"], t)
        diff = np.abs(ref.astype(int) - mine.astype(int))
        assert diff.max() == 0, (e["key"], t, "与论文帧图不一致", int(diff.max()), float(diff.mean()))
    return list(s["frames"])


def encode(png_dir, out, fps):
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", f"{fps:g}", "-i", f"{png_dir}/%05d.png",
           "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p", "-g", "250",
           "-movflags", "+faststart", out]
    subprocess.run(cmd, check=True)
    probe = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                            "stream=nb_read_frames,r_frame_rate,width,height", "-of", "csv=p=0", out],
                           check=True, capture_output=True, text=True).stdout.strip().split(",")
    return probe


def render_videos(e, d, tmp):
    T, fps = d["T"], d["fps"]
    dirs = {v: f"{tmp}/{e['key']}/{v}" for v in VARIANTS}
    for p in dirs.values():
        shutil.rmtree(p, ignore_errors=True); os.makedirs(p)
    last_both = None
    for t in range(T):
        raw = d["frames"][t]
        masks = draw(raw, t, T, d["objs"], show_traj=False)
        traj = draw(raw, t, T, d["objs"], show_mask=False)
        both = draw(raw, t, T, d["objs"])
        tiles = np.concatenate([np.concatenate([raw, masks], 1), np.concatenate([traj, both], 1)], 0)
        sbs = np.concatenate([raw, both], 1)
        for v, im in (("tiles", tiles), ("both", both), ("traj", traj), ("masks", masks), ("input", raw), ("sbs", sbs)):
            cv2.imwrite(f"{dirs[v]}/{t:05d}.png", im)
        last_both = both
    cv2.imwrite(f"{OUT}/posters/{e['key']}.jpg", last_both, [cv2.IMWRITE_JPEG_QUALITY, 92])
    info = {}
    for v in VARIANTS:
        out = f"{OUT}/videos/{e['key']}_{v}.mp4"
        w, h, rate, n = encode(dirs[v], out, fps)
        num, den = rate.split("/")
        assert int(n) == T and abs(int(num) / int(den) - fps) < 1e-6, (e["key"], v, n, rate)
        info[v] = dict(path=f"videos/{e['key']}_{v}.mp4", w=int(w), h=int(h), frames=int(n), bytes=os.path.getsize(out))
    return info


def load_qa_and_preds():
    want_q = {e["qid"] for e in EX}
    want_v = {"scene_" + e["clip"] for e in EX}
    t8 = {}
    for ln in open(f"{M}/captions/t8_train.jsonl"):
        if "scene_" in ln:
            r = json.loads(ln)
            if r["video_id"] in want_v:
                t8.setdefault(r["video_id"], []).append(r["answer"])
    qrec = {}
    for sp in ("test", "heldout_test", "val", "train"):
        for ln in open(f"{QA}/{sp}.jsonl"):
            r = json.loads(ln)
            if r["id"] in want_q:
                r["_split"] = sp; qrec[r["id"]] = r
    preds = {}
    for key, fn, _ in MODELS:
        meta = json.load(open(f"{PREDS}/{fn}.meta.json"))
        assert meta["base_model"] == "Qwen3-VL-8B-Instruct" and "NOT zero-shot" in meta["note"], fn
        preds[key] = {json.loads(ln)["id"]: json.loads(ln) for ln in open(f"{PREDS}/{fn}.jsonl")}
    assert "pB_rgbsw_9002_s0/final" in json.load(open(f"{PREDS}/Ours_RGB-arm.meta.json"))["checkpoint"]
    assert json.load(open(f"{PREDS}/Ours_TrajRGB-final.meta.json"))["checkpoint"] == "runs_sft110k/pB_s2_cos1ep_s0/final"
    preds["qwen_zs"] = {json.loads(ln)["id"]: json.loads(ln) for ln in open(f"{PREDS}/Qwen3-VL-8B.jsonl")}
    return t8, qrec, preds


def parse_output(raw):
    lines = raw.split("\n")
    assert lines[0] == "<evidence>" and lines[-2] == "</evidence>", raw
    m = re.fullmatch(r"<answer>([A-D])</answer>", lines[-1])
    assert m, raw
    return lines, m.group(1)


def clip_record(e, d, t8, qrec, preds, sel_frames, vinfo):
    v = "scene_" + e["clip"]
    caps = t8[v]
    caps = [x for x in caps if x.startswith("Moving in the world frame")] + [x for x in caps if not x.startswith("Moving in the world frame")]
    assert caps and caps[0].startswith("Moving in the world frame"), e["key"]
    q = qrec[e["qid"]]
    assert q["_split"] == e["split"] and q["_split"] in ("test", "heldout_test"), (e["key"], q["_split"])
    gp = q["gt_payload"]
    assert q["frame_range"] == [0, gp["T"] - 1] and gp["T"] == d["T"], (e["key"], q["frame_range"], gp["T"], d["T"])
    by_iid = {o["iid"]: o for o in d["report"]}
    if q["event_subtype"] == "cam_disp_grid":
        assert q["involved_inst_ids"] == [] and "the camera move" in q["question"], e["key"]
        obj = None
    else:
        assert len(q["involved_inst_ids"]) == 1 and q["involved_inst_ids"][0] in by_iid, (e["key"], q["involved_inst_ids"])
        obj = by_iid[q["involved_inst_ids"][0]]
        assert gp["obj"] == "the " + obj["label"] and gp["obj"] in q["question"], (e["key"], gp["obj"], obj["label"])
    unit = gp["unit"]
    assert unit in ("m", "m/s") and (unit == "m/s") == (q["event_subtype"] == "speed_grid")
    keyline = "speed_mps:" if unit == "m/s" else ("camera_displacement_m:" if obj is None else "displacement_m:")
    outs = {}
    for key, fn, head in MODELS:
        p = preds[key][e["qid"]]
        lines, letter = parse_output(p["raw"])
        assert letter == p["pred"], (e["key"], key)
        assert sum(ln.startswith(keyline) for ln in lines) == 1
        val = float(next(ln for ln in lines if ln.startswith(keyline)).split(":")[1])
        outs[key] = dict(head=head, lines=lines, letter=letter, correct=(letter == q["answer"]), value=val)
    zs = preds["qwen_zs"][e["qid"]]
    outs["qwen_zs"] = dict(head="Qwen3-VL-8B · zero-shot", letter=zs["pred"], correct=(zs["pred"] == q["answer"]))
    return dict(
        key=e["key"], name=e["name"], clip=e["clip"], fig=e["fig"], split=q["_split"], T=d["T"], fps=d["fps"],
        duration_s=round((d["T"] - 1) / d["fps"], 3), paper_frames=sel_frames, n_masklets_rerun=d["n_masklets"],
        objects=d["report"], captions=caps,
        question=dict(id=q["id"], text=q["question"], options=q["options"], answer=q["answer"], subtype=q["event_subtype"],
                      gt_value=gp["value"], gt_unit=unit, i=gp["i"], j=gp["j"], keyline=keyline,
                      obj=(gp.get("obj") if obj else None), obj_iid=(obj["iid"] if obj else None),
                      obj_color=(obj["color"] if obj else None)),
        outputs=outs, videos=vinfo, poster=f"posters/{e['key']}.jpg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", default="")
    ap.add_argument("--tmp", default=os.environ.get("DEMO_TMP", "/tmp/demo_web_frames"))
    a = ap.parse_args()
    keys = set(a.keys.split(",")) if a.keys else None
    t8, qrec, preds = load_qa_and_preds()
    data_p = f"{OUT}/data.js"
    old = {}
    if os.path.exists(data_p):
        txt = open(data_p).read()
        old = {c["key"]: c for c in json.loads(txt[txt.index("=") + 1:].rstrip().rstrip(";"))["clips"]}
    recs = []
    for e in EX:
        if keys and e["key"] not in keys:
            assert e["key"] in old, (e["key"], "未渲染过又没被选中")
            recs.append(old[e["key"]]); continue
        d = load_clip(e)
        sel = paper_check(e, d)
        vinfo = render_videos(e, d, a.tmp)
        r = clip_record(e, d, t8, qrec, preds, sel, vinfo)
        recs.append(r)
        print(f"[{e['key']}] {e['clip']} T={d['T']} fps={d['fps']:g} objects={len(d['report'])} "
              f"with_mask={sum(o['mask_matched'] for o in d['report'])} paper_frames_identical={sel} "
              f"tiles={vinfo['tiles']['bytes'] / 1e6:.2f}MB both={vinfo['both']['bytes'] / 1e6:.2f}MB "
              f"| frames-only {r['outputs']['rgb']['letter']} / ours {r['outputs']['ours']['letter']} / key {r['question']['answer']}",
              flush=True)
    payload = dict(generated="2026-09-26", models={k: h for k, _, h in MODELS}, rainbow="violet = start, red = end",
                   clips=recs)
    with open(data_p, "w") as f:
        f.write("window.DEMO_DATA = " + json.dumps(payload, ensure_ascii=False, indent=1) + ";\n")
    tot = sum(c["videos"]["tiles"]["bytes"] for c in recs)
    print(f"wrote {data_p}; {len(recs)} clips; tiles total {tot / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
