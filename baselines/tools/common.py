"""TrajQA-B baseline 协议公共件。

与我们自己的评测 (stage0_4/src/training/g1/{data,eval}.py) **逐字一致**,
任何改动都会让 baseline 与我方数字不可比。三处必须一致:
  1) 抽帧    = np.linspace(0, T-1, k).round() 取整,整段 clip(B 集所有题 frame_range=[0,T-1])
  2) 缩放    = max_side 等比缩放(原始帧已是 512x512;我方 RGB 臂用 384)
  3) prompt  = build_prompt_text() 的 MCQ 分支,逐字
视觉侧默认 **不给时间信息**(video_mode="images"),与我方 RGB 臂相同。
"""
from __future__ import annotations
import glob, json, os, re
import numpy as np
from PIL import Image

LETTERS = ("A", "B", "C", "D")
NUMERIC3 = ("cam_disp_grid", "disp_grid", "speed_grid")   # 论文主指标:数值三族合并

PLAIN_TAIL = "\nAnswer with the single letter of the correct option."
COT_TAIL = ("\nThink step by step about the metric (world-frame, in meters) quantity being asked, "
            "then end your reply with 'Answer: X' where X is the option letter.")


def frames_dir(pkg_root: str, video_id: str) -> str:
    clip = video_id[len("scene_"):] if video_id.startswith("scene_") else video_id
    return os.path.join(pkg_root, "frames", clip)


def sample_frames(pkg_root: str, video_id: str, k: int = 8, max_side: int = 384):
    """整段 clip 均匀抽 k 帧 -> List[PIL.Image]。与 g1/data.py:sample_frames 同算法。"""
    fs = sorted(glob.glob(os.path.join(frames_dir(pkg_root, video_id), "*.jpg")))
    if not fs:
        raise FileNotFoundError(f"no frames for {video_id}")
    idx = (list(range(len(fs))) if len(fs) <= k
           else np.linspace(0, len(fs) - 1, k).round().astype(int).tolist())
    out = []
    for i in idx:
        im = Image.open(fs[i]).convert("RGB")
        w, h = im.size
        if max_side and max(w, h) > max_side:
            s = max_side / max(w, h)
            im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
        out.append(im)
    return out, idx


def build_prompt(item, style: str = "plain") -> str:
    """与 g1/data.py:build_prompt_text(item, arm='rgb') 的 MCQ 分支逐字一致(style='plain')。"""
    lines = [f"Question: {item['question']}", "Options:"]
    for L in LETTERS:
        if L in item["options"]:
            lines.append(f"{L}. {item['options'][L]}")
    tail = PLAIN_TAIL if style == "plain" else COT_TAIL
    return "\n".join(lines) + tail


_ANS_RE = re.compile(r"[Aa]nswer\s*[:：]?\s*\(?\s*([A-Da-d])\b")


def extract_letter(txt: str, style: str = "plain") -> str:
    """抽选项字母。plain: 取首个出现的字母(与 g1/eval.py 一致);cot: 先找 'Answer: X',
    找不到退回**最后**一个出现的字母(CoT 里前面的字母多是在复述选项)。"""
    if style != "plain":
        m = None
        for m in _ANS_RE.finditer(txt):
            pass
        if m:
            return m.group(1).upper()
        return next((c for c in txt.strip()[::-1].upper() if c in LETTERS), "?")
    return next((c for c in txt.strip().upper() if c in LETTERS), "?")


def load_split(pkg_root: str, splits, limit=None):
    """splits: 'heldout_test' / 'test' / 'val' / 'train' / 'all' 的列表或逗号串。"""
    if isinstance(splits, str):
        splits = [s.strip() for s in splits.split(",")]
    want = None if "all" in splits else set(splits)
    out = []
    with open(os.path.join(pkg_root, "qa", "questions.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            if want is None or d["split"] in want:
                out.append(d)
                if limit and len(out) >= limit:
                    break
    return out


def load_answers(pkg_root: str):
    with open(os.path.join(pkg_root, "qa", "answers.jsonl")) as f:
        return {d["id"]: d for d in map(json.loads, f)}
