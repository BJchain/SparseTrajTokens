"""G1 数据管线:QA jsonl (+ frames + tokens) → Qwen3-VL 批输入 + 轨迹批。

两臂共用同一 QA / 帧:
- arm="rgb"   : 只喂帧 + 问题(轨迹 token 通道关闭)。
- arm="traj"  : 帧 + 问题 + "Scene trajectory: <traj>*N"(N=该 scene 的轨迹 token 数)。

对齐关系(已验证):
  video_id="scene_<clip>"  → Stereo4D:帧 data/stereo4d/frames/<clip>/%05d.jpg,
                             token data/tokens_v4td/<video_id>.pkl
  video_id="val_XXXXX"     → MOVi:帧 data/movi_frames/<video_id>/%05d.jpg,
                             token outputs/movi_gt/scene_<video_id>/scene_tokens.pkl

短答案 SFT:target = 正确选项字母;loss 只算答案 token(prompt 全 -100)。
"""
from __future__ import annotations

from functools import lru_cache
import glob
import json
import os
import pickle
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .traj_serialize import (serialize_scene, serialize_scene_labeled, records_to_arrays,
                             group_spans, ROLE2ID)
from .modeling_traj import TRAJ_TOKEN

# 相对本仓库根(stage0_4)运行
# 目录可用环境变量覆盖(v4 批次/110k 换 token 目录不改代码;默认 = G1 原路径)
STEREO_FRAMES = os.environ.get("G1_STEREO_FRAMES", "data/stereo4d/frames")
# 2026-08-22 默认切到最新批:v4 三次 B 样条 + 绝对时间轴(506 场景)。
# 旧的 milestone0/stereo4d_tokens 是 08-02 的 895 场景批,早于 08-08 坐标系契约变更与
# 08-16 final_repr 字段修复 —— **已 stale**,只作对照臂用(设 G1_STEREO_TOKENS 切回)。
STEREO_TOKENS = os.environ.get("G1_STEREO_TOKENS", "data/tokens_v4td")
MOVI_FRAMES = "data/movi_frames"
MOVI_TOKENS = os.environ.get("G1_MOVI_TOKENS", "outputs/movi_gt_v4t")

LETTERS = ("A", "B", "C", "D")
_VERT_ROLES = {ROLE2ID["ROT_VERT"], ROLE2ID["TRANS_VERT"], ROLE2ID["PL_VERT"], ROLE2ID["STATIC_POINT"]}


# ----------------------------------------------------------------- 路径解析
def frames_dir_for(video_id: str) -> str:
    if video_id.startswith("val_"):
        return os.path.join(MOVI_FRAMES, video_id)
    clip = video_id[len("scene_"):] if video_id.startswith("scene_") else video_id
    return os.path.join(STEREO_FRAMES, clip)


def token_path_for(video_id: str) -> str:
    if video_id.startswith("val_"):
        return os.path.join(MOVI_TOKENS, f"scene_{video_id}", "scene_tokens.pkl")
    return os.path.join(STEREO_TOKENS, f"{video_id}.pkl")


def sample_frames(video_id: str, k: int, max_side: int = 384,
                  frame_range=None, return_idx: bool = False):
    """默认返回 List[Image];return_idx=True 时返回 (imgs, 绝对帧号列表)。
    绝对帧号 = 在该 clip 完整帧序列里的下标,用于查绝对秒(与 token 同一时间轴)。"""
    fs = sorted(glob.glob(os.path.join(frames_dir_for(video_id), "*.jpg")))
    if not fs:
        raise FileNotFoundError(f"no frames for {video_id} at {frames_dir_for(video_id)}")
    # 区间式问句:RGB 只从 [t0,t1] 窗口采样,与轨迹读同一区间,保证公平。
    lo, hi = 0, len(fs) - 1
    if frame_range is not None:
        lo = max(0, int(frame_range[0]))
        hi = min(len(fs) - 1, int(frame_range[1]))
        if hi <= lo:
            hi = min(len(fs) - 1, lo + 1)
    win = fs[lo:hi + 1] if hi >= lo else fs
    fs = win if win else fs
    if len(fs) <= k:
        idx = list(range(len(fs)))
    else:
        idx = np.linspace(0, len(fs) - 1, k).round().astype(int).tolist()
    out = []
    for i in idx:
        im = Image.open(fs[i]).convert("RGB")
        w, h = im.size
        if max(w, h) > max_side:
            s = max_side / max(w, h)
            im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
        out.append(im)
    return (out, [lo + i for i in idx]) if return_idx else out


@lru_cache(maxsize=8192)
def scene_time_axis(video_id: str):
    """该 clip 的绝对时间轴 (times_s tuple, fps) —— 与轨迹 token 的 tsec 同一个钟。
    取自 SceneTokens.frame_times_s(tokenizer 从 gt npz 的 timestamps_us 算)。
    拿不到返回 (None, None):**不用帧号冒充秒**,调用方据此关掉时间戳而不是编一个。"""
    p = token_path_for(video_id)
    if not os.path.exists(p):
        return None, None
    # 这里**不吞异常**:load 失败通常是 sys.path 缺 src/tokenization(SceneTokens 反序列化不了),
    # 吞掉会把"环境没配好"伪装成"这个 clip 没时间轴",于是时间戳全体静默消失。
    with open(p, "rb") as f:
        sc = pickle.load(f)
    t = getattr(sc, "frame_times_s", None)
    if t is None:
        return None, None
    fps = getattr(sc, "fps_effective", None)
    return tuple(float(x) for x in np.asarray(t).reshape(-1)), (float(fps) if fps else None)


# ----------------------------------------------------------------- 轨迹 arrays(带封顶)
def _cap_records(recs, max_traj: int):
    """轨迹 token 数封顶:保留所有结构/头/参考 token + **事件节点**(knot=reversal/impact,
    v4 角点共位双节点是反转/冲击的载体,不参与下采样),均匀下采样其余顶点类 token。"""
    if len(recs) <= max_traj:
        return recs, 0
    from .traj_serialize import KNOT2ID
    _EVENT_KNOTS = (KNOT2ID["reversal"], KNOT2ID["impact"])
    keep_idx = [i for i, r in enumerate(recs)
                if r.role not in _VERT_ROLES or r.knot in _EVENT_KNOTS]
    vert_idx = [i for i, r in enumerate(recs)
                if r.role in _VERT_ROLES and r.knot not in _EVENT_KNOTS]
    budget = max_traj - len(keep_idx)
    if budget <= 0:
        sel = set(keep_idx[:max_traj])
    else:
        if len(vert_idx) > budget:
            pick = np.linspace(0, len(vert_idx) - 1, budget).round().astype(int)
            vert_idx = [vert_idx[j] for j in sorted(set(pick.tolist()))]
        sel = set(keep_idx) | set(vert_idx)
    capped = [recs[i] for i in sorted(sel)]
    return capped, len(recs) - len(capped)


def traj_arrays_for(video_id: str, max_traj: int, frame_range=None):
    with open(token_path_for(video_id), "rb") as f:
        scene = pickle.load(f)
    recs, labels = serialize_scene_labeled(scene, frame_range=frame_range)
    recs, dropped = _cap_records(recs, max_traj)
    spans = group_spans(recs)                       # [(group_id, count), ...] 按序
    return records_to_arrays(recs), len(recs), dropped, spans, labels


def build_traj_block(spans, labels) -> str:
    """实例接地(+ P0 嵌套部件接地):轨迹 token 按物体分组、物体内再按运动部件分组,
    每物体前缀其 label、每部件前缀 [part k],让问句对物体/部件的指代都能对齐到轨迹。
    格式: "[scene] <..> [the person] [part 1] <..> [part 2] <..> [the ball] <..> ...\""""
    out = []
    cur_g = None
    for gid, pid, cnt in spans:
        if gid < 0:
            out.append("[scene] " + TRAJ_TOKEN * cnt)
            cur_g = gid
            continue
        if gid != cur_g:
            lab = labels[gid] if 0 <= gid < len(labels) else "the object"
            out.append(f"[{lab}]")
            cur_g = gid
        if pid > 0:
            out.append(f"[part {pid}] " + TRAJ_TOKEN * cnt)
        else:
            out.append(TRAJ_TOKEN * cnt)
    return "Scene trajectory tokens (grouped by object and motion part):\n" + " ".join(out) + "\n"


# ----------------------------------------------------------------- Dataset
class G1Dataset(Dataset):
    def __init__(self, jsonl_paths: List[str], sources: Optional[set] = None,
                 limit: Optional[int] = None, require_frames: bool = True,
                 allow_open: bool = False):
        """allow_open=True:接受开放式条目(options 为空、answer 为自由文本;T8 描述,1a 用)。
        默认 False = 只收 MCQ(G1 原行为,逐位不变)。"""
        self.items = []
        # 丢题计数:**不静默**。切换 token 目录时最容易发生"悄悄少了 20% 的题"而无人察觉。
        drop = dict(no_frames=0, no_tokens=0, open_form=0, bad_answer=0)
        n_read = 0
        for jp in jsonl_paths:
            if not os.path.exists(jp):
                print(f"[G1Dataset] WARN: {jp} 不存在,跳过(movi_qa 曾被清理误删;见 memory)", flush=True)
                continue
            with open(jp) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    n_read += 1
                    vid = d["video_id"]
                    if require_frames and not os.path.isdir(frames_dir_for(vid)):
                        drop["no_frames"] += 1
                        continue
                    if not os.path.exists(token_path_for(vid)):
                        drop["no_tokens"] += 1
                        continue
                    ans = d["answer"]
                    is_open = not d.get("options")
                    if is_open and not allow_open:
                        drop["open_form"] += 1
                        continue
                    if not is_open and ans not in LETTERS:
                        drop["bad_answer"] += 1
                        continue
                    self.items.append(d)
        kept = len(self.items)
        print(f"[G1Dataset] 读入 {n_read} 题 → 保留 {kept} ({100*kept/max(1,n_read):.1f}%);"
              f"丢弃 {dict((k, v) for k, v in drop.items() if v)}", flush=True)
        if drop["no_tokens"]:
            print(f"[G1Dataset] 其中 {drop['no_tokens']} 题因 **该场景在 token 目录里没有** 被丢 "
                  f"(token 目录 = {STEREO_TOKENS} / {MOVI_TOKENS})", flush=True)
        if limit:
            self.items = self.items[:limit]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


STRUCT_TAIL = ("\nFirst list the trajectory facts that determine the answer inside <evidence></evidence> "
               "(world-frame metric positions/distances/times), then give the option letter inside <answer></answer>.")


def build_target_text(item, structured: bool = False) -> str:
    """assistant 目标文本。structured=True 且条目带 evidence → 整段 <evidence>…</evidence><answer>L</answer>(全段算 loss)。"""
    if structured and item.get("evidence") and item.get("options"):
        return f"<evidence>\n{item['evidence']}\n</evidence>\n<answer>{item['answer']}</answer>"
    return item["answer"]


def build_prompt_text(item, arm: str, traj_block: str = "", structured: bool = False) -> str:
    opts = item.get("options")
    if opts:                                        # MCQ(原逻辑,逐位不变)
        lines = [f"Question: {item['question']}", "Options:"]
        for L in LETTERS:
            if L in opts:
                lines.append(f"{L}. {opts[L]}")
        body = "\n".join(lines)
        tail = STRUCT_TAIL if (structured and item.get("evidence")) else "\nAnswer with the single letter of the correct option."
    else:                                           # 开放式(T8 描述)
        body = f"Question: {item['question']}"
        tail = "\nAnswer:"
    if arm not in ("traj", "traj_only"):
        traj_block = ""
    return traj_block + body + tail


VIDEO_MODES = ("images", "tstamp", "video")


def build_visual_content(vid, imgs, fidx, video_mode, vmeta, warned=None):
    """按 video_mode 组装 prompt 里的视觉部分;video 模式顺带把 VideoMetadata 追加进 vmeta。

    images = 8 张独立图,没有任何时间信息(现役默认);
    tstamp = 每帧前插一句字面时间戳 "<x.x seconds>",视觉 token 数不变,与轨迹 tsec 同一个钟;
    video  = 走 Qwen3-VL 原生视频通道(processor 生成时间戳)。⚠️帧两两合并,视觉 token 会变少。
    拿不到时间轴时退回纯图并打印一次 —— **不用帧号冒充秒**。"""
    plain = [{"type": "image", "image": im} for im in imgs]
    if not imgs or video_mode == "images":
        return plain
    times, fps = scene_time_axis(vid)
    secs = ([times[i] for i in fidx]
            if (times is not None and fidx and max(fidx) < len(times)) else None)
    if secs is None or (video_mode == "video" and not fps):
        if warned is not None and vid not in warned:
            warned.add(vid)
            if len(warned) <= 3:
                print(f"[video_mode={video_mode}] {vid} 无绝对时间轴,该样本退回纯图"
                      f"(不用帧号冒充秒)", flush=True)
        return plain
    if video_mode == "tstamp":
        out = []
        for sec, im in zip(secs, imgs):
            out.append({"type": "text", "text": f"<{sec:.1f} seconds>"})
            out.append({"type": "image", "image": im})
        return out
    from transformers.video_utils import VideoMetadata
    vmeta.append(VideoMetadata(total_num_frames=len(times), fps=float(fps),
                               duration=float(times[-1]), video_backend="manual",
                               frames_indices=list(fidx)))
    return [{"type": "video", "video": imgs}]


# ----------------------------------------------------------------- Collator
class G1Collator:
    def __init__(self, processor, traj_token_id: int, arm: str,
                 n_frames: int = 8, max_traj: int = 512, max_side: int = 384,
                 rgb_dropout: float = 0.0, seed: int = 0, video_mode: str = "images",
                 structured: bool = False, traj_fn=None):
        self.proc = processor
        self.structured = bool(structured)          # 2026-09-13:结构化中间监督(evidence + answer)
        # 2026-09-20:取轨迹 arrays 的函数可注入(默认 = traj_arrays_for,逐位不变)。
        # 内联验证的 donor 口径(val_diag)传 eval_permute.Corruptor,签名相同。
        self.traj_fn = traj_fn
        self.tid = traj_token_id
        self.arm = arm
        self.n_frames = n_frames
        self.max_traj = max_traj
        self.max_side = max_side
        # modality dropout:训练时按概率把某样本的 RGB 帧丢掉(变成该样本 traj-only),
        # 逼模型学会用轨迹,消除"有 RGB 就忽略轨迹"的 modality dominance。仅 traj 臂生效。
        self.rgb_dropout = rgb_dropout
        # 视觉通道给不给时间(2026-08-22):
        #   images  = 8 张独立图,**没有任何时间信息**(08-04 起的现役行为,默认,逐位不变)
        #   tstamp  = 每帧前插一句字面时间戳 "<x.x seconds>",视觉 token 数不变(512),
        #             与轨迹 token 的 tsec 同一个钟 → 两个模态终于能对上表
        #   video   = 走 Qwen3-VL 原生视频通道(processor 自己生成时间戳)。⚠️帧会两两合并,
        #             视觉 token 512→196,**RGB 基线口径随之改变**,不能和旧结果直接比
        if video_mode not in VIDEO_MODES:
            raise ValueError(f"video_mode 必须是 images/tstamp/video,收到 {video_mode!r}")
        self.video_mode = video_mode
        self._no_time = set()          # 拿不到时间轴的 clip(不静默:首次会打印)
        import random as _r
        self._rng = _r.Random(seed)

    def _visual_content(self, vid, imgs, fidx, vmeta):
        return build_visual_content(vid, imgs, fidx, self.video_mode, vmeta, self._no_time)

    def __call__(self, batch: List[dict]):
        convs, answers, traj_list, n_per = [], [], [], []
        vmeta = []                      # video 模式:与批内视频出现顺序对齐
        use_frames = (self.arm != "traj_only")
        inject = self.arm in ("traj", "traj_only")
        for item in batch:
            vid = item["video_id"]
            this_frames = use_frames and not (
                self.rgb_dropout > 0 and self._rng.random() < self.rgb_dropout)
            imgs, fidx = ((sample_frames(vid, self.n_frames, self.max_side,
                                         frame_range=item.get("frame_range"), return_idx=True))
                          if this_frames else ([], []))
            n_traj = 0
            traj_block = ""
            if inject:
                arr, n_traj, _, spans, labels = (self.traj_fn or traj_arrays_for)(
                    vid, self.max_traj, frame_range=item.get("frame_range"))
                traj_list.append(arr)
                traj_block = build_traj_block(spans, labels)
            n_per.append(n_traj)
            content = self._visual_content(vid, imgs, fidx, vmeta)
            content.append({"type": "text", "text": build_prompt_text(item, self.arm, traj_block, self.structured)})
            convs.append([{"role": "user", "content": content}])
            answers.append(build_target_text(item, self.structured))

        # video 模式:把 metadata 交给 processor(否则它会按 fps 重采样并越界)
        vkw = (dict(video_metadata=vmeta, do_sample_frames=False)
               if (self.video_mode == "video" and vmeta) else {})
        # prompt-only(算掩码长度)
        prompt_ids = self.proc.apply_chat_template(
            convs, tokenize=True, add_generation_prompt=True, padding=False, **vkw)
        # full = prompt + assistant answer
        full_convs = [c + [{"role": "assistant", "content": [{"type": "text", "text": a}]}]
                      for c, a in zip(convs, answers)]
        enc = self.proc.apply_chat_template(
            full_convs, tokenize=True, add_generation_prompt=False,
            return_dict=True, return_tensors="pt", padding=True, **vkw)

        input_ids = enc["input_ids"]
        attn = enc["attention_mask"]
        labels = input_ids.clone()
        labels[attn == 0] = -100
        # 逐样本:mask 掉 prompt 段(左侧非 pad 起点 + prompt 长度)
        for i in range(len(batch)):
            plen = len(prompt_ids[i])
            n_pad = int((attn[i] == 0).sum()) if self.proc.tokenizer.padding_side == "left" else 0
            labels[i, : n_pad + plen] = -100
        labels[input_ids == self.tid] = -100  # 保险:轨迹占位不参与 loss

        out = {k: v for k, v in enc.items()}
        out["labels"] = labels

        traj_batch = None
        if inject and traj_list:
            cat = {k: np.concatenate([a[k] for a in traj_list], axis=0) for k in traj_list[0]}
            traj_batch = {k: torch.from_numpy(v) for k, v in cat.items()}
        out["_traj_batch"] = traj_batch
        out["_n_traj_total"] = int(sum(n_per))
        return out
