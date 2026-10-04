"""matched permutation / corruption 评测(2026-09-13,只做推理不重训):
同一 checkpoint,同一批题,只在推理时改轨迹**内容**,格式(object-slot 结构、token 数、角色/身份/部件通道、时间戳)逐位不变。
  clean  原样(A)
  donor  **matched donor**:把每个连续的顶点 run(PL_VERT/ROT_VERT/TRANS_VERT/STATIC_POINT/SE3_REF,按 (role, 长度) 匹配)
         的 vec3(+knot 类型)换成**另一个场景**同 (role, 长度) 的 run 的内容;时间戳(u/tsec)、角色、实例/部件/keypoint 桶、
         SCENE 头、OBJ_HDR 全部保留 → 破坏的是 trajectory↔video 的对应,不是数据格式(B)。找不到同长度供体的 run
         退化为节点顺序反转(记账)。
  C(RGB+∅)= eval.py --mode rgb(无 token),不在本脚本内。
用法(stage0_4 下):PYTHONPATH=src/training python -m g1.eval_permute --ckpt <run>/final [--no-adapter] --corrupt donor \
      --qa data/trajqa_b/final/heldout_test.jsonl --on all --out results/eval/xxx.json
其余参数与 g1.eval 相同。"""
from __future__ import annotations
import argparse, hashlib, json, os, sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, "src/tokenization"); sys.path.insert(0, "src"); sys.path.insert(0, "src/training")
import schema  # noqa
from g1 import eval as G
from g1.traj_serialize import ROLE2ID

VERT_ROLES = {ROLE2ID[r] for r in ("PL_VERT", "ROT_VERT", "TRANS_VERT", "STATIC_POINT", "SE3_REF")}
_orig_traj_arrays_for = G.traj_arrays_for


def _runs(role):
    """连续同角色顶点 run → [(start, end_exclusive, role)]"""
    out = []; i = 0; n = len(role)
    while i < n:
        if int(role[i]) in VERT_ROLES:
            j = i
            while j < n and role[j] == role[i]: j += 1
            out.append((i, j, int(role[i]))); i = j
        else:
            i += 1
    return out


class Corruptor:
    def __init__(self, kind, seed=0):
        self.kind = kind; self.seed = seed; self.pool = defaultdict(list); self.cache = {}
        self.stats = defaultdict(int)

    def key_of(self, vid, fr):
        return (vid, None if fr is None else (int(fr[0]), int(fr[1])))

    def build_pool(self, items, max_traj, whole):
        for it in items:
            vid = it["video_id"]; fr = None if whole else it.get("frame_range")
            k = self.key_of(vid, fr)
            if k in self.cache: continue
            arr, n, dropped, spans, labels = _orig_traj_arrays_for(vid, max_traj, frame_range=fr)
            self.cache[k] = (arr, n, dropped, spans, labels)
            for (a, b, r) in _runs(arr["role"]):
                self.pool[(r, b - a)].append((vid, arr["vec3"][a:b].copy(), arr["knot"][a:b].copy(),
                                              arr["scalar"][a:b].copy(), arr["tsec"][a:b].copy()))
        self.stats["pool_scenes"] = len(self.cache); self.stats["pool_keys"] = len(self.pool)

    def __call__(self, vid, max_traj, frame_range=None):
        k = self.key_of(vid, frame_range)
        if k not in self.cache:
            self.cache[k] = _orig_traj_arrays_for(vid, max_traj, frame_range=frame_range)
        arr, n, dropped, spans, labels = self.cache[k]
        if self.kind == "clean":
            return arr, n, dropped, spans, labels
        out = {kk: v.copy() for kk, v in arr.items()}
        rng = np.random.RandomState(int(hashlib.md5(f"{self.seed}|{vid}|{frame_range}".encode()).hexdigest()[:8], 16))
        for ri, (a, b, r) in enumerate(_runs(arr["role"])):
            cands = [c for c in self.pool.get((r, b - a), []) if c[0] != vid]
            if cands:
                d = cands[rng.randint(len(cands))]
                out["vec3"][a:b] = d[1]; out["knot"][a:b] = d[2]; self.stats["donor_runs"] += 1
                if self.kind == "donor_full" and b - a >= 2 and arr["has_scalar"][a:b].all():
                    # 节点时间也换成供体的(线性重标到接收者该 run 的 [u0,u1] / [t0,t1]):时间结构属于轨迹内容,不是格式
                    u_r, u_d = arr["scalar"][a:b], d[3]; t_r = arr["tsec"][a:b]
                    du = (u_d[-1] - u_d[0]); frac = (u_d - u_d[0]) / du if du > 1e-9 else np.linspace(0, 1, b - a)
                    out["scalar"][a:b] = u_r[0] + frac * (u_r[-1] - u_r[0])
                    if arr["has_tsec"][a:b].all():
                        out["tsec"][a:b] = t_r[0] + frac * (t_r[-1] - t_r[0])
                    self.stats["time_swapped_runs"] += 1
            else:                                            # 无同长度供体:节点顺序反转(仍是本场景真实内容)
                out["vec3"][a:b] = arr["vec3"][a:b][::-1]; out["knot"][a:b] = arr["knot"][a:b][::-1]; self.stats["reversed_runs"] += 1
            self.stats["tokens_touched"] += (b - a)
        self.stats["tokens_total"] += n
        return out, n, dropped, spans, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corrupt", choices=["clean", "donor", "donor_full"], default="donor")
    ap.add_argument("--seed", type=int, default=0)
    # 2026-09-17 修:原来把模式写死 "traj"(带帧)。对 token-only 训练的臂,帧是分布外输入,
    # donor 掉分会混入"突然多了帧"这一项 → 那类臂的 donor 必须用 traj_only 口径。
    ap.add_argument("--mode", choices=["traj", "traj_only"], default="traj")
    args, rest = ap.parse_known_args()
    sys.argv = [sys.argv[0]] + rest + ["--mode", args.mode]
    p = argparse.ArgumentParser()        # 与 g1.eval 同名参数(--mode 固定 traj)
    p.add_argument("--base", default=os.environ.get("QWEN3VL_PATH", "checkpoints/Qwen3-VL-8B-Instruct"))
    p.add_argument("--ckpt", required=True); p.add_argument("--mode", default="traj"); p.add_argument("--no-adapter", action="store_true")
    p.add_argument("--qa", nargs="+", required=True); p.add_argument("--out", required=True)
    p.add_argument("--val-frac", type=float, default=0.1); p.add_argument("--test-frac", type=float, default=0.0)
    p.add_argument("--split-mode", choices=["scene", "video"], default="scene"); p.add_argument("--on", choices=["val", "test", "all"], default="val")
    p.add_argument("--batch", type=int, default=8); p.add_argument("--n-frames", type=int, default=8); p.add_argument("--max-traj", type=int, default=512)
    p.add_argument("--max-side", type=int, default=384); p.add_argument("--video-mode", default="images"); p.add_argument("--limit", type=int, default=0)
    p.add_argument("--frames-whole-clip", action="store_true")
    p.add_argument("--structured", action="store_true"); p.add_argument("--max-new", type=int, default=160)   # 2026-09-14:结构化臂透传
    eargs = p.parse_args()
    # 先算评测集(与 g1.eval 同一套划分逻辑)以建供体池
    ds = G.G1Dataset(eargs.qa)
    if eargs.on == "all": val = list(ds.items)
    else:
        _, v, t = G.split3_items(ds.items, eargs.val_frac, eargs.test_frac, eargs.split_mode); val = t if eargs.on == "test" else v
    if eargs.limit: val = val[:eargs.limit]
    cor = Corruptor(args.corrupt, args.seed)
    cor.build_pool(val, eargs.max_traj, eargs.frames_whole_clip)
    G.traj_arrays_for = cor                      # 钩住 g1.eval 里的名字
    print(f"[permute] corrupt={args.corrupt} pool_scenes={cor.stats['pool_scenes']} pool_keys={cor.stats['pool_keys']}", flush=True)
    G.run(eargs)
    st = dict(cor.stats); st["corrupt"] = args.corrupt
    json.dump(st, open(eargs.out + ".permute_stats.json", "w"), indent=1)
    print("[permute] stats", st, flush=True)


if __name__ == "__main__":
    main()
