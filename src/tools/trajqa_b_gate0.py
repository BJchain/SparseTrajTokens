"""Gate 0:B 集逐题 token 侧复算(纯 CPU;2026-09-14)。

动机:v5 的"答案可从 token 复现"核验在 final_repr 米制空间做,漏了 (a) 归一化尺度层(token 坐标 = (p−center)/scale,
米制要乘 NORM_SCALE)(b) 模型看到的是**节点记录**(serialize_scene + _cap_records 512 封顶),不是样条系数。
本工具只用模型真正看到的东西:逐 keypoint 的 (u, vec3) 节点序列 + NORM_CENTER/NORM_SCALE + SCENE_HDR(T) +
节点 tsec(这里用 frame_times_s 插值,等价于"模型会在节点 tsec 间线性插值"),把每题的量复算一遍:
  节点折线在整数帧线性插值 → keypoint 中位数 = 质心 → ×scale + center → 米制 → 复算位移/速率/区间比较/开始时刻/时长/相机位移/运动来源。
已知近似:刚体 ROT_VERT 是相对 run 首帧的旋转向量而 rot_ref 不在 token 里 → 只能按 R=exp(φ) 近似;相机伪实例的平移流=光心归一化坐标(精确)。
输出 <out>.jsonl:id, tok_value(数值族)/ tok_answer(二分族), n_kp, dropped_by_cap, note。build 阶段据此剔除不可复现题并记账。
用法:PYTHONPATH=src/training python src/tools/trajqa_b_gate0.py <candidates.jsonl> <tokens_dir> <out.jsonl> [workers]"""
import json, os, pickle, sys
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
sys.path[:0] = ["src/tokenization", "src", "src/training"]
import schema  # noqa
from g1.traj_serialize import serialize_scene, ROLE2ID, KINDS
from g1.data import _cap_records
from g1.time_qa_gen import _motion_duration_s
from scipy.spatial.transform import Rotation

MAX_TRAJ = 512
MIN_KP = 1


def decode(scene):
    """序列化 + 封顶 → 各物体的节点结构(只保留模型可见的量)。"""
    recs = serialize_scene(scene, frame_range=None)
    recs, dropped = _cap_records(recs, MAX_TRAJ)
    R = ROLE2ID
    centroid_param = getattr(scene, "rigid_param", "origin") == "centroid"
    center = scale = T = None
    groups = defaultdict(lambda: dict(kind=None, pl=defaultdict(list), trans=[], rot=[], ref=[], static=[]))
    for r in recs:
        if r.role == R["NORM_CENTER"]: center = np.asarray(r.vec3, float)
        elif r.role == R["NORM_SCALE"]: scale = float(r.scalar)
        elif r.role == R["SCENE_HDR"]: T = int(round(r.scalar))
        elif r.group >= 0:
            g = groups[r.group]
            if r.role == R["OBJ_HDR"]: g["kind"] = KINDS[r.kind] if r.kind < len(KINDS) else None
            elif r.role == R["PL_VERT"]: g["pl"][(r.part, r.kp)].append((float(r.scalar), np.asarray(r.vec3, float)))
            elif r.role == R["TRANS_VERT"]: g["trans"].append((float(r.scalar), np.asarray(r.vec3, float)))
            elif r.role == R["ROT_VERT"]: g["rot"].append((float(r.scalar), np.asarray(r.vec3, float)))
            elif r.role == R["SE3_REF"]: g["ref"].append(np.asarray(r.vec3, float))
            elif r.role == R["STATIC_POINT"]: g["static"].append(np.asarray(r.vec3, float))
    return dict(center=center, scale=scale, T=T, groups=groups, dropped=dropped, centroid_param=centroid_param)


def _interp_nodes(nodes, frames, T):
    """节点 (u, vec3) 折线在帧 frames 上插值;帧在节点跨度外 → nan。"""
    if not nodes: return np.full((len(frames), 3), np.nan)
    nodes = sorted(nodes, key=lambda x: x[0])
    fu = np.array([u * (T - 1) for u, _ in nodes]); P = np.stack([p for _, p in nodes])
    out = np.full((len(frames), 3), np.nan)
    fr = np.asarray(frames, float)
    m = (fr >= fu.min() - 1e-6) & (fr <= fu.max() + 1e-6)
    if len(nodes) == 1:
        out[np.abs(fr - fu[0]) < 0.5] = P[0]; return out
    for a in range(3):
        out[m, a] = np.interp(fr[m], fu, P[:, a])
    return out


def kp_trajs(D, g, frames):
    """物体 g 的逐 keypoint 归一化轨迹 (M, len(frames), 3);静态 → 常量;刚体 → exp(φ)·ref + trans(近似)。"""
    G = D["groups"][g]; T = D["T"]
    if G["kind"] == "nonrigid":
        if not G["pl"]: return None
        return np.stack([_interp_nodes(v, frames, T) for v in G["pl"].values()])
    if G["kind"] == "rigid":
        # v4.1 tokens(rigid_param=centroid):平移流 = 质心轨迹(精确),ROT_VERT = 绝对 rotvec,SE3_REF = 以质心为原点的形状
        # → p_j(u) = exp(φ(u))·ref_j + trans(u)。旧 tokens(origin 口径)下这条近似不成立(Gate 0 会剔)。
        tr = _interp_nodes(G["trans"], frames, T)
        if not G["ref"]: return tr[None]
        ref = np.stack(G["ref"])
        if G["rot"] and D.get("centroid_param"):
            phi = _interp_nodes(G["rot"], frames, T)
            out = np.full((len(ref), len(frames), 3), np.nan)
            for k in range(len(frames)):
                if np.isnan(phi[k, 0]) or np.isnan(tr[k, 0]): continue
                out[:, k, :] = (Rotation.from_rotvec(phi[k]).as_matrix() @ ref.T).T + tr[k]
            return out
        return (ref[:, None, :] + tr[None])
    if G["kind"] == "static":
        if not G["static"]: return None
        return np.repeat(np.stack(G["static"])[:, None, :], len(frames), axis=1)
    return None


def centroid(D, g, frames, camera=False):
    """米制质心 (len(frames),3);相机 = 平移流本身。"""
    T = D["T"]
    if camera:
        tr = _interp_nodes(D["groups"][g]["trans"], frames, T); return tr * D["scale"] + D["center"], 1
    if D["groups"][g]["kind"] == "rigid" and D["groups"][g]["trans"] and D.get("centroid_param"):     # 质心参数化:平移流即质心
        tr = _interp_nodes(D["groups"][g]["trans"], frames, T); return tr * D["scale"] + D["center"], len(D["groups"][g]["ref"])
    A = kp_trajs(D, g, frames)
    if A is None: return None, 0
    c = np.nanmedian(A, axis=0) if D["groups"][g]["kind"] == "nonrigid" else np.nanmean(A, axis=0)
    nk = int(np.isfinite(A[:, :, 0]).sum(0).min()) if A.shape[1] else 0
    return c * D["scale"] + D["center"], nk


def run_video(args):
    vid, tok_path, items = args
    out = []
    try:
        scene = pickle.load(open(tok_path, "rb")); D = decode(scene); T = D["T"]
        times = np.asarray(scene.frame_times_s, float)
        gid = {int(o.instance_id): gi for gi, o in enumerate(scene.objects)}
        cam_g = next((gi for gi, o in enumerate(scene.objects) if o.label == "camera"), None)
        K, w2c = scene.camera_K, scene.camera_w2c
        for d in items:
            fam = d["event_subtype"]; gp = d["gt_payload"]; r = dict(id=d["id"], dropped_by_cap=D["dropped"], note="")
            try:
                if fam in ("disp_grid", "speed_grid"):
                    g = gid.get(d["involved_inst_ids"][0]); c, nk = centroid(D, g, [gp["i"], gp["j"]]); r["n_kp"] = nk
                    if c is None or np.isnan(c).any(): r["note"] = "no_token_pos"
                    else:
                        dd = float(np.linalg.norm(c[1] - c[0]))
                        r["tok_value"] = dd if fam == "disp_grid" else dd / max(float(times[gp["j"]] - times[gp["i"]]), 1e-6)
                elif fam in ("interval_disp", "interval_speed"):
                    g = gid.get(d["involved_inst_ids"][0]); c, nk = centroid(D, g, [gp["i"], gp["j"], gp["k"]]); r["n_kp"] = nk
                    if c is None or np.isnan(c).any(): r["note"] = "no_token_pos"
                    else:
                        d1, d2 = float(np.linalg.norm(c[1] - c[0])), float(np.linalg.norm(c[2] - c[1]))
                        if fam == "interval_speed":
                            d1 /= max(float(times[gp["j"]] - times[gp["i"]]), 1e-6); d2 /= max(float(times[gp["k"]] - times[gp["j"]]), 1e-6)
                        r["tok_answer"] = 1 if d1 >= d2 else 2; r["tok_ratio"] = round(max(d1, d2) / max(min(d1, d2), 1e-6), 3)
                elif fam in ("start_time_grid", "motion_duration_binary", "motion_duration_mcq"):
                    g = gid.get(d["involved_inst_ids"][0]); A = kp_trajs(D, g, list(range(T)))
                    if A is None: r["note"] = "no_token_pos"
                    else:
                        traj = A * D["scale"] + D["center"]; vis = np.isfinite(traj[:, :, 0])
                        md = _motion_duration_s(dict(trajectories=np.nan_to_num(traj), visibility=vis), times, K, w2c)
                        r["n_kp"] = int(A.shape[0])
                        if md is None: r["note"] = "no_motion_run"
                        else:
                            dur, fa, fb = md
                            if fam == "start_time_grid": r["tok_value"] = float(times[fa])
                            elif fam == "motion_duration_mcq": r["tok_value"] = float(dur)
                            else: r["tok_answer"] = "Yes" if dur > gp["threshold_s"] else "No"
                elif fam == "cam_disp_grid":
                    if cam_g is None: r["note"] = "no_camera_group"
                    else:
                        c, _ = centroid(D, cam_g, [gp["i"], gp["j"]], camera=True)
                        r["tok_value"] = float(np.linalg.norm(c[1] - c[0])) if not np.isnan(c).any() else None
                        if r["tok_value"] is None: r["note"] = "no_token_pos"
                elif fam == "motion_source":
                    g = gid.get(d["involved_inst_ids"][0]); kind = D["groups"][g]["kind"] if g is not None else None
                    if kind == "static": r["tok_answer"] = "camera"
                    else:
                        c, nk = centroid(D, g, [gp["i"], gp["j"]]); r["n_kp"] = nk
                        if c is None or np.isnan(c).any(): r["note"] = "no_token_pos"
                        else: r["tok_disp"] = float(np.linalg.norm(c[1] - c[0])); r["tok_answer"] = "object" if r["tok_disp"] >= 0.25 else "camera"
                else:
                    r["note"] = "family_not_handled"
            except Exception as e:
                r["note"] = f"exc:{type(e).__name__}:{str(e)[:50]}"
            out.append(r)
    except Exception as e:
        out = [dict(id=d["id"], note=f"scene_exc:{type(e).__name__}:{str(e)[:50]}") for d in items]
    return out


def main():
    src, tok, dst = sys.argv[1:4]; nw = int(sys.argv[4]) if len(sys.argv) > 4 else 24
    by = defaultdict(list)
    for l in open(src):
        d = json.loads(l); by[d["video_id"]].append(d)
    jobs = [(v, os.path.join(tok, v + ".pkl"), L) for v, L in by.items()]
    n = 0; notes = defaultdict(int)
    with Pool(nw) as pool, open(dst, "w") as fo:
        for rows in pool.imap_unordered(run_video, jobs, chunksize=8):
            for r in rows:
                fo.write(json.dumps(r) + "\n"); n += 1
                if r.get("note"): notes[r["note"].split(":")[0] + (":" + r["note"].split(":")[1] if r["note"].startswith("exc") or r["note"].startswith("scene_exc") else "")] += 1
    print(f"gate0: {n} 题,备注 {dict(notes)}")


if __name__ == "__main__":
    main()
