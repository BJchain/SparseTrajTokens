#!/usr/bin/env python3
"""Verify the three supplementary Hub repos file-by-file: python verify_extra.py [proxy] [frames|ablations|archive ...]
(maintainer tool; expects the local staging dir ~/tmp/hf_stage and the old workspace paths of the training server)
  frames    : 64 个 tar 的大小 + sha256(Hub 的 lfs.sha256)对 frames_MANIFEST.tsv;README/INDEX/MANIFEST 是否在
  ablations : 52 个 run:.uploaded 标记、远端文件集合(路径+大小)== 本地按同一 ignore 规则过滤后的集合
  archive   : tar.gz 大小 + sha256 对本地 .sha256;.sha256/archive_files.txt/README/MANIFEST 是否在"""
import os, sys
from pathlib import Path
os.environ.pop("HF_ENDPOINT", None)
args = sys.argv[1:]
proxy = args.pop(0) if args and args[0].startswith("http") else "http://127.0.0.1:7897"
for k in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy"): os.environ[k] = proxy
modes = args or ["archive", "ablations", "frames"]
from huggingface_hub import HfApi
from huggingface_hub.utils import filter_repo_objects
api = HfApi(endpoint="https://huggingface.co"); NS = "Chenfy0407"
STAGE = Path.home() / "tmp" / "hf_stage"; OLD = Path("/data/home/chenfy/3D_VLM_Fusion/stage0_4")
ARCH = Path("/var/tmp/chenfy_stage")
RELEASED = {"1a_fit_tail24k_s0", "pB_wsd_tok_s0_tail7k_d20", "pB_s2_cos1ep_s0", "pB_rgbsw_9002_s0"}
bad = 0

def remote(repo, rtype):
    out = {}
    for f in api.list_repo_tree(repo, repo_type=rtype, recursive=True, expand=True):
        if type(f).__name__ == "RepoFile":
            out[f.path] = (f.size, f.lfs.sha256 if getattr(f, "lfs", None) else None)
    return out

def need(rem, path, label):
    global bad
    if path in rem: print(f"  ok   {label} {path} ({rem[path][0]/1e6:.1f} MB)")
    else: print(f"  MISS {label} {path}"); bad += 1

if "archive" in modes:
    rem = remote(f"{NS}/sparsetraj-archive", "dataset"); print(f"== archive: {len(rem)} remote files")
    for tgz in sorted(ARCH.glob("workspace_archive_*.tar.gz")):
        shaf = next(p for p in ARCH.iterdir() if p.name.startswith(tgz.name.replace(".tar.gz", "")) and p.name.endswith(".sha256"))
        sha = shaf.read_text().split()[0]
        r = rem.get(tgz.name)
        ok = r is not None and r[0] == tgz.stat().st_size and r[1] == sha
        print(f"  {'ok  ' if ok else 'BAD '} {tgz.name} local {tgz.stat().st_size} sha {sha[:12]} | remote {r}")
        bad += 0 if ok else 1
        need(rem, shaf.name, "sha file")
    for p in ("archive_files.txt", "README.md", "MANIFEST.tsv"): need(rem, p, "aux")

if "ablations" in modes:
    rem = remote(f"{NS}/SparseTrajTokens-ablations", "model"); print(f"== ablations: {len(rem)} remote files")
    runs = []
    for d in sorted((OLD / "runs_sft110k").iterdir()):
        if d.is_symlink() or not d.is_dir() or d.name in RELEASED or d.name.startswith("_") or d.name == "eval": continue
        if (d / "final").is_dir() and any((d / "final").glob("*.safetensors")) or (d / "final" / "traj_encoder.pt").exists():
            runs.append(("runs_sft110k", d))
    for d in sorted((OLD / "runs_t2").iterdir()):
        if d.is_dir() and (d / "final").is_dir(): runs.append(("runs_t2", d))
    for d in sorted((OLD / "runs_ablation_0824").iterdir()):
        if d.is_dir(): runs.append(("runs_ablation_0824", d))
    done = 0
    for group, d in runs:
        rp = f"{group}/{d.name}"
        ign = ["**/optim.pt", "optim.pt", "resume*/**", "**/resume*/**", "step*/**", "**/*.lock", "**/__pycache__/**"] \
            if group != "runs_ablation_0824" else ["**/optim.pt", "**/__pycache__/**"]
        local_all = [str(p.relative_to(d)) for p in d.rglob("*") if p.is_file()]
        exp = {rel: (d / rel).stat().st_size for rel in filter_repo_objects(local_all, ignore_patterns=ign)}
        got = {k[len(rp) + 1:]: v[0] for k, v in rem.items() if k.startswith(rp + "/") and not k.endswith("/.uploaded")}
        marker = f"{rp}/.uploaded" in rem
        missing = {k for k in exp if k not in got}; extra = {k for k in got if k not in exp}
        sizebad = {k for k in exp if k in got and exp[k] != got[k]}
        if marker and not missing and not extra and not sizebad:
            done += 1; print(f"  ok   {rp}: {len(exp)} files")
        else:
            bad += 1; print(f"  BAD  {rp}: marker={marker} local {len(exp)} remote {len(got)} missing {sorted(missing)[:4]} extra {sorted(extra)[:4]} size {sorted(sizebad)[:4]}")
    print(f"  -> {done}/{len(runs)} runs complete"); need(rem, "README.md", "card")

if "frames" in modes:
    rem = remote(f"{NS}/stereo4d-frames-42k", "dataset"); print(f"== frames: {len(rem)} remote files")
    man = {}
    for line in open(STAGE / "frames_MANIFEST.tsv"):
        k, s, h = line.rstrip("\n").split("\t"); man[k] = (int(s), h)
    okn = 0
    for i in range(64):
        k = f"frames/frames-{i:02d}.tar"; r = rem.get(k); m = man.get(k)
        if r and m and r[0] == m[0] and r[1] == m[1]: okn += 1
        else:
            bad += 1; print(f"  BAD  {k}: manifest {m and (m[0], m[1][:12])} remote {r and (r[0], r[1] and r[1][:12])}")
    print(f"  -> {okn}/64 shards verified (size + sha256)")
    for p in ("README.md", "frames/INDEX.tsv", "MANIFEST.tsv"): need(rem, p, "aux")
print("== RESULT:", "ALL OK" if bad == 0 else f"{bad} problems"); sys.exit(1 if bad else 0)
