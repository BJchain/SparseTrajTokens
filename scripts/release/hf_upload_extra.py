#!/usr/bin/env python3
"""Supplementary uploads made before the training server was decommissioned (2026-10-18). Same robustness as
hf_upload.py (timeouts on every Hub call, proxy fallback, resumable, sha256 manifest), three targets:

  --mode archive    <ns>/sparsetraj-archive          (dataset, private)  tar.gz of every non-binary file of the old
                                                     workspace + third-party source snapshots + notes + Claude memory
  --mode ablations  <ns>/SparseTrajTokens-ablations  (model, private)    final checkpoints of the ablation / search runs
                                                     that are not in the main model repo
  --mode frames     <ns>/stereo4d-frames-42k         (dataset, public, gated) the 42,151 frame directories used for
                                                     training/evaluation, 64 tar shards aligned with TrajQA-B/scenes

  python scripts/release/hf_upload_extra.py --namespace Chenfy0407 --mode frames [--shards even|odd|1,3,5]
"""
from __future__ import annotations

import argparse, hashlib, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hf_upload as H                                   # build_tar / sha256 / shard_of / N_SHARDS

OLD = Path(os.environ.get("OLD_WORKSPACE", "/data/home/chenfy/3D_VLM_Fusion/stage0_4"))
NEW = Path(__file__).resolve().parents[2]
STAGE = Path(os.environ.get("HF_STAGE_DIR", str(Path.home() / "tmp" / "hf_stage")))
ARCHIVE_DIR = Path(os.environ.get("ARCHIVE_DIR", "/var/tmp/chenfy_stage"))
RELEASED = {"1a_fit_tail24k_s0", "pB_wsd_tok_s0_tail7k_d20", "pB_s2_cos1ep_s0", "pB_rgbsw_9002_s0"}
CARDS = {"archive": "archive_card.md", "ablations": "ablations_card.md", "frames": "frames_card.md"}
REPOS = {"archive": ("sparsetraj-archive", "dataset"), "ablations": ("SparseTrajTokens-ablations", "model"),
         "frames": ("stereo4d-frames-42k", "dataset")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--namespace", required=True)
    ap.add_argument("--mode", choices=list(REPOS), required=True)
    ap.add_argument("--shards", default="")
    ap.add_argument("--index-only", action="store_true",
                    help="frames: upload README + frames/INDEX.tsv + MANIFEST.tsv (deduplicated) and exit, no shards")
    a = ap.parse_args()
    name, rtype = REPOS[a.mode]; repo = f"{a.namespace}/{name}"
    private = a.mode != "frames"

    os.environ.pop("HF_ENDPOINT", None)
    from huggingface_hub import HfApi, configure_http_backend, get_token
    import huggingface_hub.utils._http as _hfhttp
    import requests
    _default_factory = _hfhttp._default_backend_factory

    def _session_with_timeout():
        s = _default_factory(); orig = s.request

        def request(*x, **k):
            k.setdefault("timeout", (30, int(os.environ.get("HF_READ_TIMEOUT", "900"))))
            return orig(*x, **k)
        s.request = request
        return s
    configure_http_backend(_session_with_timeout)
    api = HfApi(endpoint="https://huggingface.co")
    PROXIES = [p for p in os.environ.get("HF_UPLOAD_PROXIES", "").split(",") if p]

    def set_proxy(p):
        for k in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy"):
            os.environ[k] = p

    def online():
        tok = get_token(); why = []
        for p in PROXIES or [os.environ.get("HTTPS_PROXY", "")]:
            for _ in range(2):
                try:
                    r = requests.get("https://huggingface.co/api/whoami-v2", timeout=(10, 30),
                                     proxies={"https": p, "http": p} if p else None,
                                     headers={"authorization": f"Bearer {tok}"} if tok else {})
                    if r.status_code in (200, 401, 403):
                        if p and p != os.environ.get("HTTPS_PROXY"):
                            print(f"  switching proxy -> {p}", flush=True)
                        if p:
                            set_proxy(p)
                        return True
                    why.append(f"{p}: HTTP {r.status_code}")
                except Exception as e:
                    why.append(f"{p}: {e.__class__.__name__} {str(e)[:80]}")
        print("  probe: " + " | ".join(why), flush=True)
        return False

    def retry(what, fn, max_attempts=500):
        for attempt in range(1, max_attempts + 1):
            try:
                return fn()
            except Exception as e:
                wait = min(60 * attempt, 600)
                print(f"  {what} failed ({e.__class__.__name__}: {str(e)[:160]}), retry in {wait}s "
                      f"[{time.strftime('%m-%d %H:%M')}]", flush=True)
                time.sleep(wait)
                while not online():
                    print(f"  hub unreachable, waiting 300s [{time.strftime('%m-%d %H:%M')}]", flush=True)
                    time.sleep(300)
        raise SystemExit(f"giving up on {what}")

    def remote_size(path):
        try:
            info = api.get_paths_info(repo, [path], repo_type=rtype)
        except Exception:
            return None
        if not info:
            return None
        x = info[0]
        return getattr(x, "size", None) or (x.lfs.size if getattr(x, "lfs", None) else None)

    def upload(local: Path, path, msg):
        retry(f"upload {path}", lambda: api.upload_file(path_or_fileobj=str(local), path_in_repo=path, repo_id=repo,
                                                        repo_type=rtype, commit_message=msg))

    manifest = STAGE / f"{a.mode}_MANIFEST.tsv"; STAGE.mkdir(parents=True, exist_ok=True)
    done = {}
    if manifest.exists():
        for line in open(manifest):
            k, size, h = line.rstrip("\n").split("\t"); done[k] = (int(size), h)

    def record(path, local: Path):
        h = H.sha256(local); s = local.stat().st_size
        with open(manifest, "a") as f:
            f.write(f"{path}\t{s}\t{h}\n")
        done[path] = (s, h)

    def send_tar(rp, entries):
        if rp in done and remote_size(rp) == done[rp][0]:
            print(f"  skip (exists) {rp}"); return
        online()
        local = STAGE / Path(rp).name; t0 = time.time()
        H.build_tar(local, entries); sz = local.stat().st_size; t1 = time.time()
        upload(local, rp, f"add {rp}"); record(rp, local); local.unlink()
        print(f"  {rp} {sz/1e9:.2f} GB  tar {t1-t0:.0f}s  upload {time.time()-t1:.0f}s "
              f"({sz/1e6/max(time.time()-t1,1e-6):.2f} MB/s)", flush=True)

    print(f"== {a.mode} -> {repo} ({rtype}, {'private' if private else 'public+gated'})", flush=True)
    retry("create repo", lambda: api.create_repo(repo, repo_type=rtype, private=private, exist_ok=True))
    if a.mode == "frames":
        retry("gate repo", lambda: api.update_repo_settings(repo, repo_type=rtype, gated="manual"))
    card = NEW / "scripts/release" / CARDS[a.mode]
    if card.exists() and (not a.shards or a.index_only):
        upload(card, "README.md", "card")

    if a.mode == "archive":
        for f in sorted(ARCHIVE_DIR.glob("workspace_archive_*")) + [ARCHIVE_DIR / "archive_files.txt"]:
            if not f.exists(): continue
            if remote_size(f.name) == f.stat().st_size:
                print(f"  skip (exists) {f.name}"); continue
            t0 = time.time(); upload(f, f.name, f"add {f.name}")
            print(f"  {f.name} {f.stat().st_size/1e9:.2f} GB in {time.time()-t0:.0f}s", flush=True)

    elif a.mode == "ablations":
        runs = []
        for d in sorted((OLD / "runs_sft110k").iterdir()):
            if d.is_symlink() or not d.is_dir() or d.name in RELEASED or d.name.startswith("_") or d.name == "eval": continue
            if (d / "final").is_dir() and any((d / "final").glob("*.safetensors")) or (d / "final" / "traj_encoder.pt").exists():
                runs.append(("runs_sft110k", d))
        for d in sorted((OLD / "runs_t2").iterdir()):
            if d.is_dir() and (d / "final").is_dir(): runs.append(("runs_t2", d))
        for d in sorted((OLD / "runs_ablation_0824").iterdir()):
            if d.is_dir(): runs.append(("runs_ablation_0824", d))
        print(f"  {len(runs)} run directories", flush=True)
        for group, d in runs:
            rp = f"{group}/{d.name}"
            marker = f"{rp}/.uploaded"
            if remote_size(marker) is not None:
                print(f"  skip (done) {rp}"); continue
            online(); t0 = time.time()
            ign = ["**/optim.pt", "optim.pt", "resume*/**", "**/resume*/**", "step*/**", "**/*.lock",
                   "**/__pycache__/**"] if group != "runs_ablation_0824" else ["**/optim.pt", "**/__pycache__/**"]
            if os.environ.get("ABL_PER_FILE"):
                # Flaky proxy fallback: upload_folder puts every non-LFS file (json/log/jsonl) base64-inline into ONE
                # commit POST (~215 MB for runs_ablation_0824/stage0_4) that never completes on a 0.2 MB/s link that
                # drops every few minutes. Same file set (same ignore rules), but committed in <=16 MB batches, each
                # retried; files already present with the right size are skipped.
                from huggingface_hub import CommitOperationAdd
                from huggingface_hub.utils import filter_repo_objects
                rels = sorted(filter_repo_objects([str(p.relative_to(d)) for p in d.rglob("*") if p.is_file()],
                                                  ignore_patterns=ign))
                have = {}
                try:
                    for f in api.list_repo_tree(repo, path_in_repo=rp, repo_type=rtype, recursive=True):
                        if type(f).__name__ == "RepoFile": have[f.path] = f.size
                except Exception:
                    pass
                todo = [r for r in rels if have.get(f"{rp}/{r}") != (d / r).stat().st_size]
                batches, cur, cur_sz = [], [], 0
                for r in todo:
                    sz = (d / r).stat().st_size
                    if cur and cur_sz + sz > 16e6:
                        batches.append(cur); cur, cur_sz = [], 0
                    cur.append(r); cur_sz += sz
                if cur: batches.append(cur)
                print(f"  {rp}: {len(todo)}/{len(rels)} files to upload in {len(batches)} commits", flush=True)
                for i, b in enumerate(batches, 1):
                    ops = [CommitOperationAdd(path_in_repo=f"{rp}/{r}", path_or_fileobj=str(d / r)) for r in b]
                    retry(f"commit {rp} [{i}/{len(batches)}]", lambda: api.create_commit(
                        repo_id=repo, repo_type=rtype, operations=ops, commit_message=f"add {rp} part {i}/{len(batches)}"))
                    print(f"  {rp} commit {i}/{len(batches)} ok ({sum((d / r).stat().st_size for r in b)/1e6:.1f} MB)", flush=True)
            else:
                retry(f"upload folder {rp}", lambda: api.upload_folder(
                    folder_path=str(d), path_in_repo=rp, repo_id=repo, repo_type=rtype, ignore_patterns=ign,
                    commit_message=f"add {rp}"))
            mk = STAGE / ".uploaded"; mk.write_text(time.strftime("%Y-%m-%d %H:%M"))
            upload(mk, marker, f"mark {rp}")
            print(f"  {rp} done in {time.time()-t0:.0f}s", flush=True)

    elif a.mode == "frames":
        fr = NEW / "data" / "stereo4d" / "frames"
        clips = sorted(p for p in fr.iterdir() if p.is_dir())
        # Priority tiers (the server is being decommissioned; upload what matters most first):
        #   A = clips of the QA evaluation splits (val / test / held-out)  -> needed to re-evaluate the released model
        #   B = clips of the TrajQA-B training split                        -> needed to retrain
        #   C = everything else (caption-only scenes)
        import json
        tier_of = {}
        for split, t in (("heldout_test", "A"), ("test", "A"), ("val", "A"), ("train", "B")):
            for line in open(NEW / "data" / "trajqa_b" / "final" / f"{split}.jsonl"):
                c = json.loads(line)["video_id"][len("scene_"):]
                tier_of[c] = min(tier_of.get(c, "Z"), t)
        tiers = {"A": [], "B": [], "C": []}
        for p in clips:
            tiers[tier_of.get(p.name, "C")].append(p)
        n_a, n_b, n_c = (len(tiers[k]) for k in "ABC")
        s_a = max(1, round(H.N_SHARDS * n_a / len(clips))); s_b = max(1, round(H.N_SHARDS * n_b / len(clips)))
        s_c = H.N_SHARDS - s_a - s_b
        print(f"  tiers: A(eval) {n_a} clips/{s_a} shards, B(train) {n_b}/{s_b}, C(other) {n_c}/{s_c}", flush=True)
        buckets = {i: [] for i in range(H.N_SHARDS)}
        base = 0
        for k, n_sh in (("A", s_a), ("B", s_b), ("C", s_c)):
            for p in tiers[k]:
                buckets[base + int(hashlib.md5(p.name.encode()).hexdigest(), 16) % n_sh].append(p)
            base += n_sh
        if a.shards in ("odd", "even"):
            sel = {i for i in range(H.N_SHARDS) if i % 2 == (1 if a.shards == "odd" else 0)}
        elif a.shards:
            sel = {int(x) for x in a.shards.split(",")}
        else:
            sel = set(range(H.N_SHARDS))
        if not a.shards or a.index_only:
            idx = STAGE / "frames_INDEX.tsv"
            with open(idx, "w") as f:
                f.write("clip\tshard\n")
                for i in range(H.N_SHARDS):
                    for p in buckets[i]:
                        f.write(f"{p.name}\tframes/frames-{i:02d}.tar\n")
            upload(idx, "frames/INDEX.tsv", "index")
        if a.index_only:
            # manifest may hold duplicate (identical) lines when a shard was re-sent after a failed remote_size probe
            dedup = STAGE / "frames_MANIFEST.dedup.tsv"
            lines = sorted(set(open(manifest).read().splitlines())) if manifest.exists() else []
            dedup.write_text("\n".join(lines) + "\n")
            upload(dedup, "MANIFEST.tsv", f"manifest ({len(lines)} shards, sha256)")
            print(f"== index-only done: {len(lines)} shards in MANIFEST", flush=True); return
        for i in range(H.N_SHARDS):
            if i not in sel: continue
            send_tar(f"frames/frames-{i:02d}.tar", [(p, p.name) for p in buckets[i]])
    if a.mode == "archive" and not manifest.exists():          # record the sha256 of the archive tarballs too
        for f in sorted(ARCHIVE_DIR.glob("workspace_archive_*.tar.gz")):
            record(f.name, f)
    if (not a.shards or a.mode != "frames") and manifest.exists():
        upload(manifest, "MANIFEST.tsv", "manifest (sha256 of every tar)")
    print("== done", flush=True)


if __name__ == "__main__":
    main()
