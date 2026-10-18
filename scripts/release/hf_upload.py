#!/usr/bin/env python3
"""Package and upload the data / checkpoint release to Hugging Face (resumable; shard by shard).

  python scripts/release/hf_upload.py --namespace <user_or_org> [--private] [--dry-run] [--only model|data]

Layout on the Hub (both repos private until released):
  <ns>/SparseTrajTokens            (model)   checkpoints/{stage1_projector,stage2_token_only,stage3_final,rgb_only_arm}, SHA256SUMS
  <ns>/TrajQA-B                    (dataset) trajqa_b/ captions/ curation/ results/ baselines/ paper_assets/
                                             tokens_v4td.tar  figure_scenes.tar  scenes/scenes-NN.tar (64 shards) + scenes/INDEX.tsv
Frames (data/stereo4d/frames), raw mp4 tars and third-party data (data/external) are NOT uploaded (YouTube / third-party content).
Every tar is built in a staging dir, hashed, uploaded, then deleted, so the staging dir never holds more than one shard.
Re-running skips every file that already exists on the Hub with the same size (SHA256 of each tar is recorded in MANIFEST.tsv).
"""
from __future__ import annotations

import argparse, hashlib, io, json, os, sys, tarfile, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STAGE = Path(os.environ.get("HF_STAGE_DIR", str(Path.home() / "tmp" / "hf_stage")))
N_SHARDS = 64
MODEL_REPO, DATA_REPO = "SparseTrajTokens", "TrajQA-B"


def sha256(p: Path, bufsize=1 << 24) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(bufsize), b""):
            h.update(b)
    return h.hexdigest()


def shard_of(name: str) -> int:
    return int(hashlib.md5(name.encode()).hexdigest(), 16) % N_SHARDS


def add_tree(tar: tarfile.TarFile, src: Path, arc: str):
    """Add a directory; symlinked files (gt_stereo4d.npz -> frames/) are stored as regular files."""
    for e in sorted(os.scandir(src), key=lambda e: e.name):
        a = f"{arc}/{e.name}"
        if e.is_dir(follow_symlinks=False):
            add_tree(tar, Path(e.path), a)
        elif e.is_symlink():
            tar.add(os.path.realpath(e.path), arcname=a, recursive=False)
        else:
            tar.add(e.path, arcname=a, recursive=False)


def build_tar(out: Path, entries: list[tuple[Path, str]]):
    tmp = out.with_suffix(".tar.part")
    with tarfile.open(tmp, "w", format=tarfile.PAX_FORMAT) as tar:
        for src, arc in entries:
            if src.is_dir():
                add_tree(tar, src, arc)
            else:
                tar.add(str(src), arcname=arc, recursive=False)
    tmp.rename(out)


def plan_data() -> list[dict]:
    """Return the list of artefacts for the dataset repo: {'repo_path', 'kind': 'file'|'folder'|'tar', ...}."""
    d = ROOT / "data"
    items: list[dict] = []
    for sub in ("trajqa_b", "captions", "curation"):
        items.append(dict(kind="folder", src=d / sub, repo_path=sub))
    items.append(dict(kind="folder", src=ROOT / "results", repo_path="results", ignore=["README.md"]))
    items.append(dict(kind="folder", src=ROOT / "baselines", repo_path="baselines",
                      ignore=["tools/*", "**/__pycache__/*"]))
    items.append(dict(kind="folder", src=ROOT / "paper" / "figure_assets", repo_path="paper_assets/figure_assets"))
    items.append(dict(kind="folder", src=ROOT / "paper" / "demo", repo_path="paper_assets/demo",
                      ignore=["*.py", "README.md", "index.html", "data.js"]))
    items.append(dict(kind="file", src=d / "tokens_v4td" / "_meta.json", repo_path="tokens_v4td_meta.json"))
    items.append(dict(kind="tar", repo_path="tokens_v4td.tar",
                      entries=[(d / "tokens_v4td", "tokens_v4td")]))
    items.append(dict(kind="tar", repo_path="figure_scenes.tar",
                      entries=[(d / "figure_scenes", "figure_scenes")]))
    scenes = sorted(p for p in (d / "scenes").iterdir() if p.name.startswith("scene_"))
    buckets: dict[int, list[Path]] = {i: [] for i in range(N_SHARDS)}
    for p in scenes:
        buckets[shard_of(p.name)].append(p)
    index = STAGE / "INDEX.tsv"
    STAGE.mkdir(parents=True, exist_ok=True)
    with open(index, "w") as f:
        f.write("scene\tshard\n")
        for i in range(N_SHARDS):
            for p in buckets[i]:
                f.write(f"{p.name}\tscenes/scenes-{i:02d}.tar\n")
    items.append(dict(kind="file", src=index, repo_path="scenes/INDEX.tsv"))
    for i in range(N_SHARDS):
        items.append(dict(kind="tar", repo_path=f"scenes/scenes-{i:02d}.tar",
                          entries=[(p, p.name) for p in buckets[i]]))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--namespace", required=True)
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", choices=["model", "data"], default=None)
    ap.add_argument("--max-shards", type=int, default=0, help="debug: stop after N scene shards")
    ap.add_argument("--shards", default="", help="only these scene shards, e.g. 'odd' / 'even' / '3,7,12' "
                    "(run several instances through different proxies in parallel; the rest is skipped)")
    a = ap.parse_args()
    if a.shards in ("odd", "even"):
        shard_sel = {i for i in range(N_SHARDS) if i % 2 == (1 if a.shards == "odd" else 0)}
    elif a.shards:
        shard_sel = {int(x) for x in a.shards.split(",")}
    else:
        shard_sel = None

    os.environ.pop("HF_ENDPOINT", None)                      # hf-mirror is read-only; uploads go to huggingface.co
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    from huggingface_hub import HfApi, configure_http_backend
    import huggingface_hub.utils._http as _hfhttp

    # Every Hub API call gets a timeout: the proxies used here sometimes accept a connection and never answer,
    # and huggingface_hub issues most requests without one, so a single stalled call would hang the job forever.
    _default_factory = _hfhttp._default_backend_factory

    def _session_with_timeout():
        s = _default_factory()
        orig = s.request

        def request(*a, **k):
            k.setdefault("timeout", (30, int(os.environ.get("HF_READ_TIMEOUT", "900"))))
            return orig(*a, **k)
        s.request = request
        return s
    configure_http_backend(_session_with_timeout)
    api = HfApi(endpoint="https://huggingface.co")
    model_id, data_id = f"{a.namespace}/{MODEL_REPO}", f"{a.namespace}/{DATA_REPO}"
    manifest = STAGE / "MANIFEST.tsv"
    STAGE.mkdir(parents=True, exist_ok=True)
    done = {}
    if manifest.exists():
        for line in open(manifest):
            k, size, h = line.rstrip("\n").split("\t"); done[k] = (int(size), h)

    def remote_size(repo, path, rtype):
        try:
            info = api.get_paths_info(repo, [path], repo_type=rtype)
        except Exception:
            return None
        if not info:
            return None
        x = info[0]
        return getattr(x, "size", None) or (x.lfs.size if getattr(x, "lfs", None) else None)

    PROXIES = [p for p in os.environ.get("HF_UPLOAD_PROXIES", "").split(",") if p]   # e.g. http://127.0.0.1:17891,http://127.0.0.1:17890

    def set_proxy(p: str):
        for k in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy"):
            os.environ[k] = p

    def online() -> bool:
        """Reachability probe WITH a timeout (the proxy here sometimes accepts the connection and then hangs, or
        drops the TLS handshake). Any HTTP answer counts as reachable. If several proxies are configured, the
        first one that answers becomes the active proxy for the following requests."""
        import requests
        from huggingface_hub import get_token
        tok = get_token()
        cands = PROXIES or [os.environ.get("HTTPS_PROXY", "")]
        why = []
        for p in cands:
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
        print("  probe: " + " | ".join(why), flush=True)        # 留痕:到底是哪条代理怎么失败的
        return False

    def retry(what, fn, max_attempts=int(os.environ.get("HF_MAX_ATTEMPTS", "500"))):
        """Proxy outages here last hours: keep retrying with a capped back-off, and wait until the Hub is
        reachable again before each attempt (a 1.4 GB upload must not be restarted blindly every minute)."""
        for attempt in range(1, max_attempts + 1):
            try:
                return fn()
            except Exception as e:  # rate limit / network: back off and retry
                wait = min(60 * attempt, 600)
                print(f"  {what} failed ({e.__class__.__name__}: {str(e)[:160]}), retry in {wait}s "
                      f"[{time.strftime('%m-%d %H:%M')}]", flush=True)
                time.sleep(wait)
                while not online():
                    print(f"  hub unreachable, waiting 300s [{time.strftime('%m-%d %H:%M')}]", flush=True)
                    time.sleep(300)
        raise SystemExit(f"giving up on {what}")

    def upload(repo, rtype, local: Path, path, msg):
        retry(f"upload {path}", lambda: api.upload_file(path_or_fileobj=str(local), path_in_repo=path, repo_id=repo,
                                                        repo_type=rtype, commit_message=msg))

    def record(path, local: Path):
        h = sha256(local); s = local.stat().st_size
        with open(manifest, "a") as f:
            f.write(f"{path}\t{s}\t{h}\n")
        done[path] = (s, h)

    # ---------------- model repo ----------------
    if a.only in (None, "model"):
        print(f"== model repo {model_id}")
        if not a.dry_run:
            retry("create model repo", lambda: api.create_repo(model_id, repo_type="model", private=a.private, exist_ok=True))
            upload(model_id, "model", ROOT / "scripts/release/model_card.md", "README.md", "model card")
        ck = ROOT / "checkpoints"
        files = sorted(p for p in ck.rglob("*") if p.is_file() and p.name != "README.md")
        for p in files:
            rel = str(p.relative_to(ck)); sz = p.stat().st_size
            if a.dry_run:
                print(f"  {rel} ({sz/1e6:.1f} MB)"); continue
            if remote_size(model_id, rel, "model") == sz:
                print(f"  skip (exists) {rel}"); continue
            t0 = time.time(); upload(model_id, "model", p, rel, f"add {rel}")
            print(f"  {rel} {sz/1e6:.1f} MB in {time.time()-t0:.0f}s ({sz/1e6/max(time.time()-t0,1e-6):.2f} MB/s)", flush=True)

    # ---------------- dataset repo ----------------
    if a.only in (None, "data"):
        print(f"== dataset repo {data_id}")
        items = plan_data()
        if a.dry_run:
            for it in items:
                if it["kind"] == "tar":
                    n = sum(sum(f.stat().st_size for f in src.rglob("*") if f.is_file()) if src.is_dir() else src.stat().st_size
                            for src, _ in it["entries"])
                    print(f"  tar  {it['repo_path']}  ~{n/1e9:.2f} GB  ({len(it['entries'])} entries)")
                else:
                    print(f"  {it['kind']:6s} {it['repo_path']}")
            return
        retry("create dataset repo", lambda: api.create_repo(data_id, repo_type="dataset", private=a.private, exist_ok=True))
        upload(data_id, "dataset", ROOT / "scripts/release/dataset_card.md", "README.md", "dataset card")
        n_shards = 0
        for it in items:
            rp = it["repo_path"]
            if shard_sel is not None:                     # parallel instance: only its own scene shards
                if not rp.startswith("scenes/scenes-") or int(rp[len("scenes/scenes-"):-4]) not in shard_sel:
                    continue
            if it["kind"] == "folder":
                print(f"  folder {rp}", flush=True)
                retry(f"upload folder {rp}", lambda: api.upload_folder(
                    folder_path=str(it["src"]), path_in_repo=rp, repo_id=data_id, repo_type="dataset",
                    ignore_patterns=it.get("ignore", []) + ["**/__pycache__/*", "*.pyc"], commit_message=f"add {rp}"))
                continue
            if it["kind"] == "file":
                sz = it["src"].stat().st_size
                if remote_size(data_id, rp, "dataset") == sz:
                    print(f"  skip (exists) {rp}"); continue
                upload(data_id, "dataset", it["src"], rp, f"add {rp}"); print(f"  file {rp}", flush=True)
                continue
            # tar shard
            if rp.startswith("scenes/scenes-"):
                n_shards += 1
                if a.max_shards and n_shards > a.max_shards:
                    break
            if rp in done and remote_size(data_id, rp, "dataset") == done[rp][0]:
                print(f"  skip (exists) {rp}"); continue
            local = STAGE / Path(rp).name
            online()                                      # re-pick the preferred proxy if it came back
            t0 = time.time()
            build_tar(local, it["entries"])
            sz = local.stat().st_size; t1 = time.time()
            upload(data_id, "dataset", local, rp, f"add {rp}")
            record(rp, local); local.unlink()
            print(f"  {rp} {sz/1e9:.2f} GB  tar {t1-t0:.0f}s  upload {time.time()-t1:.0f}s "
                  f"({sz/1e6/max(time.time()-t1,1e-6):.2f} MB/s)", flush=True)
        # manifest last
        upload(data_id, "dataset", manifest, "MANIFEST.tsv", "manifest (sha256 of every tar)")
        print("== done")


if __name__ == "__main__":
    main()
