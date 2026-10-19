---
license: apache-2.0
tags:
  - archive
---

# SparseTrajTokens — workspace archive (private)

Snapshot taken on 2026-10-18, two days before the training server was decommissioned. One `tar.gz`
(`workspace_archive_20261018.tar.gz`, 8.9 GB raw / 2.1 GB compressed, SHA-256 in the `.sha256` file;
`archive_files.txt` lists every member) with everything from the old workspace that is not a large binary:

* `3D_VLM_Fusion/` — all source code of every stage and version (incl. exploratory ones not in the public repo),
  every design / finding / experiment-record markdown, all QA sets of every version (`milestone0/*`), all evaluation
  outputs (`runs_sft110k/eval`, `runs_t2`, …), training logs and `meta.json` of every run, figures, LaTeX sources,
  shell launch scripts and logs — excluding frames, scene products, conda environments, third-party checkpoints and
  model weights.
* `3D_VLM_Fusion/stage0_4/third_party/*` — source snapshots (no weights) of SAM 3, SAM 3D Objects, Fast-SAM3D,
  PartSAM, Track4World, TripoSR, SpaTrackerV2, stereo4d-code, cc_torch, tapnet as used, with local patches.
* `tmp/sft110k/` — the original training / evaluation launch scripts and their logs.
* `SparseTrajTokens/docs`, `scripts`, `environment`, `third_party` — the release repository's internal notes
  (Chinese), release tooling, exact `pip freeze` of every environment and the third-party patches.
* `.claude/projects/` — the assistant's project memory and session transcripts (provenance of every number).

Supplement (added 2026-10-20, after the last uploads finished): `workspace_archive_20261020_claude_projects.tar.gz`
is a fresh snapshot of `.claude/projects/` (assistant memory and session transcripts); it supersedes the copy inside
the 2026-10-18 archive, which lacks the final two days (the frames / ablations / archive uploads themselves).
`MANIFEST.tsv` lists the size and SHA-256 of both tarballs.

Large binaries live elsewhere: `TrajQA-B` (data, results), `SparseTrajTokens` (released weights),
`SparseTrajTokens-ablations` (other checkpoints), `stereo4d-frames-42k` (frames).
