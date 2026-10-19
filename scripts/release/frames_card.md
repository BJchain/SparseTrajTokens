---
license: other
license_name: stereo4d-derived-research-only
license_link: https://stereo4d.github.io/
extra_gated_prompt: "These frames are decoded from YouTube videos of the Stereo4D dataset. By requesting access you confirm that you will use them only for non-commercial research, will not redistribute them, and accept the terms of the original Stereo4D release and of the video owners."
extra_gated_fields:
  Affiliation: text
  Intended use: text
task_categories:
  - visual-question-answering
tags:
  - video
  - stereo4d
  - frames
pretty_name: Stereo4D frames used by TrajQA-B (42,151 clips)
---

# Stereo4D frames used by TrajQA-B

The exact video frames used to train and evaluate *Sparse Trajectory Tokens for Metric Motion Queries in
Multimodal Language Models*: for each of the 42,151 processed Stereo4D clips, the rectified left-eye frames at
512×512 (`%05d.jpg`, adaptive stride, 20–38 frames per clip), the camera-only `gt_stereo4d.npz` (camera poses and
timestamps) and `stereo4d_meta.json`. Scenes, trajectory tokens and questions are in the companion dataset
`TrajQA-B`; the code repository extracts these frames deterministically from the official Stereo4D release
(byte-identical), so this repo only saves that step.

**Access is gated**: the frames are derived from YouTube videos and are provided for non-commercial research use
only, under the terms of the Stereo4D release; please do not redistribute them.

## Layout

`frames/frames-NN.tar` (64 shards, ≈1.6 GB each) contain `<clip>/…`; `frames/INDEX.tsv` maps each clip to its shard.
The shard index of a clip is the same as in `TrajQA-B/scenes/`, so shard `NN` of both repos covers the same clips.
`MANIFEST.tsv` lists the SHA-256 of every tar.

```bash
mkdir -p data/stereo4d/frames && for t in frames/frames-*.tar; do tar xf "$t" -C data/stereo4d/frames; done
```
