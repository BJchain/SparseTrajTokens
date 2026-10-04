# Notes for figure/trajqa_b_baseline.png (moved out of the figure on 2026-09-25)

- Main metric = Numeric-3 combined (cam_disp_grid + disp_grid + speed_grid), 4-way MCQ, chance = 25.0.  All numbers are plain accuracy with frames given.

- Frames are sampled uniformly over the WHOLE clip and NEVER cropped to the asked interval; greedy, max_new_tokens=4, first A/B/C/D taken.

- Non-num. = mean over the four small families (n=76-146), reported not interpreted.

- Base models are zero-shot; our arms are fine-tuned on TrajQA-B train (videos disjoint from both splits), so the comparison is protocol-mismatched (same caveat as MLLM4D-Bench Table 1). Ours: same 7,922 Q on both splits; 8 frames sampled INSIDE the asked interval + trajectory tokens from the 4D scene representation; structured evidence->answer output, greedy. Recipe = projector warm-up -> token-only stage -> joint RGB+traj stage (cosine, 1 epoch). Reference (not shown): the same fine-tuning WITHOUT tokens (RGB-only arm) scores 48.3 (heldout 47.8 / test 48.8; preds/Ours_RGB-arm.jsonl).

- MLLM-4D-SFT: official usage (transformers 4.57, 32 frames via the official Qwen3-VL video processor with timestamps, flash-attn 2; max_new_tokens=16 + official answer parser, every output was a single letter). Its SFT set MLLM4D-2M is built from Stereo4D like TrajQA-B: 2,635 of the 3,103 eval clips are in it (same clip id). Sub-rows split its questions by that (n = Numeric-3 count); "NOT in" = the clip's YouTube video never appears in MLLM4D-2M (264 Numeric-3 questions whose video appears only via other clips are in neither). Its 25.0 is a shortcut, not chance guessing: in MLLM4D-2M's 856k numeric QAs the answer is the 2nd/3rd-smallest option 99.4% of the time; on TrajQA-B it picks the 3rd-smallest option in 92.6% of numeric questions, and TrajQA-B's answer positions are balanced (25% each).

- MLLM-4D-SFT split by whether the eval clip is in its SFT set (Numeric-3): clips IN its SFT data (n=6,332) 24.9 / 24.5 / 25.2 / 25.4 / 24.4 / 24.7 / 68.0; clips NOT in (video never in MLLM4D-2M; n=904) 26.1 / 24.1 / 29.3 / 24.7 / 28.7 / 25.6 / 75.7 (columns as in the table); see RESULTS.md and logs/overlap_mllm4d.json.
