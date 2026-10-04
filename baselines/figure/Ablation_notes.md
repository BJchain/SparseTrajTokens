# Notes for figure/Ablation.png (kept out of the figure)

- Ablation on TrajQA-B (heldout_test + test, 7,922 Q). Main metric = Numeric-3 combined (cam_disp_grid + disp_grid + speed_grid, n = 7,500), 4-way MCQ, chance = 25.0; Heldout / Test = Numeric-3 per split; By family = per-family accuracy over both splits; Non-num. = mean over the four small families.

- All four rows are fine-tuned on TrajQA-B train (videos disjoint from both splits). Rows 1, 3, 4 are the SAME final model (projector warm-up -> token-only stage -> joint RGB+traj stage, cosine 1 epoch) with inputs changed at inference; row 2 is a separately fine-tuned RGB-only arm (same data and budget family, no trajectory tokens at training or inference; SPEC sec 17). Frames: 8, sampled inside the asked interval; structured evidence->answer output, greedy.

- Row 3 (donor control) = the final model given frames + the trajectory tokens of ANOTHER scene (format, token count, timestamps unchanged): Numeric-3 29.6 (heldout 30.6 / test 28.8), i.e. at the question-only floor, so the gain of row 4 comes from the CONTENT of the tokens, not from their format. RGB-only arm's own question-only floor: 25.7 (heldout 24.3 / test 26.9).

- Not in the figure: trajectory tokens only (final model, no frames) = Numeric-3 82.2 (heldout 81.5 / test 82.8; cam/obj/speed 89.4 / 77.7 / 75.5); frames add -0.1 on top of tokens (row 4 - tokens only); tokens add +33.8 on top of frames (row 4 - row 2). TrajQA-B is built so that the asked quantities are not recoverable from RGB alone, so these gains characterise the benchmark, not fusion in general.
