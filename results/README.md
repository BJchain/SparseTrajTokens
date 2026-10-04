# Results

Distributed with the data release (not in git).

| Path | Content |
|---|---|
| `eval/final/final_{hld,dev}_{traj,trajonly,donor,blind}.json(.items.jsonl)` | Final model (`checkpoints/stage3_final`) on the held-out (`hld`, 3,430 q.) and test (`dev`, 4,492 q.) splits under the four input conditions; `.items.jsonl` = per-question output (generated evidence, parsed answer, correctness) |
| `eval/final/final_{heldout,dev}.jsonl` | Pooled metrics per condition (`src/tools/evidence_metrics.py`): numeric-3 accuracy, per-family accuracy, evidence position errors, within-20% rates |
| `eval/final/cmp_{heldout,dev}_{rgb,rgbblind}.json(.items.jsonl)`, `final_compare.jsonl` | Frames-only arm (`checkpoints/rgb_only_arm`) on the same splits (`cmp_*joint*` rows in `final_compare.jsonl` belong to an earlier joint model and are not used in the paper) |
| `eval/final/logs/` | stdout of the final evaluation runs |
| `eval/selection_val/` | Every validation-split evaluation used for checkpoint / schedule selection: Stage 1 probes (`1afit_*`, `1a_fit_probe.jsonl`), Stage 2 schedule search (`1bwsd_*`, `1b_wsd.jsonl`), Stage 3 schedule search incl. the selected cosine run (`s2wsd_*`, `1b_s2_wsd.jsonl`), frames-only budget sweep (`rgbsw_*`, `rgb_sweep_val.jsonl`) |
| `training_logs/<run>/` | `meta.json` (all hyper-parameters), `train_log.jsonl`, `val_curve.jsonl`, `PREREG.md` (selection rule written before the run), `stdout.log` for every run on the final recipe's search path |
| `tracker_bakeoff/scoreboard.json` | Tracker bake-off numbers (appendix table) |

Baseline predictions and the main/ablation tables live in `baselines/` (`preds/`, `results_table.md`, `figure/`).
