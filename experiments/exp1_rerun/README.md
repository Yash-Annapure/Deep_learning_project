# Experiment 1 rerun: the baseline on the same ruler

**Why:** the original `hybrid` runs of exp1 chose their best epoch on the old 4 validation sites; every later run chose
on the current 10. To compare every step with the same ruler, the baseline was trained again (same settings,
QUANTIZE off, the no-data fix in place: 0.09% of training pixels now ignored).

| Run | Val mIoU | Test mIoU | Test water IoU |
|---|---|---|---|
| `hybrid` (original, exp1) | 0.375 | 0.335 | 0.423 |
| `hybrid_seed1` (original, exp1) | 0.383 | 0.335 | 0.417 |
| `hybrid` (rerun) | 0.376 | 0.302 | 0.245 |
| `hybrid_seed1` (rerun) | 0.375 | 0.319 | 0.348 |

**Conclusions**
- Validation is unchanged (0.379 -> 0.376 mean): on validation, context adds nothing and AGREE adds +0.03.
- On test, the rerun baseline is lower (0.311 vs 0.335 mean), so the total test gain to `hybrid_ctx_agree` is +0.09.
- Test water is fragile: the same settings give 0.245 to 0.423 depending on seed and on which epoch is picked.
  Water claims rest on 3 test sites and need fresh test sites to confirm.
- The rerun is the baseline in `03_report_figures.ipynb`.

Folders: `results/` scores, `runs/` pictures and logs (both in git), `weights/` models (not in git).
