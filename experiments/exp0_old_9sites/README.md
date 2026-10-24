# Experiment 0: first dataset (9 sites, 2024 only)

The first training runs, on the original 9 training sites with 2024 data only. Kept for the report as the starting point.
Runs are named `old_*` (for example `old_hybrid`, `old_hybrid_ctx`).

**Result:** test mIoU 0.21 to 0.25, test water IoU 0.065 to 0.105. The model saw water in dry regions where there was
none, and the validation sites could not show it (no dry places in them).

**What it led to:** more and drier training sites, more years, and dry validation sites (experiment 1).

Folders: `results/` scores, `notebooks/` the training notebooks with their outputs, `weights/` models (not in git).
