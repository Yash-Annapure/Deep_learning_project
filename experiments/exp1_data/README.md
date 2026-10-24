# Experiment 1: what the data does

**Question:** on the bigger dataset (21 training sites, 2022 to 2025), what helps: more sites or more years, a label
filter, or a radar filter?

**Setup:** U-Net, ResNet-34 from scratch, hybrid labels, 15 epochs unless stated. Queue entries 0 to 7 in
`run_experiments.py`, plus the first `hybrid` run (seed 0) trained in the notebook. Validation numbers below are from
`--rescore` on the 10 current validation sites.

| Run | What changes | Val mIoU | Test mIoU | Test water IoU |
|---|---|---|---|---|
| `hybrid` | baseline, all 4 years | 0.375 | 0.335 | 0.423 |
| `hybrid_seed1` | same, second seed (**the prototype's model**) | 0.383 | 0.335 | 0.417 |
| `hybrid_y24` | 2024 only | 0.347 | 0.350 | |
| `hybrid_y24_seed1` | 2024 only, second seed | 0.338 | 0.335 | |
| `hybrid_e60_y24` | 2024 only, 60 epochs (same steps as 4 years) | 0.323 | 0.260 | 0.078 |
| `hybrid_e60_y24_seed1` | same, second seed | 0.362 | 0.280 | 0.132 |
| `hybrid_agree` | filled labels only where WorldCover agrees | 0.378 | 0.362 | 0.516 |
| `hybrid_sharpen` | sharpened radar | 0.373 | 0.291 | 0.124 |
| `hybrid_smooth` | speckle-filtered radar | 0.374 | 0.293 | 0.204 |

**Conclusions**
- The dry training sites fixed the false water (test water IoU from about 0.1 to over 0.4).
- Extra years prevent overfitting: 2024 only for as many steps as the 4-year run falls to 0.26 to 0.28 test mIoU.
- Sharpening and smoothing both hurt on the dry test sites.
- The agreement filter helps test water and bare, but not crops, and not validation.
- `hybrid_seed1` was chosen for the prototype (best validation mIoU).

Folders: `results/` scores, `runs/` pictures and logs (both in git), `notebooks/` the notebook of the first `hybrid`
run with its outputs, `weights/` models (not in git).
