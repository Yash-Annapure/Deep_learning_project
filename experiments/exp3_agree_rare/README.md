# Experiment 3: cleaner labels and more rare-class examples, on top of context

**Question:** context (exp2) lifted water from 0.42 to 0.68 test IoU and mIoU by 0.03, but left the other classes weak:
the look-alikes (crops, low vegetation, trees), whose labels are messy, and the rare classes (flooded vegetation, bare;
3% of pixels each). Do cleaner labels and more rare-class windows fix them, alone or together?

**Setup:** everything as `hybrid_ctx` in exp2 (U-Net, context, all 4 years, 15 epochs). Two settings change:
- `AGREE`: a filled (yearly) label is kept only where ESA WorldCover agrees with it.
- `OVERSAMPLE = 3`: training windows with at least 5% flooded vegetation or bare pixels are drawn 3x as often (same
  number of draws per epoch). The notebook cell "What oversampling changes" shows what the model then sees.

| | Normal labels | AGREE |
|---|---|---|
| **No oversampling** | `hybrid_ctx`, `hybrid_ctx_seed1` (exp2, baseline) | `hybrid_ctx_agree` |
| **OVERSAMPLE 3** | `hybrid_ctx_over3` | `hybrid_ctx_agree_over3` |

Two seeds each, queue entries 18 to 23 in `run_experiments.py`.

**New in this round:** per-class IoU on the 10 validation sites (`val_iou`) is saved for every run, and `--rescore`
adds it to older runs. Exp2 showed that single-class test scores on 3 sites move a lot between seeds, so decisions use
validation per class first and test second.

**Decision rule (set before the results):** a setting is kept if, averaged over 2 seeds, it improves its target classes
(flooded vegetation and bare for oversampling; crops, low vegetation and trees for AGREE) on validation, test water
IoU stays above 0.6, and mIoU does not drop.

**Results:** pending.

Folders: `results/` scores, `runs/` pictures and logs (both in git), `weights/` models (not in git).
