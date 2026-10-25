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

**Results** (mean of 2 seeds; difference to context alone in brackets):

| | mIoU | Water | Trees | Low veg | Flooded | Crops | Bare |
|---|---|---|---|---|---|---|---|
| **Test:** context (baseline) | 0.366 | 0.684 | 0.343 | 0.418 | 0.028 | 0.154 | 0.133 |
| + AGREE | **0.404** (+0.038) | 0.673 | 0.351 | 0.448 (+0.03) | 0.075 (+0.05) | 0.231 (+0.08) | 0.226 (+0.09) |
| + OVERSAMPLE 3 | 0.363 | 0.547 (-0.14) | 0.337 | 0.431 | 0.038 | 0.188 | 0.187 |
| + both | 0.394 | 0.639 | 0.326 | 0.435 | 0.084 | 0.230 | 0.227 |
| **Val:** context (baseline) | 0.380 | 0.560 | 0.410 | 0.333 | 0.066 | 0.304 | 0.195 |
| + AGREE | **0.408** (+0.028) | 0.594 | 0.370 (-0.04) | 0.352 (+0.02) | 0.146 (+0.08) | 0.359 (+0.06) | 0.221 |
| + OVERSAMPLE 3 | 0.383 | 0.529 | 0.397 | 0.317 | 0.057 | 0.362 | 0.205 |
| + both | 0.401 | 0.577 | 0.377 | 0.357 | 0.126 | 0.343 | 0.228 |

What the model saw with OVERSAMPLE 3: 19.7% of training windows boosted; flooded vegetation 2.2% -> 4.6% and bare
2.5% -> 4.8% of labelled pixels per epoch, water 5.3% -> 7.5% (a side effect), built 36.3% -> 30.2%.

**Conclusions against the rule**
- **AGREE: kept.** Both seeds beat both baseline seeds on test and on validation mIoU, and the two seeds agree closely
  (0.401 / 0.407). Crops and low vegetation improve on both sets, test water stays above 0.6. Trees is the exception:
  -0.04 on validation, so the rule is only partly met (2 of 3 target classes); kept for the large overall gain, with
  trees as a known trade-off to investigate.
- **Oversampling: dropped.** Twice the rare-class exposure did not improve flooded vegetation on validation, and the
  extra water exposure cost water (test 0.68 -> 0.55; validation water precision 0.67-0.78 vs 0.88 with AGREE).
  On top of AGREE it adds nothing either.
- **Insight:** flooded vegetation responded to cleaner labels (validation 0.07 -> 0.15), not to more exposure. Its
  problem is label quality more than the number of examples.
- New best model: `hybrid_ctx_agree_seed1` (best validation mIoU 0.410; test 0.407). Using it in the prototype needs
  context support there, and because AGREE uses ESA WorldCover in training, the prototype's independent check must
  move to another map (Esri 10 m land cover).

Folders: `results/` scores, `runs/` pictures and logs (both in git), `weights/` models (not in git).
