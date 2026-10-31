# Experiment 4: the best model on all 50 sites

**Question:** does the exp3 winner (`hybrid_ctx_agree`: U-Net, context, AGREE) get better with 8 more training sites,
chosen for the weak classes (flooded vegetation: mopti, khulna, sudd; bare: jaisalmer; crops: wad_madani, gorakhpur;
built: niamey; trees: tai)?

**Setup:** exactly as `hybrid_ctx_agree` in exp3, two seeds, queue entries 26 and 27 in `run_experiments.py`. Only the
data changes: 29 training sites instead of 21. Validation now has 12 sites (bangweulu and gao added), so after the runs
`run_experiments.py --rescore` scores the 34-site models on the same 12 sites; the old numbers stay in `*_oldval`.

The 6 holdout sites (kumasi, kano, chiang_mai, gulu_rural, kayes_rural, anuradhapura_rural) are not scored here.

**Decision rule (set before the results):** the 50-site model becomes the prototype's model, and the baseline for the
coherence experiment, if averaged over 2 seeds its validation mIoU on the 12 sites is at least that of the 34-site
`hybrid_ctx_agree` runs, and no class drops by more than 0.05 validation IoU.

**Results** (2 seeds each; validation = the same 12 sites for both, after `--rescore`; test = the 3 development sites):

| | Val mIoU | Val water (precision) | Test mIoU | Test water |
|---|---|---|---|---|
| 34 sites (exp3 `hybrid_ctx_agree`) | 0.391 (0.389 / 0.392) | 0.473 (0.61) | 0.404 | 0.673 |
| **50 sites** | **0.450** (0.455 / 0.445) | **0.539** (0.73) | 0.395 (0.400 / 0.390) | 0.678 |

Per class, validation IoU (mean of 2 seeds):

| water | trees | low veg | flooded veg | crops | built | bare |
|---|---|---|---|---|---|---|
| 0.473 -> 0.538 (+0.065) | 0.338 -> 0.368 (+0.030) | 0.355 -> 0.372 (+0.017) | **0.189 -> 0.360 (+0.171)** | 0.340 -> 0.355 (+0.014) | 0.776 -> 0.785 (+0.010) | **0.261 -> 0.371 (+0.111)** |

**Conclusion against the rule: met.** Validation mIoU +0.059 (both 50-site seeds beat both 34-site seeds), and every class
improves. `hybrid_ctx_agree` (seed 0, best validation mIoU 0.455) becomes the prototype's model and the baseline for
the coherence experiment.

- The largest gains are in the classes the new sites were chosen for: flooded vegetation (mopti, khulna, sudd) nearly
  doubles and bare (jaisalmer) gains 0.11. In exp3, showing the existing rare-class windows 3x as often did nothing for
  flooded vegetation; new, different examples did. More varied data beat more exposure.
- The 3 development test sites do not move (0.395 vs 0.404, within the seed spread). They are a city, a town and
  savanna, which the new sites were not aimed at. Part of the validation gain is also that validation now includes a
  swamp (bangweulu) and a desert town (gao), where the new training sites pay off: the new data helps where it was
  aimed, not everywhere. The 6 holdout sites give the final, unbiased number.
