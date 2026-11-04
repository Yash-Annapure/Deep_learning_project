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

## What the model relies on (channel ablation, seed 0)

`run_experiments.py --ablation experiments/exp4_50sites/weights/hybrid_ctx_agree.pt`; full table and heatmap in
`ablation_hybrid_ctx_agree.md` / `.png`. Validation (12 sites), one part of the input taken away at a time; the
numbers are the change in IoU against the full input (baseline mIoU 0.455).

| Class (baseline IoU) | Recognised by | Evidence |
|---|---|---|
| built (0.791) | the yearly median (bright all year) and VH; the month itself barely matters | median removed −0.79 (to 0.00); month → median −0.03 |
| low vegetation (0.394) | the yearly context; hardly the month | median removed −0.39; this month removed −0.08 |
| crops (0.357) | how this month differs from the typical year | month → median −0.34 (to 0.02) |
| bare (0.364) | this month, both polarisations, and its season | VV −0.31, VH −0.36, month → median −0.19 |
| flooded vegetation (0.397) | everything: the month, its season and fine texture; the most fragile class | VV −0.33, VH −0.37, blur −0.29, month → median −0.16 |
| trees (0.332) | this month's radar and its fine texture; barely the median | VV −0.27, VH −0.22, blur −0.17, median −0.07 |
| water (0.547) | the month and the median together; little texture | VH −0.26, median −0.38, blur −0.07 |

- **Two strategies.** Stable classes (built, low vegetation) are recognised from the yearly picture, seasonal classes
  (crops, bare, flooded vegetation) from how this month departs from it. The monthly maps, and so the changes the
  prototype reports, are driven by each month's own signal, not only by the typical year.
- **Crops exist only through their season:** with the month replaced by the yearly median, crops IoU falls from 0.357
  to 0.021. The same effect makes a majority vote over 12 months call most fields bare or low vegetation (the reason
  for the 3-month cropland rule in the prototype's yearly map).
- **It explains the largest confusion left** (true trees predicted as low vegetation, 0.43): low vegetation is the
  class the model falls back on from context, while trees need the month's signal and texture.
- **The model reads the relationship between VV and VH:** taking away VH alone (−0.247) hurts more than taking away
  the whole month (−0.238). A real VV next to an average VH is a combination it never saw; with both gone it falls back
  on the yearly median.
- **Fine texture matters for trees and flooded vegetation** (blurred to 20 m: −0.17, −0.29), not for low vegetation or
  built (−0.01, −0.05).

**Caveat:** a channel replaced by its average gives inputs that never occurred in training, so the single-channel and
median rows overstate the pure loss of information (the VV/VH point shows it). "Month → yearly median" is a realistic
input and the most trustworthy row. **Consequence for the prototype:** the median is computed from the months that
have radar; with few months (coverage gaps) the context the model relies on most is weaker.
