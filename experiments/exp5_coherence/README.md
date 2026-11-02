# Experiment 5: does InSAR coherence help the radar?

**Question:** does the exp4 model (`hybrid_ctx_agree`, 50 sites) get better when it also sees each month's InSAR
coherence? Coherence measures how much a pixel stays the same between two passes 12 days apart (coherence.py, ASF HyP3).
In the proof of concept at Pakokku and Kisumu it separated built from trees (0.83 / 0.87) far better than brightness
(~0.58), so the expected gains are in **built, bare and trees**.

**Setup:** exactly as exp4, plus `COHERENCE=True`, two seeds, queue entries 28 and 29 in `run_experiments.py`. Two more
input channels (6 in all): the month's coherence (0 where there is none) and a flag, 1 where coherence is real (option C:
a plain 0 would read as "changed completely", like water). 30% of training windows have both channels zeroed
(coherence dropout), so the model also works where coherence is missing: site-months without pairs, and the prototype,
which cannot compute coherence live. Run only once every train and val site-year has its coherence (`data/coherence/*.done`).

**Decision rule (set before the results, option (b)):** coherence is worth its extra pipeline if, averaged over 2 seeds
on the 12 validation sites, against exp4 (mean of its 2 seeds):
- built **or** bare validation IoU rises by at least **0.03** (exp4: built 0.785, bare 0.371), and
- validation mIoU is not lower (exp4: 0.450), and
- no class drops by more than 0.05 validation IoU.

Otherwise the prototype keeps the exp4 model and coherence is reported as a negative result.

**Second measurement (also set before the results): the same exp5 models with coherence zeroed** (`val_miou_nocoh`
from `--rescore`), which is what the prototype gets at a new location, where coherence can't be computed live.
- If it is at least exp4's validation mIoU (0.450) with no class down more than 0.05, the prototype switches to the
  exp5 model: training with coherence then also helped the radar-only predictions.
- If not, the prototype keeps exp4, whatever the first rule says, and coherence's gain is reported as needing coherence
  at prediction time. Stricter than exp4's rule
because coherence makes the system more complex and the prototype cannot use it live, so it has to earn a clear gain
where it was predicted.

**Coherence data** (`coherence.py --verify`): present on 98% of training, 93% of validation and 100% of test
site-month pixels; gao (validation) has coherence for under half its pixels. Best alignment with the labels at or
within 1-2 px of (0, 0) for most sites (a lean of ~1 px, half a coherence pixel), at most +0.7% (median) better than (0, 0).
Validation mean coherence per class: flooded vegetation 0.32, water 0.41, trees 0.51, low vegetation 0.66, crops 0.71,
bare 0.71, built 0.78.

**Results** (2 seeds each, same 12 validation sites; exp4 after `--rescore`):

| | Val mIoU (seeds) | Test mIoU | Test water |
|---|---|---|---|
| exp4 (no coherence) | 0.450 (0.455 / 0.445) | 0.395 | 0.678 |
| **exp5 with coherence** | **0.460** (0.464 / 0.456) | 0.403 | 0.727 |
| exp5, coherence zeroed (as in the prototype) | 0.415 | | |

Per class, validation IoU (mean of 2 seeds):

| | water | trees | low veg | flooded veg | crops | built | bare |
|---|---|---|---|---|---|---|---|
| exp4 | 0.538 | 0.368 | 0.372 | 0.360 | 0.355 | 0.785 | 0.371 |
| exp5 with coherence | 0.554 (+0.015) | 0.344 (−0.024) | 0.392 (+0.020) | **0.410 (+0.050)** | 0.340 (−0.015) | 0.789 (+0.004) | 0.388 (+0.017) |
| exp5, coherence zeroed | 0.516 (−0.022) | 0.334 (−0.034) | 0.362 (−0.010) | 0.258 (−0.102) | 0.323 (−0.031) | 0.780 (−0.005) | 0.333 (−0.038) |

**Conclusion against the rules: not met. The prototype keeps exp4.**
- Rule (b): mIoU is higher (+0.010, both exp5 seeds above both exp4 seeds) and no class drops more than 0.05, but
  neither built (+0.004) nor bare (+0.017) gains the required 0.03. The predicted gain did not happen.
- Second rule: with coherence zeroed the model is worse than exp4 (0.415, flooded vegetation −0.102), so training
  with coherence did not improve the radar-only predictions; the model came to rely on it, and 30% dropout was not
  enough to keep the radar-only path as good.
- What coherence did do: flooded vegetation +0.050 and water +0.015, classes that were not predicted to gain.
  Physically plausible: flooded vegetation has the lowest coherence of all classes (0.32), since water and plants
  change between two passes 12 days apart, so coherence marks it where brightness alone is ambiguous.
- Why built did not gain: it was already at 0.785 from brightness, context and spatial pattern. The proof of concept
  compared coherence with raw brightness at single pixels, not with what the full model already extracts.
- A model using coherence would need it at prediction time, which takes hours to days per place (ASF HyP3), so it is
  reported as a finding, not used in the prototype.
