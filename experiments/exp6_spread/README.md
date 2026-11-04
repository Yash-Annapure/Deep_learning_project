# Experiment 6: does each pixel's yearly variability separate trees, crops and low vegetation?

**Question:** the exp4/exp5 confusion matrices (validation) show the largest error left: of the true trees, 0.39-0.43
are predicted as low vegetation, and of the true crops, 0.32-0.35. These classes differ less in a single month than in
how they change over the year: trees (and buildings) stay steady, crops swing (bare, green, harvested), grass changes
moderately. The model sees each month plus the yearly median, which hides that. Does adding each pixel's standard
deviation of VV and VH over the site-year's months (2 more channels, `SPREAD=True`) help?

Unlike coherence, the spread is computed from the same 12 months the prototype already fetches, so it works at any
location.

**Setup:** exactly as exp4 (`hybrid_ctx_agree`: U-Net, context, AGREE), plus `SPREAD=True`. Two seeds, queue entries
30 and 31 in `run_experiments.py`. Input channels: VV, VH, median VV, median VH, spread VV, spread VH.

**Decision rule (set before the results):** averaged over 2 seeds on the 12 validation sites, against exp4 (mean of
its 2 seeds), the prototype switches to the exp6 model if:
- trees **or** crops validation IoU rises by at least **0.03** (exp4: trees 0.368, crops 0.355), and
- validation mIoU is not lower (exp4: 0.450), and
- no class drops by more than 0.05 validation IoU.

**Results:** pending.
