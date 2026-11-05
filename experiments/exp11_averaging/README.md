# Experiment 11: averaging the model's answers (no training)

**Question:** exp6-9 found no input or architecture change that helps. Does the same model get better if its class
probabilities are averaged over things that should not change the answer, before the class is picked?
- **seeds**: the 2 training runs of one setup (the luck of one run). Literature: about +0.3 to +1.5 mIoU points.
- **views**: the 8 rotations and flips of the image. The training rotates and flips every batch, so all 8 look
  familiar to the model, although radar looks sideways and a turned image is not physically the same. Literature: 0 to
  +1.5 points, sometimes nothing for a single model.
- **months**: each month averaged with the months before and after (3-month mean) where both have radar; first and
  last month unchanged. The prototype already removes one-month flickers with `despike` (hard: only where the two
  neighbours agree); the 3-month mean does the same for confident flips and also weighs how sure the model was. Aimed
  at the professor's criterion, change across months: a false change hurts that more than a slightly worse map.
  Literature: temporal models (hidden Markov models) remove spurious changes in land-cover time series.

**Setup:** `averaging.py` (written 2026-11-05, before any result). Every combination (2 x 2 x 3 = 12) is scored on
full 10 km scenes, as the prototype sees them, against the real monthly labels (pixels filled from the yearly label do
not count):
- mIoU;
- **false changes**: of the pixels whose Dynamic World class stays the same from one month with radar to the next,
  the share where the model's class changes;
- **missed changes**: of the pixels whose Dynamic World class changes, the share where the model's stays. Dynamic
  World's monthly labels flicker too (clouds), so neither rate can reach 0; they compare the options, they are not an
  error rate.

**Which models and sites decide:** the final model cannot judge itself, because the validation sites are in its
training data. So the decision is made with the two exp8 ResNet-50 models from scratch (seeds 0 and 1, the setup the
final model uses), trained on the 29 train sites, on the **12 validation and 62 `final` sites (74, all unseen by
those two models)**:

    CUDA_VISIBLE_DEVICES=0 uv run python averaging.py val,final experiments/exp8_ssl4eo/weights/hybrid_r50_ctx_agree.pt experiments/exp8_ssl4eo/weights/hybrid_r50_ctx_agree_seed1.pt

**Decision rules (set before the results), on those 74 sites:**
1. **Seeds:** the prototype averages the 2 final seeds if that beats one seed (the mean of the two seeds' own scores),
   1 view, no month smoothing, by at least **0.005** mIoU, and no class drops by more than 0.02 IoU.
2. **Views:** the prototype averages the 8 views if that beats 1 view (one seed, no month smoothing) by at least
   **0.005** mIoU, and no class drops by more than 0.02 IoU.
3. **Months:** the 3-month mean replaces `despike` (one seed, 1 view) if its mIoU is at least despike's, its
   false-change rate is lower, and its missed-change rate is at most **5 points** higher. Otherwise despike stays.

The prototype then uses what passed, together; that combination's row is reported as a check that they stack. The
script prints the verdicts with the table (`results_val_final.md`, `.png`).

**The holdout:** the final model (both seeds) is scored once on the 14 holdout sites with all 12 combinations
(`averaging.py holdout ...`), next to exp10's evaluation. The combination was chosen above, so the holdout chooses
nothing; every row is reported.

**Results** (2026-11-05, 74 sites, the two 29-site exp8 ResNet-50 models from scratch; '1 seed' = the mean of the
two seeds' own scores; `results_val_final.md` / `.png` on the server):

| Seeds · views · months | mIoU | false changes | missed changes | water | trees | low veg | flooded veg | crops | built | bare |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 seed · 1 view · none | 0.411 | 18.5% | 71.0% | 0.591 | 0.510 | 0.271 | 0.220 | 0.400 | 0.667 | 0.219 |
| 2 seeds · 1 view · none | 0.417 | 17.2% | 72.4% | 0.591 | 0.516 | 0.274 | 0.234 | 0.406 | 0.675 | 0.224 |
| 1 seed · 1 view · despike | 0.416 | 10.3% | 82.0% | 0.590 | 0.514 | 0.274 | 0.226 | 0.411 | 0.673 | 0.224 |
| 2 seeds · 1 view · despike | 0.421 | 9.6% | 82.7% | 0.590 | 0.520 | 0.276 | 0.238 | 0.416 | 0.680 | 0.228 |
| 1 seed · 1 view · 3-month mean | 0.419 | 9.5% | 83.7% | 0.587 | 0.516 | 0.276 | 0.236 | 0.419 | 0.674 | 0.225 |
| **2 seeds · 1 view · 3-month mean** | **0.425** | **9.0%** | 84.2% | 0.588 | 0.521 | 0.278 | 0.250 | 0.424 | 0.681 | 0.231 |
| 1 seed · 8 views · none | 0.414 | 17.5% | 72.1% | 0.591 | 0.513 | 0.273 | 0.223 | 0.404 | 0.673 | 0.221 |
| 2 seeds · 8 views · none | 0.419 | 16.6% | 73.1% | 0.592 | 0.518 | 0.275 | 0.234 | 0.409 | 0.679 | 0.225 |
| 1 seed · 8 views · despike | 0.418 | 9.9% | 82.6% | 0.591 | 0.517 | 0.275 | 0.227 | 0.414 | 0.678 | 0.226 |
| 2 seeds · 8 views · despike | 0.422 | 9.3% | 83.1% | 0.591 | 0.521 | 0.276 | 0.237 | 0.418 | 0.683 | 0.229 |
| 1 seed · 8 views · 3-month mean | 0.421 | 9.2% | 84.2% | 0.587 | 0.519 | 0.277 | 0.237 | 0.421 | 0.679 | 0.227 |
| 2 seeds · 8 views · 3-month mean | 0.426 | 8.7% | 84.5% | 0.588 | 0.523 | 0.279 | 0.248 | 0.425 | 0.684 | 0.232 |

**Conclusion against the rules:**
1. **Met:** 2 seeds +0.006 mIoU, no class lower (water equal).
2. **Not met:** 8 views +0.003. The prototype stays at 1 view (and 8x less work per month).
3. **Met:** the 3-month mean against despike: mIoU +0.003, false changes −0.7 points, missed changes +1.7 points
   (the guard allows 5).

**The prototype uses 2 seeds · 1 view · 3-month mean:** against what it showed before (1 seed, despike) mIoU
0.416 → 0.425 and false changes 10.3% → 9.0%; against the raw maps 0.411 → 0.425 and 18.5% → 9.0%. Removing
one-month flips is the largest single effect on change: it halves the false changes. The largest class gain is
flooded vegetation (0.220 → 0.250). Every row with month smoothing has slightly lower water IoU (−0.003): a flood
that lasts one month is smoothed away like a flicker, so the app keeps its switch for the raw maps. Missed changes
rise with any smoothing because Dynamic World's own monthly labels flicker (clouds); the rate compares the options, it
is not the share of real changes missed.
