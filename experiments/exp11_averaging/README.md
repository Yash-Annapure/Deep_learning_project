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

**Results:** pending.
