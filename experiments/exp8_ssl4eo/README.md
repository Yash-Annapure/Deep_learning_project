# Experiment 8: an encoder pretrained on Sentinel-1 (SSL4EO-S12)

**Question:** exp2 found that ImageNet weights (photos) did not help. Does an encoder pretrained on Sentinel-1 itself
help? SSL4EO-S12 is a ResNet-50 trained self-supervised (MoCo) on millions of Sentinel-1 VV/VH images, so it has
already learned what radar textures look like (weights: torchgeo `ResNet50_Weights.SENTINEL1_GRD_MOCO`).

**Setup:** the exp4 setup (U-Net, context, AGREE), with a ResNet-50 encoder in two versions, two seeds each (queue
entries 34-37 in `run_experiments.py`):
- **ResNet-50 from scratch**: the control. Without it, "pretraining helped" and "a bigger encoder helped" (ResNet-50
  vs exp4's ResNet-34) could not be told apart.
- **ResNet-50 from SSL4EO-S12** (`SSL4EO=True`):
  - the input is turned back into dB and normalised as in pretraining (mean −12.59 / −20.26 dB, std 5.26 / 5.91 for VV / VH);
  - the first layer was pretrained on 2 channels; the yearly median channels get a copy of its VV/VH filters, and all 4 are halved to keep the signal strength;
  - the encoder learns at a tenth of the learning rate (fine-tuning, so 15 epochs do not wash out what pretraining
    learned), the decoder at the full rate.

**Decision rules (set before the results), averaged over 2 seeds on the 12 validation sites:**
1. **Does Sentinel-1 pretraining help?** Yes if ResNet-50 SSL4EO beats ResNet-50 from scratch by at least **0.020**
   validation mIoU.
2. **Prototype:** it switches to ResNet-50 SSL4EO if that model beats exp4 (0.450) by at least **0.010** validation
   mIoU and no class drops by more than 0.05 validation IoU.
3. **Added after seeing one result** (ResNet-50 scratch, seed 0: 0.467), before scratch seed 1 and both SSL4EO runs
   were known: rules 1-2 did not say what happens if the control itself is the best model. ResNet-50 from scratch
   becomes the prototype's model if it beats exp4 (0.450) by at least **0.010** validation mIoU and no class drops by
   more than 0.05. If it and SSL4EO both pass, the higher validation mIoU wins. Since this rule was written after one
   seed was seen, it is mildly optimistic; the holdout shows whether the choice holds.
4. **Tiebreak on fresh sites** (added after all 4 runs, before the 6 `final` sites were exported; it replaces the
   last sentence of rule 3): see the conclusion for why. Both setups (the 29-site models, 2 seeds each) are scored
   once on all `final` sites, which no model has seen (`run_experiments.py --tiebreak`): 5 in the tropics and, added
   on 2026-11-05 before anything was scored, about 59 in Europe, the Americas, East Asia, Australia, southern Africa and Anatolia, where neither model has seen any
   place at all. The setup with the higher mean mIoU there trains the final model. The sites then join the final
   model's training data; their scores are reported, and the holdout sites stay untouched.

**Results** (mean of 2 seeds; validation = the same 12 sites; test = the 3 development sites):

| | Val mIoU | water | trees | low veg | flooded veg | crops | built | bare | Test mIoU | Test water |
|---|---|---|---|---|---|---|---|---|---|---|
| exp4 (ResNet-34) | 0.450 | 0.538 | 0.368 | 0.372 | 0.360 | 0.355 | 0.785 | 0.371 | 0.395 | 0.678 |
| ResNet-50, scratch | 0.461 | 0.539 | 0.362 | 0.384 | 0.404 | 0.361 | 0.795 | 0.380 | 0.390 | 0.685 |
| ResNet-50, SSL4EO | 0.467 | 0.562 | 0.364 | 0.396 | 0.453 | 0.334 | 0.793 | 0.366 | 0.330 | 0.393 |

Test water per seed: scratch 0.720 / 0.649, SSL4EO 0.264 / 0.522.

**Conclusion against the rules:**
1. **Not met** (+0.006, the bar is +0.020): pretraining on Sentinel-1 adds little over the same encoder from scratch.
2. **Met**: SSL4EO +0.017 over exp4; the largest drop is crops −0.021.
3. **Met**: scratch +0.011 over exp4; the largest drop is trees −0.006. So the gain over exp4 comes mostly from
   the larger encoder, which the scratch control was there to show.

Both pass, and the rule-3 tiebreak would pick SSL4EO by 0.006, less than the gap between scratch's own two seeds
(seed 0 is tied, 0.467 vs 0.468). On the 3 test sites, both SSL4EO seeds score below both scratch seeds, mostly on
water (0.26 / 0.52 vs 0.72 / 0.65), the class the prototype's flood examples depend on. A difference of 0.006 should
not decide that, and the test sites have been looked at too often to decide it either, so rule 4 lets new data decide.
The flooded vegetation gain from SSL4EO (+0.093 over exp4) is the largest single-class gain of exp6-9.

Tiebreak (rule 4): pending.
