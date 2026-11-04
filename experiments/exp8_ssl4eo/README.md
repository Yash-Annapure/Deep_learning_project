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

**Results:** pending.
