# Experiment 7: DeepLabV3+ instead of U-Net

**Question:** does a DeepLabV3+ decoder separate the classes better than the U-Net decoder on the exp4 setup? DeepLabV3+
looks at several scales at once (atrous spatial pyramid pooling), which can help with large uniform areas, but it
predicts at a quarter of the resolution and scales up, so edges may come out softer than U-Net's. exp2 found no gain
from U-Net++ over U-Net; this tests a decoder built on a different idea.

**Setup:** exactly as exp4 (`hybrid_ctx_agree`: ResNet-34 encoder, context, AGREE), with `ARCH="DeepLabV3Plus"`. Two
seeds, queue entries 32 and 33 in `run_experiments.py`.

**Decision rule (set before the results):** averaged over 2 seeds on the 12 validation sites, against exp4 (0.450),
the prototype switches to DeepLabV3+ if validation mIoU is at least **0.010 higher** and no class drops by more than
0.05 validation IoU. Otherwise it is reported as a negative result. Edges are also compared by eye on the run's
detail pictures, but the number decides.

**Results** (validation IoU, mean of 2 seeds, the same 12 sites; test = the 3 development sites):

| | Val mIoU | water | trees | low veg | flooded veg | crops | built | bare | Test mIoU |
|---|---|---|---|---|---|---|---|---|---|
| exp4 (U-Net) | 0.450 | 0.538 | 0.368 | 0.372 | 0.360 | 0.355 | 0.785 | 0.371 | 0.395 |
| exp7 (DeepLabV3+) | 0.456 | 0.515 | 0.329 | 0.375 | 0.391 | 0.366 | 0.795 | 0.421 | 0.368 |
| change | +0.006 | −0.023 | −0.039 | +0.003 | +0.031 | +0.011 | +0.010 | +0.050 | −0.027 |

**Conclusion against the rule: not met** (+0.006, the bar is +0.010). DeepLabV3+ trades classes rather than
separating them better: bare land (+0.050) and flooded vegetation (+0.031) gain, while trees (−0.039) and water
(−0.023) lose, and the 3 test sites drop 0.027. Trees, the class that depended most on fine texture in the exp4
ablation (blurred to 20 m: −0.17), lose the most, which fits a decoder that predicts at a quarter of the resolution.
The prototype keeps the U-Net. Reported as a negative result.
