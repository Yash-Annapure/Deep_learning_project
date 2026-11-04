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

**Results:** pending.
