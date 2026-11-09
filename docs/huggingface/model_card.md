---
license: other
library_name: pytorch
pipeline_tag: image-segmentation
datasets:
- YashAnnapure/sentinel1-lulc-dataset
tags:
- sentinel-1
- sar
- land-cover
- semantic-segmentation
- unet
- remote-sensing
---

# Sentinel-1 monthly land cover: U-Net (ResNet-50)

The final model of the project [Monthly Land Cover from Sentinel-1 Radar](https://github.com/Yash-Annapure/Deep_learning_project)
(SRH Data Engineering 1, November 2026). It labels every 10 m pixel of a Sentinel-1 radar image with one of seven
land-cover classes, from radar alone.

## Files

| File | What it is |
|---|---|
| `hybrid_r50_ctx_agree_final.pt` | seed 0 (the prototype uses both seeds and averages their probabilities) |
| `hybrid_r50_ctx_agree_final_seed1.pt` | seed 1 |

Each file is a PyTorch `state_dict` for `segmentation_models_pytorch.Unet`.

## Input and output

- **Input:** 4 × H × W, float32, values 0 to 1, on a 10 m grid:
  1. the month's Sentinel-1 VV, median composite in dB, clipped to [-25, 5] and scaled to 0 to 1;
  2. the month's VH, clipped to [-35, -5] and scaled the same way;
  3. and 4. the median of the twelve monthly VV and VH composites (the yearly context), scaled the same way.
- **Output:** 7 class scores per pixel, in this order: water, trees, low vegetation, flooded vegetation, crops, built,
  bare.

## Use

```python
import segmentation_models_pytorch as smp
import torch

model = smp.Unet("resnet50", encoder_weights=None, in_channels=4, classes=7)
model.load_state_dict(torch.load("hybrid_r50_ctx_agree_final.pt", map_location="cpu"))
model.eval()
```

The easiest route is the project's `prototype.py`, which fetches the radar from Google Earth Engine, builds the input
exactly as in training and runs both seeds:

```bash
uv run python prototype.py --lat -0.0917 --lon 34.768 --start 2024-01
```

## Training

- U-Net with a ResNet-50 encoder, trained from scratch on 103 sites (2022 to 2025), 256 × 256 windows.
- Labels: Dynamic World monthly labels; where a month has none, the yearly label at loss weight 0.3, kept only where
  ESA WorldCover agrees.
- Weighted cross-entropy (1 / sqrt of class frequency), AdamW (learning rate 1e-3, weight decay 1e-4), cosine
  schedule, 15 epochs, random 90° rotations and flips. Last epoch kept.

## Results

Scored once on 14 holdout sites the model never saw, IoU against Dynamic World, mean of the two seeds:

| mIoU | tropics (6) | Europe and the Americas (8) | built | water | trees | low veg. | crops | bare | flooded veg. |
|---|---|---|---|---|---|---|---|---|---|
| 0.481 | 0.366 | 0.493 | 0.807 | 0.790 | 0.633 | 0.422 | 0.409 | 0.238 | 0.070 |

## Limitations

- The labels are themselves a model (Dynamic World); scores measure agreement with it, not ground truth.
- Flooded vegetation and bare ground are not reliable. There is no snow or ice class, so snowy and steep places get
  the nearest of the seven classes.
- Abrupt, subtle changes (for example a forest fire) can be diluted by the yearly context input.

## Licence

No licence is granted yet. The weights are public for the review of the course project; please ask before reusing
them.
