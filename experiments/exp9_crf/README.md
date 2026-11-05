# Experiment 9: a CRF after the model (post-processing, no training)

**Question:** does a conditional random field (CRF) after the model separate the classes better? A CRF pushes
neighbouring pixels towards the same class unless an image shows an edge between them, which on optical images
sharpens class boundaries. On radar the month's image is full of speckle, so its "edges" are everywhere; the CRF is
therefore guided by the **yearly median radar** (already an input channel of the exp4 model), where 12 months
average the speckle away and real boundaries (fields, shores, rivers) remain.

**Method:** a convolutional CRF (Teichmann & Cipolla 2018, "Convolutional CRFs for semantic segmentation"):
mean-field inference of a fully connected CRF restricted to a k x k window, written in PyTorch (`lulc.conv_crf`), so
no extra package (pydensecrf does not build on our Python). Potts model with an appearance kernel (distance and
difference in the median radar) and a smoothness kernel (distance only), 5 iterations. Applied to the exp4 model
(`hybrid_ctx_agree`, seed 0) with `run_experiments.py --crf`.

Synthetic check (noisy predictions on two regions split by an edge in the guide): the CRF removes the noise and keeps
the boundary; with a guide that has no edge, 12.5% of the boundary pixels go wrong, so the guide is what keeps real
boundaries.

**Decision rule (set before the results):** two settings, fixed in advance and not tuned on validation:
mild (window 7 px, theta_xy 3, theta_i 0.10) and strong (window 11 px, theta_xy 5, theta_i 0.10). The prototype uses
the CRF if one of them raises validation mIoU by at least **0.010** over the same model without it, with no class down
by more than 0.05. Picking the better of two settings on validation is mildly optimistic; the holdout score shows
whether it holds.

**Results** (validation, the 12 sites, exp4 `hybrid_ctx_agree` seed 0; `crf_hybrid_ctx_agree.md`):

| CRF | mIoU | water | trees | low veg | flooded veg | crops | built | bare |
|---|---|---|---|---|---|---|---|---|
| none (argmax) | 0.455 | 0.547 | 0.332 | 0.394 | 0.397 | 0.357 | 0.791 | 0.364 |
| mild (7 px, theta_xy 3) | 0.456 | 0.544 | 0.327 | 0.396 | 0.402 | 0.358 | 0.793 | 0.371 |
| strong (11 px, theta_xy 5) | 0.454 | 0.536 | 0.320 | 0.396 | 0.403 | 0.357 | 0.793 | 0.375 |

**Conclusion against the rule: not met** (+0.001 and −0.001; the bar is +0.010). The CRF smooths, but the model's
output is already about as smooth as the median radar's boundaries allow: bare and flooded vegetation gain a little,
trees and water lose a little, and the stronger setting moves further in the same direction. The prototype uses no
CRF. Temporal post-processing (removing one-month flickers) is what the prototype uses instead.
