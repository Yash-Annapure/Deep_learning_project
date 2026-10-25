# Experiment 2: does a different model keep more detail?

**Question:** the prototype's maps are blobby: in Ouagadougou, parks, grass strips and street trees vanish into "built"
(built 84.5% vs 72.1% in ESA WorldCover, low vegetation 0.9% vs 9.2%). Does a sharper decoder, a pretrained encoder,
or a less noisy input fix that?

**Setup:** the same data and training as `hybrid` in experiment 1 (all 4 years, 15 epochs); only the model or its
input changes. Each variant twice (seeds 0 and 1). Queue entries 8 to 17 in `run_experiments.py`.

| | Plain U-Net | U-Net++ | ImageNet U-Net |
|---|---|---|---|
| **No context** | `hybrid`, `hybrid_seed1` (exp1, baseline) | `hybrid_upp` | `hybrid_imnet` |
| **+ context** (yearly median radar as 2 extra channels) | `hybrid_ctx` | `hybrid_upp_ctx` | `hybrid_imnet_ctx` |

**Decision rule (set before the results):** a variant wins if all three hold:
1. **Eye:** in the detail check, tree lines and small patches are visibly clearer and less blobby, in **both** seeds.
2. **Trees IoU** (test), averaged over the 2 seeds, beats the baseline by more than about 0.015 (the gap between two
   seeds of the same run; below that it is luck).
3. **mIoU** does not drop.

**Visual check:** the notebook's "Detail check" cell shows a 4 km piece of each test site: radar, label, baseline,
this run. Saved in `runs/<name>/`.

**Results** (mean of 2 seeds, test sites unless marked):

| Variant | Test mIoU | Water | Trees | Low veg | Val mIoU | Cost |
|---|---|---|---|---|---|---|
| Baseline (`hybrid`) | 0.335 | 0.420 | 0.322 | 0.451 | 0.379 | 1x |
| **+ context** | **0.367** | **0.684** | 0.343 | 0.418 | 0.380 | 1x |
| U-Net++ | 0.324 | 0.281 | 0.333 | 0.419 | 0.381 | 2x |
| U-Net++ + context | 0.372 | 0.655 | 0.312 | 0.435 | 0.386 | 2x |
| ImageNet | 0.312 | 0.277 | 0.345 | 0.435 | 0.375 | 1x |
| ImageNet + context | 0.328 | 0.388 | - | - | 0.391 | 1x |

**Conclusions**
- No variant passes the rule for trees: architecture does not fix the blobby maps. Low vegetation improves nowhere.
- Context is the big win **on the test sites**, for water: 0.42 to 0.62-0.75 test IoU in every plain and U-Net++
  context run, mIoU +0.03. The likely reason: the yearly median separates permanent water from surfaces that are dark
  only in some months (smooth dry soil).
- **Correction (found after the rescore, 2026-10-25):** on the 10 validation sites, context changes nothing (mIoU
  0.379 -> 0.380, water 0.57 -> 0.56). The gain is a test-site effect, and context was chosen as the base after
  looking at test scores, so for context the test sites are no longer an untouched final exam. Confirming it needs
  test sites that were never looked at.
- U-Net++ adds nothing worth its 2x cost; ImageNet is unstable (one seed's water IoU 0.110, the other 0.443).
- Single-class test scores on 3 sites jump a lot between seeds (much more than the 0.015 assumed), so from exp3 on,
  per-class validation scores on 10 sites are saved too.
- Validation curves flatten around epoch 10 of 15 while the learning rate is still about 15-25% of its start, so
  longer training is not the lever.
- Plain U-Net + context becomes the base for exp3.

Folders: `results/` scores, `runs/` pictures and logs (both in git), `weights/` models (not in git).
