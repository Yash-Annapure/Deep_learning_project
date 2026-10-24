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

**Decision rule (set before the results):** a variant wins if, averaged over its 2 seeds, it beats the baseline on
**test low-vegetation IoU and trees IoU** without mIoU dropping, by more than about 0.015 (the gap between two seeds
of the same run). Below that it is luck.

**Visual check:** the notebook's "Detail check" cell shows a 4 km piece of each test site: radar, label, baseline,
this run. Saved in `runs/<name>/`.

**Results:** pending.

Folders: `results/` scores, `runs/` pictures and logs (both in git), `weights/` models (not in git).
