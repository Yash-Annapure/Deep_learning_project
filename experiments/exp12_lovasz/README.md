# Experiment 12: the Lovász-softmax loss

**Question:** the model is trained with cross-entropy but judged by IoU. The Lovász-softmax loss is a smooth stand-in
for 1 − IoU (Berman et al., CVPR 2018: +3.8 mIoU over cross-entropy; the Jaccard metric losses paper, 2023, found
gains on 4 datasets including the DeepGlobe land-cover set). Does adding it help here?

**Setup** (written 2026-11-05, before any run): the final model's setup (exp8's ResNet-50 from scratch, context,
AGREE) with `LOVASZ=True` in `02_train.ipynb`: the loss is the weighted cross-entropy as before **plus** the Lovász
term (alone it is known to train unstably), the Lovász term only on real monthly labels, the pixels that are scored.
Everything else unchanged. Queue entries 42-43 (29 train sites, 2 seeds); 44-45 the final model with it.

**Decision rule (set before the results):** the two 29-site runs are scored on the **62 `final` sites** (unseen by
them, all regions; scored as in exp8 rule 4, and the 12 validation sites proved too noisy there) next to the 29-site
runs without Lovász (exp8 scratch, 0.392). The final model with Lovász trains (44-45) if, mean of 2 seeds, it beats
them by at least **0.010** mIoU and no class drops by more than **0.05** IoU. `run_experiments.py --lovasz` applies
this and starts 44-45 itself; validation scores are reported, not used.

If it passes, the prototype and the holdout use the Lovász final model instead of exp10's (with exp11's averaging:
2 seeds, 1 view, 3-month mean), and exp10's final model is reported on the holdout next to it. If it does not, exp10's
final model stays. Either way the holdout is scored once, after this decision.

    nohup bash -c "CUDA_VISIBLE_DEVICES=0 uv run python -u run_experiments.py --only 42,43 > lovasz.log 2>&1 && CUDA_VISIBLE_DEVICES=0 uv run python -u run_experiments.py --lovasz >> lovasz.log 2>&1" > /dev/null 2>&1 &

**Results:** pending.
