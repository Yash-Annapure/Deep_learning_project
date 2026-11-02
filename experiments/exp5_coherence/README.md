# Experiment 5: does InSAR coherence help the radar?

**Question:** does the exp4 model (`hybrid_ctx_agree`, 50 sites) get better when it also sees each month's InSAR
coherence? Coherence measures how much a pixel stays the same between two passes 12 days apart (coherence.py, ASF HyP3).
In the proof of concept at Pakokku and Kisumu it separated built from trees (0.83 / 0.87) far better than brightness
(~0.58), so the expected gains are in **built, bare and trees**.

**Setup:** exactly as exp4, plus `COHERENCE=True`, two seeds, queue entries 28 and 29 in `run_experiments.py`. Two more
input channels (6 in all): the month's coherence (0 where there is none) and a flag, 1 where coherence is real (option C:
a plain 0 would read as "changed completely", like water). 30% of training windows have both channels zeroed
(coherence dropout), so the model also works where coherence is missing: site-months without pairs, and the prototype,
which cannot compute coherence live. Run only once every train and val site-year has its coherence (`data/coherence/*.done`).

**Decision rule (set before the results, option (b)):** coherence is worth its extra pipeline if, averaged over 2 seeds
on the 12 validation sites, against exp4 (mean of its 2 seeds):
- built **or** bare validation IoU rises by at least **0.03** (exp4: built 0.785, bare 0.371), and
- validation mIoU is not lower (exp4: 0.450), and
- no class drops by more than 0.05 validation IoU.

Otherwise the prototype keeps the exp4 model and coherence is reported as a negative result. Stricter than exp4's rule
because coherence makes the system more complex and the prototype cannot use it live, so it has to earn a clear gain
where it was predicted.

**Results:** pending.
