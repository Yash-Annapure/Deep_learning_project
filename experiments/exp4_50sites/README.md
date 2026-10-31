# Experiment 4: the best model on all 50 sites

**Question:** does the exp3 winner (`hybrid_ctx_agree`: U-Net, context, AGREE) get better with 8 more training sites,
chosen for the weak classes (flooded vegetation: mopti, khulna, sudd; bare: jaisalmer; crops: wad_madani, gorakhpur;
built: niamey; trees: tai)?

**Setup:** exactly as `hybrid_ctx_agree` in exp3, two seeds, queue entries 26 and 27 in `run_experiments.py`. Only the
data changes: 29 training sites instead of 21. Validation now has 12 sites (bangweulu and gao added), so after the runs
`run_experiments.py --rescore` scores the 34-site models on the same 12 sites; the old numbers stay in `*_oldval`.

The 6 holdout sites (kumasi, kano, chiang_mai, gulu_rural, kayes_rural, anuradhapura_rural) are not scored here.

**Decision rule (set before the results):** the 50-site model becomes the prototype's model, and the baseline for the
coherence experiment, if averaged over 2 seeds its validation mIoU on the 12 sites is at least that of the 34-site
`hybrid_ctx_agree` runs, and no class drops by more than 0.05 validation IoU.
