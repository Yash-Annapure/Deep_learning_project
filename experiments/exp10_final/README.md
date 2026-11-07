# Experiment 10: the final model, and the one-time holdout evaluation

**What it is:** after exp6-9 have chosen the setup on validation, that setup is trained once more on all the data that
is not held out, and then scored once on the holdout sites. Written down before any of exp6-10's results exist.

**Training data:** 103 sites:
- train (29);
- val (12): bangweulu and gao bring kinds of place the training sites lack, and new kinds of place gave the largest
  gains in exp4;
- 62 sites with role `final`, one year each, chosen for kinds of place the data lacks, screened with Dynamic
  World and checked on satellite view (`01_data_export.ipynb`, sections 13 and 14):
  - 5 in the tropics: oil palm next to forest (Riau, Sumatra), rice outside Asia (Lac Alaotra, Madagascar), irrigated
    fields against desert (Nile valley), steep hills (Huye, Rwanda), lake shore farming (Lake Tana, Ethiopia);
  - about 51 in Europe and the Americas (added 2026-11-05; the first 7 aimed at the weaknesses in
    `experiments/new_regions_diagnosis.md`): all 50 earlier sites lie between 27 N and 25 S, so the
    model had never seen a temperate place, while the prototype is meant to work at any coordinates;
  - 7 in East Asia, Australia, temperate southern Africa and Anatolia, where the map was still empty after that.

  `final` sites are trained on by this model only. Before that they are scored once, to break the exp8 tie
  between ResNet-50 from scratch and from SSL4EO (exp8 rule 4); no other decision uses them.

**No training before this data is in:** the final model, and the exp8 tiebreak, run only after every `final` and new
holdout site has been exported.

**Setup:** the best setup by the rules of exp6-9 (exp4's if none of them passes), with `FINAL=True`: trained on
`("train", "val", "final")`, 15 epochs, and the **last epoch kept** (validation is in the training data, so it cannot
pick an epoch). Two seeds. Any change suggested by the exp6-9 pictures is made before this run; none after it.

**Commitment (made before the results):**
- The prototype uses the final model, seed 0, whatever the holdout says. The holdout chooses nothing; it only
  measures.
- The holdout is scored once (`run_experiments.py --holdout`), for: exp4 seed 0 (the current prototype, trained on
  29 sites), both exp8 ResNet-50 setups trained on 29 sites (the one exp8 rule 4 did not pick included, so the
  report shows whether the tiebreak was right), and the final model (both seeds). All numbers are reported, including
  any that make the final model look worse.
- The holdout has 14 sites: the first 6 (tropics, chosen 2026-10-27) and 8 in Europe and the Americas (4 cities,
  4 countryside, section 14). Both groups are reported separately as well as together, so the gain from the new
  regions' training data shows where it was aimed.
- The validation scores of the final model are not reported as results (validation is in its training data).
- **Added 2026-11-05, before the final model finished and before any holdout score:** exp11 decides, with the 29-site
  models, whether the prototype averages the final model's 2 seeds, 8 views of the image and neighbouring months. If
  it does, the prototype uses the final model (seed 0, plus seed 1 if exp11 rule 1 passes) with what exp11 keeps,
  instead of seed 0 alone; the holdout is then also scored with `averaging.py holdout` (all 12 combinations reported).
- **Added 2026-11-05, same conditions:** if exp12's rule passes, the final model with the Lovász loss (exp12, entries
  44-45) takes this model's place in the prototype; this model is still scored on the holdout next to it.

**Results** (holdout, scored once on 2026-11-07, `run_experiments.py --holdout`, windows as in validation; 14 sites:
6 tropics, 8 Europe and the Americas; mean of 2 seeds where a setup has two):

| Model | Sites trained on | mIoU | tropics (6) | new regions (8) | water | trees | low veg | flooded veg | crops | built | bare |
|---|---|---|---|---|---|---|---|---|---|---|---|
| exp4 (the first prototype model, ResNet-34) | 29 | 0.433 | 0.357 | 0.437 | 0.743 | 0.568 | 0.364 | 0.037 | 0.339 | 0.793 | 0.189 |
| exp8 ResNet-50 from scratch | 29 | 0.423 | 0.366 | 0.414 | 0.720 | 0.558 | 0.325 | 0.042 | 0.334 | 0.788 | 0.194 |
| exp8 ResNet-50 SSL4EO | 29 | 0.414 | 0.338 | 0.423 | 0.695 | 0.532 | 0.370 | 0.032 | 0.312 | 0.799 | 0.156 |
| **exp10 final model** (ResNet-50 from scratch) | **103** | **0.481** | 0.366 | **0.493** | 0.790 | 0.633 | 0.422 | 0.070 | 0.409 | 0.807 | 0.238 |

Seeds: final 0.481 / 0.481, scratch 0.420 / 0.425, SSL4EO 0.414 / 0.414, exp4 seed 0 only.

**What the holdout says:**
- **The new sites did the work.** Same setup, 29 → 103 sites: mIoU 0.423 → 0.481 (+0.058). In the new regions
  0.414 → 0.493 (+0.080), where the 62 sites were aimed; in the tropics 0.366 → 0.366, unchanged, not hurt.
- **The final model is the best on every class**, against every 29-site model (largest gains: crops +0.075, trees
  +0.075, low vegetation +0.097 over the 29-site setup). Its two seeds agree to the third decimal.
- **Against the first prototype model (exp4): +0.048** overall, +0.056 in the new regions, +0.009 in the tropics.
- **The exp8 tiebreak held overall:** from scratch 0.423 against SSL4EO 0.414, though by region it is mixed
  (tropics: scratch +0.028; new regions: SSL4EO +0.009).
- **Not confirmed:** exp8's gain of ResNet-50 over ResNet-34 (+0.011 on validation). On the holdout both 29-site
  ResNet-50 models score below exp4 (0.423 and 0.414 against 0.433). The final model's gain over exp4 therefore
  comes from the data, not from the larger encoder.
- **Weak spots that remain:** flooded vegetation (0.070) and bare ground (0.238); the tropical holdout sites score
  lower than the new regions (0.366 against 0.493) for every model, so they are harder sites, not a regression.

The prototype uses the final model as committed (both seeds averaged, 3-month mean; exp11). Its holdout score with
that averaging, on full scenes, is in `experiments/exp11_averaging/results_holdout.md`.
