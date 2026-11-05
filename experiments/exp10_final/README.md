# Experiment 10: the final model, and the one-time holdout evaluation

**What it is:** after exp6-9 have chosen the setup on validation, that setup is trained once more on all the data that
is not held out, and then scored once on the holdout sites. Written down before any of exp6-10's results exist.

**Training data:** about 90 sites:
- train (29);
- val (12): bangweulu and gao bring kinds of place the training sites lack, and new kinds of place gave the largest
  gains in exp4;
- about 49 sites with role `final`, one year each, chosen for kinds of place the data lacks, screened with Dynamic
  World and checked on satellite view (`01_data_export.ipynb`, sections 13 and 14):
  - 5 in the tropics: oil palm next to forest (Riau, Sumatra), rice outside Asia (Lac Alaotra, Madagascar), irrigated
    fields against desert (Nile valley), steep hills (Huye, Rwanda), lake shore farming (Lake Tana, Ethiopia);
  - about 44 in Europe and the Americas (added 2026-11-05): all 50 earlier sites lie between 27 N and 25 S, so the
    model had never seen a temperate place, while the prototype is meant to work at any coordinates.

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

**Results:** pending.
