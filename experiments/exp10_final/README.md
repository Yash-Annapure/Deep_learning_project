# Experiment 10: the final model, and the one-time holdout evaluation

**What it is:** after exp6-9 have chosen the setup on validation, that setup is trained once more on all the data that
is not held out, and then scored once on the 6 holdout sites. Written down before any of exp6-10's results exist.

**Training data:** 47 sites:
- train (29);
- val (12): bangweulu and gao bring kinds of place the training sites lack, and new kinds of place gave the largest
  gains in exp4;
- 6 sites with role `final`, chosen for kinds of place the data lacks and checked on satellite view: oil palm next to
  forest (Riau, Sumatra), orchards between rice fields (Mekong Delta), rice outside Asia (Lac Alaotra, Madagascar),
  irrigated fields against desert (Nile valley), steep hills (Huye, Rwanda), highland farming (Lake Tana, Ethiopia).
  `final` sites are used by this model only; exp6-9 never see them, so their comparisons stay fair.

**Setup:** the best setup by the rules of exp6-9 (exp4's if none of them passes), with `FINAL=True`: trained on
`("train", "val", "final")`, 15 epochs, and the **last epoch kept** (validation is in the training data, so it cannot
pick an epoch). Two seeds. Any change suggested by the exp6-9 pictures is made before this run; none after it.

**Commitment (made before the results):**
- The prototype uses the final model, seed 0, whatever the holdout says. The holdout chooses nothing; it only
  measures.
- The holdout is scored once (`run_experiments.py --holdout`), for: exp4 seed 0 (the current prototype, trained on
  29 sites), the chosen setup trained on 29 sites if different, and the final model (both seeds). All numbers are
  reported, including any that make the final model look worse.
- The validation scores of the final model are not reported as results (validation is in its training data).

**Results:** pending.
