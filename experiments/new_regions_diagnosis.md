# Where the model is weak outside the tropics (diagnosis, 2026-11-05, no training)

**Why:** all 50 training/validation/test/holdout sites lie between 27 N and 25 S (Africa, South/South-East Asia), while
the prototype is meant to work at any coordinates. Before choosing sites for Europe and the Americas, the best model so
far was run on 6 places it has never seen, plus one familiar place, to find out *what* goes wrong there, so the new
sites can target it (as in exp4, where sites aimed at the weak classes gave the largest gain).

**How:** `prototype.py` with `MODEL_WEIGHTS=experiments/exp8_ssl4eo/weights/hybrid_r50_ctx_agree.pt` (ResNet-50 from
scratch, seed 0: validation 0.467, test 0.398), 12 months from 2025-06. The model's yearly map (each pixel's most
frequent class; crops if crops in 3+ months) is compared with Dynamic World's yearly map (the source of our training
labels) and with ESA WorldCover 2021 (a second opinion). These are not the holdout sites, which stay untouched.
Grand Island (Nebraska) and Santa Maria (Rio Grande do Sul) sit on towns.

| Place | Seen in training? | Model vs Dynamic World | Model vs WorldCover | Dynamic World vs WorldCover |
|---|---|---|---|---|
| Gezira, Sudan (reference) | similar place (wad_madani) | 88% | 85% | 95% |
| Munich, Germany | no | 91% | 64% | 68% |
| Heidelberg, Germany | no | 86% | 71% | 80% |
| Rio Grande do Sul, Brazil | no | 82% | 70% | 73% |
| Picardy, France | no | 77% | 72% | 90% |
| Nebraska, USA | no | 75% | 57% | 67% |
| Massachusetts, USA | no | 72% | 70% | 83% |

Largest mix-ups against Dynamic World (labels -> model, % of the 10 km square):

| Place | Mix-ups |
|---|---|
| Massachusetts | trees -> low vegetation 15.7; trees -> built 2.6; crops -> low vegetation 2.4 |
| Picardy | crops -> low vegetation 14.7; crops -> trees 3.6 |
| Nebraska | built -> bare 8.6; crops -> bare 4.9; low vegetation -> bare 2.7 |
| Heidelberg | trees -> low vegetation 6.9; trees -> built 2.5 |
| Rio Grande do Sul | trees -> low vegetation 4.7; trees -> built 3.7 |
| Munich | trees -> built 4.2 |
| Gezira (reference) | crops -> low vegetation 9.5 |

**Findings:**
- **Not broken outside the tropics.** Against its own label source the model scores 72-91% at places it has never
  seen, against 88% at the familiar one. Munich and Heidelberg are as good as Gezira.
- **"Trees called built" in towns is a difference between the two reference maps, not a model error.** Against
  WorldCover the model calls many trees in towns "built" (Munich 30% of the square), but where it does, Dynamic World
  says "built" too (68-94% of those pixels, 45% in Massachusetts): Dynamic World labels leafy streets and gardens as
  built, WorldCover as trees. The model learned its labels; more data would not change this. Reported, not targeted.
- **Three real weaknesses**, where the model disagrees with its own labels:
  1. **trees -> low vegetation** in temperate forest (Massachusetts 16%, Heidelberg 7%): most likely leafless
     deciduous forest in winter, which the training data (tropics) never had;
  2. **crops -> low vegetation** (Picardy 15%; also Gezira 10%, so partly a general weakness): winter cereals are green
     for months, like grass;
  3. **-> bare** on the dry Great Plains (Nebraska: a low-density town 9%, crops 5%): harvested fields and sparse towns
     on dry soil.

**Consequence for the site list** (`01_data_export.ipynb`, section 14): a block of sites aimed at these three goes
first after the holdout sites: deciduous forest (Spessart, Pennsylvania, Vermont), winter cereals and spring wheat
(East Anglia, North Dakota), Great Plains rangeland and a plains town (Nebraska Sandhills, Lubbock). No extra sites for
"trees called built". Tropical America moves to the end of the priority order: the model already knows tropical kinds
of place.
