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

## After: the same 6 places with the final model (2026-11-05)

**How:** `prototype.py` with the final model (exp10, 103 sites, 2 seeds averaged), the same 12 months from 2025-06,
raw monthly maps for both models (no month smoothing), so only the model differs. The earlier runs are kept in
`prototype_runs_before/` on the server.

| Place | Model vs Dynamic World, before → after | Model vs WorldCover, before → after |
|---|---|---|
| Nebraska, USA | 75% → **89%** | 57% → 64% |
| Massachusetts, USA | 72% → **89%** | 70% → **82%** |
| Picardy, France | 77% → **96%** | 72% → **90%** |
| Munich, Germany | 91% → 91% | 64% → 68% |
| Heidelberg, Germany | 86% → **93%** | 71% → 79% |
| Rio Grande do Sul, Brazil | 82% → 86% | 70% → 69% |

**The three weaknesses, month by month** (% of the 10 km square, June 2025 to May 2026):

| Place, class | Before | After |
|---|---|---|
| Massachusetts, trees | 66 62 66 63 53 51 **32** 42 56 45 59 58 | 80 80 79 79 78 76 77 77 76 74 76 78 |
| Picardy, "water" | 6 4 4 3 3 3 3 4 3 4 **31 42** | 2 2 2 2 2 2 2 2 2 2 2 4 |
| Picardy, crops | 50 41 61 56 37 37 32 30 **13** 31 36 33 | 81 80 80 80 79 79 78 78 77 80 81 80 |
| Nebraska, bare | 9 5 11 5 13 **62 77 74 85 70 66 50** | 1 0 1 0 1 3 4 3 10 3 3 2 |
| Nebraska, crops | 51 49 53 47 45 12 3 2 1 10 13 19 | 49 50 50 49 49 52 52 51 52 52 53 51 |

**Findings:**
- **All three weaknesses are gone at these places.** Leafless forest in winter stays trees (Massachusetts, December:
  32% → 77%); freshly rolled spring fields are no longer called water (Picardy, May: 42% → 4%), a confusion the
  first diagnosis had missed; harvested Great Plains fields stay crops instead of bare (Nebraska, February: bare
  85% → 10%).
- **WorldCover agrees too**, not only Dynamic World (the model's teacher): Picardy +18 points, Massachusetts +12.
- **The swings the old model showed here were false changes.** By Dynamic World's definitions a deciduous forest is
  trees and a field is crops all year, so the new model's flat curves are right. A change map must not report winter
  leaf fall as deforestation; the tropical demos (Bagre, Tonle Sap, Satkhira, Lake Debo) show that real seasonal
  change still comes through.
- **Not an unbiased score:** the new sites were chosen to fix exactly these weaknesses, and Munich and Heidelberg are
  about 40 km from a new training site. The holdout (8 sites in Europe and the Americas, scored once) gives the
  unbiased number.
