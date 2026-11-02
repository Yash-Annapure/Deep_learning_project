# Prototype demo: notes for the meeting

## What it does
- **You give it:** a latitude, a longitude and a start month.
- **It gives back:** 12 monthly land-cover maps of a 10 km × 10 km square around that point, the share of each class per month, and how much each class changed, month to month and from the first to the last month.
- **Input data:** Sentinel-1 radar only. Radar sees through clouds, so rainy-season months work too.
- **7 classes:** water, trees, low vegetation (grass and shrub merged), flooded vegetation, crops, built, bare.
- **Model:** a U-Net trained on 50 sites in Africa and Asia (29 used for training). It learned from Dynamic World labels, Google's land-cover map made from optical images. Validation score: mIoU 0.45 on 12 sites it never saw.
- **Speed:** a new place takes about 1 minute (downloading the radar is most of it). A place that has run before opens instantly.

## What is on the screen, top to bottom

| Part | What it shows | How to read it |
|---|---|---|
| **Summary row** | The 3 classes that changed most from the first to the last month, plus how many months had radar | "Water: 18% of the area, −13.6 points" means water covered 13.6 percentage points less of the square at the end than at the start |
| **Month slider** | The map for one month; **▶ Play** runs through all 12 | Scrub through to see seasons: floods spreading and drying, fields being sown |
| **Magenta highlight** | Pixels whose class changed since the previous month | Big connected areas (a lake edge moving) are real change; scattered dots are mostly radar noise |
| **% pixels changed** | Share of the square that changed class | Usually larger than the change in shares: 1% trees→crops plus 1% crops→trees is 2% of pixels changed but 0 points of net change |
| **Trends tab** | Line chart and tables of the share per class, monthly change and total change | Numbers are in percentage points |
| **Compared with other maps tab** | Our map for the whole year next to Dynamic World (our teacher) and ESA WorldCover 2021, with 3 agreement numbers | See below |
| **All 12 months tab** | All months at once, small | Quick overview |

Months with too little radar coverage are left out (shown grey), and changes are then measured against the last month that had data. A half-covered month would otherwise show changes that never happened.

## The comparison with other maps
- **Three maps:** ours (radar only), Dynamic World (optical, the labels we trained on), WorldCover 2021 (optical + radar).
- **Same rule for all three:** each pixel's most frequent class over the year, but **crops if crops in 3 or more months**. That's WorldCover's definition of cropland ("planted at least once a year"). Fields carry a crop only part of the year, so a plain majority vote would call them bare or grass.
- **What the numbers say:** our map agrees best with Dynamic World, its teacher. The two optical maps also disagree with each other a lot:

| Place | Ours vs Dynamic World | Ours vs WorldCover | Dynamic World vs WorldCover |
|---|---|---|---|
| Gezira (Sudan) | 97% | 94% | 95% |
| Bagré reservoir (Burkina Faso) | 73% | 55% | 47% |
| Tonle Sap (Cambodia) | 56% | 37% | 40% |

  Gezira's numbers are high mainly because ~95% of the square is crops: any map that says "crops" everywhere would
  score about that. The useful check there is the small features: the model finds most villages in the right places,
  but misses a tree patch and the thin tree lines along the canals.

  None of the three maps is the truth. In seasonal wetlands, two established maps agree on only 40% of pixels.
  At Bagré, our map agrees with WorldCover more (55%) than its own teacher does (47%): it doesn't copy some of
  Dynamic World's local mistakes (for example small "built" blobs on bare soil south of the river).

## Demo places (12 months from June 2025)

| Place | What changes | What it shows |
|---|---|---|
| Lake Debo, Mali | water −13.6 points, low vegetation +11.3 | The Niger flood plain filling and drying |
| Tonle Sap, Cambodia | flooded vegetation −21.8 points | The lake swelling into the flooded forest and pulling back |
| Gezira, Sudan | crops +54 points, bare −52 | Irrigated fields from bare soil to crops |
| Bagré reservoir, Burkina Faso | water +5.8, bare −5.9 | Reservoir level and the fields around it |
| Satkhira, Bangladesh | water −7.8, flooded vegetation +7.2 | Shrimp ponds: radar confuses shallow ponds with flooded vegetation |
| Nairobi, Kenya | every class under 1 point | **Control:** a city that shouldn't change, and doesn't |

## Limits we know about (and will discuss in the report)
- **Change ≠ land-use change.** Seasons show up as change: harvested fields look bare, the wet season floods land. Classification errors also add to the change percentages.
- **Radar confuses some classes:** crops vs grass (similar echo), flooded rice paddies vs water, shallow ponds vs flooded vegetation.
- **Towns look too big and blobby:** strong echoes from buildings cover the yards between them. Thin features (roads, dams) disappear.
- **Coverage gaps:** since Sentinel-1B failed (Dec 2021), some places have months, or whole years, without radar. Coverage is best from 2025, when Sentinel-1C joined.
- **All scores measure agreement with Dynamic World,** which is itself a model, not ground truth.

## In progress
- **InSAR coherence experiment (done):** coherence measures how stable a pixel stays between two radar passes 12 days apart. We predicted it would help built and bare, and wrote the decision rule down before training (built or bare +0.03).
  - With coherence: validation mIoU 0.450 → **0.460** (both seeds better), but the gain came in **flooded vegetation (+0.05)** and water, not built (+0.004) or bare (+0.017). **Rule not met.**
  - With coherence left out (as in the prototype, since it takes hours to days to compute for a new place): 0.415, worse than without it.
  - So the prototype keeps the radar-brightness model, and coherence goes into the report as a finding: it marks flooded vegetation (the class with the lowest coherence, 0.32), but it's needed at prediction time.
- **6 untouched holdout sites,** scored once at the end, for the final unbiased number.

## Questions for the professor
1. Is comparing against Dynamic World (our teacher) and WorldCover side by side a fair way to show quality?
2. Is the "crops if crops in 3+ months" rule for the yearly map acceptable?
3. Are percentage points per class the right way to report change, or would you also like a "% of pixels changed" figure in the report?
4. Coherence helps only when it's available (flooded vegetation +0.05). Is reporting it as a finding, not using it in the prototype, the right call?
5. Which kind of places would you like to see in the demo video?
