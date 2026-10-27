# Report notes

Running list of points to use in the report. Add to it as we go.

## Date matching between radar and labels (dataset and preprocessing section)

- Accepted time windows in the literature range from about 30 days to a whole season, so a monthly radar window sits inside normal practice.
  - SEN12MS: pairing window is a meteorological season (about 3 months). Sentinel-1 was exported as-is (not affected by clouds); Sentinel-2 was a cloud-free mosaic over the season; labels were annual MODIS maps (2016). Source: Schmitt et al. 2019, https://arxiv.org/pdf/1906.07789
  - SSL4EO-S12: 30-day interval around four reference dates (Mar 20, Jun 21, Sep 22, Dec 21). Kept a location only if Sentinel-2 had under 10% cloud and a matching Sentinel-1 image existed for all four dates. Source: Wang et al. 2023, https://arxiv.org/pdf/2211.07044
- A yearly label is looser in time than what SEN12MS used for imagery, but it is in line with SEN12MS's annual MODIS labels.
- SSL4EO-S12 handled clouds by discarding cloudy locations (its own limitations section lists "geographical bias due to cloud filtering"). We cannot do that, because the project must work in cloudy, data-poor regions.

Draft sentence: "Unlike SSL4EO-S12, which discards locations without cloud-free optical coverage, we keep cloudy tropical sites, since the model is meant to work in exactly those regions."

## Main variant comparison (learner's choice, 2026-10-21)

Hypothesis from the learner: hybrid labels (monthly label where it exists, yearly label as a lower-weight fill) give a better model than monthly labels only.
- Variant A: monthly labels only (filled pixels set to ignore). Variant B: hybrid labels. Everything else identical (same encoder from scratch, same split, same training setup).
- Score both on the same test pixels, real monthly labels only, otherwise the hybrid model benefits from the yearly fill. Also break scores down by month, especially wet-season months (note: test labels are scarce there, so those scores rest on thin data).
- The result may go either way; both outcomes are reportable.
- Learner's prediction, written before running: monthly-only will score higher overall (hybrid may hide problems not yet visible); the biggest difference will be in harsh-weather months (monsoon, rain), where missing or misidentified data shows up. Claude's note: monthly-only gets few labels in those months, so a mixed result (hybrid better in the rainy season, monthly-only better elsewhere) is also plausible.

## Second experiment: time context (variant C), rule written before running (2026-10-22)

Motivation: the first hybrid model labels many smooth non-water surfaces (bare ground, dry fields, grass) as water (only 7% of predicted water was truly water on test; false water was clustered, darker than its surroundings). Learner's idea: give the model the year's context, because vegetation and soil change with the seasons while water stays dark.
- Measurement behind it (training sites, VV): yearly median is -14.0 dB for water vs -7.6 to -11.4 dB for other classes; a single threshold on the yearly median catches 81% of water with 11% false alarms, a single month catches 59% with 18%. Water's month-to-month variability is actually higher (2.9 dB vs 1.3-2.0), so the useful signal is the typical level, not stability.
- Variant C: hybrid labels plus 2 extra channels (yearly median VV, VH), 4 input channels. Same labels, model and settings otherwise.
- Success rule, judged on VALIDATION sites (the test sites are only read once at the end; the first test results were seen once before any change): worked = water IoU up by at least ~0.05 points AND clearly more than the run-to-run spread, and mIoU not lower. Didn't help = gain within the run-to-run spread. Worse = water IoU or mIoU drops by more than the spread. Spread is measured by training each setup with two seeds.
- Prototype consequence: compute the 12 monthly images, then their median; any 12-month window contains each calendar month once, so it matches training.
- Trade-off to mention: the yearly median is the same for all 12 months, so it may stabilise predictions (fewer fake seasonal changes) but could damp real month-to-month change.

### Result of the time-context experiment (validation sites, 2 seeds each)

| | mIoU | water IoU | water precision |
|---|---|---|---|
| baseline (hybrid), seeds 0 / 1 | 0.273 / 0.273 | 0.474 / 0.409 | 0.587 / 0.516 |
| + yearly median context, seeds 0 / 1 | 0.300 / 0.289 | 0.464 / 0.405 | 0.661 / 0.511 |

Verdict (learner's reading, matches the numbers): a small improvement in mIoU (about +0.02, higher on both seeds), nothing clear on water (difference about -0.007, smaller than the seed-to-seed spread of about 0.06). By the rule written beforehand the idea did not clearly help on its target. Caveat: on validation sites the baseline already has decent water IoU (~0.44), unlike the test sites (0.065), so validation could not test the false-water problem well. The failure shows on dry test regions (Sahel, northern Ghana, Nashik) that the validation sites do not represent; this is a finding about generalisation to unlike regions for the critical discussion.

## Planned next changes (decided 2026-10-22)

Professor's feedback: data quality and quantity first; look at newer architectures; sharpening/scaling the inputs; watch imbalance and coverage; synthetic data for under-covered cases.
Order agreed: (1) expand real data: dry-region sites from other regions than the test sites (Dodoma, Kitui, Solapur, Pakokku, Khon Kaen, Adama, town plus rural square each) and years 2022, 2023, 2025 for training sites (2026 later, incomplete year); (2) label-quality filter using ESA WorldCover agreement (2021 only, partly S1-based, so not the independent case-study map); (3) after the data results: label-safe synthetic variation (random dB brightness jitter, re-sampled speckle noise, oversampling of dry-site and false-water patches); (4) architecture comparison (U-Net++, Segformer, FPN are one-line swaps in segmentation_models_pytorch) and a spatial speckle filter test.
Why not generative synthetic radar: the failure is false water on dry non-water ground, so the missing examples are dry smooth non-water surfaces; a generator would need real dry-region data to learn from and may add artefacts. Mention as future work.

## Result of the expanded dataset (2026-10-23): first run on 21 train sites x 4 years

Setup: hybrid labels, 2 input channels, U-Net ResNet-34 from scratch, 15 epochs (about 49,000 training patches, about 45,000 steps; the old runs had about 5,300 patches and 30 epochs, about 10,000 steps), seed 0, one run. Validation and test sites unchanged (4 and 3 sites, 2024).

| | old hybrid (9 sites, 2024), range of 5 old runs | **new hybrid (21 sites, 2022-2025)** |
|---|---|---|
| validation mIoU | 0.260 to 0.300 | **0.350** |
| validation water IoU | 0.405 to 0.474 | **0.566** |
| validation water precision | 0.511 to 0.661 | **0.910** |
| test OA | 0.541 to 0.648 | **0.688** |
| test mIoU | 0.211 to 0.253 | **0.335** |
| test water IoU | 0.065 to 0.105 | **0.423** |
| test flooded vegetation IoU | 0.009 to 0.035 | 0.102 |
| test low vegetation IoU | 0.200 to 0.411 | 0.456 |
| test built IoU | 0.765 to 0.814 | 0.813 |
| test crops IoU | 0.065 to 0.204 | 0.163 (no gain) |
| test bare IoU | 0.004 to 0.071 | 0.073 |

Reading: the false-water failure on the dry test regions is largely fixed by adding dry-region training sites and more years (test water IoU about 4x). Caveats to state: one seed (old runs had a spread of about 0.06 on water IoU and 0.04 on test mIoU, but this gain is larger than any of that); the change bundles more sites, more years, more steps and fewer epochs, so it cannot say which part helped (planned ablation: the same 21 sites with YEARS=(2024,)); the test numbers are for reporting, the validation numbers also improved; rare classes (flooded vegetation, bare) and crops are still weak, consistent with the low agreement between Dynamic World and WorldCover for those classes.

## Sites against years, label filter and radar filters (queue results, 2026-10-24)

All runs: hybrid labels, 2 input channels, U-Net ResNet-34 from scratch, 15 epochs unless noted. Noise level of the 4-year setup from 2 seeds: test mIoU 0.335 / 0.335, validation mIoU 0.350 / 0.362, test water IoU 0.423 / 0.417.

| run | val mIoU | test mIoU | test OA | test water IoU | test bare IoU | test crops IoU |
|---|---|---|---|---|---|---|
| old runs, 9 sites 2024 (range of 5) | 0.26 to 0.30 | 0.21 to 0.25 | 0.54 to 0.65 | 0.065 to 0.105 | 0.004 to 0.071 | 0.07 to 0.20 |
| 21 sites, 2024 only, 2 seeds | 0.326 / 0.337 | 0.350 / 0.335 | 0.682 / 0.693 | 0.467 / 0.382 | 0.272 / 0.247 | 0.229 / 0.144 |
| 21 sites, 4 years, 2 seeds | 0.350 / 0.362 | 0.335 / 0.335 | 0.688 / 0.690 | 0.423 / 0.417 | 0.073 / 0.110 | 0.163 / 0.140 |
| + WorldCover agreement filter (AGREE) | 0.342 | 0.362 | 0.696 | 0.516 | 0.212 | 0.174 |
| + sharpen filter | 0.337 | 0.291 | 0.661 | 0.124 | 0.109 | 0.169 |
| + smooth filter | 0.362 | 0.293 | 0.676 | 0.204 | 0.098 | 0.152 |

- Sites, not years: 21 sites with 2024 only, at about the same number of training steps as the old runs, already matches the 4-year runs. Targeted diversity of places (dry regions) fixed the false water; repeating the same places in other years added little on the main scores. (Pending: 60-epoch 2024-only runs, which match the 4-year run's step count.)
- Agreement filter (keep a filled yearly label only where WorldCover agrees): test mIoU +0.03 and test water IoU +0.10, test bare +0.12, but crops and low vegetation barely moved, so it did not fix the crops / low-vegetation confusion; validation mIoU did not improve (0.342 against 0.350 / 0.362). Single seed; mixed evidence.
- Radar filters: neither sharpening nor smoothing helped. Validation was unchanged, but on the dry test sites both hurt (test mIoU -0.04, water IoU 0.42 to 0.12 and 0.20). Raw radar is best here. Single seed each, but the effect is much larger than the seed noise.
- Methodological finding: the validation sites do not include dry regions, so they cannot see effects that show only on the dry test sites (the filters, the agreement filter). Plan: add dry-region validation sites.
- Target remark: scoring is against Dynamic World, which agrees with WorldCover on only about 60% of pixels, so pixel accuracy near 95% is not a meaningful goal; report per-class IoU and compare with a baseline and with the label agreement.

## Years matter after all: overfitting at equal training steps (queue runs 6-7, confirmed with 2 seeds)

| run (21 sites) | training steps | val mIoU | test mIoU | test water IoU |
|---|---|---|---|---|
| 2024 only, 15 epochs (2 seeds) | about 12k | 0.326 / 0.337 | 0.350 / 0.335 | 0.467 / 0.382 |
| 2024 only, 60 epochs (2 seeds) | about 46k | 0.343 / 0.350 | 0.260 / 0.280 | 0.078 / 0.132 |
| 4 years, 15 epochs (2 seeds) | about 46k | 0.350 / 0.362 | 0.335 / 0.335 | 0.423 / 0.417 |

With the same number of steps, training on one year overfits (every image seen 60 times) and the false water returns on the dry test sites; four years of the same places (different weather and moisture) prevent this and also make results more stable between seeds. Conclusion: at short training, sites matter most; for longer training, the extra years are needed. Validation (no dry sites at the time) did not detect the overfitting; the model was picked by its best validation score, so the bad weights still won. This motivated the dry-region validation sites.

## Sentinel-1 limitations, with evidence from this project (critical discussion)

| Limitation | Plain words | Seen in this project |
|---|---|---|
| No colour | radar measures structure and moisture; structurally similar classes look alike | crops / low vegetation / shrub confusion (crops IoU about 0.16) |
| Smooth = dark | smooth surfaces scatter the signal away; calm water and smooth dry ground both look dark | false water in dry regions (fixed mostly by dry-region training sites) |
| Wind on water | rough water looks brighter | some water missed |
| Speckle | grainy noise, needs averaging | only 2-6 scenes per month, noisy monthly composites |
| Coarse effective resolution | IW GRD is about 20 m true resolution on a 10 m grid; edges and small objects become mixed pixels | errors along class borders |
| Moisture and season | rain and wet soil change backscatter without any land-use change | seasonal effects that can look like change |
| Side-looking geometry | radar shadow and distortion in hilly terrain; buildings look different from each side | orbit direction changes between years at 7 of 28 sites |
| C-band wavelength | sees the canopy top, not underneath | flooded vegetation IoU about 0.10 |
| Coverage gaps | some months have no scene in the chosen direction | 12 missing site-months |

Short version: Sentinel-1 is good at classes with a distinct structure or wetness (built-up bright, calm water dark) and weak at structurally similar classes (crops, grass, shrub, bare) and at borders.

## Improvement ideas queued for after prototype v1 (learner's ideas, 2026-10-24)

Weak classes split into two groups: look-alikes with plenty of data (crops 21% of training pixels, trees 16%) and rare classes (flooded vegetation 3%, bare 3%).
1. Rare classes, synthetic examples built from REAL pixels (label-safe): copy-paste of real rare-class patches (radar + label) into other training images; oversampling of windows that contain rare classes. Not generative synthetic radar.
2. Use the model's known confusions to teach it (learner's idea): hard-example mining (show wrongly predicted patches more often), focal loss (more weight on uncertain pixels), cost-sensitive loss (specific mix-ups such as crops->low vegetation cost more). Combine with the label-agreement filter, because part of the crops / low vegetation confusion is in the labels (the two label sources disagree on about 40% of those pixels).
3. Look-alikes may need a better signal than more data: the pattern across months (fields change through the growing season, grass and scrub less).

## Planned model variant for later (not iteration 1)

Use extra data during training only, so the final model stays Sentinel-1-only (the brief requires Sentinel-1 as the only input). Options, in order of effort:
- Better labels for the built class from high-resolution-derived building footprints (e.g. Google Open Buildings; coverage of our regions not verified).
- Multimodal radar + optical pre-training. SSL4EO-S12 reports a 1-3% gain for radar downstream tasks vs radar-only pre-training (BigEarthNet). Check pretrained weights are available.
- Distillation from an optical teacher (more work).
Cautions: Sentinel-2 is 10 m, not high definition; truly high-resolution imagery is commercial/limited. Ask the professor whether optical pre-training is allowed under "Sentinel-1 only". Fits the brief's "comparison of model variants".

## Site coordinates

- All 10 original site centres were cross-checked against OpenStreetMap's Nominatim geocoder (city-level results, 2026-10-21). Every site agrees within about 2 km, which is negligible for 10 km squares. Coordinates in the notebook were left unchanged so they match the data already downloaded.
- Lesson for the report or README: a plain place-name search can return an administrative region (division, province) instead of the city (seen for Morogoro, Mymensingh, Battambang). Use a city filter (`featuretype=city`) and check the result type.
- Rural sites were found by searching a 3x3 grid (about 28 km spacing) around geocoded anchor towns, then screening on the Dynamic World label (built share under 15%, at least 90% labelled).

## Label gaps in our data (observed 2024 export, 10 sites)

- 75 of 120 site-months have at most 20% of label pixels missing. Gaps follow each site's rainy season; Kisumu is the only clean site. Radar is complete in all months.

## Prototype case study: Ouagadougou, Jan-Dec 2024 (unseen city, 2026-10-24)

- Location 12.3682, -1.5271. Not a training site; nearest sites are the test sites 305-329 km away. Model hybrid_seed1, descending passes, no radar in June and July.
- The model's typical map (each pixel's most frequent class over the year) agrees with ESA WorldCover 2021 on 80% of pixels.
- Caveat: built covers 72% of the area in WorldCover, so a map that says "built" everywhere would already score 72%. The 80% is only about 8 points above that baseline.
- Class shares, model vs WorldCover: built 84.5 vs 72.1 (+12.4 pp), low vegetation 0.9 vs 9.2 (-8.3), trees 3.9 vs 7.7 (-3.8), water 3.0 vs 2.2, crops 7.3 vs 6.5, bare 0.6 vs 2.1.
- What the model gets right: the reservoir chain, the forest/park in the north-east, and the airport strip.
- Where it is wrong: it absorbs small green patches (parks, grass strips, scattered trees inside the city) into built and draws smooth blobs instead of fine detail. Likely causes: built radar returns (double bounce) dominate mixed 10 m pixels, speckle, and the model learned to smooth.
- Report point: overall agreement is inflated by a dominant class, so per-class agreement is needed.

## Copy-paste for rare classes: risks noted before building it (2026-10-25)

- Shortcut learning: a pasted piece has an artificial border (sudden brightness jump, often a straight or rectangular edge). The model can learn "sharp artificial edge = rare class" instead of the class's own texture, then fire on real straight edges (roads, field borders).
- Mismatches that also give the paste away: different orbit direction or incidence angle between the two sites (brightness level jump), different season (wet-season wetland pasted into a dry-season window), and with CONTEXT the yearly-median channels must be pasted too or the input disagrees with itself.
- Mitigations: paste along the class's own label shape (not a rectangle), only between windows of the same month and pass direction, paste all channels including context, and apply it to a fraction of batches only.
- WorldCover must not be used to place or label pasted pieces, or the independent check is no longer independent.

## Idea: Sentinel-1 interferometric coherence (to discuss with the professor, 2026-10-25)

- What: Sentinel-1 also records the phase of the radar wave. Comparing two passes 12 days apart gives coherence (0 to 1): stable surfaces such as buildings and bare rock stay high, vegetation and water fall low. Our GRD input (brightness only) does not contain it.
- Why: it targets our built vs bare vs vegetation confusions directly, and it is still Sentinel-1, so it should fit "Sentinel-1 only" (to confirm with the professor).
- What it would not fix: the blobby small patches in cities (coherence products are tens of metres per pixel), and flooded vegetation (a C-band limit; L-band, e.g. ALOS-2 or NISAR, sees water under canopy).
- Sources: (1) process Sentinel-1 SLC data from the Copernicus Data Space with ESA SNAP (any date, finest pixels, very heavy); (2) ASF HyP3 on-demand processing (free with an Earthdata login, Python package, coarser pixels, monthly quota, queue time); (3) the global seasonal coherence dataset (Kellndorfer et al., AWS) - 2020 only, about 90 m, seasonal.
- Deciding constraint: the prototype must fetch every model input for any place and any 12 months. A 2020 dataset cannot, and HyP3 would turn a 1-minute run into hours. A full pipeline would take most of the remaining time.
- Plan agreed: a small proof of concept only - coherence for one site and a few months via HyP3, one figure showing how coherence and backscatter separate built, bare and vegetation. The report presents it as evidence plus future work.

### Questions for the professor
1. Does interferometric coherence count as "Sentinel-1 as the only input"?
2. Is a one-site proof of concept enough for the report, with the full pipeline as future work?
3. If an AGREE model (labels filtered with ESA WorldCover) becomes the prototype model, is switching the independent check to Esri's 10 m land cover map acceptable?

## Brainstorm ideas (learner, 2026-10-25)

- Biomes: tag each site with its biome (RESOLVE Ecoregions 2017 in Earth Engine) and report scores per biome; also use it to pick new sites so every biome is covered. Biome as a model input is risky with 21 training sites (2-4 per biome, shortcut learning); the CONTEXT channel already acts as a learned biome signal.
- PCA: on 2 channels it only rotates them (about VV+VH = brightness, VH-VV = cross-pol ratio, a radar vegetation indicator; the first network layer can compute this itself). Useful across time instead: PCA of the 12 monthly values per pixel gives "mean brightness" and "seasonal swing", which separates crops from grass and scrub. Also a good false-colour figure for the report.

## Review of the whole pipeline (2026-10-25)

Fixed:
- Test sites were used for decisions (exp1 filters and agreement filter, exp2 context): they are now a "development test". Fresh test sites, looked at only once at the end, go into the next export.
- The exp1 baseline chose its best epoch on the old 4 validation sites, later runs on the current 10: baseline rerun queued (exp1_rerun) so every step uses the same validation set.
- Pixels without radar: the export fills them with -999 (scaled to the darkest value, water-like). One training month (morogoro 2022-06) was 88% empty; overall 0.09% of training pixels. Training now ignores them; the prototype fills them the same way, gives them no class, and skips a month with more than 5% uncovered (a half-empty month would appear as large fake change).

For the report (limits, not bugs):
- Every score is agreement with Dynamic World, itself a model, not ground truth; Dynamic World and WorldCover agree on only about 60% of pixels, which caps any score.
- Windows overlap (stride 128), so pixels inside a site count up to 4 times in the scores and printed pixel counts are inflated; comparisons between runs are unaffected.
- Validation scores are slightly optimistic because the best epoch is chosen on validation.
- The prototype picks the orbit direction among scenes with both VV and VH, the export among all IW scenes; almost always the same.

Checked and fine: every validation/test site is at least 82 km from any training site; no NaN in the 1,152 radar files; AGREE does not change validation or test windows; the prototype's compositing, scaling and WorldCover mapping match the export; the rescore reproduces training-time scores exactly.

## Professor feedback (2026-10-27)

- Coherence: the main obstacle he saw was storage on the datalab server. Solution: a second server (the learner has access) with fast NVMe read-write storage, so data is downloaded straight onto it instead of laptop -> Drive -> server.
- Dataset: grow from 34 to 50 sites and re-split.
- Still to ask at the next lecture (2026-10-29): is a one-site coherence proof of concept enough; does the prototype also need coherence, or can coherence stay a research variant (a prototype run would need about 24 radar phase scenes, several GB each, processed on the fly); can the independent check switch to Esri.
