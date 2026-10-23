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
