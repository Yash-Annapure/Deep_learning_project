# Report notes

Running list of points to use in the report. Add to it as we go.

## Date matching between radar and labels (dataset and preprocessing section)

- Accepted time windows in the literature range from about 30 days to a whole season, so a monthly radar window sits inside normal practice.
  - SEN12MS: pairing window is a meteorological season (about 3 months). Sentinel-1 was exported as-is (not affected by clouds); Sentinel-2 was a cloud-free mosaic over the season; labels were annual MODIS maps (2016). Source: Schmitt et al. 2019, https://arxiv.org/pdf/1906.07789
  - SSL4EO-S12: 30-day interval around four reference dates (Mar 20, Jun 21, Sep 22, Dec 21). Kept a location only if Sentinel-2 had under 10% cloud and a matching Sentinel-1 image existed for all four dates. Source: Wang et al. 2023, https://arxiv.org/pdf/2211.07044
- A yearly label is looser in time than what SEN12MS used for imagery, but it is in line with SEN12MS's annual MODIS labels.
- SSL4EO-S12 handled clouds by discarding cloudy locations (its own limitations section lists "geographical bias due to cloud filtering"). We cannot do that, because the project must work in cloudy, data-poor regions.

Draft sentence: "Unlike SSL4EO-S12, which discards locations without cloud-free optical coverage, we keep cloudy tropical sites, since the model is meant to work in exactly those regions."

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
