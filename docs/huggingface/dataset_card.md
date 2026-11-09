---
license: cc-by-4.0
pretty_name: Monthly Sentinel-1 land cover, 120 sites
task_categories:
- image-segmentation
tags:
- sentinel-1
- sar
- land-cover
- remote-sensing
- dynamic-world
size_categories:
- 1K<n<10K
---

# Monthly Sentinel-1 land cover dataset (120 sites)

Monthly Sentinel-1 radar composites paired with monthly land-cover labels, for 10 km × 10 km squares at 120 sites
worldwide. Made for the project [Monthly Land Cover from Sentinel-1 Radar](https://github.com/Yash-Annapure/Deep_learning_project)
(SRH Data Engineering 1, November 2026). All data was exported from Google Earth Engine on a 10 m grid
(1000 × 1000 pixels per site).

## Sites

`sites.csv` lists every site with its centre and role:

| Role | Sites | Years | Used for |
|---|---|---|---|
| train | 29 | 2022 to 2025 | training |
| val | 12 | 2024 | choosing epochs, judging experiments |
| test | 3 | 2024 | early development checks |
| final | 62 | one year | training the final model only |
| holdout | 14 | one year | scored once, at the end |

## Files

| Path | Contents |
|---|---|
| `<site>_<year>-<month>.npz` | one site-month: `radar` (float32, 2 × 1000 × 1000: VV and VH in dB, median of the month's scenes from one orbit direction; -999 where there is no radar), `label` (uint8, Dynamic World class with the highest month-averaged probability; 255 = no label), `conf` (float32, that probability), `n_scenes`, `pass_dir` |
| `yearly/<site>_<year>.npz` | the yearly Dynamic World label and its confidence, same layout |
| `worldcover/<site>.npz` | ESA WorldCover 2021, converted to Dynamic World class numbers |
| `coherence/<site>_<year>-<month>.npz` | Sentinel-1 interferometric coherence from ASF HyP3 (`coh`, `n_pairs`; NaN where no pair covers a pixel), for the sites used in experiment 5 |
| `sites.csv` | name, latitude, longitude, role |

Labels use Dynamic World's numbering: 0 water, 1 trees, 2 grass, 3 flooded vegetation, 4 crops, 5 shrub and scrub,
6 built, 7 bare, 8 snow and ice. The project merges grass and shrub into one *low vegetation* class and ignores
snow, giving seven classes (`REMAP` in the project's `lulc.py`).

```python
import numpy as np

z = np.load("kisumu_2024-01.npz")
radar, label = z["radar"], z["label"]        # (2, 1000, 1000) dB, (1000, 1000) Dynamic World class
```

The project's `lulc.LULCDataset` cuts these files into training windows and applies the label rules.

## Sources and attribution

- Contains modified Copernicus Sentinel data (2021 to 2025), accessed through Google Earth Engine.
- Dynamic World V1: produced for the Dynamic World Project by Google in partnership with the National Geographic
  Society and the World Resources Institute. CC BY 4.0. Brown et al. (2022), *Scientific Data* 9, 251.
- ESA WorldCover 10 m 2021 v200: © ESA WorldCover project 2021, contains modified Copernicus Sentinel data (2021)
  processed by the ESA WorldCover consortium. CC BY 4.0. Zanaga et al. (2022).
- Coherence: processed with ASF HyP3 (Alaska Satellite Facility).

This dataset is shared under CC BY 4.0, with the attributions above.
