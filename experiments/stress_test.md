# Stress test of the prototype (final model, 2 seeds averaged)

Agreement: the model's map of the year against Dynamic World (its training labels) and ESA WorldCover. Biggest change: first to last month with radar, in points (= km²). Seconds: fetch and model, 0 when reused.

| Group | Place | Months with radar | Nearest training site | vs Dynamic World | vs WorldCover | Biggest change | s | Look for |
|---|---|---|---|---|---|---|---|---|
| everyday | Paris, France: city | 12/12 | 92 km (beauce cereal) | 91% | 82% | built -0.3, trees +0.2 | 0 | built, stable |
| everyday | Iowa, USA: corn belt | 12/12 | 63 km (iowa corn) | 82% | 62% | trees -1.9, crops +1.0 | 0 | crops, harvest in autumn |
| everyday | Pampas, Argentina: cropland | 12/12 | 183 km (pampa crops) | 81% | 75% | crops +10.1, low vegetation -9.5 | 0 | crops |
| everyday | Amazonas, Brazil: rainforest | 12/12 | 230 km (amazon varzea) | 100% | 100% | water +0.0, trees +0.0 | 0 | trees, stable |
| everyday | Nile delta, Egypt: farmland | 12/12 | 332 km (minya nile) | 96% | 86% | bare -4.8, crops +4.0 | 0 | crops, villages |
| everyday | Lucknow, India: city and fields | 12/12 | 265 km (gorakhpur) | 87% | 76% | built -0.4, water +0.3 | 0 | built, crops |
| everyday | Furano, Japan: farmland and forest | 12/12 | 1,512 km (korea paddies) | 91% | 82% | crops -2.0, low vegetation +1.1 | 0 | crops, trees |
| everyday | Longreach, Australia: cattle country | 12/12 | 1,235 km (murray irrigation) | 52% | 34% | crops +5.8, low vegetation -4.7 | 0 | low vegetation, bare |
| everyday | Lahti, Finland: lakes and forest | 12/12 | 600 km (dalarna boreal) | 89% | 79% | built -2.6, trees +2.4 | 0 | trees, water; above 60 N |
| everyday | Krasnoyarsk krai, Russia: taiga | 12/12 | 3,365 km (north china wheat) | 97% | 97% | low vegetation +26.5, trees -26.3 | 0 | trees; far north, far from training data |
| edge | Open Atlantic | – | – | – | – | – | 2 | **ValueError: no Sentinel-1 scenes at this location in these 12 months** |
| edge | Grand Erg Oriental, Algeria: sand sea | 12/12 | 1,289 km (jaen olives) | 74% | 74% | water +6.1, bare -6.1 | 0 | bare, stable |
| edge | Greenland ice margin | 3/12 | 2,702 km (irish midlands) | – | nan% | low vegetation +5.1, bare -4.0 | 0 | ice and snow: no such class, expect confusion |
| edge | Everest, Nepal: high mountains | 12/12 | 343 km (gorakhpur) | 0% | 2% | water -19.0, low vegetation +15.3 | 0 | steep terrain, snow: expect confusion |
| edge | Bernese Alps, Switzerland | 12/12 | 157 km (black forest) | 35% | 40% | low vegetation +27.5, water -16.5 | 0 | mountains, glacier, valley towns |
| edge | Dubai, UAE: coast, city, desert | 12/12 | 1,573 km (jaisalmer) | 71% | 66% | low vegetation +3.7, crops -3.6 | 0 | water, built, bare |
| event | Kakhovka reservoir, Ukraine: dam destroyed 6 June 2023 | 12/12 | 482 km (danube delta) | 71% | 50% | water -59.0, bare +34.1 | 0 | water falls from June 2023 |
| event | Tulare Lake, California: lake returned in spring 2023 | 12/12 | 412 km (sacramento rice) | 53% | 49% | water +30.9, crops -26.3 | 0 | water rises from March 2023 |
| event | Manchar Lake, Pakistan: overflowed in the floods of August 2022 | 12/12 | 330 km (jaisalmer) | 87% | 77% | bare -12.6, water +6.9 | 0 | water rises Aug-Oct 2022 |
| event | Pacific Palisades, California: fire of January 2025 | 12/12 | 297 km (imperial valley) | 76% | 76% | low vegetation -3.6, trees +3.3 | 0 | trees/low veg to bare in Jan 2025 |

## Change events, month by month (smoothed maps, % of the square)

| Event | Class | m1 | m2 | m3 | m4 | m5 | m6 | m7 | m8 | m9 | m10 | m11 | m12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Kakhovka reservoir, Ukraine: dam destroyed 6 June 2023 (m1 = 2023-01) | water | 73 | 76 | 76 | 76 | 76 | 38 | 12 | 10 | 10 | 9 | 12 | 14 |
| Tulare Lake, California: lake returned in spring 2023 (m1 = 2022-10) | water | 35 | 41 | 36 | 37 | 37 | 50 | 61 | 69 | 71 | 74 | 68 | 66 |
| Manchar Lake, Pakistan: overflowed in the floods of August 2022 (m1 = 2022-04) | water | 68 | 67 | 67 | 74 | 96 | 96 | 96 | 95 | 92 | 84 | 77 | 75 |
| Pacific Palisades, California: fire of January 2025 (m1 = 2024-07) | bare | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
