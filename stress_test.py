"""Stress test: the prototype at places of the professor's kind, own coordinates anywhere, plus edge cases and real
change events. Each place runs once through prototype.run (a run already saved is reused); experiments/stress_test.md
gets one row per place: months with radar, distance to the nearest training site, agreement with Dynamic World and
WorldCover, the biggest change, seconds, or the failure. For the change events, the month-by-month share of the class
that should change.

    uv run python -u stress_test.py > stress.log 2>&1        (on the server: Earth Engine and the model)
"""
import os
import time

import numpy as np

import prototype as P

E = "everyday"
PLACES = [  # (group, name, lat, lon, first month, what to look for)
    (E, "Paris, France: city", 48.857, 2.352, "2025-06", "built, stable"),
    (E, "Iowa, USA: corn belt", 42.03, -93.62, "2025-06", "crops, harvest in autumn"),
    (E, "Pampas, Argentina: cropland", -34.60, -60.90, "2025-06", "crops"),
    (E, "Amazonas, Brazil: rainforest", -3.47, -62.37, "2025-06", "trees, stable"),
    (E, "Nile delta, Egypt: farmland", 30.90, 31.10, "2025-06", "crops, villages"),
    (E, "Lucknow, India: city and fields", 26.85, 80.95, "2025-06", "built, crops"),
    (E, "Furano, Japan: farmland and forest", 43.34, 142.38, "2025-06", "crops, trees"),
    (E, "Longreach, Australia: cattle country", -23.44, 144.25, "2025-06", "low vegetation, bare"),
    (E, "Lahti, Finland: lakes and forest", 61.00, 25.60, "2025-06", "trees, water; above 60 N"),
    (E, "Krasnoyarsk krai, Russia: taiga", 61.00, 90.00, "2025-06", "trees; far north, far from training data"),
    ("edge", "Open Atlantic", 30.00, -40.00, "2025-06", "no land: expect a clear 'no Sentinel-1 scenes' message"),
    ("edge", "Grand Erg Oriental, Algeria: sand sea", 30.50, 7.50, "2025-06", "bare, stable"),
    ("edge", "Greenland ice margin", 67.10, -49.70, "2025-06", "ice and snow: no such class, expect confusion"),
    ("edge", "Everest, Nepal: high mountains", 27.99, 86.92, "2025-06", "steep terrain, snow: expect confusion"),
    ("edge", "Bernese Alps, Switzerland", 46.54, 7.96, "2025-06", "mountains, glacier, valley towns"),
    ("edge", "Dubai, UAE: coast, city, desert", 25.20, 55.27, "2025-06", "water, built, bare"),
    ("event", "Kakhovka reservoir, Ukraine: dam destroyed 6 June 2023", 47.525, 34.40, "2023-01", "water falls from June 2023"),   # 47.45, 34.45 was the shore town
    ("event", "Tulare Lake, California: lake returned in spring 2023", 36.05, -119.80, "2022-10", "water rises from March 2023"),
    ("event", "Dadu, Pakistan: floods from August 2022", 26.70, 67.75, "2022-04", "water rises Aug-Oct 2022"),
    ("event", "Pacific Palisades, California: fire of January 2025", 34.07, -118.55, "2024-07", "trees/low veg to bare in Jan 2025"),
]
WATCH = {"water falls": "water", "water rises": "water", "to bare": "bare"}   # the class to follow for each event


def main():
    P.init_ee()
    models = P.load_model()
    sites = P.training_sites(P.WEIGHTS)
    rows, series = [], []
    for group, name, lat, lon, start, look in PLACES:
        path = f"{P.RUNS}/{lat:.4f}_{lon:.4f}_{start}.pkl.gz"
        t = time.time()
        try:
            res = P.load(path) if os.path.exists(path) else P.run_and_save(lat, lon, start, models, name)[0]
        except Exception as e:                                    # a failure is a result here: say what it was
            rows.append(f"| {group} | {name} | – | – | – | – | – | {time.time() - t:.0f} | **{type(e).__name__}: {e}** |")
            print(f"{name}: {type(e).__name__}: {e}", flush=True)
            continue
        have = [i for i, m in enumerate(res["maps"]) if m is not None]
        km, near, _ = P.familiarity(lat, lon, sites)
        dw = res.get("dw")
        agree_dw = f"{P.agreement(res['typical'], dw):.0%}" if dw is not None else "–"
        big = ", ".join(f"{c.replace('_', ' ')} {p:+.1f}" for c, p in P.biggest_changes(res)[:2])
        rows.append(f"| {group} | {name} | {len(have)}/12 | {km:,.0f} km ({near.replace('_', ' ')}) | {agree_dw} | "
                    f"{res['agreement']:.0%} | {big} | {time.time() - t:.0f} | {look} |")
        print(rows[-1], flush=True)
        cls = next((c for key, c in WATCH.items() if key in look), None)
        if group == "event" and cls:
            shares = P.summarise(res.get("smooth") or res["maps"])["shares"][:, P.CLASSES.index(cls)]
            series.append(f"| {name} (m1 = {start}) | {cls} | " + " | ".join("–" if v != v else f"{v:.0f}" for v in shares) + " |")
    md = ["# Stress test of the prototype (final model, 2 seeds averaged)", "",
          "Agreement: the model's map of the year against Dynamic World (its training labels) and ESA WorldCover. "
          "Biggest change: first to last month with radar, in points (= km²). Seconds: fetch and model, 0 when reused.", "",
          "| Group | Place | Months with radar | Nearest training site | vs Dynamic World | vs WorldCover | Biggest change | s | Look for |",
          "|---|---|---|---|---|---|---|---|---|", *rows, "",
          "## Change events, month by month (smoothed maps, % of the square)", "",
          "| Event | Class | " + " | ".join(f"m{i + 1}" for i in range(12)) + " |", "|---" * 14 + "|", *series]
    open("experiments/stress_test.md", "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
