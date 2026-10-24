"""Prototype logic: a location and a start month -> 12 monthly land-cover maps, class shares and their changes.

Used by the Streamlit page (app.py), and runnable on its own as a quick check:
    uv run python prototype.py --lat -0.0917 --lon 34.768 --start 2024-01
which saves a picture of the 12 maps and prints the tables.

Settings through environment variables: EE_PROJECT (Google Cloud project for Earth Engine) and MODEL_WEIGHTS.
"""
import argparse
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor

import ee
import numpy as np
import segmentation_models_pytorch as smp
import torch

from lulc import CLASSES, COLORS, REMAP, scale_radar

PROJECT = os.environ.get("EE_PROJECT", "deep-learning-project-512013")
WEIGHTS = os.environ.get("MODEL_WEIGHTS", "experiments/exp1_data/weights/hybrid_seed1.pt")
SIZE_PX, SCALE = 1000, 10          # a 10 km square of 10 m pixels, the same as the training data
NO_DATA = 255
WC_TO_DW = {10: 1, 20: 5, 30: 2, 40: 4, 50: 6, 60: 7, 70: 8, 80: 0, 90: 3, 95: 1}       # ESA WorldCover -> Dynamic World index, as in 01_data_export.ipynb
device = "cuda" if torch.cuda.is_available() else "cpu"

# ponytail: the Earth Engine helpers below repeat the ones in 01_data_export.ipynb (same maths, so the prototype prepares
# radar exactly like the training data). If they ever change, change both, or make the notebook import this module.


def init_ee():
    ee.Initialize(project=PROJECT)


def load_model(path=WEIGHTS):
    model = smp.Unet("resnet34", encoder_weights=None, in_channels=2, classes=len(CLASSES)).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.eval()


def months_from(start):
    """'2024-03' -> the 12 (year, month) pairs from March 2024 to February 2025."""
    y, m = map(int, start.split("-"))
    return [((y * 12 + m - 1 + i) // 12, (m - 1 + i) % 12 + 1) for i in range(12)]


def make_grid(lat, lon):
    """SIZE_PX x SIZE_PX pixels of SCALE metres centred on the point, in the local UTM projection (square pixels)."""
    zone = int((lon + 180) // 6) + 1
    epsg = f"EPSG:{(32600 if lat >= 0 else 32700) + zone}"
    x, y = ee.Geometry.Point([lon, lat]).transform(epsg, 1).coordinates().getInfo()
    half = SIZE_PX * SCALE / 2
    grid = {"dimensions": {"width": SIZE_PX, "height": SIZE_PX},
            "affineTransform": {"scaleX": SCALE, "shearX": 0, "translateX": x - half,
                                "shearY": 0, "scaleY": -SCALE, "translateY": y + half},
            "crsCode": epsg}
    return grid, ee.Geometry.Rectangle([x - half, y - half, x + half, y + half], epsg, False)


def s1(region, start, end):
    return (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(region).filterDate(start, end)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")))


def pick_pass(region, start, end):
    """The orbit direction with the most scenes in the 12-month window (one direction only, as in training)."""
    counts = s1(region, start, end).aggregate_histogram("orbitProperties_pass").getInfo()
    return max(counts, key=counts.get) if counts else None


def _download_once(image, grid):
    raw = ee.data.computePixels({"expression": image, "fileFormat": "NPY", "grid": grid})
    arr = np.load(io.BytesIO(raw))
    return np.stack([arr[name] for name in arr.dtype.names])


def download(image, grid):
    """Image -> array (bands, H, W). If Earth Engine runs out of memory, fetch 2x2 tiles and stitch them (same numbers)."""
    try:
        return _download_once(image, grid)
    except ee.EEException as e:
        if "memory" not in str(e).lower():
            raise
    size, out = grid["dimensions"]["width"] // 2, None
    for i in range(2):
        for j in range(2):
            tile = json.loads(json.dumps(grid))
            tile["dimensions"] = {"width": size, "height": size}
            tile["affineTransform"]["translateX"] += j * size * SCALE
            tile["affineTransform"]["translateY"] -= i * size * SCALE
            part = _download_once(image, tile)
            out = np.zeros((part.shape[0], size * 2, size * 2), part.dtype) if out is None else out
            out[:, i * size:(i + 1) * size, j * size:(j + 1) * size] = part
    return out


def fetch_month(region, grid, pass_dir, year, month):
    """Median of the month's Sentinel-1 scenes (VV, VH in dB), or None if there is no scene that month."""
    start = ee.Date.fromYMD(year, month, 1)
    col = s1(region, start, start.advance(1, "month")).filter(ee.Filter.eq("orbitProperties_pass", pass_dir))
    n = col.size().getInfo()
    return (download(col.select(["VV", "VH"]).median(), grid).astype("float32"), n) if n else (None, 0)


def worldcover(grid):
    """ESA WorldCover 2021 for the same square, in our 7 classes (255 where it has no counterpart). An independent map:
    the prototype's model was trained on Dynamic World only."""
    wc = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
    img = wc.remap(list(WC_TO_DW), list(WC_TO_DW.values()), NO_DATA).unmask(NO_DATA)
    return REMAP[download(img, grid)[0].astype(np.uint8)]


@torch.no_grad()
def predict(model, radar_db):
    """(2, H, W) radar in dB -> (H, W) class map. Scaled exactly like the training data, padded to a multiple of 32."""
    h, w = radar_db.shape[1:]
    x = torch.from_numpy(scale_radar(radar_db))[None].to(device)
    x = torch.nn.functional.pad(x, (0, -w % 32, 0, -h % 32), mode="reflect")
    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
        return model(x).argmax(1)[0, :h, :w].cpu().numpy().astype(np.uint8)


def run(lat, lon, start, model, workers=6):
    """The whole prototype: 12 maps, % per class each month, month-to-month change and first-to-last change (in points)."""
    months = months_from(start)
    grid, region = make_grid(lat, lon)
    y0, m0 = months[0]
    window = ee.Date.fromYMD(y0, m0, 1)
    pass_dir = pick_pass(region, window, window.advance(12, "month"))
    if pass_dir is None:
        raise ValueError("no Sentinel-1 scenes at this location in these 12 months")
    with ThreadPoolExecutor(max_workers=workers) as pool:          # the 12 months (and WorldCover) are fetched at the same time
        wc_job = pool.submit(worldcover, grid)
        fetched = list(pool.map(lambda ym: fetch_month(region, grid, pass_dir, *ym), months))
        wc = wc_job.result()
    maps = [predict(model, radar) if radar is not None else None for radar, _ in fetched]

    C = len(CLASSES)
    shares = np.full((12, C), np.nan)                              # % of the area per class, per month
    for i, mp in enumerate(maps):
        if mp is not None:
            shares[i] = np.bincount(mp.ravel(), minlength=C)[:C] / mp.size * 100
    change = np.full((12, C), np.nan)                              # points of change against the previous month WITH data
    have = [i for i in range(12) if maps[i] is not None]
    if not have:
        raise ValueError("none of the 12 months has radar at this location")
    for prev, cur in zip(have, have[1:]):
        change[cur] = shares[cur] - shares[prev]
    total = shares[have[-1]] - shares[have[0]] if len(have) >= 2 else np.full(C, np.nan)

    # independent check: the model's "typical" map (each pixel's most frequent class over the months) against WorldCover
    votes = np.stack([sum((maps[i] == c).astype(np.uint8) for i in have) for c in range(C)])
    typical = votes.argmax(0).astype(np.uint8)
    valid = wc != NO_DATA
    wc_shares = np.bincount(wc[valid], minlength=C)[:C] / max(valid.sum(), 1) * 100
    typical_shares = np.bincount(typical.ravel(), minlength=C)[:C] / typical.size * 100
    agreement = float((typical[valid] == wc[valid]).mean()) if valid.any() else float("nan")
    return {"months": [f"{y}-{m:02d}" for y, m in months], "maps": maps, "shares": shares, "change": change,
            "total": total, "scenes": [n for _, n in fetched], "pass_dir": pass_dir, "lat": lat, "lon": lon,
            "typical": typical, "wc": wc, "typical_shares": typical_shares, "wc_shares": wc_shares, "agreement": agreement}


def maps_figure(result):
    """The 12 maps in a 3 x 4 grid, with one colour per class and grey for months without radar."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(COLORS)
    fig, ax = plt.subplots(3, 4, figsize=(16, 14), layout="constrained")   # constrained: keeps room for every title
    for a, month, mp, n in zip(ax.ravel(), result["months"], result["maps"], result["scenes"]):
        if mp is None:
            a.set_facecolor("lightgrey")
            a.text(0.5, 0.5, "no radar\nthis month", ha="center", va="center", transform=a.transAxes)
        else:
            a.imshow(mp, cmap=cmap, vmin=-0.5, vmax=len(CLASSES) - 0.5, interpolation="nearest")
        a.set_box_aspect(1)                                    # every panel square, including "no radar" ones
        a.set_title(f"{month}  ({n} scenes)")
        a.set_xticks([]); a.set_yticks([])
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS], CLASSES, loc="outside lower center", ncol=len(CLASSES))
    fig.suptitle(f"Land cover at {result['lat']:.4f}, {result['lon']:.4f} (10 km x 10 km), {result['pass_dir'].lower()} passes")
    return fig


def comparison_figure(result):
    """WorldCover 2021 next to the model's typical map, plus where the two agree."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(COLORS)
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.8), layout="constrained")
    ax[0].imshow(np.ma.masked_equal(result["wc"], NO_DATA), cmap=cmap, vmin=-0.5, vmax=len(CLASSES) - 0.5, interpolation="nearest")
    ax[0].set_title("ESA WorldCover 2021 (independent)")
    ax[1].imshow(result["typical"], cmap=cmap, vmin=-0.5, vmax=len(CLASSES) - 0.5, interpolation="nearest")
    ax[1].set_title("model: each pixel's most frequent class")
    same = np.where(result["wc"] == NO_DATA, np.nan, (result["typical"] == result["wc"]).astype(float))
    ax[2].imshow(np.ma.masked_invalid(same), cmap="RdYlGn", vmin=0, vmax=1, interpolation="nearest")
    ax[2].set_title(f"agree (green) / disagree (red): {result['agreement']:.0%}")
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS], CLASSES, loc="outside lower center", ncol=len(CLASSES))
    return fig


def check(result):
    """Self-check of the calculation: each month's shares add up to 100, and changes add up to 0 (area moves between classes)."""
    for i, mp in enumerate(result["maps"]):
        if mp is not None:
            assert abs(result["shares"][i].sum() - 100) < 1e-6, result["months"][i]
            assert set(np.unique(mp)) <= set(range(len(CLASSES)))
    rows = ~np.isnan(result["change"]).any(axis=1)
    assert np.allclose(result["change"][rows].sum(axis=1), 0, atol=1e-6)
    if not np.isnan(result["total"]).any():
        assert abs(result["total"].sum()) < 1e-6
    assert abs(result["typical_shares"].sum() - 100) < 1e-6
    assert (result["wc"] != NO_DATA).sum() == 0 or abs(result["wc_shares"].sum() - 100) < 1e-6
    assert np.isnan(result["agreement"]) or 0 <= result["agreement"] <= 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--lat", type=float, required=True)
    p.add_argument("--lon", type=float, required=True)
    p.add_argument("--start", required=True, help="first month, e.g. 2024-01")
    p.add_argument("--out", default="prototype_runs")
    a = p.parse_args()
    init_ee()
    res = run(a.lat, a.lon, a.start, load_model())
    check(res)
    os.makedirs(a.out, exist_ok=True)
    name = f"{a.out}/{a.lat:.4f}_{a.lon:.4f}_{a.start}"
    maps_figure(res).savefig(f"{name}_maps.png", dpi=80, bbox_inches="tight")
    comparison_figure(res).savefig(f"{name}_worldcover.png", dpi=80, bbox_inches="tight")
    np.set_printoptions(suppress=True)
    print("% of area per class each month:\n", " " * 8, " ".join(f"{c[:8]:>9s}" for c in CLASSES))
    for m, row in zip(res["months"], res["shares"]):
        print(f"{m:8s}", " ".join(f"{v:9.1f}" for v in row))
    print("\ntotal change first -> last month (points):", {c: round(float(v), 1) for c, v in zip(CLASSES, res["total"])})
    print("\nindependent check, ESA WorldCover 2021 vs the model's typical map:")
    print("  WorldCover %:", {c: round(float(v), 1) for c, v in zip(CLASSES, res["wc_shares"])})
    print("  model %     :", {c: round(float(v), 1) for c, v in zip(CLASSES, res["typical_shares"])})
    print(f"  pixels where they agree: {res['agreement']:.0%}")
    print(f"\nscenes per month: {res['scenes']}  ({res['pass_dir']})   pictures saved: {name}_maps.png, {name}_worldcover.png   self-check OK")
