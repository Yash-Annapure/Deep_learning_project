"""Prototype logic: a location and a start month -> 12 monthly land-cover maps, class shares and their changes.

Used by the Streamlit page (app.py), and runnable on its own as a quick check:
    uv run python prototype.py --lat -0.0917 --lon 34.768 --start 2024-01
which saves a picture of the 12 maps, the result itself (prototype_runs/*.pkl.gz, which app.py can open without Earth
Engine or a model) and prints the tables.
    uv run python prototype.py --demos
does the same for every place in DEMOS.

Settings through environment variables: EE_PROJECT (Google Cloud project for Earth Engine) and MODEL_WEIGHTS.
"""
import argparse
import gzip
import io
import json
import os
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed

import ee
import numpy as np
import segmentation_models_pytorch as smp
import torch

from lulc import CLASSES, COLORS, NODATA_RADAR, REMAP, missing, scale_radar

PROJECT = os.environ.get("EE_PROJECT", "deep-learning-project-512013")
WEIGHTS = os.environ.get("MODEL_WEIGHTS", "experiments/exp4_50sites/weights/hybrid_ctx_agree.pt")   # radar + context, 4 channels
RUNS = "prototype_runs"
# places with change to show, 12 months from June 2025 (Sentinel-1A and 1C both flying: the best coverage).
# ponytail: coordinates picked from maps, check each square on satellite view before recording
DEMOS = {
    "Lake Debo, Mali: Niger flood plain": (15.30, -4.15),
    "Tonle Sap, Cambodia: seasonal lake": (13.15, 104.10),     # north shore: 12.95, 104.00 was open lake, 100% water all year
    "Satkhira, Bangladesh: shrimp ponds": (22.55, 89.10),
    "Gezira, Sudan: irrigated crops": (14.50, 33.20),
    "Bagre reservoir, Burkina Faso": (11.48, -0.55),
    "Nairobi, Kenya: city (little change expected)": (-1.29, 36.82),
}
DEMO_START = "2025-06"
SIZE_PX, SCALE = 1000, 10          # a 10 km square of 10 m pixels, the same as the training data
NO_DATA = 255
WC_TO_DW = {10: 1, 20: 5, 30: 2, 40: 4, 50: 6, 60: 7, 70: 8, 80: 0, 90: 3, 95: 1}       # ESA WorldCover -> Dynamic World index, as in 01_data_export.ipynb
device = "cuda" if torch.cuda.is_available() else "cpu"

# ponytail: the Earth Engine helpers below repeat the ones in 01_data_export.ipynb (same maths, so the prototype prepares
# radar exactly like the training data). If they ever change, change both, or make the notebook import this module.


def init_ee():
    ee.Initialize(project=PROJECT)


def load_model(path=WEIGHTS):
    model = smp.Unet("resnet34", encoder_weights=None, in_channels=4, classes=len(CLASSES)).to(device)
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


def fetch_month(region, grid, pass_dir, year, month, max_gap=0.05):
    """Median of the month's Sentinel-1 scenes (VV, VH in dB), filled with NODATA_RADAR where no scene covers a pixel,
    as in the training export. None if there is no scene, or if more than max_gap of the square is uncovered: a half-empty
    month would show up as a large change that never happened."""
    start = ee.Date.fromYMD(year, month, 1)
    col = s1(region, start, start.advance(1, "month")).filter(ee.Filter.eq("orbitProperties_pass", pass_dir))
    n = col.size().getInfo()
    if not n:
        return None, 0
    radar = download(col.select(["VV", "VH"]).median().unmask(NODATA_RADAR), grid).astype("float32")
    return (None if missing(radar).mean() > max_gap else radar), n


def worldcover(grid):
    """ESA WorldCover 2021 for the same square, in our 7 classes (255 where it has no counterpart). A second opinion, not
    fully independent: training used it to filter the filled labels (AGREE), though the model never learned from it directly."""
    wc = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
    img = wc.remap(list(WC_TO_DW), list(WC_TO_DW.values()), NO_DATA).unmask(NO_DATA)
    return REMAP[download(img, grid)[0].astype(np.uint8)]


def dynamic_world(region, grid, year, month):
    """Dynamic World (the training labels) for the same 12 months: each month's most frequent label, in our classes;
    None for a month without any (Dynamic World is made from optical images, so a cloudy month can have none)."""
    dw = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1").filterBounds(region).select("label")
    empty = ee.ImageCollection([ee.Image.constant(0).rename("label").toByte().selfMask()])   # fully masked: keeps an empty month's mode() valid
    start = ee.Date.fromYMD(year, month, 1)
    img = ee.Image.cat([dw.filterDate(start.advance(i, "month"), start.advance(i + 1, "month")).merge(empty).mode()
                        .unmask(NO_DATA).toByte().rename(f"m{i}") for i in range(12)])
    maps = REMAP[download(img, grid).astype(np.uint8)]                 # shrub -> low vegetation, 255 stays no data
    return [mp if (mp != NO_DATA).any() else None for mp in maps]


@torch.no_grad()
def predict(model, radar_db, context):
    """(2, H, W) radar in dB + (2, H, W) scaled context -> (H, W) class map. Scaled exactly like the training data,
    padded to a multiple of 32. Pixels without radar get NO_DATA, so they never count in the shares."""
    h, w = radar_db.shape[1:]
    x = torch.from_numpy(np.concatenate([scale_radar(radar_db), context]))[None].to(device)
    x = torch.nn.functional.pad(x, (0, -w % 32, 0, -h % 32), mode="reflect")
    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
        out = model(x).argmax(1)[0, :h, :w].cpu().numpy().astype(np.uint8)
    out[missing(radar_db)] = NO_DATA
    return out


def run(lat, lon, start, model, workers=14, progress=lambda done: None):
    """The whole prototype: 12 maps, % per class each month, month-to-month change and first-to-last change (in points).
    progress(n) is called as each of the 12 months arrives."""
    months = months_from(start)
    grid, region = make_grid(lat, lon)
    y0, m0 = months[0]
    window = ee.Date.fromYMD(y0, m0, 1)
    pass_dir = pick_pass(region, window, window.advance(12, "month"))
    if pass_dir is None:
        raise ValueError("no Sentinel-1 scenes at this location in these 12 months")
    with ThreadPoolExecutor(max_workers=workers) as pool:          # the 12 months (and WorldCover) are fetched at the same time
        wc_job, dw_job = pool.submit(worldcover, grid), pool.submit(dynamic_world, region, grid, y0, m0)
        jobs = [pool.submit(fetch_month, region, grid, pass_dir, *ym) for ym in months]
        for n, _ in enumerate(as_completed(jobs), 1):
            progress(n)
        fetched = [j.result() for j in jobs]
        wc, dw = wc_job.result(), year_map(dw_job.result())
    radars = [r for r, _ in fetched if r is not None]
    if not radars:
        raise ValueError("none of the 12 months has radar at this location")
    # context, as in training: each pixel's median radar over the months (training: the calendar year; here the 12 months)
    context = np.median(np.stack([scale_radar(r) for r in radars]), axis=0)
    maps = [predict(model, radar, context) if radar is not None else None for radar, _ in fetched]

    C = len(CLASSES)
    shares = np.full((12, C), np.nan)                              # % of the area per class, per month
    for i, mp in enumerate(maps):
        if mp is not None:
            ok = mp != NO_DATA                                     # pixels with radar
            shares[i] = np.bincount(mp[ok], minlength=C)[:C] / ok.sum() * 100
    change = np.full((12, C), np.nan)                              # points of change against the previous month WITH data
    have = [i for i in range(12) if maps[i] is not None]
    for prev, cur in zip(have, have[1:]):
        change[cur] = shares[cur] - shares[prev]
    total = shares[have[-1]] - shares[have[0]] if len(have) >= 2 else np.full(C, np.nan)

    return {"months": [f"{y}-{m:02d}" for y, m in months], "maps": maps, "shares": shares, "change": change,
            "total": total, "scenes": [n for _, n in fetched], "pass_dir": pass_dir, "lat": lat, "lon": lon,
            "wc": wc, "dw": dw, **second_opinion(maps, wc)}


CROP_MONTHS = 3   # crops in at least this many months = cropland, WorldCover's definition (planted at least once a year)


def year_map(maps):
    """One map for the whole year from monthly maps: each pixel's most frequent class, except that a pixel that is crops
    in CROP_MONTHS or more months is crops (fields carry a crop only part of the year, so a majority vote would call most
    of them bare or low vegetation). NO_DATA where no month has a class; None if no month has a map."""
    have = [mp for mp in maps if mp is not None]
    if not have:
        return None
    votes = np.stack([sum((mp == c).astype(np.uint8) for mp in have) for c in range(len(CLASSES))])
    out = np.where(votes.sum(0) > 0, votes.argmax(0), NO_DATA).astype(np.uint8)
    out[votes[CLASSES.index("crops")] >= CROP_MONTHS] = CLASSES.index("crops")
    return out


def agreement(a, b):
    """Share of pixels where two maps give the same class, over the pixels where both have one."""
    ok = (a != NO_DATA) & (b != NO_DATA)
    return float((a[ok] == b[ok]).mean()) if ok.any() else float("nan")


def second_opinion(maps, wc):
    """The model's map for the whole year (year_map) next to WorldCover."""
    C = len(CLASSES)
    typical = year_map(maps)
    has = typical != NO_DATA
    valid = (wc != NO_DATA) & has
    return {"typical": typical,
            "typical_shares": np.bincount(typical[has], minlength=C)[:C] / max(has.sum(), 1) * 100,
            "wc_shares": np.bincount(wc[valid], minlength=C)[:C] / max(valid.sum(), 1) * 100,
            "agreement": agreement(typical, wc)}


def save(result, path):
    with gzip.open(path, "wb") as f:
        pickle.dump(result, f)


def load(path):
    """A result saved by save(). Pickle runs code from the file: open only results you made yourself."""
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def biggest_changes(result, n=3):
    """The n classes whose share moved most from the first to the last month with radar: [(class, points), ...]."""
    order = np.argsort(-np.nan_to_num(np.abs(result["total"])))[:n]
    return [(CLASSES[i], float(result["total"][i])) for i in order if np.isfinite(result["total"][i])]


LUT = np.full((256, 3), 255, np.uint8)                # class index -> RGB; NO_DATA (and anything else) white
LUT[:len(COLORS)] = [[int(c[i:i + 2], 16) for i in (1, 3, 5)] for c in COLORS]


def changed(result, i):
    """(H, W) True where month i's class differs from the previous month with radar; None for months without radar and
    for the first month with radar (nothing to compare with)."""
    mp = result["maps"][i]
    prev = next((result["maps"][j] for j in range(i - 1, -1, -1) if result["maps"][j] is not None), None)
    if mp is None or prev is None:
        return None
    return (mp != prev) & (mp != NO_DATA) & (prev != NO_DATA)


def changed_share(result, i):
    """% of the square whose class changed since the previous month with radar (NaN if there is nothing to compare).
    Larger than the change in class shares, which only counts the net flow: 1% trees -> crops plus 1% crops -> trees
    is 2% of pixels changed but 0 points of change in either share."""
    ch = changed(result, i)
    return float("nan") if ch is None else float(ch.mean() * 100)


def month_image(result, i, show_change=False):
    """Month i as an RGB picture (H, W, 3) for the slider, or None if that month has no radar. show_change: pixels that
    changed since the previous month with radar in magenta, everything else faded."""
    mp = result["maps"][i]
    if mp is None:
        return None
    img = LUT[mp]
    ch = changed(result, i) if show_change else None
    if ch is not None:
        img = np.where(ch[..., None], np.array([255, 0, 255], np.uint8), (img * 0.3 + 255 * 0.7).astype(np.uint8))
    return img


def maps_figure(result):
    """The 12 maps in a 3 x 4 grid, with one colour per class and grey for months without radar."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(COLORS)
    fig, ax = plt.subplots(3, 4, figsize=(16, 14), layout="constrained")   # constrained: keeps room for every title
    for a, month, mp, n in zip(ax.ravel(), result["months"], result["maps"], result["scenes"]):
        if mp is None:
            a.set_facecolor("lightgrey")
            a.text(0.5, 0.5, "no full radar\ncoverage this month", ha="center", va="center", transform=a.transAxes)
        else:
            a.imshow(np.ma.masked_equal(mp, NO_DATA), cmap=cmap, vmin=-0.5, vmax=len(CLASSES) - 0.5, interpolation="nearest")
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
    ax[0].set_title("ESA WorldCover 2021 (second opinion)")
    ax[1].imshow(np.ma.masked_equal(result["typical"], NO_DATA), cmap=cmap, vmin=-0.5, vmax=len(CLASSES) - 0.5, interpolation="nearest")
    ax[1].set_title("model: each pixel's most frequent class")
    same = np.where((result["wc"] == NO_DATA) | (result["typical"] == NO_DATA), np.nan, (result["typical"] == result["wc"]).astype(float))
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
            assert set(np.unique(mp)) <= set(range(len(CLASSES))) | {NO_DATA}
            assert month_image(result, i, True).shape == mp.shape + (3,)
            assert np.isnan(changed_share(result, i)) or 0 <= changed_share(result, i) <= 100
    rows = ~np.isnan(result["change"]).any(axis=1)
    assert np.allclose(result["change"][rows].sum(axis=1), 0, atol=1e-6)
    if not np.isnan(result["total"]).any():
        assert abs(result["total"].sum()) < 1e-6
    assert abs(result["typical_shares"].sum() - 100) < 1e-6
    assert (result["wc"] != NO_DATA).sum() == 0 or abs(result["wc_shares"].sum() - 100) < 1e-6
    assert np.isnan(result["agreement"]) or 0 <= result["agreement"] <= 1
    big = biggest_changes(result)
    assert all(abs(a[1]) >= abs(b[1]) for a, b in zip(big, big[1:]))
    crop, grass = (np.full((2, 2), CLASSES.index(c), np.uint8) for c in ("crops", "low_vegetation"))
    assert second_opinion([crop] * CROP_MONTHS + [grass] * 9, grass)["typical"][0, 0] == CLASSES.index("crops")
    assert second_opinion([crop] * (CROP_MONTHS - 1) + [grass] * 9, grass)["typical"][0, 0] == CLASSES.index("low_vegetation")


def run_and_save(lat, lon, start, model, place=None):
    res = run(lat, lon, start, model)
    res["place"] = place or f"{lat:.4f}, {lon:.4f}"
    check(res)
    os.makedirs(RUNS, exist_ok=True)
    name = f"{RUNS}/{lat:.4f}_{lon:.4f}_{start}"
    save(res, f"{name}.pkl.gz")
    maps_figure(res).savefig(f"{name}_maps.png", dpi=80, bbox_inches="tight")
    comparison_figure(res).savefig(f"{name}_worldcover.png", dpi=80, bbox_inches="tight")
    return res, name


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--lat", type=float)
    p.add_argument("--lon", type=float)
    p.add_argument("--start", help="first month, e.g. 2024-01")
    p.add_argument("--demos", action="store_true", help="every place in DEMOS, from DEMO_START")
    a = p.parse_args()
    if not a.demos and None in (a.lat, a.lon, a.start):
        p.error("give --lat, --lon and --start, or --demos")
    init_ee()
    model = load_model()
    if a.demos:
        for place, (lat, lon) in DEMOS.items():
            try:
                res, name = run_and_save(lat, lon, DEMO_START, model, place)
            except ValueError as e:              # no radar there: say so and go on with the next place
                print(f"{place}: {e}")
                continue
            print(f"{place}: saved {name}.pkl.gz   biggest changes: {biggest_changes(res)}", flush=True)
        raise SystemExit
    res, name = run_and_save(a.lat, a.lon, a.start, model)
    np.set_printoptions(suppress=True)
    print("% of area per class each month:\n", " " * 8, " ".join(f"{c[:8]:>9s}" for c in CLASSES))
    for m, row in zip(res["months"], res["shares"]):
        print(f"{m:8s}", " ".join(f"{v:9.1f}" for v in row))
    print("\ntotal change first -> last month (points):", {c: round(float(v), 1) for c, v in zip(CLASSES, res["total"])})
    print("\nsecond opinion, ESA WorldCover 2021 vs the model's typical map:")
    print("  WorldCover %:", {c: round(float(v), 1) for c, v in zip(CLASSES, res["wc_shares"])})
    print("  model %     :", {c: round(float(v), 1) for c, v in zip(CLASSES, res["typical_shares"])})
    print(f"  pixels where they agree: {res['agreement']:.0%}")
    print(f"\nscenes per month: {res['scenes']}  ({res['pass_dir']})   saved: {name}.pkl.gz, {name}_maps.png, {name}_worldcover.png   self-check OK")
