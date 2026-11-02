"""Coherence for every site-year, from ASF HyP3: the batch version of 04_coherence_poc.ipynb.

    python coherence.py            # plan: pairs and credits per site-year, costs nothing
    python coherence.py --submit   # submit what is missing, then exit (no waiting on a paid server)
    python coherence.py --collect  # one pass: download, align and average what has finished, report what still waits, exit
    python coherence.py --check    # self-check of the pair and date logic, offline
    python coherence.py --verify   # before training: coverage per role, alignment with the radar, mean per class

Safe to stop and rerun: jobs are found by name (never paid twice), finished site-years are skipped (a .done file
in data/coherence/), and each download is deleted as soon as it is aligned, so the disk only holds zips in flight.
Output: data/coherence/<site>_<year>-<month>.npz with coh (float16, NaN where no pair covers a pixel) and n_pairs.
"""
import collections
import csv
import glob
import os
import sys
import time
import warnings
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import asf_search as asf
import hyp3_sdk
import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject
from rasterio.warp import transform as warp_points

LOOKS = "5x1"                     # about 20 m pixels, the finest HyP3 option
RAW_DIR = os.environ.get("COHERENCE_RAW", "coherence_raw")
OUT_DIR = "data/coherence"
SIZE_PX, SCALE = 1000, 10
warnings.filterwarnings("ignore", "Mean of empty slice")   # pixels no pair covers stay NaN, as intended


def site_grid(lat, lon):
    """The same 10 km UTM grid as the radar export (make_grid in 01_data_export.ipynb), without Earth Engine."""
    zone = int((lon + 180) // 6) + 1
    epsg = f"EPSG:{(32600 if lat >= 0 else 32700) + zone}"
    (x,), (y,) = warp_points("EPSG:4326", epsg, [lon], [lat])
    half = SIZE_PX * SCALE / 2
    return epsg, rasterio.Affine(SCALE, 0, x - half, 0, -SCALE, y + half)


def site_years():
    """(site, year, lat, lon, orbit direction) for every site-year with a yearly label, as LULCDataset finds them."""
    out = []
    for r in csv.DictReader(open("data/sites.csv")):
        for f in sorted(glob.glob(f"data/yearly/{r['name']}_[0-9][0-9][0-9][0-9].npz")):
            year = int(f[-8:-4])
            for month_file in sorted(glob.glob(f"data/{r['name']}_{year}-[0-9][0-9].npz")):
                with np.load(month_file) as z:
                    if "pass_dir" in z.files:   # some early exports lack it
                        out.append((r["name"], year, float(r["lat"]), float(r["lon"]), str(z["pass_dir"])))
                        break
            else:
                print(f"WARNING: {r['name']} {year} skipped, no month file says which orbit direction it uses")
    return out


def twelve_day_pairs(acq, year):
    """{burst: {(date, scene)}} -> [(burst, ref date, ref scene, secondary scene)]: consecutive passes exactly
    12 days apart whose first pass is in `year` (a gap makes a 24-day pair, which decorrelates more: skipped)."""
    pairs = []
    for burst, a in sorted(acq.items()):
        a = sorted(a)
        for (d1, s1), (d2, s2) in zip(a, a[1:]):
            if (datetime.fromisoformat(d2) - datetime.fromisoformat(d1)).days == 12 and d1.startswith(str(year)):
                pairs.append((burst, d1, s1, s2))
    return pairs


def find_pairs(site, year, lat, lon, pass_dir):
    """Every Sentinel-1A burst pair over the site's exact square. 1A only: from 2025 Sentinel-1C flies 6 days
    behind it, and mixing them would double the pairs (and credits) in 2025 only. The search's platform= option
    is not used because with the burst dataset it also returns other satellites and products."""
    crs, a = site_grid(lat, lon)
    x0, y0, x1, y1 = a.c, a.f, a.c + SIZE_PX * SCALE, a.f - SIZE_PX * SCALE
    lons, lats = warp_points(crs, "EPSG:4326", [x0, x0, x1, x1, x0], [y0, y1, y1, y0, y0])   # counter-clockwise
    wkt = "POLYGON((" + ",".join(f"{x} {y}" for x, y in zip(lons, lats)) + "))"
    for attempt in range(4):          # the search service turns away bursts of requests now and then
        try:
            found = asf.search(dataset=asf.DATASET.SLC_BURST, intersectsWith=wkt, start=f"{year}-01-01",
                               end=f"{year + 1}-01-13", polarization=asf.POLARIZATION.VV, flightDirection=pass_dir)
            break
        except Exception:
            if attempt == 3:
                raise
            time.sleep(30)
    acq = {"SENTINEL-1A": collections.defaultdict(set), "SENTINEL-1C": collections.defaultdict(set)}
    for r in found:
        p = r.properties
        if p["platform"] in acq and p["sceneName"].endswith("-BURST"):
            acq[p["platform"]][p["burst"]["fullBurstID"]].add((p["startTime"][:10], p["sceneName"]))
    # 1C only where 1A has nothing (some 2025 descending site-years)
    return twelve_day_pairs(acq["SENTINEL-1A"], year) or twelve_day_pairs(acq["SENTINEL-1C"], year)


def month_of(scene):
    """'S1_325416_IW3_20250626T031920_VV_5876-BURST' -> '2025-06'"""
    d = scene.split("_")[3]
    return f"{d[:4]}-{d[4:6]}"


def coherence_on_grid(zip_path, crs, affine):
    """A product's coherence map on the site's 10 m grid (NaN where the burst does not cover the square)."""
    with zipfile.ZipFile(zip_path) as z:
        inner = next(n for n in z.namelist() if n.endswith("_corr.tif"))
    out = np.full((SIZE_PX, SIZE_PX), np.nan, np.float32)
    with rasterio.open(f"zip://{os.path.abspath(zip_path)}!/{inner}") as src:
        reproject(rasterio.band(src, 1), out, src_nodata=src.nodata if src.nodata is not None else 0,
                  dst_transform=affine, dst_crs=crs, dst_nodata=np.nan, resampling=Resampling.bilinear)
    return out


# site-years whose jobs from 2026-10-27 stayed PENDING at HyP3 while newer jobs finished: submitted again under a new
# name, so the stuck jobs neither block finish() nor count as already submitted
RESUBMITTED = ({(s, y) for s in ("khon_kaen_rural", "adama_rural") for y in range(2022, 2026)} | {("pakokku_rural", 2025)}
               | {(s, 2024) for s in ("maradi", "maradi_rural", "gaborone", "gaborone_rural", "anantapur", "anantapur_rural")})


def job_name(site, year):
    return f"coh_{site}_{year}" + ("_r2" if (site, year) in RESUBMITTED else "")


def done_file(site, year):
    return f"{OUT_DIR}/{site}_{year}.done"


def finish(hyp3, site, year, lat, lon, pass_dir):
    """Once every job of the site-year has finished: download, align and delete each product (8 at a time),
    save the monthly averages and mark the site-year done. Returns False while jobs are still running."""
    batch = hyp3.find_jobs(name=job_name(site, year))
    if not batch.complete():
        return False
    crs, affine = site_grid(lat, lon)
    ok = [j for j in batch if j.succeeded()]

    def one(job):
        path = os.path.join(RAW_DIR, job.files[0]["filename"])
        if not os.path.exists(path):
            job.download_files(RAW_DIR)
        try:
            coh = coherence_on_grid(path, crs, affine)
        except (zipfile.BadZipFile, rasterio.errors.RasterioIOError):
            os.remove(path)       # a download cut off halfway: fetched again next round
            raise
        os.remove(path)
        return month_of(job.job_parameters["granules"][0]), coh

    t = time.time()
    with ThreadPoolExecutor(8) as pool:
        maps = list(pool.map(one, ok))
    per_month = collections.defaultdict(list)
    for month, coh in maps:
        per_month[month].append(coh)
    cover = []
    for month, m in sorted(per_month.items()):
        coh = np.nanmean(np.stack(m), 0)
        np.savez_compressed(f"{OUT_DIR}/{site}_{month}.npz", coh=coh.astype(np.float16), n_pairs=len(m))
        cover.append(np.isfinite(coh).mean())
    open(done_file(site, year), "w").write(f"{len(ok)} of {len(batch)} jobs succeeded, months {sorted(per_month)}\n")
    print(f"{site} {year}: {len(ok)} of {len(batch)} jobs, {len(per_month)} months, square covered "
          f"{min(cover, default=0):.0%} (worst month), {time.time() - t:.0f} s", flush=True)
    return True


def check():
    acq = {"b1": {("2024-01-01", "a"), ("2024-01-13", "b"), ("2024-02-06", "c"), ("2024-12-27", "d"), ("2025-01-08", "e")},
           "b2": {("2023-12-26", "x"), ("2024-01-07", "y")}}
    got = [(b, s1, s2) for b, _, s1, s2 in twelve_day_pairs(acq, 2024)]
    assert got == [("b1", "a", "b"), ("b1", "d", "e")], got   # b->c is a 24-day gap; x->y starts in 2023
    assert month_of("S1_325416_IW3_20250626T031920_VV_5876-BURST") == "2025-06"
    print("check OK")


def verify():
    """Before training with coherence: how much of each train/val/test site-month has coherence, whether it lines up
    with the radar (the shift that best matches the two should be 0, 0), and its mean per class on validation."""
    from lulc import CLASSES, REMAP, missing
    roles = {r["name"]: r["role"] for r in csv.DictReader(open("data/sites.csv"))}
    cover, shifts, per_class = collections.defaultdict(list), collections.Counter(), collections.defaultdict(list)
    for site, role in roles.items():
        if role == "holdout":
            continue
        best = None
        for f in sorted(glob.glob(f"data/{site}_[0-9][0-9][0-9][0-9]-[0-9][0-9].npz")):
            year = f[-11:-7]
            if not os.path.exists(f"data/yearly/{site}_{year}.npz"):
                continue                            # LULCDataset skips these months too
            cf = f"{OUT_DIR}/{os.path.basename(f)}"
            coh = np.load(cf)["coh"].astype(np.float32) if os.path.exists(cf) else np.full((SIZE_PX, SIZE_PX), np.nan)
            assert np.nanmin(coh, initial=0) >= 0 and np.nanmax(coh, initial=0) <= 1.0001, cf
            cover[role].append(np.isfinite(coh).mean())
            cover[role, site].append(cover[role][-1])
            if best is None or cover[role][-1] > best[0]:
                best = (cover[role][-1], f, coh, year)
        if best is None or best[0] < 0.5:
            continue
        _, f, coh, year = best                      # alignment on the site's best-covered month
        vv = np.load(f)["radar"]
        ok = np.isfinite(coh) & ~missing(vv)
        inner = (slice(10, -10),) * 2               # away from the edges that np.roll wraps around
        corr = {(dy, dx): abs(np.corrcoef(coh[inner][ok[inner]], np.roll(vv[0], (dy, dx), (0, 1))[inner][ok[inner]])[0, 1])
                for dy in range(-3, 4) for dx in range(-3, 4)}
        shifts[max(corr, key=corr.get)] += 1
        if role == "val":
            label = REMAP[np.load(f"data/yearly/{site}_{year}.npz")["label"]]
            for c in range(len(CLASSES)):
                per_class[c].append(coh[ok & (label == c)])
    for role in ("train", "val", "test"):
        sites = {k[1]: np.mean(v) for k, v in cover.items() if isinstance(k, tuple) and k[0] == role}
        print(f"{role:5s}: {len(cover[role])} site-months, coherence on {np.mean(cover[role]):.0%} of their pixels; "
              f"sites under 50%: {sorted(s for s, v in sites.items() if v < 0.5) or 'none'}")
    print("best shift (rows, cols) between coherence and radar, count of sites:", dict(shifts.most_common()))
    print("validation mean coherence per class:",
          {c: round(float(np.concatenate(per_class[i]).mean()), 2) for i, c in enumerate(CLASSES) if per_class[i]})


if __name__ == "__main__":
    if "--check" in sys.argv:
        check()
        sys.exit()
    if "--verify" in sys.argv:
        verify()
        sys.exit()
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(RAW_DIR, exist_ok=True)
    todo = [sy for sy in site_years() if not os.path.exists(done_file(*sy[:2]))]
    if "--collect" in sys.argv:    # one pass over jobs already submitted: no search, no new jobs, no credits; then exit
        hyp3 = hyp3_sdk.HyP3()
        status = collections.Counter()
        for sy in todo:
            jobs = hyp3.find_jobs(name=job_name(*sy[:2]))
            if not jobs:
                continue               # never submitted (e.g. sites added later): left for a --submit run
            try:
                if finish(hyp3, *sy):
                    continue
            except Exception as e:
                print(f"{sy[0]} {sy[1]}: {e!r}, retried next time", flush=True)
            status.update(j.status_code for j in jobs)
        print(f"{len(glob.glob(f'{OUT_DIR}/*.done'))} site-years done; jobs still waiting at HyP3: {dict(status)}")
        sys.exit()
    with ThreadPoolExecutor(3) as pool:
        pairs = dict(zip(todo, pool.map(lambda sy: find_pairs(*sy), todo)))
    for sy, p in pairs.items():
        print(f"{sy[0]:22s} {sy[1]}  {sy[4][:4].lower()}  {len({b for b, *_ in p})} bursts  {len(p):3d} pairs")
    print(f"{len(todo)} site-years to do, {sum(map(len, pairs.values()))} pairs (1 credit each, minus any already submitted)")
    if "--submit" not in sys.argv:
        sys.exit()

    hyp3 = hyp3_sdk.HyP3()
    new = {}
    for sy, p in pairs.items():
        submitted = {tuple(j.job_parameters["granules"]) for j in hyp3.find_jobs(name=job_name(*sy[:2]))}
        new[sy] = [(s1, s2) for _, _, s1, s2 in p if (s1, s2) not in submitted]
    need, credits = sum(map(len, new.values())), hyp3.check_credits()
    print(f"{need} jobs to submit, {credits} credits left", flush=True)
    if need > credits:
        sys.exit("not enough credits: nothing submitted. Wait for next month's credits or drop site-years.")
    for sy, p in new.items():
        jobs = [hyp3_sdk.HyP3.prepare_insar_isce_burst_job(s1, s2, name=job_name(*sy[:2]), looks=LOOKS) for s1, s2 in p]
        for i in range(0, len(jobs), 200):    # ponytail: 200 per request to stay under the API's batch limit, unverified
            hyp3.submit_prepared_jobs(jobs[i:i + 200])
    print("submitted; collect the results later with --collect", flush=True)
