"""Dataset and radar scaling shared by training and the prototype.

Each sample is a 256x256 window of one site-month: scaled radar, a 7-class label, the label's confidence,
and a flag saying whether the label came from the yearly map (filled) instead of that month's own label.
"""
import csv
import glob
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from torch.utils.data import Dataset

CLASSES = ["water", "trees", "low_vegetation", "flooded_vegetation", "crops", "built", "bare"]
COLORS = ["#2b83ba", "#1a7837", "#a6d96a", "#5ab4ac", "#fdae61", "#d7191c", "#7a5c3e"]   # one per class, in CLASSES order (bare: dark brown, unlike crops)
IGNORE = 255

# Dynamic World index -> our index. DW order: water, trees, grass, flooded, crops, shrub, built, bare, snow.
# Shrub is merged into low_vegetation, snow has no examples and becomes ignore, 255 stays ignore.
REMAP = np.full(256, IGNORE, dtype=np.uint8)
REMAP[:8] = [0, 1, 2, 3, 4, 2, 5, 6]

# dB ranges roughly matching the 1st to 99th percentile of the training sites; values outside are clipped
VV_RANGE, VH_RANGE = (-25.0, 5.0), (-35.0, -5.0)
NODATA_RADAR = -999.0   # what the export (and the prototype) put where a month has no radar for a pixel


def missing(radar):
    """(H, W): True where a pixel has no radar. Scaling would turn the fill into the darkest value (water-like), so
    training ignores these pixels and the prototype gives them no class."""
    return (radar <= NODATA_RADAR / 2).any(0)


def scale_radar(radar):
    """radar: (2, H, W) in dB, bands VV then VH. Clips each band to its range and rescales to 0..1."""
    out = np.empty(radar.shape, dtype=np.float32)
    for i, (lo, hi) in enumerate((VV_RANGE, VH_RANGE)):
        out[i] = (np.clip(radar[i], lo, hi) - lo) / (hi - lo)
    return out


def _blur(x, sigma):
    """Gaussian blur of each band of x, shape (bands, H, W), done as two 1-D passes."""
    r = max(1, int(3 * sigma))
    k = torch.exp(-torch.arange(-r, r + 1, dtype=torch.float32) ** 2 / (2 * sigma ** 2))
    k = (k / k.sum()).view(1, 1, 1, -1)
    t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))[:, None]            # (bands, 1, H, W)
    t = torch.nn.functional.conv2d(torch.nn.functional.pad(t, (r, r, 0, 0), mode="reflect"), k)
    t = torch.nn.functional.conv2d(torch.nn.functional.pad(t, (0, 0, r, r), mode="reflect"), k.transpose(2, 3))
    return t[:, 0].numpy()


def filter_radar(radar, kind):
    """Optional spatial filter on the dB values, applied before scaling. The prototype must use the same one.
    None: nothing. 'smooth': a light Gaussian blur, a simple speckle filter (less grain, softer edges).
    'sharpen': an unsharp mask, the original plus its own detail (sharper edges, but the speckle grain is boosted too)."""
    if kind is None:
        return radar
    blur = _blur(radar, 1.0)
    if kind == "smooth":
        return blur
    if kind == "sharpen":
        return radar + (radar - blur)
    raise ValueError(f"unknown filter {kind!r}")


def window_starts(size, patch, stride):
    """Start positions of windows along one axis. A last window is added so the far edge is covered."""
    starts = list(range(0, size - patch + 1, stride))
    if starts[-1] != size - patch:
        starts.append(size - patch)
    return starts


def _read(f):
    """Reads one monthly file. Decompressing releases Python's lock, so several files can be read at once in threads."""
    with np.load(f) as d:
        return d["radar"], d["label"], d["conf"]


class LULCDataset(Dataset):
    """role is 'train', 'val' or 'test', taken from data/sites.csv, so a site is only ever in one split.

    Label rule: the month's own label where it exists, otherwise the site's yearly label (filled = True).
    Confidence cutoffs and loss weights are not applied here; the training step does that.

    years=None uses every year that has a yearly label file for the site (training sites can have several,
    test and validation sites only 2024). Pass for example years=(2024,) to reproduce a single-year dataset.

    agree=True is the label-quality filter: the yearly label (used only where a month has no label of its own) is kept
    only where it agrees with ESA WorldCover (data/worldcover/<site>.npz, same classes after remapping). Everywhere
    else it becomes ignore. Monthly labels are not touched, so real changes are not filtered out.

    context=True adds two channels to every window: the site-year's median radar (VV, VH), the same for all
    12 months. Water stays dark all year, while dry soil and grass brighten in some months.

    filter=None, 'smooth' or 'sharpen' applies filter_radar to the radar before it is scaled (see there).

    quantize=True stores radar and confidence as 8-bit numbers (0 to 255 instead of 0 to 1), half the memory of the
    default 16-bit. The step is 1/255 of the range (about 0.12 dB for VV), far below the speckle noise. Windows are
    converted back to 0..1 floats when they are read, so users of the dataset see the same thing either way.
    """

    def __init__(self, role, data_dir="data", years=None, patch=256, stride=128, min_labelled=0.05, context=False,
                 agree=False, quantize=False, workers=8, filter=None):
        roles = {r["name"]: r["role"] for r in csv.DictReader(open(f"{data_dir}/sites.csv"))}
        self.patch = patch
        self.channels = 2 + 2 * context
        self.quantize = quantize
        store = (lambda a: np.round(a * 255).astype(np.uint8)) if quantize else (lambda a: a.astype(np.float16))
        self.radar, self.label, self.conf, self.filled, self.site, self.month, self.year = [], [], [], [], [], [], []
        self.items = []  # (index of site-month, row, col)
        for site in sorted(s for s, r in roles.items() if r == role):
            # the digit pattern keeps "dodoma" from also picking up the files of "dodoma_rural"
            found = sorted(int(f[-8:-4]) for f in glob.glob(f"{data_dir}/yearly/{site}_[0-9][0-9][0-9][0-9].npz"))
            if agree:
                wc = REMAP[np.load(f"{data_dir}/worldcover/{site}.npz")["label"]]  # WorldCover in our classes
            for year in found if years is None else [y for y in found if y in years]:
                yearly = np.load(f"{data_dir}/yearly/{site}_{year}.npz")
                fill = yearly["label"]  # the yearly label that stands in where a month has no label of its own
                if agree:
                    fill = np.where(REMAP[fill] == wc, fill, IGNORE)  # keep it only where both sources say the same class
                first = len(self.radar)  # this site-year's months are self.radar[first:] once the loop below is done
                def prepare(f, fill=fill, yearly=yearly):
                    """Everything for one month, in a worker thread: read it, rescale the radar, build the label and
                    list the windows worth keeping. Numpy releases Python's lock on big arrays, so threads overlap."""
                    radar_raw, label_raw, conf_raw = _read(f)
                    own = label_raw != IGNORE  # pixels where this month has its own label
                    label = REMAP[np.where(own, label_raw, fill)]
                    label[missing(radar_raw)] = IGNORE  # no radar, nothing to learn from
                    h, w = label.shape
                    keep = [(r, c) for r in window_starts(h, patch, stride) for c in window_starts(w, patch, stride)
                            if (label[r:r + patch, c:c + patch] != IGNORE).mean() >= min_labelled]
                    return (store(scale_radar(filter_radar(radar_raw, filter))), label, store(np.where(own, conf_raw, yearly["conf"])),
                            ~own, keep)

                files = sorted(glob.glob(f"{data_dir}/{site}_{year}-*.npz"))
                with ThreadPoolExecutor(max_workers=workers) as pool:  # one site-year at a time keeps memory small
                    months = list(pool.map(prepare, files))
                for f, (radar, label, conf, filled, keep) in zip(files, months):
                    k = len(self.radar)
                    self.radar.append(radar)
                    self.label.append(label)
                    self.conf.append(conf)
                    self.filled.append(filled)
                    self.site.append(site)
                    self.month.append(int(f[-6:-4]))  # file names end in _YYYY-MM.npz
                    self.year.append(year)
                    self.items.extend((k, r, c) for r, c in keep)
                if context:  # median over the site-year's months; scaling is monotonic, so scale-then-median equals median-then-scale
                    med = np.median(np.stack(self.radar[first:]), axis=0)
                    med = np.round(med).astype(np.uint8) if quantize else med.astype(np.float16)
                    for k in range(first, len(self.radar)):
                        self.radar[k] = np.concatenate([self.radar[k], med])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        k, r, c = self.items[i]
        win = (slice(r, r + self.patch), slice(c, c + self.patch))
        div = 255 if self.quantize else 1  # 8-bit storage is read back as 0..1
        x = torch.from_numpy(self.radar[k][:, win[0], win[1]].astype(np.float32) / div)
        y = torch.from_numpy(self.label[k][win].astype(np.int64))  # 255 = ignore, as CrossEntropyLoss expects
        conf = torch.from_numpy(self.conf[k][win].astype(np.float32) / div)
        return x, y, conf, torch.from_numpy(self.filled[k][win])


if __name__ == "__main__":  # self-check: python lulc.py
    sets = {role: LULCDataset(role, years=(2024,)) for role in ("train", "val", "test")}  # one year keeps the check light
    for role, ds in sets.items():
        x, y, conf, filled = ds[0]
        assert x.shape == (2, 256, 256) and y.shape == (256, 256) and conf.shape == filled.shape == y.shape
        assert 0 <= x.min() and x.max() <= 1
        assert set(y.unique().tolist()) <= set(range(len(CLASSES))) | {IGNORE}
        print(f"{role}: {len(ds)} patches from {len(set(ds.site))} sites, "
              f"{np.mean([f.mean() for f in ds.filled]):.2f} of pixels filled from the yearly label")
    assert not (set(sets["train"].site) & set(sets["val"].site) | set(sets["train"].site) & set(sets["test"].site)
                | set(sets["val"].site) & set(sets["test"].site)), "a site appears in two splits"
    gap = np.full((2, 8, 8), -10.0, np.float32)  # pixels without radar are found, in either band
    gap[0, 0, 0] = gap[1, 7, 7] = NODATA_RADAR
    assert missing(gap).sum() == 2 and missing(gap)[0, 0] and missing(gap)[7, 7]
    probe = np.random.default_rng(0).normal(-10, 3, (2, 64, 64)).astype(np.float32)  # the filters: shape kept, grain reduced or boosted
    assert filter_radar(probe, None) is probe
    sm, sh = filter_radar(probe, "smooth"), filter_radar(probe, "sharpen")
    assert sm.shape == sh.shape == probe.shape and np.isfinite(sm).all() and np.isfinite(sh).all()
    assert sm.std() < probe.std() < sh.std(), (sm.std(), probe.std(), sh.std())
    fs = LULCDataset("val", years=(2024,), filter="sharpen")  # and through the dataset
    assert fs[0][0].shape == (2, 256, 256) and 0 <= fs[0][0].min() and fs[0][0].max() <= 1
    q = LULCDataset("val", years=(2024,), quantize=True)  # 8-bit storage reads back as the same numbers, to within one step
    f16 = LULCDataset("val", years=(2024,))
    assert q.radar[0].dtype == np.uint8 and len(q) == len(f16)
    assert abs(q[0][0] - f16[0][0]).max() <= 1 / 255 and abs(q[0][2] - f16[0][2]).max() <= 1 / 255
    ctx = LULCDataset("val", context=True)  # context channels: 4 channels, in range, and identical for all months of a site
    assert ctx.channels == 4 and ctx[0][0].shape == (4, 256, 256) and 0 <= ctx[0][0].min() and ctx[0][0].max() <= 1
    first_month = next(i for i in range(1, len(ctx.site)) if ctx.site[i] == ctx.site[0])
    assert np.array_equal(ctx.radar[0][2:], ctx.radar[first_month][2:]) and not np.array_equal(ctx.radar[0][:2], ctx.radar[first_month][:2])
    lab = np.concatenate([l.ravel() for l in sets["train"].label])
    shares = np.bincount(lab, minlength=256) / lab.size
    print("train class shares:", {c: round(float(shares[i]), 3) for i, c in enumerate(CLASSES)}, "ignore", round(float(shares[IGNORE]), 3))
    print("ok")
