"""Dataset and radar scaling shared by training and the prototype.

Each sample is a 256x256 window of one site-month: scaled radar, a 7-class label, the label's confidence,
and a flag saying whether the label came from the yearly map (filled) instead of that month's own label.
"""
import csv
import glob

import numpy as np
import torch
from torch.utils.data import Dataset

CLASSES = ["water", "trees", "low_vegetation", "flooded_vegetation", "crops", "built", "bare"]
IGNORE = 255

# Dynamic World index -> our index. DW order: water, trees, grass, flooded, crops, shrub, built, bare, snow.
# Shrub is merged into low_vegetation, snow has no examples and becomes ignore, 255 stays ignore.
REMAP = np.full(256, IGNORE, dtype=np.uint8)
REMAP[:8] = [0, 1, 2, 3, 4, 2, 5, 6]

# dB ranges roughly matching the 1st to 99th percentile of the training sites; values outside are clipped
VV_RANGE, VH_RANGE = (-25.0, 5.0), (-35.0, -5.0)


def scale_radar(radar):
    """radar: (2, H, W) in dB, bands VV then VH. Clips each band to its range and rescales to 0..1."""
    out = np.empty(radar.shape, dtype=np.float32)
    for i, (lo, hi) in enumerate((VV_RANGE, VH_RANGE)):
        out[i] = (np.clip(radar[i], lo, hi) - lo) / (hi - lo)
    return out


def window_starts(size, patch, stride):
    """Start positions of windows along one axis. A last window is added so the far edge is covered."""
    starts = list(range(0, size - patch + 1, stride))
    if starts[-1] != size - patch:
        starts.append(size - patch)
    return starts


class LULCDataset(Dataset):
    """role is 'train', 'val' or 'test', taken from data/sites.csv, so a site is only ever in one split.

    Label rule: the month's own label where it exists, otherwise the site's yearly label (filled = True).
    Confidence cutoffs and loss weights are not applied here; the training step does that.
    """

    def __init__(self, role, data_dir="data", year=2024, patch=256, stride=128, min_labelled=0.05):
        roles = {r["name"]: r["role"] for r in csv.DictReader(open(f"{data_dir}/sites.csv"))}
        self.patch = patch
        self.radar, self.label, self.conf, self.filled, self.site, self.month = [], [], [], [], [], []
        self.items = []  # (index of site-month, row, col)
        for site in sorted(s for s, r in roles.items() if r == role):
            yearly = np.load(f"{data_dir}/yearly/{site}_{year}.npz")
            for f in sorted(glob.glob(f"{data_dir}/{site}_{year}-*.npz")):
                d = np.load(f)
                own = d["label"] != IGNORE  # pixels where this month has its own label
                label = REMAP[np.where(own, d["label"], yearly["label"])]
                k = len(self.radar)
                self.radar.append(scale_radar(d["radar"]).astype(np.float16))  # float16 halves memory
                self.label.append(label)
                self.conf.append(np.where(own, d["conf"], yearly["conf"]).astype(np.float16))
                self.filled.append(~own)
                self.site.append(site)
                self.month.append(int(f[-6:-4]))  # file names end in _YYYY-MM.npz
                h, w = label.shape
                for r in window_starts(h, patch, stride):
                    for c in window_starts(w, patch, stride):
                        if (label[r:r + patch, c:c + patch] != IGNORE).mean() >= min_labelled:
                            self.items.append((k, r, c))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        k, r, c = self.items[i]
        win = (slice(r, r + self.patch), slice(c, c + self.patch))
        x = torch.from_numpy(self.radar[k][:, win[0], win[1]].astype(np.float32))
        y = torch.from_numpy(self.label[k][win].astype(np.int64))  # 255 = ignore, as CrossEntropyLoss expects
        conf = torch.from_numpy(self.conf[k][win].astype(np.float32))
        return x, y, conf, torch.from_numpy(self.filled[k][win])


if __name__ == "__main__":  # self-check: python lulc.py
    sets = {role: LULCDataset(role) for role in ("train", "val", "test")}
    for role, ds in sets.items():
        x, y, conf, filled = ds[0]
        assert x.shape == (2, 256, 256) and y.shape == (256, 256) and conf.shape == filled.shape == y.shape
        assert 0 <= x.min() and x.max() <= 1
        assert set(y.unique().tolist()) <= set(range(len(CLASSES))) | {IGNORE}
        print(f"{role}: {len(ds)} patches from {len(set(ds.site))} sites, "
              f"{np.mean([f.mean() for f in ds.filled]):.2f} of pixels filled from the yearly label")
    assert not (set(sets["train"].site) & set(sets["val"].site) | set(sets["train"].site) & set(sets["test"].site)
                | set(sets["val"].site) & set(sets["test"].site)), "a site appears in two splits"
    lab = np.concatenate([l.ravel() for l in sets["train"].label])
    shares = np.bincount(lab, minlength=256) / lab.size
    print("train class shares:", {c: round(float(shares[i]), 3) for i, c in enumerate(CLASSES)}, "ignore", round(float(shares[IGNORE]), 3))
    print("ok")
