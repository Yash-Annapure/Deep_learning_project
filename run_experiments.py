"""Runs training experiments one after another with no browser, and saves every picture of every run.

    uv run python run_experiments.py              run the whole queue (runs that are already finished are skipped)
    uv run python run_experiments.py --list       show the queue and which runs are done
    uv run python run_experiments.py --smoke      a tiny quick version of each run, to check the setup first
    uv run python run_experiments.py --only 0,3   run only these entries of the queue (counting from 0)
    uv run python run_experiments.py --rescore    score every saved model again on the CURRENT validation sites (no training).
                                                  Use it after validation sites were added: the old validation numbers are kept
                                                  as val_*_oldval in the results files, the test numbers are not touched.

To keep it going when your browser or connection drops:
    nohup uv run python run_experiments.py > queue.log 2>&1 &        then: tail -f queue.log

Each run is the training notebook (02_train.ipynb) with the settings below changed. For every run you get a folder
runs/<name>/ with log.txt (the training log, with seconds per epoch) and every picture the notebook draws, as numbered PNGs.
The weights go to weights/<name>.pt and the scores to results_<name>.json, as in the notebook.
"""
import ast
import json
import os
import re
import subprocess
import sys
import time

NOTEBOOK = "02_train.ipynb"

# The queue. Each entry changes some settings of the notebook's first code cell. Order = what runs first.
# Settings you can change: MONTHLY_ONLY, CONTEXT, AGREE, FILTER, YEARS, SEED, EPOCHS, BATCH, LR, QUANTIZE.
EXPERIMENTS = [
    dict(YEARS=(2024,), SEED=0),                  # 0  sites effect: 21 sites, 2024 only (about the same steps as the old runs)
    dict(YEARS=(2024,), SEED=1),                  # 1
    dict(SEED=1),                                 # 2  second seed of the 4-year run (seed 0 is done)
    dict(AGREE=True, SEED=0),                     # 3  crops: keep a filled label only where WorldCover agrees
    dict(FILTER="sharpen", SEED=0),               # 4  your professor's sharpening idea
    dict(FILTER="smooth", SEED=0),                # 5  the standard speckle filter
    dict(YEARS=(2024,), EPOCHS=60, SEED=0),       # 6  years effect: 2024 only but as many steps as the 4-year run
    dict(YEARS=(2024,), EPOCHS=60, SEED=1),       # 7
]


def settings_cell_with(overrides, smoke):
    """The notebook's first code cell with the chosen settings swapped in."""
    nb = json.load(open(NOTEBOOK, encoding="utf-8"))
    cell = next("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    changes = dict(overrides, **({"SMOKE": True} if smoke else {}))
    for key, value in changes.items():
        cell, n = re.subn(rf"^({key}\s*=\s*)[^#\n]*?(\s*(#.*)?)$", lambda m: f"{m.group(1)}{value!r}{m.group(2)}",
                          cell, count=1, flags=re.M)
        if n != 1:
            sys.exit(f"setting {key} not found in the notebook's settings cell")
    return cell


def run_name(overrides, smoke):
    g = {}
    exec(settings_cell_with(overrides, smoke), g)
    return g["NAME"]


def child(overrides, smoke):
    """Runs one experiment in this process: the notebook's code cells in order, saving every figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nb = json.load(open(NOTEBOOK, encoding="utf-8"))
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    cells[0] = settings_cell_with(overrides, smoke)
    g = {"__name__": "__main__"}
    count = [0]

    def save_figures(*a, **k):
        folder = f"runs/{g['NAME']}"
        os.makedirs(folder, exist_ok=True)
        for num in plt.get_fignums():
            fig = plt.figure(num)
            axes = fig.get_axes()
            title = (fig._suptitle.get_text() if fig._suptitle else (axes[0].get_title() if axes else "")) or "figure"
            count[0] += 1
            slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")[:45]
            fig.savefig(f"{folder}/{count[0]:02d}_{slug}.png", dpi=80, bbox_inches="tight")
        plt.close("all")

    plt.show = save_figures
    for source in cells:
        exec(compile(source, NOTEBOOK, "exec"), g)


def rescore():
    """Scores every saved model (weights/*.pt) again on the current validation sites and updates its results file."""
    import glob
    import numpy as np
    import segmentation_models_pytorch as smp
    import torch
    from torch.utils.data import DataLoader
    from lulc import CLASSES, IGNORE, LULCDataset

    C = len(CLASSES)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    datasets = {}                                              # one validation dataset per way of preparing the radar
    for w in sorted(glob.glob("weights/*.pt")):
        name = os.path.basename(w)[:-3]
        if not os.path.exists(f"results_{name}.json") or name.endswith("_smoke"):
            continue
        ctx = "_ctx" in name                                   # the run name says how its input was built
        kind = "sharpen" if "sharpen" in name else "smooth" if "smooth" in name else None
        if (ctx, kind) not in datasets:
            datasets[(ctx, kind)] = LULCDataset("val", context=ctx, filter=kind)
        ds = datasets[(ctx, kind)]
        model = smp.Unet("resnet34", encoder_weights=None, in_channels=ds.channels, classes=C).to(device)
        model.load_state_dict(torch.load(w, map_location=device))
        model.eval()
        cm = np.zeros((C, C), np.int64)
        with torch.no_grad():
            for x, y, _, filled in DataLoader(ds, batch_size=64):
                with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
                    pred = model(x.to(device)).argmax(1).cpu()
                keep = (y != IGNORE) & ~filled                 # real monthly labels only, as in the notebook
                cm += np.bincount((y[keep] * C + pred[keep]).numpy(), minlength=C * C).reshape(C, C)
        tp = np.diag(cm).astype(float)
        iou = tp / np.maximum(tp + (cm.sum(0) - tp) + (cm.sum(1) - tp), 1)
        r = json.load(open(f"results_{name}.json"))
        for k in ("val_miou", "val_water_iou", "val_water_precision"):
            if k in r and k + "_oldval" not in r:
                r[k + "_oldval"] = r[k]                        # keep what the old validation set said
        r["val_miou"] = float(iou[cm.sum(1) > 0].mean())
        r["val_water_iou"] = float(iou[0])
        r["val_water_precision"] = float(tp[0] / max(cm[:, 0].sum(), 1))
        json.dump(r, open(f"results_{name}.json", "w"))
        print(f"{name:34s} val mIoU {r['val_miou']:.3f} (was {r.get('val_miou_oldval', float('nan')):.3f})   "
              f"water IoU {r['val_water_iou']:.3f}   water precision {r['val_water_precision']:.3f}", flush=True)


def show_scores(names):
    print("\nSUMMARY (validation = the numbers to decide with, test = the final exam)")
    print(f"{'run':34s}{'val mIoU':>10s}{'val water':>11s}{'test mIoU':>11s}{'test water':>12s}")
    for n in names:
        f = f"results_{n}.json"
        if os.path.exists(f):
            r = json.load(open(f))
            print(f"{n:34s}{r.get('val_miou', float('nan')):10.3f}{r.get('val_water_iou', float('nan')):11.3f}"
                  f"{r['miou']:11.3f}{r['iou'][0]:12.3f}")


def main():
    args = sys.argv[1:]
    smoke = "--smoke" in args
    if "--one" in args:                                        # internal: one run in this process
        child(ast.literal_eval(args[args.index("--one") + 1]), smoke)
        return
    if "--rescore" in args:
        rescore()
        return
    only = None
    if "--only" in args:
        only = {int(i) for i in args[args.index("--only") + 1].split(",")}
    todo = [(i, e) for i, e in enumerate(EXPERIMENTS) if only is None or i in only]
    names = {i: run_name(e, smoke) for i, e in todo}

    if "--list" in args:
        for i, e in todo:
            done = os.path.exists(f"results_{names[i]}.json")
            print(f"{i}  {'done   ' if done else 'pending'}  {names[i]:34s} {e}")
        return

    os.makedirs("runs", exist_ok=True)
    start = time.time()
    for i, e in todo:
        name = names[i]
        if os.path.exists(f"results_{name}.json") and not smoke:
            print(f"[{i}] {name}: already done, skipped", flush=True)
            continue
        os.makedirs(f"runs/{name}", exist_ok=True)
        print(f"[{i}] {name}: started {time.strftime('%H:%M')}  {e}", flush=True)
        t = time.time()
        with open(f"runs/{name}/log.txt", "w") as log:
            code = subprocess.run([sys.executable, __file__, "--one", repr(e)] + (["--smoke"] if smoke else []),
                                  stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ, MPLBACKEND="Agg")).returncode
        ok = code == 0 and os.path.exists(f"results_{name}.json")
        print(f"[{i}] {name}: {'finished' if ok else 'FAILED (see runs/' + name + '/log.txt)'} in {(time.time() - t) / 60:.1f} min, "
              f"{len(os.listdir(f'runs/{name}')) - 1} pictures saved in runs/{name}/", flush=True)
    print(f"\nqueue finished in {(time.time() - start) / 60:.1f} min")
    show_scores(names.values())


if __name__ == "__main__":
    main()
