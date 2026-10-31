"""Runs training experiments one after another with no browser, and saves every picture of every run.

    uv run python run_experiments.py              run the whole queue (runs that are already finished are skipped)
    uv run python run_experiments.py --list       show the queue and which runs are done
    uv run python run_experiments.py --smoke      a tiny quick version of each run, to check the setup first
    uv run python run_experiments.py --only 0,3   run only these entries of the queue (counting from 0)
    uv run python run_experiments.py --exp exp2_architecture   run only the runs of this experiment
    uv run python run_experiments.py --rescore    score every saved model again on the CURRENT validation sites (no training).
                                                  Use it after validation sites were added: the old validation numbers are kept
                                                  as val_*_oldval in the results files, the test numbers are not touched.

To keep it going when your browser or connection drops:
    nohup uv run python -u run_experiments.py > queue.log 2>&1 &        then: tail -f queue.log

Each run is the training notebook (02_train.ipynb) with the settings below changed. Everything a run makes goes to
experiments/<experiment>/: runs/<name>/ gets log.txt (the training log, with seconds per epoch) and every picture the
notebook draws, as numbered PNGs; weights/<name>.pt the weights; results/results_<name>.json the scores.
"""
import ast
import json
import os
import re
import subprocess
import sys
import time

NOTEBOOK = "02_train.ipynb"

# The queue, one list per experiment (= folder under experiments/). Each entry changes some settings of the notebook's
# first code cell. Order = what runs first; the numbers count across all experiments, for --only.
# Settings you can change: MONTHLY_ONLY, ARCH, PRETRAINED, CONTEXT, AGREE, OVERSAMPLE, FILTER, YEARS, SEED, EPOCHS, BATCH,
# LR, QUANTIZE.
QUEUES = {
    "exp1_data": [                                # what the data does: sites vs years, label filter, radar filters
    dict(YEARS=(2024,), SEED=0),                  # 0  sites effect: 21 sites, 2024 only (about the same steps as the old runs)
    dict(YEARS=(2024,), SEED=1),                  # 1
    dict(SEED=1),                                 # 2  second seed of the 4-year run (seed 0 is done)
    dict(AGREE=True, SEED=0),                     # 3  crops: keep a filled label only where WorldCover agrees
    dict(FILTER="sharpen", SEED=0),               # 4  your professor's sharpening idea
    dict(FILTER="smooth", SEED=0),                # 5  the standard speckle filter
    dict(YEARS=(2024,), EPOCHS=60, SEED=0),       # 6  years effect: 2024 only but as many steps as the 4-year run
    dict(YEARS=(2024,), EPOCHS=60, SEED=1),       # 7
    ],
    # architecture x context, compared with the two hybrid runs of exp1. One of each first, then the second seeds,
    # so half-way through there is already one number per variant.
    "exp2_architecture": [
    dict(CONTEXT=True, QUANTIZE=False, SEED=0),                          # 8   context only
    dict(ARCH="UnetPlusPlus", QUANTIZE=False, SEED=0),                   # 9   U-Net++ only
    dict(PRETRAINED=True, QUANTIZE=False, SEED=0),                       # 10  ImageNet only
    dict(ARCH="UnetPlusPlus", CONTEXT=True, QUANTIZE=False, SEED=0),     # 11  U-Net++ + context
    dict(PRETRAINED=True, CONTEXT=True, QUANTIZE=False, SEED=0),         # 12  ImageNet + context
    dict(CONTEXT=True, QUANTIZE=False, SEED=1),                          # 13
    dict(ARCH="UnetPlusPlus", QUANTIZE=False, SEED=1),                   # 14
    dict(PRETRAINED=True, QUANTIZE=False, SEED=1),                       # 15
    dict(ARCH="UnetPlusPlus", CONTEXT=True, QUANTIZE=False, SEED=1),     # 16
    dict(PRETRAINED=True, CONTEXT=True, QUANTIZE=False, SEED=1),         # 17
    ],
    # on top of context (the exp2 winner for water): cleaner labels x more rare-class windows, a 2x2 grid whose
    # fourth corner (context alone) is hybrid_ctx from exp2
    "exp3_agree_rare": [
    dict(CONTEXT=True, AGREE=True, QUANTIZE=False, SEED=0),                   # 18  context + cleaner labels
    dict(CONTEXT=True, OVERSAMPLE=3, QUANTIZE=False, SEED=0),                 # 19  context + rare windows 3x
    dict(CONTEXT=True, AGREE=True, OVERSAMPLE=3, QUANTIZE=False, SEED=0),     # 20  both
    dict(CONTEXT=True, AGREE=True, QUANTIZE=False, SEED=1),                   # 21
    dict(CONTEXT=True, OVERSAMPLE=3, QUANTIZE=False, SEED=1),                 # 22
    dict(CONTEXT=True, AGREE=True, OVERSAMPLE=3, QUANTIZE=False, SEED=1),     # 23
    ],
    # the exp1 baseline again, now with its best epoch chosen on the current 10 validation sites (the original runs
    # chose on the old 4), so every step of the comparison is measured with the same ruler
    "exp1_rerun": [
    dict(QUANTIZE=False, SEED=0),                                              # 24
    dict(QUANTIZE=False, SEED=1),                                              # 25
    ],
    # the exp3 winner (context + AGREE) again, now on all 50 sites (29 train, 12 val): the prototype's model, and the
    # baseline that coherence must beat. Compare after --rescore, which scores the 34-site models on the same 12 val sites.
    "exp4_50sites": [
    dict(CONTEXT=True, AGREE=True, QUANTIZE=False, SEED=0),                   # 26
    dict(CONTEXT=True, AGREE=True, QUANTIZE=False, SEED=1),                   # 27
    ],
}
EXPERIMENTS = [dict(e, EXPERIMENT=x) for x, runs in QUEUES.items() for e in runs]


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


def run_paths(overrides, smoke):
    """(run name, its experiment folder), as the notebook will compute them."""
    g = {}
    exec(settings_cell_with(overrides, smoke), g)
    return g["NAME"], g["OUT"]


def results_file(name, out):
    return f"{out}/results/results_{name}.json"


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
        folder = f"{g['OUT']}/runs/{g['NAME']}"
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
    """Scores every saved model (experiments/*/weights/*.pt) again on the current validation sites and updates its results file."""
    import glob
    import numpy as np
    import segmentation_models_pytorch as smp
    import torch
    from torch.utils.data import DataLoader
    from lulc import CLASSES, IGNORE, LULCDataset

    C = len(CLASSES)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    datasets = {}                                              # one validation dataset per way of preparing the radar
    for w in sorted(glob.glob("experiments/*/weights/*.pt")):
        name = os.path.basename(w)[:-3]
        f = results_file(name, os.path.dirname(os.path.dirname(w)))
        if not os.path.exists(f) or name.endswith("_smoke"):
            continue
        ctx = "_ctx" in name                                   # the run name says how its input was built
        kind = "sharpen" if "sharpen" in name else "smooth" if "smooth" in name else None
        if (ctx, kind) not in datasets:
            datasets[(ctx, kind)] = LULCDataset("val", context=ctx, filter=kind)
        ds = datasets[(ctx, kind)]
        arch = smp.UnetPlusPlus if "_upp" in name else smp.Unet
        model = arch("resnet34", encoder_weights=None, in_channels=ds.channels, classes=C).to(device)
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
        r = json.load(open(f))
        for k in ("val_miou", "val_water_iou", "val_water_precision"):
            if k in r and k + "_oldval" not in r:
                r[k + "_oldval"] = r[k]                        # keep what the old validation set said
        r["val_miou"] = float(iou[cm.sum(1) > 0].mean())
        r["val_iou"] = iou.tolist()
        r["val_water_iou"] = float(iou[0])
        r["val_water_precision"] = float(tp[0] / max(cm[:, 0].sum(), 1))
        json.dump(r, open(f, "w"))
        print(f"{name:34s} val mIoU {r['val_miou']:.3f} (was {r.get('val_miou_oldval', float('nan')):.3f})   "
              f"water IoU {r['val_water_iou']:.3f}   water precision {r['val_water_precision']:.3f}", flush=True)


def show_scores(paths):
    print("\nSUMMARY (validation = the numbers to decide with, test = the final exam)")
    print(f"{'run':34s}{'val mIoU':>10s}{'val water':>11s}{'test mIoU':>11s}{'test water':>12s}")
    for n, out in paths:
        f = results_file(n, out)
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
    only = {int(i) for i in args[args.index("--only") + 1].split(",")} if "--only" in args else None
    exp = args[args.index("--exp") + 1] if "--exp" in args else None
    todo = [(i, e) for i, e in enumerate(EXPERIMENTS)
            if (only is None or i in only) and (exp is None or e["EXPERIMENT"] == exp)]
    paths = {i: run_paths(e, smoke) for i, e in todo}

    if "--list" in args:
        for i, e in todo:
            done = os.path.exists(results_file(*paths[i]))
            print(f"{i:2d}  {'done   ' if done else 'pending'}  {paths[i][0]:34s} {e}")
        return

    start = time.time()
    for i, e in todo:
        name, out = paths[i]
        if os.path.exists(results_file(name, out)) and not smoke:
            print(f"[{i}] {name}: already done, skipped", flush=True)
            continue
        folder = f"{out}/runs/{name}"
        os.makedirs(folder, exist_ok=True)
        print(f"[{i}] {name}: started {time.strftime('%H:%M')}  {e}", flush=True)
        t = time.time()
        with open(f"{folder}/log.txt", "w") as log:
            code = subprocess.run([sys.executable, "-u", __file__, "--one", repr(e)]   # -u: each log line appears at once + (["--smoke"] if smoke else []),
                                  stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ, MPLBACKEND="Agg")).returncode
        ok = code == 0 and os.path.exists(results_file(name, out))
        print(f"[{i}] {name}: {'finished' if ok else f'FAILED (see {folder}/log.txt)'} in {(time.time() - t) / 60:.1f} min, "
              f"{len(os.listdir(folder)) - 1} pictures saved in {folder}/", flush=True)
    print(f"\nqueue finished in {(time.time() - start) / 60:.1f} min")
    show_scores(paths.values())


if __name__ == "__main__":
    main()
