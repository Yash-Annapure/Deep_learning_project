"""Runs training experiments one after another with no browser, and saves every picture of every run.

    uv run python run_experiments.py              run the whole queue (runs that are already finished are skipped)
    uv run python run_experiments.py --list       show the queue and which runs are done
    uv run python run_experiments.py --smoke      a tiny quick version of each run, to check the setup first
    uv run python run_experiments.py --only 0,3   run only these entries of the queue (counting from 0)
    uv run python run_experiments.py --exp exp2_architecture   run only the runs of this experiment
    uv run python run_experiments.py --rescore    score every saved model again on the CURRENT validation sites (no training).
                                                  Use it after validation sites were added: the old validation numbers are kept
                                                  as val_*_oldval in the results files, the test numbers are not touched.
    uv run python run_experiments.py --ablation experiments/<exp>/weights/<run>.pt
                                                  what that model relies on: validation IoU with one input part taken
                                                  away at a time, saved as experiments/<exp>/ablation_<run>.md and .png
    uv run python run_experiments.py --crf experiments/<exp>/weights/<run>.pt
                                                  that model's validation scores without and with the convolutional CRF

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
# Settings you can change: MONTHLY_ONLY, ARCH, ENCODER, SSL4EO, PRETRAINED, CONTEXT, AGREE, OVERSAMPLE, SPREAD, COHERENCE,
# FILTER, YEARS, SEED, EPOCHS, BATCH, LR, QUANTIZE.
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
    # exp4 + InSAR coherence and its "present" flag (2 more channels, 30% coherence dropout): the only change
    "exp5_coherence": [
    dict(CONTEXT=True, AGREE=True, COHERENCE=True, QUANTIZE=False, SEED=0),   # 28
    dict(CONTEXT=True, AGREE=True, COHERENCE=True, QUANTIZE=False, SEED=1),   # 29
    ],
    # exp4 + each pixel's yearly variability (std of VV and VH) as 2 more channels: aimed at trees / crops / low
    # vegetation, the largest confusion left; unlike coherence it can be computed live
    "exp6_spread": [
    dict(CONTEXT=True, AGREE=True, SPREAD=True, QUANTIZE=False, SEED=0),      # 30
    dict(CONTEXT=True, AGREE=True, SPREAD=True, QUANTIZE=False, SEED=1),      # 31
    ],
    # exp4 with a DeepLabV3+ decoder instead of the U-Net one: the only change
    "exp7_deeplab": [
    dict(ARCH="DeepLabV3Plus", CONTEXT=True, AGREE=True, QUANTIZE=False, SEED=0),   # 32
    dict(ARCH="DeepLabV3Plus", CONTEXT=True, AGREE=True, QUANTIZE=False, SEED=1),   # 33
    ],
    # exp4 with a ResNet-50 encoder, from scratch and from SSL4EO-S12 (self-supervised on Sentinel-1): scratch is the
    # control, so "pretraining helped" is not confused with "a bigger encoder helped"
    "exp8_ssl4eo": [
    dict(ENCODER="resnet50", CONTEXT=True, AGREE=True, QUANTIZE=False, SEED=0),               # 34  ResNet-50, scratch
    dict(ENCODER="resnet50", SSL4EO=True, CONTEXT=True, AGREE=True, QUANTIZE=False, SEED=0),  # 35  ResNet-50, SSL4EO
    dict(ENCODER="resnet50", CONTEXT=True, AGREE=True, QUANTIZE=False, SEED=1),               # 36
    dict(ENCODER="resnet50", SSL4EO=True, CONTEXT=True, AGREE=True, QUANTIZE=False, SEED=1),  # 37
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


def input_of(name):
    """How a run's input was built, read from its name: the LULCDataset options."""
    kind = "sharpen" if "sharpen" in name else "smooth" if "smooth" in name else None
    return dict(context="_ctx" in name, coherence="_coh" in name, spread="_spread" in name, filter=kind)


def load_run(weights, channels):
    """A saved model, built the way its run name says (architecture from the name, input channels given)."""
    import segmentation_models_pytorch as smp
    import torch
    from lulc import CLASSES
    from lulc import ssl4eo_input
    name = os.path.basename(weights)[:-3]
    arch = smp.UnetPlusPlus if "_upp" in name else smp.DeepLabV3Plus if "_dlv3p" in name else smp.Unet
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = arch("resnet50" if "_r50" in name else "resnet34", encoder_weights=None, in_channels=channels, classes=len(CLASSES))
    if "_ssl4eo" in name:
        model = ssl4eo_input(model)                            # the input step it was trained with (it has no weights)
    model.load_state_dict(torch.load(weights, map_location=device))
    return model.to(device).eval()


def confusion(model, ds, change=None, post=None, batch=64):
    """Validation confusion matrix and IoU per class, over real monthly labels only (as in the notebook).
    change(x) may alter each batch of inputs first (ablation, coherence left out); post(logits, x) may turn the model's
    output into the class map instead of argmax (CRF)."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from lulc import CLASSES, IGNORE
    C, device = len(CLASSES), next(model.parameters()).device.type
    cm = np.zeros((C, C), np.int64)
    with torch.no_grad():
        for x, y, _, filled in DataLoader(ds, batch_size=batch):
            if change:
                x = change(x)
            x = x.to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = model(x)
            pred = (post(logits, x) if post else logits.argmax(1)).cpu()
            keep = (y != IGNORE) & ~filled
            cm += np.bincount((y[keep] * C + pred[keep]).numpy(), minlength=C * C).reshape(C, C)
    tp = np.diag(cm).astype(float)
    return cm, tp / np.maximum(tp + (cm.sum(0) - tp) + (cm.sum(1) - tp), 1)


def rescore():
    """Scores every saved model (experiments/*/weights/*.pt) again on the current validation sites and updates its results file."""
    import glob
    import numpy as np
    from lulc import LULCDataset

    datasets = {}                                              # one validation dataset per way of preparing the radar
    for w in sorted(glob.glob("experiments/*/weights/*.pt")):
        name = os.path.basename(w)[:-3]
        f = results_file(name, os.path.dirname(os.path.dirname(w)))
        if not os.path.exists(f) or name.endswith("_smoke"):
            continue
        key = tuple(input_of(name).items())
        if key not in datasets:
            datasets[key] = LULCDataset("val", **input_of(name))
        ds = datasets[key]
        model = load_run(w, ds.channels)
        coh = input_of(name)["coherence"]

        def no_coh(x):
            x[:, -2:] = 0                                      # coherence and its flag empty, as in the prototype
            return x

        cm, iou = confusion(model, ds)
        tp = np.diag(cm).astype(float)
        r = json.load(open(f))
        if coh:                                                # the same model where coherence can't be had (the prototype)
            cm0, iou0 = confusion(model, ds, no_coh)
            r["val_miou_nocoh"], r["val_iou_nocoh"] = float(iou0[cm0.sum(1) > 0].mean()), iou0.tolist()
            print(f"{name:34s} without coherence: val mIoU {r['val_miou_nocoh']:.3f}", flush=True)
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


def ablation(weights):
    """What the model relies on: its validation IoU per class when one part of its input is taken away, written to
    experiments/<experiment>/ablation_<run>.md (table) and .png (heatmap of the change per class).
    "Taken away" = replaced by that channel's average over the validation set (a value the model has seen, unlike 0,
    which would read as the darkest radar, i.e. water), or by something that keeps the rest intact (see each row)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import TwoSlopeNorm
    from torchvision.transforms.functional import gaussian_blur
    from lulc import CLASSES, LULCDataset

    name, out = os.path.basename(weights)[:-3], os.path.dirname(os.path.dirname(weights))
    opts = input_of(name)
    ds = LULCDataset("val", **opts)
    model = load_run(weights, ds.channels)
    mean = [float(np.mean([r[c].mean(dtype=np.float64) for r in ds.radar])) for c in range(ds.channels)]
    ctx = [2, 3] if opts["context"] else []
    spread = [2 + len(ctx), 3 + len(ctx)] if opts["spread"] else []

    def to_mean(*chans):
        def f(x):
            for c in chans:
                x[:, c] = mean[c]
            return x
        return f

    def month_as_median(x):
        x[:, :2] = x[:, 2:4]
        return x

    def blur_month(x):
        x[:, :2] = gaussian_blur(x[:, :2], kernel_size=13, sigma=2.0)
        return x

    def no_coh(x):
        x[:, -2:] = 0
        return x

    variants = [("everything (baseline)", "the model as trained", None),
                ("this month's VV → average", "how much it needs VV (strongest echo: built, bare, water)", to_mean(0)),
                ("this month's VH → average", "how much it needs VH (volume scattering: vegetation)", to_mean(1)),
                ("this month's radar → average", "whether the month itself matters at all, or the yearly context does the work", to_mean(0, 1))]
    if ctx:
        variants += [("this month → the yearly median", "whether it reads THIS month, or mostly the typical year "
                      "(the month's own changes, e.g. a flood, are gone)", month_as_median),
                     ("yearly median → average", "how much it needs the yearly context", to_mean(*ctx))]
    if spread:
        variants += [("yearly spread → average", "how much it needs the yearly variability", to_mean(*spread))]
    if opts["coherence"]:
        variants += [("coherence left out", "the prototype's case: coherence can't be computed live", no_coh)]
    variants += [("this month blurred (σ = 20 m)", "how much it relies on fine texture and sharp edges in the month's radar", blur_month)]

    rows = []
    for label, why, change in variants:
        cm, iou = confusion(model, ds, change)
        rows.append((label, why, iou, float(iou[cm.sum(1) > 0].mean())))
        print(f"{label.replace('→', '->').replace('σ', 'sigma'):34s} mIoU {rows[-1][3]:.3f}", flush=True)
    base, base_m = rows[0][2], rows[0][3]

    names = [c.replace("_", " ") for c in CLASSES]
    md = [f"# What `{name}` relies on: validation IoU with one part of its input taken away",
          "", f"12 validation sites, real monthly labels only. Each row replaces one part of the input (see the second "
          "column); the numbers are the change in IoU against the full input, so **more negative = relied on more**. "
          "Baseline IoU in the first row.", "",
          "| Taken away | What it tests | mIoU | " + " | ".join(names) + " | Hit hardest |",
          "|---" * (len(names) + 4) + "|"]
    for label, why, iou, m in rows:
        if label.startswith("everything"):
            cells = [f"{m:.3f}"] + [f"{v:.3f}" for v in iou] + ["–"]
        else:
            d = iou - base
            worst = [f"{names[i]} {d[i]:+.2f}" for i in np.argsort(d)[:2] if d[i] < -0.01]
            cells = [f"{m - base_m:+.3f}"] + [f"{v:+.3f}" for v in d] + [", ".join(worst) or "nothing (all within 0.01)"]
        md.append(f"| {label} | {why} | " + " | ".join(cells) + " |")
    open(f"{out}/ablation_{name}.md", "w", encoding="utf-8").write("\n".join(md) + "\n")

    delta = np.array([[m - base_m] + list(iou - base) for _, _, iou, m in rows[1:]])
    lim = max(0.05, float(np.abs(delta).max()))
    fig, ax = plt.subplots(figsize=(12, 0.6 * len(delta) + 1.8), layout="constrained")
    ax.imshow(delta, cmap="RdBu", norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")   # red = worse without it, grey 0, blue = better
    for (i, j), v in np.ndenumerate(delta):
        ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=9, color="white" if abs(v) > 0.6 * lim else "black")
    ax.set_xticks(range(len(names) + 1), ["mIoU"] + names, rotation=30, ha="right")
    ax.set_yticks(range(len(delta)), [r[0] for r in rows[1:]])
    ax.set_title(f"{name}: change in validation IoU when one input is taken away (red = relied on)")
    fig.savefig(f"{out}/ablation_{name}.png", dpi=110)
    print(f"saved {out}/ablation_{name}.md and .png")


def crf(weights):
    """The model's validation IoU per class as is and with the convolutional CRF (lulc.conv_crf, both settings fixed in
    advance, guided by the yearly median radar), written to experiments/<experiment>/crf_<run>.md."""
    import numpy as np
    from lulc import CLASSES, CRF_SETTINGS, LULCDataset, conv_crf

    name, out = os.path.basename(weights)[:-3], os.path.dirname(os.path.dirname(weights))
    assert input_of(name)["context"], "the CRF is guided by the yearly median, so the model needs context channels"
    ds = LULCDataset("val", **input_of(name))
    model = load_run(weights, ds.channels)
    rows = [("none (argmax)", *confusion(model, ds))]
    for label, s in CRF_SETTINGS.items():
        rows.append((f"{label}: window {s['k']} px, theta_xy {s['theta_xy']}, theta_i {s['theta_i']}",
                     *confusion(model, ds, post=lambda logits, x, s=s: conv_crf(logits, x[:, 2:4], **s), batch=8)))
        print(f"{label}: mIoU {rows[-1][2][rows[-1][1].sum(1) > 0].mean():.3f}", flush=True)
    names = [c.replace("_", " ") for c in CLASSES]
    md = [f"# `{name}` with a convolutional CRF (validation, 12 sites, real monthly labels)", "",
          "| CRF | mIoU | " + " | ".join(names) + " |", "|---" * (len(names) + 2) + "|"]
    for label, cm, iou in rows:
        md.append(f"| {label} | {iou[cm.sum(1) > 0].mean():.3f} | " + " | ".join(f"{v:.3f}" for v in iou) + " |")
    open(f"{out}/crf_{name}.md", "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(md))


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
    unknown = [a for a in args if a.startswith("--") and a not in
               ("--one", "--smoke", "--rescore", "--ablation", "--crf", "--only", "--exp", "--list")]
    if unknown:                                                # an unknown option must not fall through to running the queue
        sys.exit(f"unknown option(s) {unknown}; see the top of this file")
    smoke = "--smoke" in args
    if "--one" in args:                                        # internal: one run in this process
        child(ast.literal_eval(args[args.index("--one") + 1]), smoke)
        return
    if "--rescore" in args:
        rescore()
        return
    if "--ablation" in args:
        ablation(args[args.index("--ablation") + 1])
        return
    if "--crf" in args:
        crf(args[args.index("--crf") + 1])
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
            # -u: each log line appears at once
            code = subprocess.run([sys.executable, "-u", __file__, "--one", repr(e)] + (["--smoke"] if smoke else []),
                                  stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ, MPLBACKEND="Agg")).returncode
        ok = code == 0 and os.path.exists(results_file(name, out))
        print(f"[{i}] {name}: {'finished' if ok else f'FAILED (see {folder}/log.txt)'} in {(time.time() - t) / 60:.1f} min, "
              f"{len(os.listdir(folder)) - 1} pictures saved in {folder}/", flush=True)
    print(f"\nqueue finished in {(time.time() - start) / 60:.1f} min")
    show_scores(paths.values())


if __name__ == "__main__":
    main()
