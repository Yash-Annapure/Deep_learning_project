"""Runs training experiments one after another with no browser, and saves every picture of every run.

    uv run python run_experiments.py              run the whole queue (runs that are already finished are skipped)
    uv run python run_experiments.py --list       show the queue and which runs are done
    uv run python run_experiments.py --smoke      a tiny quick version of each run, to check the setup first
    uv run python run_experiments.py --only 0,3   run only these entries of the queue (counting from 0)

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
