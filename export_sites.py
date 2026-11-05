"""Downloads every site in SITES that is not on disk yet, with no browser: the export notebook's settings, Earth Engine
login and helpers, then its monthly download (section 4), yearly labels (section 8) and WorldCover (section 11), then
its sanity checks (sections 5 and 8). Files already saved are skipped, so it is safe to run again after a stop. Every
log line gets a time stamp, so the speed per site can be read off the log.

    nohup uv run python -u export_sites.py > export.log 2>&1 &        then: tail -f export.log
"""
import builtins
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NOTEBOOK = "01_data_export.ipynb"
STEPS = ["## 1.", "## 2.", "## 3.", "## 4.", "## 8.", "## 11.", "## 5.", "### Check the yearly labels"]   # in this order


def first_code_cell(cells, heading):
    """The first code cell after the markdown cell that starts with this heading."""
    i = next(i for i, c in enumerate(cells) if c["cell_type"] == "markdown" and "".join(c["source"]).startswith(heading))
    return "".join(next(c for c in cells[i + 1:] if c["cell_type"] == "code")["source"])


if __name__ == "__main__":
    _print = builtins.print
    builtins.print = lambda *a, **k: _print(time.strftime("%H:%M:%S"), *a, **k)
    plt.show = lambda: plt.savefig("export_check.png", dpi=70, bbox_inches="tight")   # the yearly-label picture
    cells = json.load(open(NOTEBOOK, encoding="utf-8"))["cells"]
    g = {"__name__": "__main__"}
    for step in STEPS:
        print(f"--- {step}")
        exec(compile(first_code_cell(cells, step), NOTEBOOK, "exec"), g)
    print("export finished")
