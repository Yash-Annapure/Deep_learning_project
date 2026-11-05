"""Invariance averaging (exp11): the model's class probabilities averaged over things that should not change the
answer, then the class with the highest average. Three kinds, each on or off:
  seeds   the 2 training runs of one setup (the luck of one run),
  views   the 8 rotations and flips of the image (the training rotates and flips every batch, so all 8 look familiar),
  months  each month averaged with the months before and after (a 3-month mean) where both have radar; the first and
          last month stay as they are, as in prototype.despike, which is the other option compared.
Every combination is scored on full 10 km scenes, as the prototype sees them, against the real monthly labels: mIoU,
and two change rates between consecutive months with radar, false changes (the model changes class where Dynamic
World does not) and missed changes (Dynamic World changes class, the model does not).

    uv run python averaging.py val,final <seed0.pt> <seed1.pt>     the exp11 decision (see its README for the rules)
    uv run python averaging.py holdout <seed0.pt> <seed1.pt>       the final model, once (exp10)
    uv run python averaging.py                                     self-check, no data or GPU needed
Writes experiments/exp11_averaging/results_<roles>.md and .png.
"""
import os
import sys
import time

import numpy as np
import torch

from lulc import CLASSES, IGNORE

C = len(CLASSES)
OUT = "experiments/exp11_averaging"
TEMPORAL = ("none", "despike", "3-month mean")


@torch.no_grad()
def views(model, x):
    """(8, C, H, W): the softmax of x (1, ch, H, W) in each of its 8 rotations and flips, each turned back to the
    original orientation (the flip undone first, then the rotation: the reverse of how they were applied). [0] is x as is."""
    out = []
    for k in range(4):
        for flip in (False, True):
            v = torch.rot90(x, k, (-2, -1))
            v = v.flip(-1) if flip else v
            with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=x.device.type == "cuda"):
                p = model(v).float().softmax(1)
            p = p.flip(-1) if flip else p
            out.append(torch.rot90(p, -k, (-2, -1))[0])
    return torch.stack(out)


def smooth(p, valid):
    """p: (T, C, H, W) probabilities of consecutive months with radar, valid: (T, H, W). A month between two others
    becomes the mean of the three where both neighbours have radar; elsewhere, and in the first and last month, it
    stays. Like despike it removes a one-month flip and keeps a change that lasts 2 months, but it also weighs how sure
    the model was."""
    out = p.clone()
    both = (valid[:-2] & valid[2:])[:, None]
    out[1:-1] = torch.where(both, (p[:-2] + p[1:-1] + p[2:]) / 3, p[1:-1])
    return out


def score(maps, label, own, valid):
    """The confusion matrix over pixels with a real monthly label, and over consecutive months where both have one:
    [unchanged pixels, of which the model changes; changed pixels, of which the model keeps the class]."""
    keep = own & valid
    cm = torch.bincount(label[keep] * C + maps[keep], minlength=C * C).view(C, C)
    both = keep[:-1] & keep[1:]
    dw, mod = label[:-1] != label[1:], maps[:-1] != maps[1:]
    ch = torch.stack([(both & ~dw).sum(), (both & ~dw & mod).sum(), (both & dw).sum(), (both & dw & ~mod).sum()])
    return cm.cpu().numpy(), ch.cpu().numpy()


def summary(cm, ch):
    """mIoU, IoU per class, false-change rate, missed-change rate."""
    tp = np.diag(cm).astype(float)
    iou = tp / np.maximum(cm.sum(0) + cm.sum(1) - tp, 1)
    return float(iou[cm.sum(1) > 0].mean()), iou, ch[1] / max(ch[0], 1), ch[3] / max(ch[2], 1)


def evaluate(ds, models, log=print):
    """Rows {(who, views, temporal): summary}, who = '1 seed' (the mean of the models' own scores) or '2 seeds'."""
    from prototype import NO_DATA, despike
    dev = next(models[0].parameters()).device
    groups = {}                                                   # (site, year) -> its site-months, in month order
    for k, key in enumerate(zip(ds.site, ds.year)):
        groups.setdefault(key, []).append(k)
    tally = {}
    for n, ((site, year), ks) in enumerate(groups.items(), 1):
        t0 = time.time()
        assert all(ds.month[a] < ds.month[b] for a, b in zip(ks, ks[1:]))
        label = torch.from_numpy(np.stack([ds.label[k] for k in ks])).long().to(dev)
        own = ~torch.from_numpy(np.stack([ds.filled[k] for k in ks])).to(dev)
        valid = label != IGNORE                                   # no radar (or no label at all): no class, as in the prototype
        h, w = label.shape[-2:]
        probs = {}
        for s, m in enumerate(models):
            one, eight = [], []
            for k in ks:
                x = torch.from_numpy(ds.radar[k].astype(np.float32))[None].to(dev)
                v = views(m, torch.nn.functional.pad(x, (0, -w % 32, 0, -h % 32), mode="reflect"))[:, :, :h, :w]
                one.append(v[0].half())
                eight.append(v.mean(0).half())
            probs[s, 1], probs[s, 8] = torch.stack(one), torch.stack(eight)
        for nv in (1, 8):
            probs["ens", nv] = sum(probs[s, nv].float() for s in range(len(models))) / len(models)
        for (who, nv), p in probs.items():
            for t in TEMPORAL:
                maps = (smooth(p.float(), valid) if t == "3-month mean" else p).argmax(1)
                if t == "despike":
                    hard = np.where(valid.cpu().numpy(), maps.cpu().numpy(), NO_DATA).astype(np.uint8)
                    maps = torch.from_numpy(np.stack(despike(list(hard)))).long().to(dev)
                cm, ch = score(maps, label, own, valid)
                old = tally.get((who, nv, t), (0, 0))
                tally[who, nv, t] = (old[0] + cm, old[1] + ch)
        log(f"{n}/{len(groups)} {site} {year}: {len(ks)} months, {time.time() - t0:.0f} s", flush=True)
    rows = {}
    for nv in (1, 8):
        for t in TEMPORAL:
            singles = [summary(*tally[s, nv, t]) for s in range(len(models))]
            rows["1 seed", nv, t] = tuple(np.mean([r[i] for r in singles], axis=0) for i in range(4))
            rows["2 seeds", nv, t] = summary(*tally["ens", nv, t])
    return rows


def rules(rows):
    """exp11's rules, as written in its README before the results: which kinds of averaging the prototype keeps."""
    def gain(a, b):                                               # mIoU gain of a over b, and its largest class drop
        return rows[a][0] - rows[b][0], float(np.min(rows[a][1] - rows[b][1]))
    g1, d1 = gain(("2 seeds", 1, "none"), ("1 seed", 1, "none"))
    g2, d2 = gain(("1 seed", 8, "none"), ("1 seed", 1, "none"))
    soft, hard = rows["1 seed", 1, "3-month mean"], rows["1 seed", 1, "despike"]
    keep = {"seeds": g1 >= 0.005 and d1 >= -0.02, "views": g2 >= 0.005 and d2 >= -0.02,
            "months": soft[0] >= hard[0] and soft[2] < hard[2] and soft[3] <= hard[3] + 0.05}
    lines = [f"1. **2 seeds**: mIoU {g1:+.3f}, largest class drop {d1:+.3f}: {'met' if keep['seeds'] else 'not met'}",
             f"2. **8 views**: mIoU {g2:+.3f}, largest class drop {d2:+.3f}: {'met' if keep['views'] else 'not met'}",
             f"3. **3-month mean instead of despike**: mIoU {soft[0] - hard[0]:+.3f}, false changes "
             f"{100 * (soft[2] - hard[2]):+.1f} points, missed changes {100 * (soft[3] - hard[3]):+.1f} points: "
             f"{'met' if keep['months'] else 'not met'}"]
    choice = ("2 seeds" if keep["seeds"] else "1 seed", 8 if keep["views"] else 1, "3-month mean" if keep["months"] else "despike")
    return lines, choice


def label_of(key):
    who, nv, t = key
    return f"{who} · {nv} view{'s' * (nv > 1)} · {t}"


def write(rows, roles, weights, seconds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [c.replace("_", " ") for c in CLASSES]
    md = [f"# exp11: averaging on the `{roles}` sites (full scenes, real monthly labels)", "",
          "Models: " + ", ".join(f"`{w}`" for w in weights) + f". {seconds / 60:.0f} min in all. '1 seed' = the mean of "
          "the two models' own scores. False changes: share of pixels whose Dynamic World class stays the same from one "
          "month with radar to the next, where the model's class changes; missed changes: the reverse.", "",
          "| Seeds · views · months | mIoU | false changes | missed changes | " + " | ".join(names) + " |",
          "|---" * (len(names) + 4) + "|"]
    for key, (m, iou, fc, mc) in rows.items():
        md.append(f"| {label_of(key)} | {m:.3f} | {100 * fc:.1f}% | {100 * mc:.1f}% | " + " | ".join(f"{v:.3f}" for v in iou) + " |")
    if "holdout" not in roles:                                    # the holdout measures; it decides nothing
        lines, choice = rules(rows)
        md += ["", "**Rules (exp11 README):**", *lines, "", f"**The prototype uses: {label_of(choice)}** "
               f"(mIoU {rows[choice][0]:.3f}, false changes {100 * rows[choice][2]:.1f}%, missed {100 * rows[choice][3]:.1f}%)"]
    os.makedirs(OUT, exist_ok=True)
    path = f"{OUT}/results_{roles.replace(',', '_')}"
    open(f"{path}.md", "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(md))

    fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=True, layout="constrained")
    y = np.arange(len(rows))
    colors = ["#2b83ba" if key[0] == "1 seed" else "#d7191c" for key in rows]       # blue 1 seed, red 2 seeds
    for ax, i, scale, title in zip(axes, (0, 2, 3), (1, 100, 100), ("mIoU: higher is better", "false changes (%): lower is better",
                                                                   "missed changes (%): lower is better")):
        ax.scatter([scale * r[i] for r in rows.values()], y, c=colors, s=40, zorder=3)
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].set_yticks(y, [label_of(k) for k in rows])
    axes[0].invert_yaxis()
    fig.suptitle(f"exp11 on the {roles} sites")
    fig.savefig(f"{path}.png", dpi=110)
    print(f"saved {path}.md and .png")


if __name__ == "__main__":
    if len(sys.argv) == 1:                                        # self-check
        torch.manual_seed(0)
        net = torch.nn.Conv2d(4, C, 1)                            # a pixel-wise model: turning the image must not change its answer
        x = torch.rand(1, 4, 32, 48)                              # not square, so a rotation that is not undone changes the shape
        assert torch.allclose(views(net, x), net(x).softmax(1).expand(8, -1, -1, -1), atol=1e-6)
        onehot = lambda seq: torch.nn.functional.one_hot(torch.tensor(seq), C).float()[:, :, None, None]
        run = lambda seq, ok: smooth(onehot(seq), ok).argmax(1).flatten().tolist()
        ok = torch.ones(4, 1, 1, dtype=torch.bool)
        assert run([0, 1, 0, 0], ok) == [0, 0, 0, 0]              # a one-month flip goes
        assert run([0, 0, 1, 1], ok) == [0, 0, 1, 1]              # a change that lasts stays, in the same month
        assert run([1, 0, 0, 1], ok) == [1, 0, 0, 1]              # first and last month untouched
        gap = ok.clone()
        gap[2] = False
        assert run([0, 1, 0, 0], gap)[1] == 1                     # a neighbour without radar: nothing to compare with
        lab = torch.tensor([[[0, 1]], [[0, 2]], [[3, 2]]])        # 3 months, 2 pixels: one changes 0 -> 3, one 1 -> 2 -> 2
        own = torch.ones_like(lab, dtype=torch.bool)
        cm, ch = score(lab, lab, own, own)
        assert summary(cm, ch)[0] == 1 and ch.tolist() == [2, 0, 2, 0]   # perfect maps: no false and no missed changes
        cm, ch = score(torch.zeros_like(lab), lab, own, own)
        assert ch.tolist() == [2, 0, 2, 2]                        # a map that never changes misses every change
        print("ok")
    else:
        from lulc import LULCDataset
        from run_experiments import input_of, load_run
        roles, weights = sys.argv[1], sys.argv[2:]
        assert len(weights) == 2, "two seeds of one setup"
        ds = LULCDataset(tuple(roles.split(",")), **input_of(os.path.basename(weights[0])[:-3]))
        t0 = time.time()
        rows = evaluate(ds, [load_run(w, ds.channels) for w in weights])
        write(rows, roles, weights, time.time() - t0)
