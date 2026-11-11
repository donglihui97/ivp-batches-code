"""Plot the convergence results of ivp_batches.py."""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DET, RAND, REF = "#2a78d6", "#eb6834", "#8a8a85"
res = json.load(open(sys.argv[1]))
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.7), constrained_layout=True)
panels = [("one", "one batch", 1.0, 1.25), ("two", "two batches", 2.0, 2.5)]
for ax, (key, title, pd, pr) in zip(axes, panels):
    rows = res[key]
    nd = np.array([r["n_det"] for r in rows]); ed = np.array([r["err_det"] for r in rows])
    nr = np.array([r["n"] for r in rows]); er = np.array([r["rms"] for r in rows])
    ax.loglog(nd, ed, "o-", color=DET, lw=1.5, ms=4, label="deterministic part $z$")
    ax.loglog(nr, er, "s-", color=RAND, lw=1.5, ms=4, label="with randomized correction")
    for p, e0, n0, fac, va in ((pd, ed[-1], nd[-1], 3.0, "bottom"), (pr, er[-1], nr[-1], 0.33, "top")):
        nn = np.array([n0 / 30, n0])
        ax.loglog(nn, fac * e0 * (nn / n0) ** (-p), "--", color=REF, lw=1)
        ax.annotate(f"$n^{{-{p:g}}}$", (nn.prod() ** 0.5, fac * e0 * 30 ** (p / 2)), fontsize=7,
                    color="#555550", ha="left" if va == "bottom" else "right", va=va,
                    xytext=(3 if va == "bottom" else -3, 0), textcoords="offset points")
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("number of queries $n$")
    ax.grid(True, which="major", color="#e6e6e3", lw=0.6)
axes[0].set_ylabel("uniform error")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, fontsize=7, ncol=2, loc="outside upper center")
fig.savefig(sys.argv[2])




