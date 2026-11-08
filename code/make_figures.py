"""Render the paper's three figures from the result files.

Usage:  python -I code/make_figures.py [outdir]
Reads   results/econ/montecarlo_curves.csv, results/econ/tornado.csv,
        results/e2/e2_summary.json
Writes  fig1_ownership_curves.pdf, fig2_sensitivity.pdf, fig3_e2_harm.pdf
        (default outdir: paper/figures)

Monochrome by design: series are told apart by line style, marker and direct
labels, so the figures survive greyscale printing.
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Match the paper's Palatino body text where TeX Gyre Pagella is installed.
for p in Path("/usr/share/texmf/fonts/opentype/public/tex-gyre").glob("texgyrepagella-*.otf"):
    font_manager.fontManager.addfont(str(p))
family = "TeX Gyre Pagella" if any(
    f.name == "TeX Gyre Pagella" for f in font_manager.fontManager.ttflist) else "serif"

INK, MID, LIGHT, GRID = "#111111", "#555555", "#8a8a8a", "#e2e2e2"
plt.rcParams.update({
    "font.family": family, "font.size": 9, "axes.titlesize": 9,
    "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.edgecolor": MID, "axes.linewidth": 0.6, "axes.labelcolor": INK,
    "xtick.color": MID, "ytick.color": MID, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 3, "text.parse_math": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
})


def money(x):
    return f"${x/1000:,.0f}k" if x >= 1000 else f"${x:,.0f}"


def crossing(bills, ps, level=0.5):
    """Bill at which P(own cheaper) first reaches `level`, interpolated in log bill."""
    for i in range(1, len(ps)):
        if ps[i - 1] < level <= ps[i]:
            lb0, lb1 = np.log(bills[i - 1]), np.log(bills[i])
            t = (level - ps[i - 1]) / (ps[i] - ps[i - 1])
            return float(np.exp(lb0 + t * (lb1 - lb0)))
    return None


# Figure 1 ---------------------------------------------------------------
rows = list(csv.DictReader(open(ROOT / "results/econ/montecarlo_curves.csv")))
series = [
    ("static prices (counterfactual)", "Prices held flat", INK, (0, (1.2, 1.6)), None),
    ("pooled adapter serving (level 3)", "Shared adapter serving", MID, (0, (5, 2)), None),
    ("lower-cost labor $25/hr", "Labor $25/h", LIGHT, "-", None),
    ("labor uncertain $20-$121/hr", "Labor $20 to $121/h (base)", INK, "-", None),
    ("US labor $121/hr", "Labor $121/h", MID, "-", None),
]
fig, ax = plt.subplots(figsize=(6.3, 3.0))
handles, labels = [], []
ax.axhline(0.5, color=GRID, lw=0.8, zorder=0)
ax.text(1.08e3, 0.515, "even odds", color=LIGHT, fontsize=7.5, va="bottom")
for key, label, color, ls, _ in series:
    pts = [(float(r["monthly_api_bill"]), float(r["p_own_cheaper"]))
           for r in rows if r["scenario"] == key]
    pts.sort()
    b, p = zip(*pts)
    lw = 1.8 if "base" in label else 1.2
    ax.plot(b, p, color=color, ls=ls, lw=lw, solid_capstyle="round")
    x50 = crossing(b, p)
    if x50:
        ax.plot([x50], [0.5], "o", ms=3.6, color=color, mec="white", mew=0.8, zorder=5)
    handles.append(ax.plot([], [], color=color, ls=ls, lw=lw)[0])
    labels.append(f"{label}: even odds at {money(x50)}")
ax.set_xscale("log")
ax.set_xlim(1e3, 1e6)
ax.set_ylim(-0.02, 1.02)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
ax.set_xticks([1e3, 1e4, 1e5, 1e6])
ax.set_xticklabels(["$1k", "$10k", "$100k", "$1M"])
ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2,
          frameon=False, fontsize=7.5, handlelength=2.6, columnspacing=1.6)
ax.set_xlabel("Monthly API bill for one narrow task (today, Claude Sonnet 5.5 prices)")
ax.set_ylabel("Share of draws in which\nowning is cheaper")
ax.grid(axis="y", color=GRID, lw=0.5)
ax.set_axisbelow(True)
fig.savefig(OUT / "fig1_ownership_curves.pdf")
plt.close(fig)

# Figure 2 ---------------------------------------------------------------
t = list(csv.DictReader(open(ROOT / "results/econ/tornado.csv")))
t = [r for r in t if abs(float(r["breakeven_low_setting"]) - float(r["breakeven_high_setting"])) > 1]
base = float(t[0]["base"])
t.sort(key=lambda r: abs(float(r["breakeven_high_setting"]) - float(r["breakeven_low_setting"])))
fig, ax = plt.subplots(figsize=(6.3, 3.1))
for i, r in enumerate(t):
    lo, hi = float(r["breakeven_low_setting"]), float(r["breakeven_high_setting"])
    a, b = min(lo, hi), max(lo, hi)
    ax.plot([a, b], [i, i], color=MID, lw=5.5, solid_capstyle="butt")
    ax.plot([lo], [i], marker="o", ms=3.4, color="white", mec=INK, mew=0.8, zorder=4)
    ax.plot([hi], [i], marker="o", ms=3.4, color=INK, mec=INK, mew=0.8, zorder=4)
    ax.text(b + 600, i, f"{money(a)} to {money(b)}", va="center", fontsize=7, color=MID)
ax.axvline(base, color=INK, lw=0.8)
ax.text(base, len(t) - 0.3, f"base {money(base)}", ha="center", va="bottom", fontsize=7.5,
        bbox=dict(fc="white", ec="none", pad=1))
ax.set_yticks(range(len(t)))
ax.set_yticklabels([r["parameter"] for r in t], fontsize=7.5)
ax.set_ylim(-0.7, len(t) - 0.1)
ax.set_xlim(14000, 52000)
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: money(x)))
ax.set_xlabel("Monthly API bill at which owning breaks even (US labor, other inputs at median)")
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.grid(axis="x", color=GRID, lw=0.5)
ax.set_axisbelow(True)
ax.plot([], [], "o", ms=3.4, color="white", mec=INK, label="first setting in the label")
ax.plot([], [], "o", ms=3.4, color=INK, label="second setting")
ax.legend(loc="lower right", frameon=False, fontsize=7, handletextpad=0.3)
fig.savefig(OUT / "fig2_sensitivity.pdf")
plt.close(fig)

# Figure 3 ---------------------------------------------------------------
e2 = json.load(open(ROOT / "results/e2/e2_summary.json"))["conditions"]
order = [("label_noise_systematic", "Systematic label errors"),
         ("empty_target", "Empty answers"),
         ("label_noise_random", "Random label errors"),
         ("truncated_target", "Answers cut short"),
         ("dup_concentrated", "Concentrated duplicates")]
by = {(c["condition"], round(c["rate"], 2)): c for c in e2}
fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.6), sharey=True, sharex=True)
for ax, metric, title in [(axes[0], "exact_match", "Greedy decoding"),
                          (axes[1], "exact_match_sampled", "Sampled at temperature 1")]:
    for i, (cond, _) in enumerate(order):
        y = len(order) - 1 - i
        for rate, dy, face in [(0.10, 0.14, "white"), (0.30, -0.14, INK)]:
            c = by[(cond, rate)]
            d, lo, hi = (100 * c[f"{metric}_{k}"] for k in ("diff", "lo", "hi"))
            ax.plot([lo, hi], [y + dy, y + dy], color=MID, lw=0.9)
            ax.plot([d], [y + dy], "o", ms=4, color=face, mec=INK, mew=0.9, zorder=4)
    ax.axvline(0, color=INK, lw=0.6)
    ax.set_title(title, loc="left", color=INK)
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.set_xlim(-36, 4)
    ax.set_xlabel("Change in exact match vs clean, points")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
axes[0].set_yticks(range(len(order)))
axes[0].set_yticklabels([lab for _, lab in reversed(order)])
axes[1].plot([], [], "o", ms=4, color="white", mec=INK, label="10% of rows")
axes[1].plot([], [], "o", ms=4, color=INK, mec=INK, label="30% of rows")
axes[1].legend(loc="lower left", frameon=False, fontsize=7, handletextpad=0.2)
fig.tight_layout(w_pad=1.2)
fig.savefig(OUT / "fig3_e2_harm.pdf")
plt.close(fig)
print("wrote", *sorted(p.name for p in OUT.glob("fig*.pdf")))
