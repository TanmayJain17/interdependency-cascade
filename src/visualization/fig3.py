#!/usr/bin/env python3
"""
make_figure3.py — Regenerate the report's Figure 3 (Direct vs Total
Failures by Scenario) directly from the production Monte Carlo outputs,
so the figure can never drift out of sync with the results again.

Everything is COMPUTED, nothing hand-typed:
  direct  = mean |seeds|  per run   (monte_carlo_failures_nyc_{s}.json)
  total   = mean |t96|    per run   (cascade_results_nyc_{s}.json)
  error   = std  |t96|    across runs (real 1,000-run spread)
  multiplier = mean_total / mean_direct

Run from project root:
    python src/analysis/make_figure3.py
Writes figure3_production.png (300 dpi) to reports/figures/.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
SIM = ROOT / "data/simulation"
OUT = ROOT / "reports/figures/figure3_production.png"

SCENARIOS = [("moderate_current", "Moderate\n(Current)"),
             ("moderate_2050",   "Moderate\n(2050)"),
             ("extreme_2080",    "Extreme\n(2080)"),
             ("geoclaw_2026",    "GeoClaw\n(2026)"),
             ("geoclaw_2050",    "GeoClaw\n(2050)"),
             ("geoclaw_2080",    "GeoClaw\n(2080)")]

plt.rcParams.update({
    "font.size": 16, "axes.labelsize": 18, "axes.titlesize": 21,
    "xtick.labelsize": 16, "ytick.labelsize": 16, "legend.fontsize": 15,
    "font.family": "DejaVu Sans",
})

direct, total, err, mult = [], [], [], []
for key, _ in SCENARIOS:
    mc = json.load(open(SIM / f"monte_carlo_failures_nyc_{key}.json"))
    cr = json.load(open(SIM / f"cascade_results_nyc_{key}.json"))
    seeds = np.array([len(r.get("failed_nodes", r.get("initial_failures", [])))
                      for r in mc], dtype=float)
    t96 = np.array([len(r["failed_nodes_t96"]) for r in cr], dtype=float)
    direct.append(seeds.mean()); total.append(t96.mean())
    err.append(t96.std()); mult.append(t96.mean() / seeds.mean())
    print(f"{key:<17} direct={seeds.mean():7.1f}  total={t96.mean():7.1f} "
          f"±{t96.std():5.1f}  A={t96.mean()/seeds.mean():.2f}x  (n={len(cr)})")

labels = [lab for _, lab in SCENARIOS]
c_pd, c_pt = "#5BACE6", "#2265A8"
c_sd, c_st = "#E5A806", "#D35400"
x = np.arange(6); width = 0.38
ymax = max(t + e for t, e in zip(total, err)) * 1.22

fig, ax = plt.subplots(figsize=(13.5, 8), dpi=200)
for i in range(6):
    cd, ct = (c_pd, c_pt) if i < 3 else (c_sd, c_st)
    ax.bar(x[i] - width/2, direct[i], width, color=cd)
    ax.bar(x[i] + width/2, total[i], width, color=ct, yerr=err[i],
           capsize=5, error_kw={"ecolor": "black", "lw": 1.8})
    ax.text(x[i] + width/2, total[i] + err[i] + 0.02 * ymax,
            f"{mult[i]:.2f}x", ha="center", va="bottom",
            fontsize=17, fontweight="bold")

ax.axvline(x=2.5, color="gray", linestyle="--", linewidth=1.8, alpha=0.7)
ax.text(1.0, 0.94 * ymax, "PLUVIAL (RAIN)", ha="center", fontsize=16,
        fontweight="bold", color="#1A4E80",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#EBF5FB",
                  edgecolor="#AED6F1"))
ax.text(4.0, 0.94 * ymax, "COASTAL SURGE", ha="center", fontsize=16,
        fontweight="bold", color="#A04000",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#FEF9E7",
                  edgecolor="#F9E79F"))
ax.text(0.9, 0.56 * ymax, "Benchmark ~3.16x\n(Brunner et al. 2024)",
        ha="center", fontsize=14,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                  edgecolor="#B0BEC5", linewidth=1.2))

ax.set_title("Direct vs Total Failures by Scenario", pad=58, fontweight="bold")
ax.set_ylabel("Number of failures", labelpad=10, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylim(0, ymax)
ax.set_axisbelow(True)
ax.yaxis.grid(True, linestyle=":", alpha=0.6, color="gray")
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)

ax.legend(handles=[Patch(facecolor=c_pd, label="Rain — direct"),
                   Patch(facecolor=c_pt, label="Rain — total"),
                   Patch(facecolor=c_sd, label="Surge — direct"),
                   Patch(facecolor=c_st, label="Surge — total")],
          loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=4,
          frameon=True, facecolor="white", edgecolor="#CFD8DC")

OUT.parent.mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.savefig(OUT, dpi=300, bbox_inches="tight")
print(f"\nWrote {OUT}")