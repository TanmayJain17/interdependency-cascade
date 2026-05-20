#!/usr/bin/env python3
"""
plot_geoclaw_depths.py

Side-by-side histograms of flooded-node depths for DEP-Extreme-2080 vs the
three GeoClaw scenarios. Used as the visual for slide 4 of the Week 9 deck.

Run from project root:
    python src/flood/plot_geoclaw_depths.py
"""

from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

NODES = Path("data/flood/nyc_infra_nodes_all_flood.geojson")
OUT = Path("outputs/geoclaw_depth_histograms.png")

# (label, column, color)
SCENARIOS = [
    ("DEP Extreme-2080",  "flood_extreme_2080_depth_m", "#185FA5"),  # blue
    ("GeoClaw 2026",      "gc_2026_depth_m",            "#D85A30"),  # coral
    ("GeoClaw 2050",      "gc_2050_depth_m",            "#993C1D"),  # darker coral
    ("GeoClaw 2080",      "gc_2080_depth_m",            "#712B13"),  # darkest
]

# HAZUS fragility threshold reference
HAZUS_SATURATION_M = 2.0

# ---- Load ----
g = gpd.read_file(NODES)
print(f"Loaded {len(g):,} nodes from {NODES}")

# ---- Plot ----
fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), sharey=False)

for ax, (label, col, color) in zip(axes, SCENARIOS):
    if col not in g.columns:
        ax.set_title(f"{label}\n(missing column)")
        continue
    d = g[col].fillna(0.0).values
    flooded = d[d > 0]
    n = len(flooded)
    pct_above_2m = 100 * (flooded >= HAZUS_SATURATION_M).sum() / max(n, 1)

    ax.hist(flooded, bins=np.arange(0, 7, 0.25), color=color,
            edgecolor="white", linewidth=0.5)
    ax.axvline(HAZUS_SATURATION_M, color="black", linestyle="--",
               linewidth=1, alpha=0.6)
    ax.text(HAZUS_SATURATION_M + 0.1, ax.get_ylim()[1] * 0.92,
            "HAZUS\nsaturates", fontsize=8, color="black", alpha=0.7)

    ax.set_title(f"{label}\n{n:,} flooded nodes  •  "
                 f"{pct_above_2m:.0f}% above 2m",
                 fontsize=11)
    ax.set_xlabel("Flood depth at node (m)", fontsize=10)
    ax.set_xlim(0, 6.5)
    ax.grid(True, axis="y", alpha=0.3)

axes[0].set_ylabel("Number of nodes", fontsize=10)

fig.suptitle("Flood depth distribution at infrastructure nodes — DEP vs GeoClaw",
             fontsize=13, fontweight="bold", y=1.02)
fig.tight_layout()

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Saved to {OUT}")
plt.show()