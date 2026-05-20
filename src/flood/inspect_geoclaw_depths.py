#!/usr/bin/env python3
"""
inspect_geoclaw_depths.py

Diagnostic: distribution of flood depths at infrastructure nodes for all 6
scenarios. Helps identify whether GeoClaw scenarios saturate the HAZUS
fragility curves (which have median ~0.1-0.6m depending on infra type).

Run from project root:
    python src/flood/inspect_geoclaw_depths.py
"""

from pathlib import Path
import geopandas as gpd
import numpy as np

NODES = Path("data/flood/nyc_infra_nodes_all_flood.geojson")
OUT = Path("outputs/geoclaw_depth_diagnostic.txt")

SCENARIOS = [
    ("DEP moderate_current", "flood_moderate_current_depth_m"),
    ("DEP moderate_2050",    "flood_moderate_2050_depth_m"),
    ("DEP extreme_2080",     "flood_extreme_2080_depth_m"),
    ("GeoClaw 2026",         "gc_2026_depth_m"),
    ("GeoClaw 2050",         "gc_2050_depth_m"),
    ("GeoClaw 2080",         "gc_2080_depth_m"),
]

# HAZUS fragility curve medians (from your fragility.py)
FRAGILITY_MEDIANS = {
    "subway": 0.1, "telecom": 0.3, "water": 0.5,
    "hospital": 0.6, "fuel": 0.5, "power": 0.6,
}

g = gpd.read_file(NODES)
lines = [f"Loaded {len(g):,} nodes from {NODES}\n"]

for label, col in SCENARIOS:
    if col not in g.columns:
        lines.append(f"\n=== {label} === MISSING COLUMN: {col}")
        continue

    d = g[col].fillna(0.0).values
    flooded = d[d > 0]
    n = len(flooded)

    lines.append(f"\n=== {label} ({col}) ===")
    lines.append(f"Flooded nodes: {n:,} / {len(g):,} ({100*n/len(g):.1f}%)")
    if n == 0:
        continue

    lines.append(f"Depth distribution at flooded nodes:")
    for q in [10, 25, 50, 75, 90, 95, 99]:
        lines.append(f"  p{q:>2}: {np.percentile(flooded, q):.2f} m")
    lines.append(f"  min/max: {flooded.min():.2f} / {flooded.max():.2f} m")

    lines.append(f"Saturation analysis (depth bins):")
    bins = [(0, 0.3), (0.3, 0.6), (0.6, 1.0), (1.0, 2.0), (2.0, 5.0), (5.0, 100)]
    for lo, hi in bins:
        n_bin = ((flooded >= lo) & (flooded < hi)).sum()
        pct = 100 * n_bin / n
        bar = "█" * int(pct / 2)
        lines.append(f"  [{lo:>4.1f} - {hi:>5.1f}) m: {n_bin:>5,} ({pct:>5.1f}%) {bar}")

    # How many flooded nodes are above the 2m saturation threshold?
    n_saturated = (flooded >= 2.0).sum()
    lines.append(f"Above 2m (fragility saturated for all infra types): "
                 f"{n_saturated:,} / {n:,} ({100*n_saturated/n:.1f}%)")

    # Per-type breakdown — depth median by infra type
    if "infra_type" in g.columns:
        flooded_mask = d > 0
        sub = g[flooded_mask][["infra_type"]].copy()
        sub["depth"] = d[flooded_mask]
        lines.append(f"Per-type median depth (flooded only):")
        for t in ["power", "telecom", "hospital", "subway", "water", "fuel"]:
            sub_t = sub[sub["infra_type"] == t]["depth"]
            if len(sub_t) == 0:
                continue
            med = np.median(sub_t)
            frag_med = FRAGILITY_MEDIANS.get(t, 0)
            ratio = med / frag_med if frag_med > 0 else 0
            sat_flag = "  <-- SATURATED" if ratio > 5 else ""
            lines.append(f"  {t:<10} n={len(sub_t):>4}  median={med:.2f} m  "
                         f"(fragility median {frag_med:.2f} m, ratio {ratio:.1f}x){sat_flag}")

OUT.parent.mkdir(parents=True, exist_ok=True)
output = "\n".join(lines)
print(output)
OUT.write_text(output)
print(f"\nSaved to {OUT}")
