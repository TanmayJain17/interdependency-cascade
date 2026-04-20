#!/usr/bin/env python3
"""
visualize_cascade_static_nyc.py

Static matplotlib visualization of citywide cascade progression under DEP
extreme_2080 scenario. Produces a slide-ready figure showing:
  - Infrastructure nodes colored by failure time (flood / 6h / 24h / late / ok)
  - DEP flood footprint as a base layer
  - NYC borough outlines for orientation
  - Top amplifier nodes highlighted

Outputs:
  outputs/nyc_cascade_map_extreme2080.png (high-res for slides)
  outputs/nyc_cascade_map_extreme2080.pdf (vector for print)

Run from project root:
    python3 src/visualization/visualize_cascade_static_nyc.py
"""

import os
import sys
import json
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "simulation"))

# Make sure OGR can handle big flood polygons
os.environ.setdefault("OGR_GEOJSON_MAX_OBJ_SIZE", "0")


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

NODES_PATH = Path("data/flood/nyc_infra_nodes_dep_flood.geojson")
CASCADE_PATH = Path("data/simulation/cascade_results_nyc_extreme_2080.json")
MC_PATH = Path("data/simulation/monte_carlo_failures_nyc_extreme_2080.json")
AMPLIFIER_PATH = Path("data/simulation/nyc_amplifier_nodes.csv")
FLOOD_PATH = Path("data/raw/flood/extreme_2080.geojson")

OUT_PNG = Path("outputs/nyc_cascade_map_extreme2080.png")
OUT_PDF = Path("outputs/nyc_cascade_map_extreme2080.pdf")

# Color palette for failure times
COLOR = {
    "ok":         "#d0d0d0",   # gray — not failed
    "flood":      "#ff8c00",   # orange — direct flood failure (t=0)
    "cascade_6":  "#dc143c",   # crimson — cascade at t=6h
    "cascade_24": "#8b0000",   # dark red — cascade at t=24h
    "cascade_late": "#4b0000", # very dark red — t=48h or t=96h
    "amplifier":  "#9400d3",   # violet — amplifier nodes
}


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def get_median_cascade_run(cascade_results):
    """Pick the Monte Carlo run closest to median total failures."""
    totals = np.array([r["total_failures"] for r in cascade_results])
    median_val = np.median(totals)
    idx = int(np.argmin(np.abs(totals - median_val)))
    return cascade_results[idx]


def compute_fail_times(cascade_run):
    """Extract node_id -> first_fail_time from a cascade run's by_timestep lists."""
    # by_timestep only gives counts; we need the actual node lists at each step
    # But cascade_results only stores 'failed_nodes_t96'. We need to re-run the
    # cascade for this scenario to get per-timestep node lists.
    # Simpler: we have failed_nodes_t96, and we have the initial failures from MC.
    # We can reconstruct by re-running cascade for this specific scenario.
    pass


def load_node_fail_times(cascade_run, mc_scenarios, graph_path):
    """Re-run cascade simulation for the median scenario to get per-step failure lists."""
    # Import here to avoid top-level import order issues
    from cascade_sim import load_graph, simulate_cascade

    # Find the MC scenario by ID
    sc_id = cascade_run["scenario_id"]
    mc_scenario = next(s for s in mc_scenarios if s["scenario_id"] == sc_id)
    initial = set(mc_scenario["failed_nodes"])

    G = load_graph(graph_path)
    cascade = simulate_cascade(G, initial, time_steps=[0, 6, 24, 48, 96])

    fail_time = {}
    prev_failed = set()
    for t in [0, 6, 24, 48, 96]:
        current = set(cascade[f"t{t}"])
        new = current - prev_failed
        for nid in new:
            fail_time[nid] = t
        prev_failed = current

    return fail_time


def assign_color(nid, fail_time, amp_ids):
    """Classify a node into one of the color buckets."""
    if nid in amp_ids and fail_time.get(nid, 999) > 0:
        # Amplifier that cascade-fails: use amplifier color (overrides cascade color)
        return COLOR["amplifier"]
    if nid not in fail_time:
        return COLOR["ok"]
    t = fail_time[nid]
    if t == 0:
        return COLOR["flood"]
    if t <= 6:
        return COLOR["cascade_6"]
    if t <= 24:
        return COLOR["cascade_24"]
    return COLOR["cascade_late"]


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def plot_cascade_map(nodes_gdf, fail_time, amp_ids, flood_gdf):
    """Produce the main two-panel figure."""
    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(2, 3, height_ratios=[3, 1], hspace=0.3, wspace=0.35)

    # -------- Main map (top, spanning full width) --------
    ax_map = fig.add_subplot(gs[0, :])

    # Base layer: DEP flood polygons (semi-transparent blue)
    if flood_gdf is not None and len(flood_gdf):
        # Dissolve all flood categories into one shape for cleaner visual
        flood_gdf.plot(
            ax=ax_map, color="#a8c8e0", alpha=0.5,
            edgecolor="#4a7fb0", linewidth=0.3, zorder=1,
        )

    # Classify each node and plot
    nodes_gdf = nodes_gdf.copy()
    nodes_gdf["color"] = nodes_gdf["node_id"].apply(
        lambda nid: assign_color(nid, fail_time, amp_ids)
    )
    nodes_gdf["marker_size"] = nodes_gdf["node_id"].apply(
        lambda nid: 40 if nid in amp_ids else 8
    )
    nodes_gdf["zorder_num"] = nodes_gdf["node_id"].apply(
        lambda nid: 5 if nid in amp_ids else (4 if fail_time.get(nid, 999) == 0 else 3)
    )

    # Plot in z-order: ok nodes first (back), flood next, cascade, amplifiers on top
    for zord in [3, 4, 5]:
        sub = nodes_gdf[nodes_gdf["zorder_num"] == zord]
        if len(sub):
            ax_map.scatter(
                sub["lon"], sub["lat"],
                c=sub["color"], s=sub["marker_size"],
                alpha=0.75 if zord == 3 else 0.9,
                edgecolors="none" if zord == 3 else "black",
                linewidths=0.3 if zord < 5 else 0.6,
                zorder=zord,
            )

    # Title + labels
    ax_map.set_title(
        "NYC Infrastructure Cascade Under Extreme 2080 Flood Scenario\n"
        "500-yr storm + 2080 sea level rise (pluvial + tidal)",
        fontsize=13, fontweight="bold", pad=10,
    )
    ax_map.set_xlabel("Longitude")
    ax_map.set_ylabel("Latitude")
    ax_map.set_aspect("equal")

    # Set bounds to NYC
    ax_map.set_xlim(-74.30, -73.68)
    ax_map.set_ylim(40.48, 40.93)

    # Light gridlines
    ax_map.grid(True, alpha=0.2, linestyle=":", linewidth=0.5)

    # Legend
    legend_elements = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLOR["ok"],
               markersize=6, label="Operational"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLOR["flood"],
               markersize=8, label="Direct flood failure (t=0)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLOR["cascade_6"],
               markersize=8, label="Cascade fail at t=6h"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLOR["cascade_24"],
               markersize=8, label="Cascade fail at t=24h"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLOR["cascade_late"],
               markersize=8, label="Cascade fail t=48-96h"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLOR["amplifier"],
               markeredgecolor="black", markersize=10, label="Amplifier node"),
        mpatches.Patch(facecolor="#a8c8e0", alpha=0.5, edgecolor="#4a7fb0",
                       label="DEP flood footprint"),
    ]
    ax_map.legend(
        handles=legend_elements, loc="lower left", framealpha=0.9,
        fontsize=9, ncol=2, title="Node status at t=96h",
    )

    # -------- Bottom left: per-type failures bar chart --------
    ax_type = fig.add_subplot(gs[1, 0])
    plot_per_type_bars(ax_type, nodes_gdf, fail_time)

    # -------- Bottom middle: per-borough failures bar chart --------
    ax_boro = fig.add_subplot(gs[1, 1])
    plot_per_borough_bars(ax_boro, nodes_gdf, fail_time)

    # -------- Bottom right: time-step progression --------
    ax_time = fig.add_subplot(gs[1, 2])
    plot_timestep_curve(ax_time, fail_time)

    return fig


def plot_per_type_bars(ax, nodes_gdf, fail_time):
    """Bar chart: for each infra type, show failed vs total."""
    types = ["power", "telecom", "hospital", "subway", "water", "fuel"]
    totals = []
    failed = []
    for t in types:
        sub = nodes_gdf[nodes_gdf["infra_type"] == t]
        totals.append(len(sub))
        failed.append(sum(1 for nid in sub["node_id"] if nid in fail_time))

    # Use percentages for fair comparison across types of different sizes
    pct = [100 * f / total if total > 0 else 0 for f, total in zip(failed, totals)]

    x = np.arange(len(types))
    bars = ax.bar(x, pct, color="#8b0000", alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(types, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("% of type failed")
    ax.set_title("Failure rate by infrastructure type", fontsize=10)
    ax.set_ylim(0, max(pct) * 1.25 if pct else 100)
    ax.grid(True, alpha=0.3, axis="y", linestyle=":")

    # Annotate with absolute count
    for i, (bar, f, tot) in enumerate(zip(bars, failed, totals)):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, h + max(pct) * 0.02,
            f"{f}/{tot}", ha="center", va="bottom", fontsize=8,
        )


def plot_per_borough_bars(ax, nodes_gdf, fail_time):
    """Bar chart: failures per borough."""
    def assign_boro(lat, lon):
        if lat < 40.65 and lon < -74.03:
            return "SI"
        if lon < -74.03:
            return "NJ"
        if lat > 40.80 and lon > -73.93:
            return "Bronx"
        if -74.02 <= lon <= -73.93 and 40.70 <= lat <= 40.88:
            return "Manhattan"
        if lon > -73.90 or (lon > -73.93 and lat > 40.70):
            return "Queens"
        return "Brooklyn"

    nodes_gdf = nodes_gdf.copy()
    nodes_gdf["boro"] = nodes_gdf.apply(
        lambda r: assign_boro(r["lat"], r["lon"]), axis=1
    )

    boros = ["Manhattan", "Brooklyn", "Queens", "Bronx", "SI"]
    totals = []
    failed = []
    for b in boros:
        sub = nodes_gdf[nodes_gdf["boro"] == b]
        totals.append(len(sub))
        failed.append(sum(1 for nid in sub["node_id"] if nid in fail_time))

    pct = [100 * f / total if total > 0 else 0 for f, total in zip(failed, totals)]
    x = np.arange(len(boros))
    bars = ax.bar(x, pct, color="#ff8c00", alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(boros, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("% of nodes failed")
    ax.set_title("Failure rate by borough", fontsize=10)
    ax.set_ylim(0, max(pct) * 1.25 if pct else 100)
    ax.grid(True, alpha=0.3, axis="y", linestyle=":")

    for bar, f, tot in zip(bars, failed, totals):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, h + max(pct) * 0.02,
            f"{f}/{tot}", ha="center", va="bottom", fontsize=8,
        )


def plot_timestep_curve(ax, fail_time):
    """Time-series line: cumulative failures at each timestep."""
    steps = [0, 6, 24, 48, 96]
    cumulative = [sum(1 for t in fail_time.values() if t <= s) for s in steps]

    ax.plot(steps, cumulative, marker="o", color="#8b0000", linewidth=2.2, markersize=7)
    ax.fill_between(steps, 0, cumulative, alpha=0.2, color="#8b0000")
    ax.set_xlabel("Time after flood (hours)")
    ax.set_ylabel("Cumulative failures")
    ax.set_title("Cascade progression over time", fontsize=10)
    ax.set_xticks(steps)
    ax.grid(True, alpha=0.3, linestyle=":")

    for s, c in zip(steps, cumulative):
        ax.annotate(
            str(c), (s, c), textcoords="offset points", xytext=(0, 6),
            ha="center", fontsize=8,
        )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    print("=" * 72)
    print("Static cascade map — NYC citywide extreme 2080")
    print("=" * 72)

    # Load nodes
    print(f"Loading nodes: {NODES_PATH}")
    nodes_gdf = gpd.read_file(NODES_PATH)
    print(f"  {len(nodes_gdf):,} nodes")

    # Load cascade results + MC scenarios
    print(f"Loading cascade results: {CASCADE_PATH}")
    with open(CASCADE_PATH) as f:
        cascade_results = json.load(f)
    with open(MC_PATH) as f:
        mc_scenarios = json.load(f)

    # Pick the median run
    median_run = get_median_cascade_run(cascade_results)
    print(f"  Median run: id={median_run['scenario_id']}, "
          f"direct={median_run['direct_failures']}, total={median_run['total_failures']}")

    # Rebuild per-node fail times for this specific scenario
    print("  Re-running cascade sim for this scenario to get per-step failures...")
    graph_path = "data/flood/nyc_infra_graph_dep_flood.graphml"
    fail_time = load_node_fail_times(median_run, mc_scenarios, graph_path)
    print(f"  {len(fail_time):,} nodes failed by t=96h")

    # Amplifier IDs
    amp_ids = set()
    if AMPLIFIER_PATH.exists():
        amp_df = pd.read_csv(AMPLIFIER_PATH)
        # Take top 50 amplifiers by frequency — these are the ones worth highlighting
        top_amps = amp_df.nlargest(50, "cascade_fail_freq")
        amp_ids = set(top_amps["node_id"].tolist())
        print(f"  Highlighting {len(amp_ids)} top amplifier nodes")

    # Load flood footprint
    flood_gdf = None
    if FLOOD_PATH.exists():
        print(f"Loading flood polygons: {FLOOD_PATH}")
        flood_gdf = gpd.read_file(FLOOD_PATH)
        # Fix invalid geometries for plotting
        from shapely.validation import make_valid
        invalid = ~flood_gdf.geometry.is_valid
        if invalid.any():
            flood_gdf.loc[invalid, "geometry"] = flood_gdf.loc[invalid, "geometry"].apply(make_valid)
        print(f"  {len(flood_gdf)} flood polygons loaded")

    # Render
    print("\nRendering figure...")
    fig = plot_cascade_map(nodes_gdf, fail_time, amp_ids, flood_gdf)

    # Save
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=220, bbox_inches="tight", facecolor="white")
    print(f"Saved: {OUT_PNG}")
    fig.savefig(OUT_PDF, bbox_inches="tight", facecolor="white")
    print(f"Saved: {OUT_PDF}")

    plt.close(fig)
    print("\nDone. Figure ready for Monday's slides.")
    return 0


if __name__ == "__main__":
    sys.exit(main())