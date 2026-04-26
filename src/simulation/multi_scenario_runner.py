#!/usr/bin/env python3
"""
multi_scenario_runner.py — Citywide cascade analysis

Orchestrates fragility + Monte Carlo + cascade simulation for the three DEP
flood scenarios (moderate_current, moderate_2050, extreme_2080) on the
citywide heterogeneous infrastructure graph.

Approach (Option A): Wraps the existing Week 5 fragility.py and cascade_sim.py
WITHOUT modifying them. For each scenario, creates a temporary nodes GeoJSON
with the appropriate DEP depth column aliased to 'flood_depth_m' and a synthetic
'gissr_division' column (-1 for external nodes, 0 for NYC nodes).

Inputs:
    data/flood/nyc_infra_nodes_dep_flood.geojson  (6,231 nodes with 3 scenarios)
    data/flood/nyc_infra_graph_dep_flood.graphml  (directed graph with buffers)

Outputs:
    data/simulation/monte_carlo_failures_nyc_{scenario}.json
    data/simulation/cascade_results_nyc_{scenario}.json
    data/simulation/nyc_scenario_comparison.json
    data/simulation/nyc_amplifier_nodes.csv
    outputs/week6_cascade_summary.txt

Run from project root (~/Desktop/RA/):
    python3 src/simulation/multi_scenario_runner.py
"""

import json
import os
import sys
import csv
from pathlib import Path
from collections import Counter

import numpy as np
import geopandas as gpd
import networkx as nx

# Make fragility.py and cascade_sim.py importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fragility import sample_initial_failures
from cascade_sim import run_all_scenarios, load_graph, get_cascade_edges


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

NODES_IN = Path("data/flood/nyc_infra_nodes_dep_flood.geojson")
GRAPH_IN = Path("data/flood/nyc_infra_graph_dep_flood.graphml")

SIM_DIR = Path("data/simulation")
OUT_DIR = Path("outputs")

SCENARIOS = ["moderate_current", "moderate_2050", "extreme_2080"]
N_MONTE_CARLO = 1000

# Amplifier thresholds (for extreme_2080 analysis)
AMP_FLOOD_THRESHOLD_M = 0.05  # node is "dry" if depth < 5 cm
AMP_FREQ_THRESHOLD = 0.50     # node is "amplifier" if it fails in >50% of runs


# -----------------------------------------------------------------------------
# Scenario preparation
# -----------------------------------------------------------------------------

def prepare_scenario_nodes(nodes_gdf, scenario_name, out_path):
    """
    Write a temp nodes GeoJSON that the Week 5 fragility module can consume.
    Adds two synthetic columns without touching the original data:
      - flood_depth_m     <- flood_{scenario}_depth_m
      - gissr_division    <- -1 if external (NJ/terminal), 0 otherwise
    """
    temp = nodes_gdf.copy()
    depth_col = f"flood_{scenario_name}_depth_m"
    if depth_col not in temp.columns:
        raise ValueError(f"Missing column {depth_col} in nodes GeoJSON")

    temp["flood_depth_m"] = temp[depth_col].fillna(0.0)

    # geopandas sometimes returns bool as string; handle both
    def is_external(x):
        return str(x).lower() in ("true", "1", "yes")

    if "external" in temp.columns:
        temp["gissr_division"] = temp["external"].apply(
            lambda x: -1 if is_external(x) else 0
        )
    else:
        temp["gissr_division"] = 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Keep file small — drop unused columns
    keep_cols = ["node_id", "infra_type", "lat", "lon", "external",
                 "flood_depth_m", "gissr_division", "geometry"]
    keep_cols = [c for c in keep_cols if c in temp.columns]
    temp[keep_cols].to_file(out_path, driver="GeoJSON")


# -----------------------------------------------------------------------------
# Per-scenario run
# -----------------------------------------------------------------------------

def run_scenario(scenario_name, nodes_gdf):
    """Run fragility + cascade for one DEP scenario. Returns (mc_scenarios, cascade_results)."""
    print(f"\n{'=' * 75}")
    print(f"SCENARIO: {scenario_name}")
    print(f"{'=' * 75}")

    # 1. Prepare temp nodes with aliased columns
    temp_nodes = SIM_DIR / f"temp_nodes_nyc_{scenario_name}.geojson"
    prepare_scenario_nodes(nodes_gdf, scenario_name, temp_nodes)
    print(f"  Prepared temp nodes: {temp_nodes}")

    # 2. Fragility: Monte Carlo initial failures
    print(f"\n  [Fragility] Sampling {N_MONTE_CARLO} Monte Carlo scenarios...")
    mc_scenarios = sample_initial_failures(
        str(temp_nodes),
        n_scenarios=N_MONTE_CARLO,
        seed=42,
        depth_scale=1.0,   # no Sandy-style scaling — each DEP scenario has its own footprint
    )

    n_failed = np.array([s["n_failed"] for s in mc_scenarios])
    print(f"  Initial failures: mean={n_failed.mean():.1f}  std={n_failed.std():.1f}  "
          f"min={n_failed.min()}  max={n_failed.max()}")

    mc_out = SIM_DIR / f"monte_carlo_failures_nyc_{scenario_name}.json"
    with open(mc_out, "w") as f:
        json.dump(mc_scenarios, f)
    print(f"  Saved: {mc_out}")

    # 3. Cascade simulation
    print(f"\n  [Cascade] Propagating through graph...")
    cascade_out = SIM_DIR / f"cascade_results_nyc_{scenario_name}.json"
    cascade_results = run_all_scenarios(
        graph_path=str(GRAPH_IN),
        scenarios_path=str(mc_out),
        output_path=str(cascade_out),
        label=f"nyc_{scenario_name}",
    )

    return mc_scenarios, cascade_results


# -----------------------------------------------------------------------------
# Cross-scenario comparison
# -----------------------------------------------------------------------------

def summarize(all_results, nodes_gdf):
    """Build comparison dict from per-scenario cascade results."""
    node_type = dict(zip(nodes_gdf["node_id"], nodes_gdf["infra_type"]))

    # Also build borough lookup so we can break down failures by borough
    def assign_borough(lat, lon):
        if lat < 40.65 and lon < -74.03:
            return "Staten Island"
        if lon < -74.03:
            return "NJ (external)"
        if lat > 40.80 and lon > -73.93:
            return "Bronx"
        if -74.02 <= lon <= -73.93 and 40.70 <= lat <= 40.88:
            return "Manhattan"
        if lon > -73.90 or (lon > -73.93 and lat > 40.70):
            return "Queens"
        return "Brooklyn"

    node_boro = {
        row["node_id"]: assign_borough(row["lat"], row["lon"])
        for _, row in nodes_gdf.iterrows()
    }

    comparison = {}
    for scenario_name, (mc, cascade) in all_results.items():
        direct = np.array([r["direct_failures"] for r in cascade])
        total = np.array([r["total_failures"] for r in cascade])
        valid = direct > 0
        amp = total[valid] / direct[valid] if valid.sum() > 0 else np.array([])

        # Per-type failures at t=96
        type_counts = Counter()
        for r in cascade:
            for nid in r["failed_nodes_t96"]:
                type_counts[node_type.get(nid, "unknown")] += 1
        n_runs = len(cascade)
        type_means = {
            t: round(type_counts.get(t, 0) / n_runs, 1)
            for t in ["power", "telecom", "hospital", "subway", "water", "fuel"]
        }

        # Per-borough failures at t=96
        boro_counts = Counter()
        for r in cascade:
            for nid in r["failed_nodes_t96"]:
                boro_counts[node_boro.get(nid, "Unknown")] += 1
        boro_means = {
            b: round(boro_counts.get(b, 0) / n_runs, 1)
            for b in ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
        }

        comparison[scenario_name] = {
            "direct_mean": round(float(direct.mean()), 1),
            "direct_std":  round(float(direct.std()), 1),
            "total_mean":  round(float(total.mean()), 1),
            "total_std":   round(float(total.std()), 1),
            "amplification_mean": round(float(amp.mean()), 2) if len(amp) > 0 else 0.0,
            "amplification_std":  round(float(amp.std()), 2)  if len(amp) > 0 else 0.0,
            "type_failures_mean":   type_means,
            "borough_failures_mean": boro_means,
            "by_timestep": {
                t: round(float(np.mean([r["by_timestep"][t] for r in cascade])), 1)
                for t in ["t0", "t6", "t24", "t48", "t96"]
            },
        }

    return comparison


# -----------------------------------------------------------------------------
# Amplifier analysis (extreme_2080 only)
# -----------------------------------------------------------------------------

def find_amplifiers_extreme(nodes_gdf, cascade_results):
    """
    Identify nodes that fail in > threshold % of extreme_2080 cascade runs
    despite having no direct flood exposure. Includes betweenness centrality
    to characterize topological importance.
    """
    print(f"\n{'=' * 75}")
    print("AMPLIFIER ANALYSIS (extreme_2080)")
    print(f"{'=' * 75}")

    depth_col = "flood_extreme_2080_depth_m"
    node_depth = dict(zip(nodes_gdf["node_id"], nodes_gdf[depth_col].fillna(0.0)))
    node_type = dict(zip(nodes_gdf["node_id"], nodes_gdf["infra_type"]))
    node_lat = dict(zip(nodes_gdf["node_id"], nodes_gdf["lat"]))
    node_lon = dict(zip(nodes_gdf["node_id"], nodes_gdf["lon"]))

    # Count failures per node across all runs
    fail_count = Counter()
    for r in cascade_results:
        for nid in r["failed_nodes_t96"]:
            fail_count[nid] += 1
    n_runs = len(cascade_results)

    # Compute betweenness centrality on cascade subgraph
    # For 6.2k nodes, approximate betweenness (k=500) is much faster than exact
    print("  Computing approximate betweenness centrality (k=500 samples)...")
    G = load_graph(str(GRAPH_IN))
    G_cascade = nx.DiGraph()
    G_cascade.add_nodes_from(G.nodes(data=True))
    for u, v, data in get_cascade_edges(G):
        G_cascade.add_edge(u, v, **data)
    bc = nx.betweenness_centrality(G_cascade, k=min(500, G_cascade.number_of_nodes()))

    # Identify amplifiers
    amplifiers = []
    for nid, freq in fail_count.items():
        frac = freq / n_runs
        depth = node_depth.get(nid, 0.0)
        if depth < AMP_FLOOD_THRESHOLD_M and frac > AMP_FREQ_THRESHOLD:
            amplifiers.append({
                "node_id": nid,
                "infra_type": node_type.get(nid, "unknown"),
                "lat": round(node_lat.get(nid, 0.0), 5),
                "lon": round(node_lon.get(nid, 0.0), 5),
                "flood_depth_m": round(depth, 3),
                "cascade_fail_freq": round(frac, 3),
                "betweenness_centrality": round(bc.get(nid, 0.0), 6),
            })

    amplifiers.sort(key=lambda x: (-x["cascade_fail_freq"], -x["betweenness_centrality"]))
    print(f"  Found {len(amplifiers)} amplifier nodes "
          f"(dry under extreme_2080 but fail via cascade in >{int(AMP_FREQ_THRESHOLD * 100)}% of runs)")

    # Save
    if amplifiers:
        csv_path = SIM_DIR / "nyc_amplifier_nodes.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=amplifiers[0].keys())
            writer.writeheader()
            writer.writerows(amplifiers)
        print(f"  Saved: {csv_path}")

    return amplifiers


# -----------------------------------------------------------------------------
# Text summary
# -----------------------------------------------------------------------------

def format_summary(comparison, amplifiers):
    lines = []
    lines.append("=" * 75)
    lines.append("WEEK 6 CASCADE ANALYSIS — NYC CITYWIDE (6,231 nodes)")
    lines.append("DEP Stormwater Flood Map scenarios (pluvial + tidal)")
    lines.append("=" * 75)
    lines.append("")

    # Main amplification table
    lines.append(f"{'Scenario':<20} | {'Direct':>14} | {'Total (t=96h)':>17} | {'Amplification':>14}")
    lines.append("-" * 75)
    for scenario in SCENARIOS:
        r = comparison[scenario]
        lines.append(
            f"{scenario:<20} | "
            f"{r['direct_mean']:>6.1f} +/- {r['direct_std']:<4.1f} | "
            f"{r['total_mean']:>7.1f} +/- {r['total_std']:<5.1f} | "
            f"{r['amplification_mean']:>7.2f}x"
        )
    lines.append("")

    # Time-step progression (extreme only — most informative)
    lines.append("Time-step progression (extreme_2080, mean across 1000 MC runs):")
    ts = comparison["extreme_2080"]["by_timestep"]
    prev = 0.0
    for t_key in ["t0", "t6", "t24", "t48", "t96"]:
        val = ts[t_key]
        delta = val - prev
        label = "(direct flood)" if prev == 0 else f"(+{delta:.0f} cascade)"
        lines.append(f"  {t_key:>4}: {val:>6.1f} failures  {label}")
        prev = val
    lines.append("")

    # Per-type breakdown (extreme)
    lines.append("Per-type failures at t=96h (extreme_2080):")
    type_means = comparison["extreme_2080"]["type_failures_mean"]
    for t in ["power", "telecom", "hospital", "subway", "water", "fuel"]:
        lines.append(f"  {t:<10}: {type_means[t]:>6.1f}")
    lines.append("")

    # Per-borough breakdown (extreme)
    lines.append("Per-borough failures at t=96h (extreme_2080):")
    boro_means = comparison["extreme_2080"]["borough_failures_mean"]
    for b in ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]:
        lines.append(f"  {b:<15}: {boro_means[b]:>6.1f}")
    lines.append("")

    # Amplifier nodes
    lines.append(f"Cascade amplifier nodes (dry but fail via cascade): {len(amplifiers)}")
    if amplifiers:
        lines.append(f"  {'Node ID':<50} | {'Type':<8} | {'Freq':>5} | {'BC':>8}")
        lines.append(f"  {'-' * 50}-+-{'-' * 8}-+-{'-' * 5}-+-{'-' * 8}")
        for a in amplifiers[:15]:
            lines.append(
                f"  {a['node_id'][:50]:<50} | "
                f"{a['infra_type']:<8} | "
                f"{a['cascade_fail_freq']:>5.3f} | "
                f"{a['betweenness_centrality']:>8.5f}"
            )
        if len(amplifiers) > 15:
            lines.append(f"  ... and {len(amplifiers) - 15} more (see nyc_amplifier_nodes.csv)")
    lines.append("")

    # Known limitations
    lines.append("Known limitations :")
    lines.append("  1. DEP flood maps exclude storm surge per their own disclaimer.")
    lines.append("     Surge-exposed infrastructure (FDR corridor hospitals, SI shore) is")
    lines.append("     systematically under-represented in flood footprint.")
    lines.append("  2. Power and fuel redundancy treated as OR (either kills the dependent")
    lines.append("     node). Proper AND-gate semantics for fuel-as-backup: Week 7+.")
    lines.append("  3. NJ substations (91) and petroleum terminals (6) excluded from")
    lines.append("     failure sampling — they can receive cascade but don't originate it.")
    lines.append("  4. Betweenness centrality is k=500 approximation (not exact) for")
    lines.append("     compute tractability on 6.2k-node graph.")
    lines.append("")
    lines.append("=" * 75)

    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    print("=" * 75)
    print("Week 6 Multi-Scenario Cascade Runner (citywide)")
    print("=" * 75)

    if not NODES_IN.exists():
        print(f"ERROR: {NODES_IN} not found. Run flood_overlay_v3.py nyc first.")
        return 1
    if not GRAPH_IN.exists():
        print(f"ERROR: {GRAPH_IN} not found.")
        return 1

    SIM_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load nodes once
    nodes_gdf = gpd.read_file(NODES_IN)
    print(f"Loaded {len(nodes_gdf):,} nodes from {NODES_IN}")

    # Run all scenarios
    all_results = {}
    for scenario in SCENARIOS:
        mc, cascade = run_scenario(scenario, nodes_gdf)
        all_results[scenario] = (mc, cascade)

    # Summarize
    comparison = summarize(all_results, nodes_gdf)
    comparison_path = SIM_DIR / "nyc_scenario_comparison.json"
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nSaved: {comparison_path}")

    # Amplifier analysis on extreme_2080 only
    _, extreme_cascade = all_results["extreme_2080"]
    amplifiers = find_amplifiers_extreme(nodes_gdf, extreme_cascade)

    # Text summary
    summary_text = format_summary(comparison, amplifiers)
    print("\n" + summary_text)
    summary_path = OUT_DIR / "week6_cascade_summary.txt"
    summary_path.write_text(summary_text)
    print(f"\nSaved: {summary_path}")

    # Cleanup temp files
    for scenario in SCENARIOS:
        temp = SIM_DIR / f"temp_nodes_nyc_{scenario}.geojson"
        if temp.exists():
            temp.unlink()

    print("\nDone. Deliverables ready for Monday meeting prep.")
    return 0


if __name__ == "__main__":
    sys.exit(main())