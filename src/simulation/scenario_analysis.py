#!/usr/bin/env python3
"""
Task 3: Flood scenario analysis — multi-scenario scaling, amplifier nodes,
betweenness centrality, and resilience cross-reference.
"""

import json
import os
import sys
import csv
import numpy as np
import networkx as nx
import geopandas as gpd
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fragility import sample_initial_failures, failure_probability
from cascade_sim import load_graph, simulate_cascade, get_cascade_edges, run_all_scenarios

# Scaling factors relative to Sandy
SCENARIO_SCALES = {
    "sandy_actual": 1.0,
    "100yr":        1.0,
    "500yr":        1.45,
}


def run_multi_scenario():
    """Task 3a: Run fragility + cascade for each flood scenario."""
    from fragility import run_and_save

    print("\n" + "=" * 60)
    print("TASK 3a: Multi-scenario flood depth scaling")
    print("=" * 60)

    nodes_path = "data/flood/lm_infra_nodes_flood.geojson"
    graph_path = "data/flood/lm_infra_graph_flood.graphml"

    scenario_results = {}

    for scenario_name, scale in SCENARIO_SCALES.items():
        if scenario_name == "100yr":
            # Same as sandy_actual, reuse
            continue

        print(f"\n--- Scenario: {scenario_name} (scale={scale}) ---")

        # Run fragility
        scenarios = run_and_save(depth_scale=scale, label=scenario_name.replace("/", "_"))

        # Run cascade
        sc_path = f"data/simulation/monte_carlo_failures_{scenario_name.replace('/', '_')}.json"
        out_path = f"data/simulation/cascade_results_{scenario_name.replace('/', '_')}.json"
        cascade_results = run_all_scenarios(
            graph_path=graph_path,
            scenarios_path=sc_path,
            output_path=out_path,
            label=scenario_name,
        )

        direct_arr = np.array([r["direct_failures"] for r in cascade_results])
        total_arr = np.array([r["total_failures"] for r in cascade_results])
        amp_valid = total_arr[direct_arr > 0] / direct_arr[direct_arr > 0]

        scenario_results[scenario_name] = {
            "direct_mean": round(float(direct_arr.mean()), 1),
            "direct_std": round(float(direct_arr.std()), 1),
            "total_mean": round(float(total_arr.mean()), 1),
            "total_std": round(float(total_arr.std()), 1),
            "amplification_mean": round(float(amp_valid.mean()), 2) if len(amp_valid) > 0 else 0,
            "amplification_std": round(float(amp_valid.std()), 2) if len(amp_valid) > 0 else 0,
            "by_timestep": {},
        }

        for t in [0, 6, 24, 48, 96]:
            key = f"t{t}"
            vals = [r["by_timestep"][key] for r in cascade_results]
            scenario_results[scenario_name]["by_timestep"][key] = {
                "mean": round(float(np.mean(vals)), 1),
                "std": round(float(np.std(vals)), 1),
            }

    # 100yr is same as sandy
    scenario_results["100yr"] = scenario_results["sandy_actual"]

    # Save
    with open("data/simulation/scenario_comparison.json", "w") as f:
        json.dump(scenario_results, f, indent=2)
    print(f"\nSaved scenario comparison to data/simulation/scenario_comparison.json")

    # Print table
    print(f"\n{'Scenario':<14} | {'Direct Failures':>16} | {'Total (t=96h)':>22} | {'Amplification':>14}")
    print("-" * 75)
    for name in ["sandy_actual", "100yr", "500yr"]:
        r = scenario_results[name]
        print(f"{name:<14} | {r['direct_mean']:>7.1f} +/- {r['direct_std']:<5.1f} | "
              f"{r['total_mean']:>9.1f} +/- {r['total_std']:<8.1f} | "
              f"{r['amplification_mean']:>8.2f}")

    return scenario_results


def find_amplifier_nodes(
    scenario_label: str = "sandy_actual",
    flood_threshold: float = 0.05,
    cascade_freq_threshold: float = 0.5,
):
    """
    Task 3b: Identify cascade amplifier nodes.

    Amplifier = node with no/low direct flood exposure that fails in >50%
    of cascade scenarios.
    """
    print(f"\n{'=' * 60}")
    print("TASK 3b: Cascade amplifier nodes")
    print("=" * 60)

    nodes_path = "data/flood/lm_infra_nodes_flood.geojson"
    graph_path = "data/flood/lm_infra_graph_flood.graphml"
    cascade_path = f"data/simulation/cascade_results_{scenario_label}.json"

    gdf = gpd.read_file(nodes_path)
    node_info = {}
    for _, row in gdf.iterrows():
        node_info[row["node_id"]] = {
            "infra_type": row["infra_type"],
            "lat": row["lat"],
            "lon": row["lon"],
            "flood_depth_m": row["flood_depth_m"] if row["flood_depth_m"] is not None else 0.0,
            "gissr_division": row["gissr_division"],
        }

    with open(cascade_path) as f:
        cascade_results = json.load(f)

    n_scenarios = len(cascade_results)

    # Count how often each node fails at t=96
    fail_count = Counter()
    for r in cascade_results:
        for nid in r["failed_nodes_t96"]:
            fail_count[nid] += 1

    # Compute betweenness centrality on cascade subgraph
    G = load_graph(graph_path)
    cascade_edges = get_cascade_edges(G)
    G_cascade = nx.DiGraph()
    G_cascade.add_nodes_from(G.nodes(data=True))
    for u, v, data in cascade_edges:
        G_cascade.add_edge(u, v, **data)

    print("  Computing betweenness centrality...")
    bc = nx.betweenness_centrality(G_cascade)

    # Find amplifier nodes
    amplifiers = []
    for nid, info in node_info.items():
        depth = info["flood_depth_m"]
        freq = fail_count.get(nid, 0) / n_scenarios

        if depth < flood_threshold and freq > cascade_freq_threshold:
            # Find upstream trigger: look at incoming cascade edges
            upstream_trigger = "unknown"
            for u, v, data in cascade_edges:
                if v == nid:
                    src_depth = node_info.get(u, {}).get("flood_depth_m", 0)
                    src_freq = fail_count.get(u, 0) / n_scenarios
                    if src_freq > 0.5:
                        upstream_trigger = u
                        break

            amplifiers.append({
                "node_id": nid,
                "infra_type": info["infra_type"],
                "lat": info["lat"],
                "lon": info["lon"],
                "flood_depth_m": depth,
                "cascade_fail_freq": round(freq, 3),
                "betweenness_centrality": round(bc.get(nid, 0), 6),
                "upstream_trigger": upstream_trigger,
            })

    amplifiers.sort(key=lambda x: -x["cascade_fail_freq"])
    print(f"  Found {len(amplifiers)} amplifier nodes")

    # Save JSON
    with open("data/simulation/amplifier_nodes.json", "w") as f:
        json.dump(amplifiers, f, indent=2)

    # Save CSV
    if amplifiers:
        with open("data/simulation/amplifier_nodes.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=amplifiers[0].keys())
            writer.writeheader()
            writer.writerows(amplifiers)

    print(f"  Saved to data/simulation/amplifier_nodes.json and .csv")

    # Print
    print(f"\n  Amplifier Nodes (flood < {flood_threshold}m, cascade fail > {cascade_freq_threshold*100}%):")
    print(f"  {'Node ID':<40} | {'Type':<10} | {'Freq':>5} | {'BC':>8} | Upstream Trigger")
    print(f"  {'-'*40}-+-{'-'*10}-+-{'-'*5}-+-{'-'*8}-+-{'-'*30}")
    for a in amplifiers[:20]:
        print(f"  {a['node_id']:<40} | {a['infra_type']:<10} | {a['cascade_fail_freq']:>5.3f} | "
              f"{a['betweenness_centrality']:>8.5f} | {a['upstream_trigger']}")
    if len(amplifiers) > 20:
        print(f"  ... and {len(amplifiers) - 20} more")

    return amplifiers, bc


def centrality_analysis(amplifiers, bc):
    """Task 3c: Betweenness centrality analysis."""
    print(f"\n{'=' * 60}")
    print("TASK 3c: Betweenness centrality analysis")
    print("=" * 60)

    amp_ids = {a["node_id"] for a in amplifiers}
    amp_bc = [bc[nid] for nid in amp_ids if nid in bc]
    non_amp_bc = [bc[nid] for nid in bc if nid not in amp_ids]

    mean_amp = np.mean(amp_bc) if amp_bc else 0
    mean_non = np.mean(non_amp_bc) if non_amp_bc else 0

    print(f"  Mean betweenness (amplifiers):     {mean_amp:.6f} (n={len(amp_bc)})")
    print(f"  Mean betweenness (non-amplifiers):  {mean_non:.6f} (n={len(non_amp_bc)})")
    if mean_non > 0:
        ratio = mean_amp / mean_non
        print(f"  Ratio: {ratio:.2f}x")
    if mean_amp > mean_non:
        print(f"  Conclusion: Topological centrality IS predictive of cascade risk")
    else:
        print(f"  Conclusion: Topological centrality is NOT more predictive than flood exposure")

    return {"mean_amp": mean_amp, "mean_non": mean_non}


def resilience_crossref(amplifiers):
    """Task 3d: NYC resilience cross-reference notes."""
    print(f"\n{'=' * 60}")
    print("TASK 3d: NYC resilience cross-reference")
    print("=" * 60)

    print("""
  NOTE: The following amplifier node locations should be manually checked
  against the NYC ArcGIS resilience viewer:
  https://experience.arcgis.com/experience/e83a49daef8a472da4a7e34dc25ac445/

  Automated GIS cross-referencing is not performed here — locations are
  provided for manual verification against NYC resiliency project footprints.
""")

    for a in amplifiers:
        print(f"  Amplifier node {a['node_id']} ({a['infra_type']}, {a['lat']:.4f}, {a['lon']:.4f})")
        print(f"    — Check against:")
        print(f"      - NYC Mayor's Office of Climate Resiliency priority areas")
        print(f"      - East Side Coastal Resiliency Project footprint")
        print(f"      - Big U / Rebuild by Design project area")
        if a["infra_type"] == "power":
            print(f"      - Con Edison post-Sandy hardening investments")
        print()


if __name__ == "__main__":
    os.chdir(os.path.expanduser("~/Desktop/RA"))

    # 3a: Multi-scenario
    scenario_results = run_multi_scenario()

    # 3b: Amplifier nodes (using sandy_actual cascade results)
    amplifiers, bc = find_amplifier_nodes()

    # 3c: Centrality analysis
    centrality_analysis(amplifiers, bc)

    # 3d: Resilience cross-reference
    resilience_crossref(amplifiers)
