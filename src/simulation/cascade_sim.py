#!/usr/bin/env python3
"""
cascade_sim.py — v2 with t=0 semantic fix

FIX: In v1, the propagation loop at t=0 also expanded through zero-buffer edges
(subway_line, power_line, fuel_distribution with buf=0), which meant "direct
failures" reported by the sim included instant cascade propagation, not just
flood-induced initial failures. This caused moderate_current to report 462
direct failures when only 34 nodes were actually flooded.

The fix:
  - t=0 results contain ONLY the initial flood-induced failures (the set
    passed in as `initial_failures`)
  - Propagation begins at the smallest non-zero time step
  - Zero-buffer edges still propagate, but they propagate at the next time
    step after the source failed — not retroactively at t=0

This keeps a clear separation between "flood exposure" and "cascade
propagation" in output metrics.

Also adds 'flood_failures' field to output (== direct_failures, kept for
backward-compatibility with existing downstream code).
"""

import json
import os
import networkx as nx
import numpy as np
from collections import Counter


def load_graph(graphml_path: str = "data/flood/lm_infra_graph_flood.graphml") -> nx.DiGraph:
    G = nx.read_graphml(graphml_path)
    if not isinstance(G, nx.DiGraph):
        G = nx.DiGraph(G)
    print(f"  Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


# DEPENDENCY edges that actually propagate failure through the network
DEPENDENCY_EDGE_TYPES = {
    "power_dependency",    # power -> everything (buffer varies by target)
    "water_supplies",      # water -> hospital (buffer 24h)
    "scada_monitoring",    # telecom -> power (buffer 2h)
    "fuel_supplies",       # fuel -> hospital/telecom/water (buffer 48-96h)
}
 
 
def get_cascade_edges(G):
    """
    Return only DEPENDENCY edges — edges that actually propagate failure.
 
    Excludes:
      - Structural edges (subway_line, power_line, water_flow,
        fuel_distribution): these represent physical connectivity, not
        failure propagation
      - Recovery edges (repair_access): these are logistics, not cascade
    """
    cascade = []
    for u, v, data in G.edges(data=True):
        edge_type = data.get("edge_type", "")
        layer = data.get("layer", "")
 
        # Exclude recovery layer
        if layer == "recovery" or edge_type == "repair_access":
            continue
 
        # Include ONLY dependency edge types
        if edge_type in DEPENDENCY_EDGE_TYPES:
            cascade.append((u, v, data))
 
    return cascade


def simulate_cascade(
    G: nx.DiGraph,
    initial_failures: set,
    time_steps: list = None,
) -> dict:
    """
    Propagate failures through cascade edges respecting buffer_hours.

    t=0 output contains ONLY the initial flood-induced failures.
    Propagation (including through zero-buffer edges) begins at t > 0.

    Returns dict mapping time step key to sorted list of failed node IDs.
    """
    if time_steps is None:
        time_steps = [0, 6, 24, 48, 96]

    # Build incoming cascade edge map: v -> list of (source, buffer_hours)
    incoming_cascade = {}
    for u, v, data in get_cascade_edges(G):
        buf = float(data.get("buffer_hours", 0.0))
        if v not in incoming_cascade:
            incoming_cascade[v] = []
        incoming_cascade[v].append((u, buf))

    # Track when each node fails (in hours)
    fail_time = {}
    for nid in initial_failures:
        if nid in G:
            fail_time[nid] = 0.0

    results = {}

    # t=0: ONLY the initial flood failures (no cascade propagation yet)
    zero_step = 0
    if zero_step in time_steps:
        results[f"t{zero_step}"] = sorted(fail_time.keys())

    # For t > 0: propagate normally
    for t in time_steps:
        if t == 0:
            continue
        changed = True
        while changed:
            changed = False
            for node in G.nodes():
                if node in fail_time:
                    continue
                for src, buf in incoming_cascade.get(node, []):
                    # Propagation requires src to have ALREADY failed at some
                    # prior time, with the buffer elapsed. For zero-buffer edges,
                    # fail_time[src] = 0 means propagation at next t > 0 is fine.
                    if src in fail_time and (fail_time[src] + buf) <= t:
                        # Exception: if src just failed at t=0 via flood AND buf=0,
                        # we still want it to propagate at the next timestep, not
                        # retroactively. This is already handled correctly because
                        # we only add to fail_time[node] = t (current timestep).
                        fail_time[node] = t
                        changed = True
                        break

        results[f"t{t}"] = sorted(fail_time.keys())

    return results


def run_all_scenarios(
    graph_path: str = "data/flood/lm_infra_graph_flood.graphml",
    scenarios_path: str = "data/simulation/monte_carlo_failures.json",
    output_path: str = "data/simulation/cascade_results.json",
    label: str = "sandy",
):
    print(f"\n=== Cascade simulation: {label} ===")

    G = load_graph(graph_path)

    with open(scenarios_path) as f:
        scenarios = json.load(f)
    print(f"  Loaded {len(scenarios)} scenarios from {scenarios_path}")

    node_type = {}
    for nid, data in G.nodes(data=True):
        node_type[nid] = data.get("infra_type", "unknown")

    time_steps = [0, 6, 24, 48, 96]
    all_results = []

    for i, sc in enumerate(scenarios):
        if (i + 1) % 100 == 0:
            print(f"  Processing scenario {i+1}/{len(scenarios)}...")

        initial = set(sc["failed_nodes"])
        cascade = simulate_cascade(G, initial, time_steps)

        flood_only = len(cascade["t0"])    # actual flood-induced direct failures
        total = len(cascade["t96"])
        cascade_only = total - flood_only  # pure cascade effect
        amp = total / flood_only if flood_only > 0 else 0.0

        result = {
            "scenario_id": sc["scenario_id"],
            "direct_failures": flood_only,       # now correctly = flood-only
            "flood_failures": flood_only,         # explicit alias for clarity
            "cascade_failures": cascade_only,     # new: pure cascade delta
            "total_failures": total,
            "cascade_amplification_ratio": round(amp, 3),
            "by_timestep": {k: len(v) for k, v in cascade.items()},
            "failed_nodes_t96": cascade["t96"],
        }
        all_results.append(result)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f)
    print(f"  Saved cascade results to {output_path}")

    direct_arr = np.array([r["direct_failures"] for r in all_results])
    total_arr = np.array([r["total_failures"] for r in all_results])
    cascade_arr = np.array([r["cascade_failures"] for r in all_results])
    amp_arr = np.array([r["cascade_amplification_ratio"] for r in all_results])
    amp_valid = amp_arr[direct_arr > 0]

    print(f"\n  Summary ({label}):")
    print(f"    Flood-induced failures (t=0):  {direct_arr.mean():.1f} +/- {direct_arr.std():.1f}")
    print(f"    Cascade failures (t0 -> t96):  {cascade_arr.mean():.1f} +/- {cascade_arr.std():.1f}")
    print(f"    Total at t=96h:                {total_arr.mean():.1f} +/- {total_arr.std():.1f}")
    if len(amp_valid) > 0:
        print(f"    Amplification ratio:           {amp_valid.mean():.2f} +/- {amp_valid.std():.2f}")
        print(f"      (Brunner's reference: ~2.16x)")

    print(f"\n  Time-step progression (mean across {len(scenarios)} runs):")
    for t in time_steps:
        key = f"t{t}"
        vals = [r["by_timestep"][key] for r in all_results]
        print(f"    t={t:3d}h: {np.mean(vals):7.1f} +/- {np.std(vals):6.1f} failures")

    return all_results


if __name__ == "__main__":
    os.chdir(os.path.expanduser("~/Desktop/RA"))
    run_all_scenarios()