#!/usr/bin/env python3
"""
Task 2: Cascade failure simulation.

Propagates initial flood failures through directed infrastructure graph,
respecting buffer_hours timing on cascade edges.
"""

import json
import os
import networkx as nx
import numpy as np
from collections import Counter


def load_graph(graphml_path: str = "data/flood/lm_infra_graph_flood.graphml") -> nx.DiGraph:
    """Load the infrastructure graph from GraphML."""
    G = nx.read_graphml(graphml_path)
    if not isinstance(G, nx.DiGraph):
        G = nx.DiGraph(G)
    print(f"  Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def get_cascade_edges(G: nx.DiGraph) -> list:
    """Return only cascade edges (exclude recovery/repair_access edges)."""
    cascade = []
    for u, v, data in G.edges(data=True):
        # Recovery edges have layer="recovery" or edge_type="repair_access"
        layer = data.get("layer", "")
        edge_type = data.get("edge_type", "")
        if layer == "recovery" or edge_type == "repair_access":
            continue
        cascade.append((u, v, data))
    return cascade


def simulate_cascade(
    G: nx.DiGraph,
    initial_failures: set,
    time_steps: list = None,
) -> dict:
    """
    Propagate failures through cascade edges respecting buffer_hours.

    Args:
        G: directed infrastructure graph
        initial_failures: set of node IDs that fail at t=0
        time_steps: list of time points (hours) to evaluate

    Returns dict mapping time step string to list of failed node IDs.
    """
    if time_steps is None:
        time_steps = [0, 6, 24, 48, 96]

    # Build adjacency for cascade edges only
    # For each node, store incoming cascade edges: (source, buffer_hours)
    incoming_cascade = {}
    for u, v, data in get_cascade_edges(G):
        buf = float(data.get("buffer_hours", 0.0))
        if v not in incoming_cascade:
            incoming_cascade[v] = []
        incoming_cascade[v].append((u, buf))

    # Track when each node fails (hour)
    fail_time = {}
    for nid in initial_failures:
        if nid in G:
            fail_time[nid] = 0.0

    results = {}

    for t in time_steps:
        # Propagate: check all non-failed nodes
        changed = True
        while changed:
            changed = False
            for node in G.nodes():
                if node in fail_time:
                    continue
                for src, buf in incoming_cascade.get(node, []):
                    if src in fail_time and (fail_time[src] + buf) <= t:
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
    """Run cascade simulation for all Monte Carlo scenarios."""
    print(f"\n=== Cascade simulation: {label} ===")

    G = load_graph(graph_path)

    with open(scenarios_path) as f:
        scenarios = json.load(f)
    print(f"  Loaded {len(scenarios)} scenarios from {scenarios_path}")

    # Get node types for breakdown
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

        direct = len(cascade["t0"])
        total = len(cascade["t96"])
        amp = total / direct if direct > 0 else 0.0

        result = {
            "scenario_id": sc["scenario_id"],
            "direct_failures": direct,
            "total_failures": total,
            "cascade_amplification_ratio": round(amp, 3),
            "by_timestep": {k: len(v) for k, v in cascade.items()},
            "failed_nodes_t96": cascade["t96"],
        }
        all_results.append(result)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f)
    print(f"  Saved cascade results to {output_path}")

    # Summary
    direct_arr = np.array([r["direct_failures"] for r in all_results])
    total_arr = np.array([r["total_failures"] for r in all_results])
    amp_arr = np.array([r["cascade_amplification_ratio"] for r in all_results])
    # Filter out scenarios with 0 direct failures for amplification stats
    amp_valid = amp_arr[direct_arr > 0]

    print(f"\n  Summary ({label}):")
    print(f"    Direct failures:  {direct_arr.mean():.1f} +/- {direct_arr.std():.1f}")
    print(f"    Total at t=96h:   {total_arr.mean():.1f} +/- {total_arr.std():.1f}")
    if len(amp_valid) > 0:
        print(f"    Amplification:    {amp_valid.mean():.2f} +/- {amp_valid.std():.2f}")
        print(f"      (Brunner's reference: ~2.16x)")

    print(f"\n  Time-step progression (mean across {len(scenarios)} runs):")
    for t in time_steps:
        key = f"t{t}"
        vals = [r["by_timestep"][key] for r in all_results]
        print(f"    t={t:3d}h: {np.mean(vals):6.1f} +/- {np.std(vals):5.1f} failures")

    return all_results


if __name__ == "__main__":
    os.chdir(os.path.expanduser("~/Desktop/RA"))
    run_all_scenarios()
