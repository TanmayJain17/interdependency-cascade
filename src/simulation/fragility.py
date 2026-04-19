#!/usr/bin/env python3
"""
Task 1: Fragility curves and Monte Carlo initial failure sampling.

HAZUS-based lognormal fragility: P(failure | depth) = Phi((ln(depth) - mu) / beta)
where mu = ln(median_depth), beta = dispersion.
"""

import json
import os
import numpy as np
import geopandas as gpd
from scipy.stats import norm
from collections import Counter

# HAZUS lognormal fragility parameters: (median_depth_m, beta)
FRAGILITY_PARAMS = {
    "power":    (0.6, 0.5),
    "telecom":  (0.3, 0.6),
    "hospital": (0.6, 0.5),
    "subway":   (0.1, 0.4),
    "water":    (0.5, 0.5),
    "fuel":     (0.5, 0.6),
}


def failure_probability(flood_depth_m: float, infra_type: str) -> float:
    """Return P(failure) in [0,1] given flood depth and infrastructure type."""
    if flood_depth_m is None or flood_depth_m <= 0:
        return 0.0
    if infra_type not in FRAGILITY_PARAMS:
        return 0.0
    median, beta = FRAGILITY_PARAMS[infra_type]
    mu = np.log(median)
    return float(norm.cdf((np.log(flood_depth_m) - mu) / beta))


def sample_initial_failures(
    nodes_geojson_path: str,
    n_scenarios: int = 1000,
    seed: int = 42,
    depth_scale: float = 1.0,
) -> list[dict]:
    """
    Monte Carlo sampling of initial flood failures.

    For each scenario, each node independently fails with probability
    P(failure | scaled_flood_depth, infra_type).

    Args:
        depth_scale: multiplier on flood_depth_m (1.0 = Sandy, 1.45 = 500yr)

    Returns list of dicts with scenario_id, failed_nodes, n_failed.
    """
    gdf = gpd.read_file(nodes_geojson_path)
    print(f"  Loaded {len(gdf)} nodes from {nodes_geojson_path}")

    # Precompute failure probabilities
    node_ids = gdf["node_id"].tolist()
    infra_types = gdf["infra_type"].tolist()
    depths = gdf["flood_depth_m"].tolist()
    divisions = gdf["gissr_division"].tolist()

    probs = []
    for i in range(len(gdf)):
        depth = depths[i] if depths[i] is not None else 0.0
        # External nodes (gissr_division == -1) get P=0
        if divisions[i] == -1:
            probs.append(0.0)
        else:
            probs.append(failure_probability(depth * depth_scale, infra_types[i]))

    probs = np.array(probs)
    n_vulnerable = np.sum(probs > 0)
    print(f"  {n_vulnerable} nodes have P(failure) > 0 (depth_scale={depth_scale})")

    # Monte Carlo sampling
    rng = np.random.default_rng(seed)
    scenarios = []
    for s in range(n_scenarios):
        rolls = rng.random(len(probs))
        failed_mask = rolls < probs
        failed = [node_ids[i] for i in range(len(node_ids)) if failed_mask[i]]
        scenarios.append({
            "scenario_id": s,
            "failed_nodes": failed,
            "n_failed": len(failed),
        })

    return scenarios


def run_and_save(depth_scale: float = 1.0, label: str = "sandy"):
    """Run fragility sampling and save results."""
    nodes_path = "data/flood/lm_infra_nodes_flood.geojson"
    if not os.path.exists(nodes_path):
        print(f"ERROR: {nodes_path} not found. Run from project root ~/Desktop/RA/")
        return None

    print(f"\n=== Fragility sampling: {label} (scale={depth_scale}) ===")
    scenarios = sample_initial_failures(nodes_path, n_scenarios=1000, depth_scale=depth_scale)

    # Summary statistics
    n_failed_arr = np.array([s["n_failed"] for s in scenarios])
    print(f"\n  Summary across 1000 Monte Carlo scenarios:")
    print(f"    Mean failures:   {n_failed_arr.mean():.1f}")
    print(f"    Median failures: {np.median(n_failed_arr):.1f}")
    print(f"    Std failures:    {n_failed_arr.std():.1f}")
    print(f"    Min/Max:         {n_failed_arr.min()} / {n_failed_arr.max()}")

    # Breakdown by type
    gdf = gpd.read_file(nodes_path)
    node_type = dict(zip(gdf["node_id"], gdf["infra_type"]))
    type_counts = Counter()
    for s in scenarios:
        for nid in s["failed_nodes"]:
            type_counts[node_type[nid]] += 1

    print(f"\n  Mean failures by type (across 1000 runs):")
    for t in ["power", "telecom", "hospital", "subway", "water", "fuel"]:
        mean_count = type_counts.get(t, 0) / len(scenarios)
        print(f"    {t:10s}: {mean_count:.1f}")

    # Validation check
    power_counts = [sum(1 for nid in s["failed_nodes"] if node_type[nid] == "power")
                    for s in scenarios]
    hospital_counts = [sum(1 for nid in s["failed_nodes"] if node_type[nid] == "hospital")
                       for s in scenarios]
    mean_power = np.mean(power_counts)
    mean_hospital = np.mean(hospital_counts)
    print(f"\n  VALIDATION (Sandy baseline):")
    print(f"    Mean failed power substations: {mean_power:.1f} (expected ~2-4)")
    print(f"    Mean failed hospitals:         {mean_hospital:.1f} (expected ~5-7)")
    if not (1.5 <= mean_power <= 5.0):
        print(f"    WARNING: Power failures outside expected range!")
    if not (3.0 <= mean_hospital <= 8.0):
        print(f"    WARNING: Hospital failures outside expected range!")

    # Save
    os.makedirs("data/simulation", exist_ok=True)
    out_path = f"data/simulation/monte_carlo_failures_{label}.json"
    with open(out_path, "w") as f:
        json.dump(scenarios, f)
    print(f"  Saved {len(scenarios)} scenarios to {out_path}")

    # Also save as the default path for sandy
    if label == "sandy":
        default_path = "data/simulation/monte_carlo_failures.json"
        with open(default_path, "w") as f:
            json.dump(scenarios, f)
        print(f"  Also saved to {default_path}")

    return scenarios


if __name__ == "__main__":
    os.chdir(os.path.expanduser("~/Desktop/RA"))
    run_and_save(depth_scale=1.0, label="sandy")
