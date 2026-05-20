#!/usr/bin/env python3
"""
Task 1: Fragility curves and Monte Carlo initial failure sampling.

HAZUS-based lognormal fragility: P(failure | depth) = Phi((ln(depth) - mu) / beta)
where mu = ln(median_depth), beta = dispersion.

Hazard regimes:
    'pluvial' — DEP scenarios (rainfall + tide). Uses HAZUS-FL curves as-is.
    'surge'   — GeoClaw scenarios (hurricane storm surge). Depths are scaled by
                SURGE_DEPTH_SCALING before the lognormal CDF, calibrated against
                the Sandy 2012 four-hospital evacuation pattern.

This is the bridge to a proper multi-feature fragility (depth + velocity + wave)
that will be implemented as a learnable layer in week 11 (Option B).
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

# Sandy-calibrated effective-depth scaling for hurricane storm surge scenarios.
#
# HAZUS-MH FL fragility curves capture static inundation depth, calibrated for
# pluvial flooding. Hurricane storm surge introduces dynamic loading (velocity,
# wave slamming, salt water corrosion) that HAZUS-FL does not represent.
#
# Calibration attempt (scripts/calibrate_surge_scaling.py, geoclaw_2026 vs Sandy
# four-hospital evacuation pattern) showed that scaling factors from 1.0 to 1.5
# all produce the same outcome at the Sandy target hospitals (P >= 0.99) and
# differ by less than 0.2% in total expected failures at t=96. The GeoClaw
# depth distribution is aggressive enough to saturate HAZUS-FL by itself; depth
# scaling has no discriminative power within the surge regime.
#
# Conclusion: keep the hazard_regime parameter (it preserves the API for the
# future learnable multi-feature fragility, Option B), but set scaling to 1.0.
# A meaningful surge correction will require multi-feature inputs (depth +
# velocity + wave) rather than a single scalar — this is the Option B target.
SURGE_DEPTH_SCALING: float = 1.0

VALID_REGIMES = ("pluvial", "surge")


def failure_probability(
    flood_depth_m: float,
    infra_type: str,
    hazard_regime: str = "pluvial",
) -> float:
    """Return P(failure) in [0,1] given flood depth and infrastructure type.

    Args:
        flood_depth_m: flood depth in meters. Non-positive values return 0.0.
        infra_type: key into FRAGILITY_PARAMS.
        hazard_regime: 'pluvial' (default, no scaling) or 'surge' (depth scaled
            by SURGE_DEPTH_SCALING before HAZUS CDF).
    """
    if flood_depth_m is None or flood_depth_m <= 0:
        return 0.0
    if infra_type not in FRAGILITY_PARAMS:
        return 0.0
    if hazard_regime not in VALID_REGIMES:
        raise ValueError(
            f"Unknown hazard_regime '{hazard_regime}'. "
            f"Expected one of {VALID_REGIMES}."
        )

    if hazard_regime == "surge":
        flood_depth_m = flood_depth_m * SURGE_DEPTH_SCALING

    median, beta = FRAGILITY_PARAMS[infra_type]
    mu = np.log(median)
    return float(norm.cdf((np.log(flood_depth_m) - mu) / beta))


def failure_probability_vectorized(
    flood_depths: np.ndarray,
    infra_types: list,
    hazard_regime: str = "pluvial",
) -> np.ndarray:
    """Vectorized HAZUS lognormal CDF for many nodes at once.

    Used by the inference wrapper to compute initial-failure probabilities
    for all 6,231 NYC nodes in a single call rather than 6,231 scalar calls.

    Args:
        flood_depths: shape [N] array of depths in meters. NaN, None, or
            non-positive values produce 0.0 failure probability.
        infra_types: length-N list/array of infrastructure type strings.
            Each must be a key in FRAGILITY_PARAMS or the corresponding row
            gets 0.0 probability.
        hazard_regime: 'pluvial' (default) or 'surge'.

    Returns:
        shape [N] np.ndarray of failure probabilities in [0, 1].
    """
    if hazard_regime not in VALID_REGIMES:
        raise ValueError(
            f"Unknown hazard_regime '{hazard_regime}'. "
            f"Expected one of {VALID_REGIMES}."
        )

    depths = np.asarray(flood_depths, dtype=np.float64)
    # Replace NaN with 0 so they fall into the "no flood" branch
    depths = np.where(np.isnan(depths), 0.0, depths)

    if hazard_regime == "surge":
        depths = depths * SURGE_DEPTH_SCALING

    probs = np.zeros(len(depths), dtype=np.float64)
    flood_mask = depths > 0

    if not flood_mask.any():
        return probs

    # Compute per-infra-type, vectorized within each type
    infra_arr = np.asarray(infra_types)
    for infra_type, (median, beta) in FRAGILITY_PARAMS.items():
        type_mask = (infra_arr == infra_type) & flood_mask
        if not type_mask.any():
            continue
        mu = np.log(median)
        z = (np.log(depths[type_mask]) - mu) / beta
        probs[type_mask] = norm.cdf(z)

    return probs


def sample_initial_failures(
    nodes_geojson_path: str,
    n_scenarios: int = 1000,
    seed: int = 42,
    depth_scale: float = 1.0,
    hazard_regime: str = "pluvial",
) -> list:
    """
    Monte Carlo sampling of initial flood failures.

    For each scenario, each node independently fails with probability
    P(failure | scaled_flood_depth, infra_type, hazard_regime).

    Args:
        nodes_geojson_path: path to a flood-tagged GeoJSON with fields
            node_id, infra_type, flood_depth_m, gissr_division.
        n_scenarios: number of Monte Carlo draws.
        seed: RNG seed.
        depth_scale: multiplier on flood_depth_m for return-period adjustment
            (1.0 = Sandy baseline, 1.45 = 500-year). Independent of regime.
        hazard_regime: 'pluvial' or 'surge'. Forwarded to failure_probability.

    Returns list of dicts with scenario_id, failed_nodes, n_failed.
    """
    gdf = gpd.read_file(nodes_geojson_path)
    print(f"  Loaded {len(gdf)} nodes from {nodes_geojson_path}")

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
            probs.append(
                failure_probability(
                    depth * depth_scale,
                    infra_types[i],
                    hazard_regime=hazard_regime,
                )
            )

    probs = np.array(probs)
    n_vulnerable = np.sum(probs > 0)
    print(f"  {n_vulnerable} nodes have P(failure) > 0 "
          f"(depth_scale={depth_scale}, regime={hazard_regime})")

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

    n_failed_arr = np.array([s["n_failed"] for s in scenarios])
    print(f"\n  Summary across 1000 Monte Carlo scenarios:")
    print(f"    Mean failures:   {n_failed_arr.mean():.1f}")
    print(f"    Median failures: {np.median(n_failed_arr):.1f}")
    print(f"    Std failures:    {n_failed_arr.std():.1f}")
    print(f"    Min/Max:         {n_failed_arr.min()} / {n_failed_arr.max()}")

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

    os.makedirs("data/simulation", exist_ok=True)
    out_path = f"data/simulation/monte_carlo_failures_{label}.json"
    with open(out_path, "w") as f:
        json.dump(scenarios, f)
    print(f"  Saved {len(scenarios)} scenarios to {out_path}")

    if label == "sandy":
        default_path = "data/simulation/monte_carlo_failures.json"
        with open(default_path, "w") as f:
            json.dump(scenarios, f)
        print(f"  Also saved to {default_path}")

    return scenarios


if __name__ == "__main__":
    os.chdir(os.path.expanduser("~/Desktop/RA"))
    run_and_save(depth_scale=1.0, label="sandy")