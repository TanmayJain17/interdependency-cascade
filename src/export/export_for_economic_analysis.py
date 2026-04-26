#!/usr/bin/env python3
"""
export_for_economic_analysis.py

Exports cascade simulation data for downstream economic impact analysis.

Produces four CSV files in exports/economic_impact/:
  1. nodes_metadata.csv
     All infrastructure nodes with id, name, borough, lat/lon, type, flood depth
     per scenario. One row per node. This is the master node reference.

  2. extreme_2080_per_timestep.csv
     Per-run, per-timestep, per-failed-node long-format table for Extreme 2080.
     Columns: run_id, timestep_h, node_id, fail_cause (flood or cascade)
     Use this to track when specific nodes failed in each Monte Carlo run.

  3. extreme_2080_aggregated.csv
     Aggregated across 1000 runs. One row per node that failed in at least
     one run. Columns: node_id, fail_freq, mean_fail_time_h, p50_fail_time_h.
     Use this for expected-value economic analysis.

  4. sandy_lm_per_timestep.csv
     Same format as file 2 but for the Lower Manhattan Sandy scenario.
     For validation against 2012 documented economic impact.

  5. sandy_lm_aggregated.csv
     Same format as file 3 but for Sandy LM.

Plus a README.md documenting schemas, known limitations, and contact info.

Run from project root:
    python3 src/export/export_for_economic_analysis.py
"""

import json
import os
import sys
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import geopandas as gpd


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Inputs — citywide
NYC_NODES_PATH = Path("data/flood/nyc_infra_nodes_dep_flood.geojson")
NYC_CASCADE_PATH = Path("data/simulation/cascade_results_nyc_extreme_2080.json")
NYC_MC_PATH = Path("data/simulation/monte_carlo_failures_nyc_extreme_2080.json")
NYC_GRAPH_PATH = Path("data/flood/nyc_infra_graph_dep_flood.graphml")

# Inputs — LM Sandy (paths may differ — check your Week 5 outputs)
LM_NODES_PATH = Path("data/flood/lm_infra_nodes_flood.geojson")
LM_CASCADE_PATH = Path("data/simulation/cascade_results_sandy_actual.json")
LM_MC_PATH = Path("data/simulation/monte_carlo_failures_sandy_actual.json")
LM_GRAPH_PATH = Path("data/flood/lm_infra_graph_flood.graphml")

# Fallback paths if the Sandy-specific files don't exist
LM_CASCADE_FALLBACK = Path("data/simulation/cascade_results.json")
LM_MC_FALLBACK = Path("data/simulation/monte_carlo_failures.json")

OUT_DIR = Path("exports/economic_impact")
TIME_STEPS = [0, 6, 24, 48, 96]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def assign_borough(lat, lon):
    """Standard borough assignment from lat/lon — matches multi_scenario_runner."""
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


def simulate_single_run(graph_path, initial_failures):
    """Re-run cascade for a specific initial-failure set to get per-timestep lists."""
    # Import here to avoid load cost if only metadata export is needed
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "simulation"))
    from cascade_sim import load_graph, simulate_cascade

    G = load_graph(str(graph_path))
    return simulate_cascade(G, set(initial_failures), time_steps=TIME_STEPS)


# -----------------------------------------------------------------------------
# File 1: nodes metadata
# -----------------------------------------------------------------------------

def export_nodes_metadata():
    """Write master node reference CSV with all metadata and flood depths."""
    print("\n[1/5] Writing nodes_metadata.csv...")
    gdf = gpd.read_file(NYC_NODES_PATH)

    df = pd.DataFrame({
        "node_id": gdf["node_id"],
        "name": gdf.get("name", gdf["node_id"]),
        "infra_type": gdf["infra_type"],
        "lat": gdf["lat"].round(6),
        "lon": gdf["lon"].round(6),
    })
    df["borough"] = df.apply(lambda r: assign_borough(r["lat"], r["lon"]), axis=1)
    df["external"] = gdf.get("external", False).astype(bool) if "external" in gdf.columns else False

    # Flood depth per scenario
    for scenario in ["moderate_current", "moderate_2050", "extreme_2080"]:
        col = f"flood_{scenario}_depth_m"
        if col in gdf.columns:
            df[col] = gdf[col].fillna(0.0).round(3)
        else:
            df[col] = 0.0

    # Reorder columns
    cols = ["node_id", "name", "infra_type", "borough", "lat", "lon", "external",
            "flood_moderate_current_depth_m",
            "flood_moderate_2050_depth_m",
            "flood_extreme_2080_depth_m"]
    df = df[cols]

    out_path = OUT_DIR / "nodes_metadata.csv"
    df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path} ({len(df)} nodes)")
    return df


# -----------------------------------------------------------------------------
# File 2 + 3: Extreme 2080 per-timestep and aggregated
# -----------------------------------------------------------------------------

def export_extreme_2080(nodes_df):
    print("\n[2/5] Writing extreme_2080_per_timestep.csv...")
    print("  This re-runs cascade for each of 1000 MC runs to get per-step failures.")
    print("  Takes ~5-10 minutes.")

    with open(NYC_MC_PATH) as f:
        mc_scenarios = json.load(f)
    print(f"  Loaded {len(mc_scenarios)} Monte Carlo scenarios")

    rows = []
    for i, sc in enumerate(mc_scenarios):
        if (i + 1) % 100 == 0:
            print(f"  Processing run {i+1}/{len(mc_scenarios)}...")

        initial = set(sc["failed_nodes"])
        cascade = simulate_single_run(NYC_GRAPH_PATH, initial)

        prev_failed = set()
        for t in TIME_STEPS:
            current = set(cascade[f"t{t}"])
            new = current - prev_failed
            for nid in new:
                cause = "flood" if t == 0 else "cascade"
                rows.append({
                    "run_id": sc["scenario_id"],
                    "timestep_h": t,
                    "node_id": nid,
                    "fail_cause": cause,
                })
            prev_failed = current

    per_step = pd.DataFrame(rows)
    out_path = OUT_DIR / "extreme_2080_per_timestep.csv"
    per_step.to_csv(out_path, index=False)
    print(f"  Saved: {out_path} ({len(per_step):,} rows)")

    # File 3: aggregated
    print("\n[3/5] Writing extreme_2080_aggregated.csv...")

    # Count per-node: how often does it fail, and what's its median fail time?
    node_fail_times = defaultdict(list)
    for _, row in per_step.iterrows():
        node_fail_times[row["node_id"]].append(row["timestep_h"])

    n_runs = len(mc_scenarios)
    agg_rows = []
    for nid, times in node_fail_times.items():
        agg_rows.append({
            "node_id": nid,
            "fail_freq": round(len(times) / n_runs, 4),
            "mean_fail_time_h": round(float(np.mean(times)), 1),
            "p50_fail_time_h": int(np.median(times)),
            "p95_fail_time_h": int(np.percentile(times, 95)) if len(times) > 1 else times[0],
        })
    agg = pd.DataFrame(agg_rows).sort_values("fail_freq", ascending=False)

    # Merge with metadata for convenience
    agg = agg.merge(
        nodes_df[["node_id", "name", "infra_type", "borough", "lat", "lon",
                  "flood_extreme_2080_depth_m"]],
        on="node_id", how="left"
    )

    out_path = OUT_DIR / "extreme_2080_aggregated.csv"
    agg.to_csv(out_path, index=False)
    print(f"  Saved: {out_path} ({len(agg):,} rows — nodes that failed in at least one run)")


# -----------------------------------------------------------------------------
# File 4 + 5: Sandy LM per-timestep and aggregated
# -----------------------------------------------------------------------------

def export_sandy_lm():
    print("\n[4/5] Writing sandy_lm_per_timestep.csv...")

    # Resolve Sandy-specific paths with fallback
    cascade_path = LM_CASCADE_PATH if LM_CASCADE_PATH.exists() else LM_CASCADE_FALLBACK
    mc_path = LM_MC_PATH if LM_MC_PATH.exists() else LM_MC_FALLBACK

    if not cascade_path.exists() or not mc_path.exists():
        print(f"  SKIPPED: Sandy LM files not found at expected paths.")
        print(f"  Looked for: {LM_CASCADE_PATH} and fallback {LM_CASCADE_FALLBACK}")
        return

    if not LM_NODES_PATH.exists():
        print(f"  SKIPPED: LM flood-tagged nodes not found: {LM_NODES_PATH}")
        return

    if not LM_GRAPH_PATH.exists():
        print(f"  SKIPPED: LM flood-tagged graph not found: {LM_GRAPH_PATH}")
        return

    # Load LM nodes with Sandy metadata
    lm_gdf = gpd.read_file(LM_NODES_PATH)
    lm_nodes = pd.DataFrame({
        "node_id": lm_gdf["node_id"],
        "name": lm_gdf.get("name", lm_gdf["node_id"]),
        "infra_type": lm_gdf["infra_type"],
        "lat": lm_gdf["lat"].round(6),
        "lon": lm_gdf["lon"].round(6),
    })
    lm_nodes["borough"] = "Manhattan"
    depth_col = "flood_depth_m"
    lm_nodes["flood_sandy_depth_m"] = (
        lm_gdf[depth_col].fillna(0.0).round(3) if depth_col in lm_gdf.columns else 0.0
    )

    # Save LM nodes metadata separately
    lm_nodes.to_csv(OUT_DIR / "sandy_lm_nodes_metadata.csv", index=False)

    # Re-run per-step cascade for each MC scenario
    with open(mc_path) as f:
        mc_scenarios = json.load(f)
    print(f"  Loaded {len(mc_scenarios)} Sandy LM Monte Carlo scenarios")

    rows = []
    for i, sc in enumerate(mc_scenarios):
        if (i + 1) % 100 == 0:
            print(f"  Processing run {i+1}/{len(mc_scenarios)}...")

        initial = set(sc["failed_nodes"])
        cascade = simulate_single_run(LM_GRAPH_PATH, initial)

        prev_failed = set()
        for t in TIME_STEPS:
            current = set(cascade[f"t{t}"])
            new = current - prev_failed
            for nid in new:
                cause = "flood" if t == 0 else "cascade"
                rows.append({
                    "run_id": sc["scenario_id"],
                    "timestep_h": t,
                    "node_id": nid,
                    "fail_cause": cause,
                })
            prev_failed = current

    per_step = pd.DataFrame(rows)
    out_path = OUT_DIR / "sandy_lm_per_timestep.csv"
    per_step.to_csv(out_path, index=False)
    print(f"  Saved: {out_path} ({len(per_step):,} rows)")

    # File 5: aggregated
    print("\n[5/5] Writing sandy_lm_aggregated.csv...")
    node_fail_times = defaultdict(list)
    for _, row in per_step.iterrows():
        node_fail_times[row["node_id"]].append(row["timestep_h"])

    n_runs = len(mc_scenarios)
    agg_rows = []
    for nid, times in node_fail_times.items():
        agg_rows.append({
            "node_id": nid,
            "fail_freq": round(len(times) / n_runs, 4),
            "mean_fail_time_h": round(float(np.mean(times)), 1),
            "p50_fail_time_h": int(np.median(times)),
            "p95_fail_time_h": int(np.percentile(times, 95)) if len(times) > 1 else times[0],
        })
    agg = pd.DataFrame(agg_rows).sort_values("fail_freq", ascending=False)

    agg = agg.merge(lm_nodes, on="node_id", how="left")
    out_path = OUT_DIR / "sandy_lm_aggregated.csv"
    agg.to_csv(out_path, index=False)
    print(f"  Saved: {out_path} ({len(agg):,} rows)")


# -----------------------------------------------------------------------------
# README
# -----------------------------------------------------------------------------

README_TEXT = """# Cascade Simulation Data — For Economic Impact Analysis

Exported from the NYC Infrastructure Cascade pipeline for use in downstream
economic impact modeling. Generated by the research pipeline under Prof. Yuki
Miura's lab (NYU Tandon).

---

## Files in this package

### 1. nodes_metadata.csv
Master reference — one row per infrastructure node in the NYC graph.

| Column | Description |
|---|---|
| node_id | Unique identifier (stable across scenarios) |
| name | Human-readable name |
| infra_type | One of: power, telecom, hospital, subway, water, fuel |
| borough | Manhattan / Brooklyn / Queens / Bronx / Staten Island / NJ (external) |
| lat, lon | WGS84 coordinates (6 decimal places) |
| external | True if node is outside NYC (NJ substations, petroleum terminals) |
| flood_*_depth_m | Flood depth in meters under each DEP scenario |

Scope: 6,231 nodes across all five boroughs plus external NJ substations and
petroleum terminals.

---

### 2. extreme_2080_per_timestep.csv
Per-run, per-timestep failure records for the DEP Extreme 2080 scenario
(500-year rainfall + 2080 sea level rise).

| Column | Description |
|---|---|
| run_id | Monte Carlo run identifier (0-999) |
| timestep_h | Hours after flood onset: 0, 6, 24, 48, or 96 |
| node_id | Which node failed at this timestep in this run |
| fail_cause | 'flood' (t=0, direct) or 'cascade' (t>0, propagated) |

Use this to track how failures propagate over time. For example, to find all
nodes that fail within 24 hours across all runs:
    df[df.timestep_h <= 24].node_id.unique()

Row count: varies by run, roughly 300-800k rows total.

---

### 3. extreme_2080_aggregated.csv
Aggregated across all 1000 Monte Carlo runs. One row per node that failed
in at least one run.

| Column | Description |
|---|---|
| node_id | Node identifier |
| fail_freq | Fraction of runs where this node failed (0.0 to 1.0) |
| mean_fail_time_h | Average time to failure across runs that failed it |
| p50_fail_time_h | Median failure time |
| p95_fail_time_h | 95th percentile failure time |
| (plus all columns from nodes_metadata for joined context) |

Use this for expected-value analysis. For example, to find the 50 nodes most
likely to fail and the typical timing:
    df.nlargest(50, 'fail_freq')

---

### 4. sandy_lm_per_timestep.csv + sandy_lm_aggregated.csv + sandy_lm_nodes_metadata.csv
Same format as files 2 and 3 but for Hurricane Sandy scenario at Lower
Manhattan scope (347 nodes).

Use this for validation against 2012 Sandy historical economic impact data.

IMPORTANT: Sandy data is LM-only. A citywide Sandy scenario would require
running the GISSR surge model for all boroughs, which is future work.

---

## Known limitations (please factor into your analysis)

1. **DEP Extreme 2080 excludes storm surge.** The DEP flood maps model pluvial
   (rainfall) and tidal flooding only. Storm surge from hurricanes is not
   included. As a result, surge-exposed infrastructure like hospitals along
   the FDR corridor (Bellevue, NYU Langone, Mount Sinai Beth Israel, Mount
   Sinai NYEE) show up as "dry" in DEP's footprint — but they still fail in
   our cascade because their upstream power and fuel suppliers are flooded.
   The Sandy LM data captures surge exposure for validation.

2. **Cascade model is rule-based.** Each dependency edge carries a buffer
   time representing how long the downstream node survives after the
   upstream source fails (e.g., hospital generator = 96h on NFPA 110 fuel
   reserve). We do NOT yet model intra-infrastructure cascade (power grid
   load redistribution via pandapower). Future versions of this data will
   have more conservative total failure counts once intra-grid cascade is
   added.

3. **Redundancy = OR-gate.** If a hospital has both a power supply and a
   fuel supply, either failing causes the hospital to fail in our model.
   In reality, redundancy should be AND-gate (both must fail). This tends
   to over-estimate failure rates for well-redundant systems.

4. **NJ substations and petroleum terminals can receive cascade but not
   originate it.** Their failure probability is always 0.0 in our fragility
   model since we don't have their flood exposure data. They may cascade-fail
   if their dependencies fail.

5. **Monte Carlo runs are noisy.** Individual run-level data varies; for
   expected-value impact estimates, use the aggregated files.

---

## Recommended usage

For direct economic impact (which blocks lose which services):
- Join `nodes_metadata.csv` with `extreme_2080_aggregated.csv` on `node_id`
- Filter to nodes with `fail_freq > 0.5` for "likely to fail" analysis
- Use `lat/lon` to spatially join with NYC census blocks

For time-dependent impact (how long is the outage):
- Use `extreme_2080_per_timestep.csv` to get failure timing distribution
- For each node, look at `mean_fail_time_h` and `p95_fail_time_h` to
  understand how quickly and how variably it fails

For Sandy validation:
- Compare `sandy_lm_aggregated.csv` failure frequencies against documented
  2012 Sandy infrastructure outages
- Four hospitals (Bellevue, NYU Langone, Mount Sinai Beth Israel, Mount
  Sinai NYEE) evacuated in 2012 due to cascade-failure — check whether
  these appear with high `fail_freq` in your Sandy LM data

---

## Questions?

Contact Tanmay Jain. Happy to explain schema, filter for specific nodes,
or generate additional scenarios (moderate_current, moderate_2050) if
useful for your analysis.

Data generated: Week 6 (April 2026)
Pipeline: NYC Infrastructure Cascade, Climate, Energy, and Risk Analytics Lab
"""


def write_readme():
    out_path = OUT_DIR / "README.md"
    out_path.write_text(README_TEXT)
    print(f"\n  Saved: {out_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    print("=" * 72)
    print("Export for Economic Impact Analysis")
    print("=" * 72)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Validate inputs
    if not NYC_NODES_PATH.exists():
        print(f"ERROR: {NYC_NODES_PATH} not found. Run flood_overlay_v3.py nyc first.")
        return 1

    nodes_df = export_nodes_metadata()
    export_extreme_2080(nodes_df)
    export_sandy_lm()
    write_readme()

    print("\n" + "=" * 72)
    print(f"Export complete. Files in: {OUT_DIR}/")
    print("=" * 72)
    print("\nPackage contents:")
    for f in sorted(OUT_DIR.iterdir()):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name:45s} {size_mb:6.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())