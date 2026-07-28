#!/usr/bin/env python3
"""
run_synthetic20.py — seed + cascade production for Gwen's 20 synthetic surge
rasters (Week 19). A thin wrapper around the PRODUCTION pipeline: it patches
multi_scenario_runner's module globals and calls its run_scenario() verbatim —
same fragility (HAZUS, regime=pluvial like all six production scenarios), same
Weibull buffers (seeds 42/43), same joint engine. No production file is
touched: outputs land in data/simulation_synthetic20/.

Design notes (locked by phase0_gwen_20maps_gate + node-depth check):
  * Rasters store DEPTH in meters on a shared EPSG:2263 grid; nodata=0.0 means
    dry land and nodata are the same value, so sampling uses RAW reads with
    0-means-dry semantics. Do not "fix" this into masked reads.
  * No land/water mask: bathymetry cells (>10 m, mid-channel) never coincide
    with infrastructure except 4 waterfront-geocoded nodes (fuel_station_
    lukoil_1058, telecom_cluster_03250/00840/00887) at 5-7 m — immaterial
    because HAZUS fragility saturates ~P=1 below 5 m anyway. Known limitation,
    documented here and in the summary JSON.
  * Scenario tag = syn_<full ts suffix> (e.g. syn_ts_808_27_3p7885): unique,
    traceable to the source file, sorts stably.
  * Power coupling: not applicable (Jesse's library covers gc scenarios only);
    these runs use the legacy power layer by construction, matching the
    current production default.
  * Resume: scenarios whose cascade_results file already exists are skipped,
    so a killed overnight restarts where it stopped.

Env knobs:
    SYN_DIR   raster directory (default /Users/tanmayjain/Downloads/nyc_synthetic_flood)
    N_MC      Monte Carlo runs per map (default 250)
    SYN_LIMIT run only the first k maps by ascending peak (cheap-first gate)

Usage (project root, conda flood):
    SYN_LIMIT=2 N_MC=2 python src/simulation/run_synthetic20.py   # cheap-first
    caffeinate -is python src/simulation/run_synthetic20.py       # full overnight

Outputs:
    data/flood/synthetic20_node_depths.csv        (versioned sampling record)
    data/simulation_synthetic20/monte_carlo_failures_nyc_syn_*.json
    data/simulation_synthetic20/cascade_results_nyc_syn_*.json
    data/analysis/synthetic20_summary.json        (peak/footprint vs cascade table)
"""

import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.warp import transform as warp_transform

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import multi_scenario_runner as msr
from src.cascade.stochastic_buffer import load_buffer_config

SYN_DIR = Path(os.environ.get(
    "SYN_DIR", "/Users/tanmayjain/Downloads/nyc_synthetic_flood"))
OUT_SIM = Path("data/simulation_synthetic20")
DEPTHS_CSV = Path("data/flood/synthetic20_node_depths.csv")
SUMMARY_JSON = Path("data/analysis/synthetic20_summary.json")
N_MC = int(os.environ.get("N_MC", "250"))
SYN_LIMIT = int(os.environ.get("SYN_LIMIT", "0"))  # 0 = all

NAME_RE = re.compile(
    r"nyc_combined_inundation_area_(ts_\d+_\d+_\d+p\d+)\.tif$")


def discover_maps():
    maps = []
    for f in sorted(os.listdir(SYN_DIR)):
        m = NAME_RE.search(f)
        if not m:
            continue
        tag = m.group(1)                                  # ts_808_27_3p7885
        ip, dp = re.search(r"(\d+)p(\d+)$", tag).groups()
        maps.append({"file": f, "tag": f"syn_{tag}",
                     "peak_m": float(f"{ip}.{dp}")})
    maps.sort(key=lambda m: m["peak_m"])
    if SYN_LIMIT:
        maps = maps[:SYN_LIMIT]
    return maps


def sample_depths(nodes_gdf, maps):
    """Sample every raster at node coordinates. RAW reads: nodata=0.0 == dry."""
    lons = nodes_gdf.geometry.x.values
    lats = nodes_gdf.geometry.y.values
    cols = {}
    for m in maps:
        with rasterio.open(SYN_DIR / m["file"]) as r:
            xs, ys = warp_transform("EPSG:4326", r.crs, list(lons), list(lats))
            vals = np.array([v[0] for v in r.sample(zip(xs, ys))],
                            dtype=np.float64)
        vals = np.nan_to_num(vals, nan=0.0)
        vals[vals < 0] = 0.0
        col = f"flood_{m['tag']}_depth_m"
        cols[col] = vals
        wet = vals > 0
        m.update(n_wet_nodes=int(wet.sum()),
                 node_depth_med=float(np.median(vals[wet])) if wet.any() else 0.0,
                 node_depth_max=float(vals.max()))
        print(f"  sampled {m['tag']}: peak {m['peak_m']:.3f} m | "
              f"wet nodes {m['n_wet_nodes']} | med {m['node_depth_med']:.2f} | "
              f"max {m['node_depth_max']:.2f}")
    for col, vals in cols.items():
        nodes_gdf[col] = vals
    rec = nodes_gdf[["node_id"] + list(cols)].copy()
    DEPTHS_CSV.parent.mkdir(parents=True, exist_ok=True)
    rec.to_csv(DEPTHS_CSV, index=False)
    print(f"  wrote sampling record: {DEPTHS_CSV}")
    return nodes_gdf


def main():
    t0 = time.time()
    maps = discover_maps()
    print(f"SYNTHETIC-20 PRODUCTION — {len(maps)} maps, N_MC={N_MC}, "
          f"outputs -> {OUT_SIM}")
    if not maps:
        sys.exit("no rasters found — check SYN_DIR")
    OUT_SIM.mkdir(parents=True, exist_ok=True)

    print("\n[1/3] Loading nodes and sampling rasters...")
    nodes_gdf = gpd.read_file(msr.NODES_IN)
    print(f"  {len(nodes_gdf):,} nodes from {msr.NODES_IN}")
    nodes_gdf = sample_depths(nodes_gdf, maps)

    print("\n[2/3] Patching production runner globals...")
    msr.SIM_DIR = OUT_SIM                       # isolate outputs
    msr.N_MONTE_CARLO = N_MC
    for m in maps:
        msr.SCENARIO_DEPTH_COL[m["tag"]] = f"flood_{m['tag']}_depth_m"
    buffer_config = load_buffer_config("config/buffer_distributions.yaml")
    print(f"  SIM_DIR -> {msr.SIM_DIR} | N_MC -> {msr.N_MONTE_CARLO} | "
          f"{len(maps)} scenario depth columns registered")

    print("\n[3/3] Running scenarios (ascending peak; resume-safe)...")
    summary = []
    for i, m in enumerate(maps, 1):
        done_marker = OUT_SIM / f"cascade_results_nyc_{m['tag']}.json"
        if done_marker.exists():
            print(f"\n[{i}/{len(maps)}] {m['tag']} — already done, skipping")
            with open(done_marker) as fh:
                runs = json.load(fh)
        else:
            t1 = time.time()
            print(f"\n[{i}/{len(maps)}] {m['tag']} (peak {m['peak_m']:.3f} m)")
            _mc, runs = msr.run_scenario(m["tag"], nodes_gdf, buffer_config)
            print(f"  scenario wall time: {(time.time()-t1)/60:.1f} min")
        direct = np.array([r["direct_failures"] for r in runs], dtype=float)
        total = np.array([r["total_failures"] for r in runs], dtype=float)
        summary.append({**{k: m[k] for k in
                           ("tag", "file", "peak_m", "n_wet_nodes",
                            "node_depth_med", "node_depth_max")},
                        "n_runs": len(runs),
                        "direct_mean": round(float(direct.mean()), 2),
                        "total_mean": round(float(total.mean()), 2),
                        "total_median": float(np.median(total)),
                        "total_p90": float(np.percentile(total, 90)),
                        "amplification_mean": round(
                            float((total / np.maximum(direct, 1)).mean()), 3)})
        SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
        with open(SUMMARY_JSON, "w") as fh:       # rewrite after every scenario
            json.dump({"n_mc": N_MC, "maps": summary,
                       "known_limitation":
                           "4 waterfront-geocoded nodes sample 5-7 m "
                           "(bathymetry-adjacent cells); immaterial under "
                           "HAZUS saturation"}, fh, indent=1)

    print(f"\n{'='*75}\nFRAGILITY CURVE (raw first look)\n"
          f"{'peak_m':>7s} {'wet_nodes':>9s} {'direct':>8s} {'total':>8s} "
          f"{'median':>8s} {'p90':>8s} {'A':>6s}")
    for s in summary:
        print(f"{s['peak_m']:7.3f} {s['n_wet_nodes']:9d} {s['direct_mean']:8.1f} "
              f"{s['total_mean']:8.1f} {s['total_median']:8.0f} "
              f"{s['total_p90']:8.0f} {s['amplification_mean']:6.2f}")
    print(f"\nTotal wall time: {(time.time()-t0)/3600:.2f} h")
    print(f"Summary: {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
