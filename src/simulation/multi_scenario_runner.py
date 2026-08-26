#!/usr/bin/env python3
"""
multi_scenario_runner.py — Citywide cascade analysis (stochastic buffers)

Orchestrates fragility + Monte Carlo + cascade simulation for the three DEP
flood scenarios (moderate_current, moderate_2050, extreme_2080) on the
citywide heterogeneous infrastructure graph.

Approach (Option A): Wraps the existing fragility.py and cascade_sim.py
WITHOUT modifying them. For each scenario, creates a temporary nodes GeoJSON
with the appropriate DEP depth column aliased to 'flood_depth_m' and a synthetic
'gissr_division' column (-1 for external nodes, 0 for NYC nodes).

Week 7 update: Buffer hours on dependency edges are now sampled per Monte Carlo
iteration from Weibull(median=edge.buffer_hours, shape=config[target_type].shape).
Engineering medians on the graph are unchanged; only variability is added.

Inputs:
    data/flood/nyc_infra_nodes_dep_flood.geojson  (6,231 nodes with 3 scenarios)
    data/flood/nyc_infra_graph_dep_flood.graphml  (directed graph with buffers)
    config/buffer_distributions.yaml              (Weibull shape per target type)

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
from cascade_joint import build_intra_context, simulate_cascade_joint
from dynamic_forcing import load_dynamic_forcing, build_record, FLOOD_CAUSE
# Make fragility.py and cascade_sim.py importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Make src.cascade.stochastic_buffer importable from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fragility import sample_initial_failures
from cascade_sim import load_graph, get_cascade_edges, simulate_cascade
from intra_power import load_power_library, build_power_join, sample_power_state
import yaml
from src.cascade.stochastic_buffer import (
    load_buffer_config,
    sample_stochastic_graph,
)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

NODES_IN = Path(os.environ.get("NODES_IN", "data/flood/nyc_infra_nodes_all_flood.geojson"))
GRAPH_IN = Path(os.environ.get("GRAPH_IN", "data/flood/nyc_infra_graph_all_flood.graphml"))

SIM_DIR = Path(os.environ.get("SIM_DIR", "data/simulation"))
OUT_DIR = Path("outputs")

SCENARIOS = [
    "moderate_current",
    "moderate_2050",
    "extreme_2080",
    "geoclaw_2026",
    "geoclaw_2050",
    "geoclaw_2080",
]

_only = os.environ.get("SCENARIO_ONLY")
if _only:
    SCENARIOS = [s for s in SCENARIOS if s == _only]

SCENARIO_DEPTH_COL = {
    "moderate_current": "flood_moderate_current_depth_m",
    "moderate_2050":    "flood_moderate_2050_depth_m",
    "extreme_2080":     "flood_extreme_2080_depth_m",
    "geoclaw_2026":     "gc_2026_depth_m",
    "geoclaw_2050":     "gc_2050_depth_m",
    "geoclaw_2080":     "gc_2080_depth_m",
}

N_MONTE_CARLO = int(os.environ.get("N_MC", "1000"))
TIME_STEPS = [0, 6, 24, 48, 96]

# Amplifier thresholds (for extreme_2080 analysis)
AMP_FLOOD_THRESHOLD_M = 0.05  # node is "dry" if depth < 5 cm
AMP_FREQ_THRESHOLD = 0.50     # node is "amplifier" if it fails in >50% of runs

# RNG seeds — independent streams for fragility vs buffer sampling
FRAGILITY_SEED = 42
BUFFER_SEED = 43

INTRA_CONFIG = Path("config/intra_cascade.yaml")


def load_power_coupling():
    """Jesse intra-power coupling (Week 19). Gated by
    config/intra_cascade.yaml -> power_coupling.enabled (default OFF).
    When enabled: covered power nodes skip HAZUS seeding and receive
    Jesse's index-paired sampled grid state at the t=6 floor with cause
    'intra_power' (replace-not-union; inter-layer edges into power stay
    live). Scenarios absent from his library fall back to legacy."""
    with open(INTRA_CONFIG) as f:
        cfg = yaml.safe_load(f) or {}
    pc = cfg.get("power_coupling") or {}
    _env = os.environ.get("POWER_COUPLING")          # per-job override; config
    if _env is not None:                             # stays 'false' at rest
        pc["enabled"] = _env == "1"
    if os.environ.get("JESSE_DIR"):
        pc["jesse_dir"] = os.environ["JESSE_DIR"]
    if not pc.get("enabled", False):
        print("  [power-coupling] disabled (legacy HAZUS power seeding)")
        return None
    lib = load_power_library(pc.get("jesse_dir", "data/external/jesse_power"))
    join = build_power_join(str(NODES_IN), lib)
    tied = pc.get("tied_rule", "matched")
    print(f"  [power-coupling] ENABLED: {len(join)} power nodes covered by "
          f"Jesse's library, scenarios={sorted(lib.scenarios)}, "
          f"tied_rule={tied}")
    return {"lib": lib, "join": join, "tied_rule": tied}


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
    depth_col = SCENARIO_DEPTH_COL[scenario_name] 
    if depth_col not in temp.columns:
        raise ValueError(f"Missing column {depth_col} in nodes GeoJSON")

    temp["flood_depth_m"] = temp[depth_col].fillna(0.0)

    def is_external(x):
        return str(x).lower() in ("true", "1", "yes")

    if "external" in temp.columns:
        temp["gissr_division"] = temp["external"].apply(
            lambda x: -1 if is_external(x) else 0
        )
    else:
        temp["gissr_division"] = 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    keep_cols = ["node_id", "infra_type", "lat", "lon", "external",
                 "flood_depth_m", "gissr_division", "geometry"]
    keep_cols = [c for c in keep_cols if c in temp.columns]
    temp[keep_cols].to_file(out_path, driver="GeoJSON")


# -----------------------------------------------------------------------------
# Per-scenario run (with inline stochastic-buffer MC loop)
# -----------------------------------------------------------------------------

def run_scenario(scenario_name, nodes_gdf, buffer_config, power_coupling=None):
    """Run fragility + stochastic-buffer cascade for one DEP scenario.
    Returns (mc_scenarios, cascade_results)."""
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
        seed=FRAGILITY_SEED,
        depth_scale=1.0,
    )

    n_failed = np.array([s["n_failed"] for s in mc_scenarios])
    print(f"  Initial failures: mean={n_failed.mean():.1f}  std={n_failed.std():.1f}  "
          f"min={n_failed.min()}  max={n_failed.max()}")

    mc_out = SIM_DIR / f"monte_carlo_failures_nyc_{scenario_name}.json"
    with open(mc_out, "w") as f:
        json.dump(mc_scenarios, f)
    print(f"  Saved: {mc_out}")

    # 3. Cascade simulation — STOCHASTIC BUFFERS
    print(f"\n  [Cascade] Propagating through graph (Weibull-sampled buffers)...")
    G = load_graph(str(GRAPH_IN))
    print(f"  Loaded graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    intra_ctx = build_intra_context(G)

    # Week 23 dynamic forcing (config/dynamic_forcing.yaml, env DYNAMIC_FORCING / DYNAMIC_MODE).
    # Disabled -> the block below is skipped and the run is byte-identical to the frozen campaign.
    dyn = load_dynamic_forcing()
    dyn_active = dyn is not None and dyn.has_scenario(scenario_name)
    if dyn is not None and not dyn_active:
        print(f"  [dynamic-forcing] no timing table for {scenario_name}; static forcing for this scenario")
    if dyn_active and power_coupling is not None and not dyn.allow_power_coupling:
        raise NotImplementedError("dynamic forcing with the power-coupling arm is not derived yet "
                                  "(Jesse's t=6 floor is peak-anchored); run the legacy arm")
    if dyn_active:
        dyn_grid = dyn.grid(scenario_name)
        dyn_tpeak = dyn.t_peak[scenario_name]
        dyn_intra = dyn.intra_origin(scenario_name)
        print(f"  [dynamic-forcing] {scenario_name}: mode={dyn.mode} t_peak={dyn_tpeak:.1f} h "
              f"intra_clock={dyn.intra_clock} grid={dyn_grid}")

    # Independent RNG for buffer sampling (fragility uses its own seed inside)
    buffer_rng = np.random.default_rng(BUFFER_SEED)

    cascade_results = []
    time_keys = [f"t{t}" for t in TIME_STEPS]

    for run_id, scenario in enumerate(mc_scenarios):
        # Sample fresh stochastic graph for this MC run
        G_stoch = sample_stochastic_graph(G, rng=buffer_rng, config=buffer_config)

        # Audit on first iteration
        if run_id == 0:
            summary = G_stoch.graph["_stochastic_buffer_summary"]
            print(f"  Stochastic buffer audit (first MC run):")
            print(f"    Sampled:          {summary['n_sampled']} edges")
            print(f"    Zero-buffer kept: {summary['n_zero_buffer']} edges")
            print(f"    Fallback used:    {summary['n_fallback']} edges")
            if summary["fallback_target_types"]:
                print(f"    WARNING — unrecognized target types: "
                      f"{summary['fallback_target_types']}")
                print(f"    Add these to config/buffer_distributions.yaml")

        # Run cascade — note: scenario field name may be 'failed_nodes' or
        # 'initial_failures' depending on your fragility.py version
        initial_failures = set(scenario.get("failed_nodes",
                                            scenario.get("initial_failures", [])))

        # Week 19 power coupling: Jesse's library replaces flood seeding +
        # intra-power cascade for covered nodes (his final state already
        # contains the flood). Dead set lands at the t=6 floor, cause
        # 'intra_power'; uncovered (NJ / out-of-footprint) nodes keep legacy.
        scheduled = None
        if power_coupling is not None and \
                power_coupling["lib"].resolve(scenario_name) is not None:
            st = sample_power_state(power_coupling["lib"], power_coupling["join"],
                                    scenario_name, run_id,
                                    tied_rule=power_coupling["tied_rule"])
            initial_failures -= st.covered_nodes          # replace, not union
            scheduled = {nid: (6.0, "intra_power") for nid in st.dead_nodes}
            if run_id == 0:
                print(f"  [power-coupling] {scenario_name}: "
                      f"{len(st.covered_nodes)} covered nodes skip HAZUS "
                      f"seeding; Jesse run {st.jesse_run_idx} kills "
                      f"{len(st.dead_nodes)} at the t=6 floor")
        """ cascade = simulate_cascade(G_stoch, initial_failures, time_steps=TIME_STEPS)

        # Per-node first-failure timestep (compact form for GNN labels):
        # iterate timesteps in order, record earliest. Nodes not in dict never failed.
        fail_time_per_node = {}
        for tk in time_keys:
            t_int = int(tk[1:])  # "t6" -> 6
            for nid in cascade[tk]:
                if nid not in fail_time_per_node:
                    fail_time_per_node[nid] = t_int """
                    
        if dyn_active:
            # seeds stay exactly as sampled; only WHEN they land changes
            seeds = {nid for nid in initial_failures if nid in G_stoch}
            seed_hours = dyn.seed_hours(scenario_name, seeds)
            if dyn.mode == "static_peak":
                # pure time translation of the frozen run: seeds are initial failures at t_origin=t_peak
                cascade, fail_time, cause = simulate_cascade_joint(
                    G_stoch, seeds, intra_ctx, time_steps=dyn_grid,
                    scheduled_failures=scheduled, t_origin=dyn_tpeak, intra_clock_origin=dyn_intra)
            else:
                sched = dict(scheduled or {})
                for nid, h in seed_hours.items():
                    if nid not in sched or h < sched[nid][0]:
                        sched[nid] = (h, FLOOD_CAUSE)
                cascade, fail_time, cause = simulate_cascade_joint(
                    G_stoch, set(), intra_ctx, time_steps=dyn_grid,
                    scheduled_failures=sched, t_origin=0.0, intra_clock_origin=dyn_intra)
            cascade_results.append(build_record(scenario.get("scenario_id", run_id), seeds, seed_hours,
                                                fail_time, cause, dyn_tpeak, dyn.post_peak_offsets_h))
        else:
            cascade, fail_time, cause = simulate_cascade_joint(
                G_stoch, initial_failures, intra_ctx, time_steps=TIME_STEPS,
                scheduled_failures=scheduled)

            fail_time_per_node = {nid: int(t) for nid, t in fail_time.items()}
            cause_counts = dict(Counter(cause.values()))

            by_timestep = {tk: len(cascade[tk]) for tk in time_keys}
            cascade_results.append({
                "scenario_id": scenario.get("scenario_id", run_id),
                "direct_failures": by_timestep["t0"],
                "cause_counts": cause_counts,
                "total_failures":  by_timestep[time_keys[-1]],
                "failed_nodes_t96": list(cascade[time_keys[-1]]),
                "by_timestep": by_timestep,
                "fail_time_per_node": fail_time_per_node,  # NEW
            })

        if (run_id + 1) % 100 == 0:
            print(f"    Completed {run_id + 1}/{len(mc_scenarios)} MC runs")

    # Save cascade results in the same format run_all_scenarios produced
    cascade_out = SIM_DIR / f"cascade_results_nyc_{scenario_name}.json"
    with open(cascade_out, "w") as f:
        json.dump(cascade_results, f)
    print(f"  Saved: {cascade_out}")

    # Quick summary print
    direct = np.array([r["direct_failures"] for r in cascade_results])
    total = np.array([r["total_failures"] for r in cascade_results])
    print(f"  [{scenario_name}] direct: {direct.mean():.1f} +/- {direct.std():.1f}, "
          f"total: {total.mean():.1f} +/- {total.std():.1f}")

    return mc_scenarios, cascade_results


# -----------------------------------------------------------------------------
# Cross-scenario comparison
# -----------------------------------------------------------------------------

def summarize(all_results, nodes_gdf):
    """Build comparison dict from per-scenario cascade results."""
    node_type = dict(zip(nodes_gdf["node_id"], nodes_gdf["infra_type"]))

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

        type_counts = Counter()
        for r in cascade:
            for nid in r["failed_nodes_t96"]:
                type_counts[node_type.get(nid, "unknown")] += 1
        n_runs = len(cascade)
        type_means = {
            t: round(type_counts.get(t, 0) / n_runs, 1)
            for t in ["power", "telecom", "hospital", "subway", "water", "fuel"]
        }

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
    despite having no direct flood exposure.
    """
    print(f"\n{'=' * 75}")
    print("AMPLIFIER ANALYSIS (extreme_2080)")
    print(f"{'=' * 75}")

    depth_col = "flood_extreme_2080_depth_m"
    node_depth = dict(zip(nodes_gdf["node_id"], nodes_gdf[depth_col].fillna(0.0)))
    node_type = dict(zip(nodes_gdf["node_id"], nodes_gdf["infra_type"]))
    node_lat = dict(zip(nodes_gdf["node_id"], nodes_gdf["lat"]))
    node_lon = dict(zip(nodes_gdf["node_id"], nodes_gdf["lon"]))

    fail_count = Counter()
    for r in cascade_results:
        for nid in r["failed_nodes_t96"]:
            fail_count[nid] += 1
    n_runs = len(cascade_results)

    print("  Computing approximate betweenness centrality (k=500 samples)...")
    G = load_graph(str(GRAPH_IN))
    G_cascade = nx.DiGraph()
    G_cascade.add_nodes_from(G.nodes(data=True))
    for u, v, data in get_cascade_edges(G):
        G_cascade.add_edge(u, v, **data)
    bc = nx.betweenness_centrality(G_cascade, k=min(500, G_cascade.number_of_nodes()))

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
          f"(dry under extreme_2080 but fail via cascade in "
          f">{int(AMP_FREQ_THRESHOLD * 100)}% of runs)")

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
    lines.append("WEEK 7 CASCADE ANALYSIS — NYC CITYWIDE (6,231 nodes)")
    lines.append("DEP + GeoClaw scenarios + STOCHASTIC buffers (Weibull)")
    lines.append("=" * 75)
    lines.append("")

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

    lines.append("Per-type failures at t=96h (extreme_2080):")
    type_means = comparison["extreme_2080"]["type_failures_mean"]
    for t in ["power", "telecom", "hospital", "subway", "water", "fuel"]:
        lines.append(f"  {t:<10}: {type_means[t]:>6.1f}")
    lines.append("")

    lines.append("Per-borough failures at t=96h (extreme_2080):")
    boro_means = comparison["extreme_2080"]["borough_failures_mean"]
    for b in ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]:
        lines.append(f"  {b:<15}: {boro_means[b]:>6.1f}")
    lines.append("")

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

    lines.append("Known limitations:")
    lines.append("  1. DEP flood maps exclude storm surge per their own disclaimer.")
    lines.append("     Surge-exposed infrastructure (FDR corridor hospitals, SI shore) is")
    lines.append("     systematically under-represented in flood footprint.")
    lines.append("  2. Power and fuel redundancy treated as OR (either kills the dependent")
    lines.append("     node). Proper AND-gate semantics for fuel-as-backup: Week 7+.")
    lines.append("  3. NJ substations (91) and petroleum terminals (6) excluded from")
    lines.append("     failure sampling — they can receive cascade but don't originate it.")
    lines.append("  4. Betweenness centrality is k=500 approximation (not exact) for")
    lines.append("     compute tractability on 6.2k-node graph.")
    lines.append("  5. Buffer Weibull shape parameters are engineering estimates pending")
    lines.append("     empirical fitting from expanded Sandy 2012 dataset.")
    lines.append("")
    lines.append("=" * 75)

    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    print("=" * 75)
    print("Week 7 Multi-Scenario Cascade Runner (citywide, stochastic buffers)")
    print("=" * 75)

    if not NODES_IN.exists():
        print(f"ERROR: {NODES_IN} not found. Run flood_overlay_v3.py nyc first.")
        return 1
    if not GRAPH_IN.exists():
        print(f"ERROR: {GRAPH_IN} not found.")
        return 1

    SIM_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load buffer distribution config once
    buffer_config = load_buffer_config()
    print(f"Loaded buffer Weibull shapes for: "
          f"{list(buffer_config['defaults'].keys())}")

    # Load nodes once
    nodes_gdf = gpd.read_file(NODES_IN)
    print(f"Loaded {len(nodes_gdf):,} nodes from {NODES_IN}")

    # Week 19: Jesse intra-power coupling (config-gated, default OFF)
    power_coupling = load_power_coupling()

    all_results = {}
    for scenario in SCENARIOS:
        mc, cascade = run_scenario(scenario, nodes_gdf, buffer_config,
                                   power_coupling=power_coupling)
        all_results[scenario] = (mc, cascade)

    comparison = summarize(all_results, nodes_gdf)
    comparison_path = SIM_DIR / "nyc_scenario_comparison.json"
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nSaved: {comparison_path}")

    if "extreme_2080" in all_results:
        _, extreme_cascade = all_results["extreme_2080"]
        amplifiers = find_amplifiers_extreme(nodes_gdf, extreme_cascade)

        summary_text = format_summary(comparison, amplifiers)
        print("\n" + summary_text)
        summary_path = OUT_DIR / "week6_cascade_summary.txt"
        summary_path.write_text(summary_text)
        print(f"\nSaved: {summary_path}")
    else:
        print("\n[note] extreme_2080 not part of this run (SCENARIO_ONLY filter) - "
              "amplifier/summary block skipped; per-scenario results saved above.")

    for scenario in SCENARIOS:
        temp = SIM_DIR / f"temp_nodes_nyc_{scenario}.geojson"
        if temp.exists():
            temp.unlink()

    print("\nDone. Stochastic-buffer cascade results ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())