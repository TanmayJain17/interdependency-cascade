#!/usr/bin/env python3
"""
calibrate_telecom_crash.py — Week 17 telecom crash-channel calibration.

Finding from the 100-run sanity pass: pre-existing inter-cascade (power
dependency) alone already puts telecom losses at ~23% of the layer for
geoclaw_2026 — right on the FCC's ~25% Sandy band — while the new intra
crash channel at crash_factor=2.0 adds ~1,100 more (49.5% total, above
Sandy's worst-county band) and is flat across GeoClaw severities
(saturated). The Sandy record says overload crashes are a minor addition
to power-driven deaths, so the target is: geoclaw_2026 telecom dead in
the 25-35% band, with crashes ~100-400, and rain (extreme_2080) well
below the regional 25%.

This script replays the FIRST N real Monte Carlo draws (same seeds, same
buffer RNG sequence per config -> apples-to-apples) through the joint
engine for each candidate telecom config and reports where each lands.

Run from project root (after the sanity run has produced the MC files):
    python src/analysis/calibrate_telecom_crash.py
    N_RUNS=50 python src/analysis/calibrate_telecom_crash.py   # optional

Expected runtime: ~5-10 min for 6 configs x 2 scenarios x 25 runs.
"""

import copy
import json
import os
import sys
import time
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "simulation"))
sys.path.insert(0, str(ROOT))

import networkx as nx
import numpy as np
import yaml

from cascade_joint import build_intra_context, simulate_cascade_joint
from src.cascade.stochastic_buffer import load_buffer_config, sample_stochastic_graph

GRAPH_FLOOD = ROOT / "data/flood/nyc_infra_graph_all_flood.graphml"
GRAPH_BASE = ROOT / "data/graph/nyc_infra_graph.graphml"
NODES = ROOT / "data/graph/nyc_infra_nodes.geojson"
SIM = ROOT / "data/simulation"
CONFIG = ROOT / "config/intra_cascade.yaml"

SCENARIOS = ["geoclaw_2026", "extreme_2080"]   # Sandy anchor + rain check
N_RUNS = int(os.environ.get("N_RUNS", "25"))
BUFFER_SEED = 43                               # same as multi_scenario_runner

# (crash_factor, surge_peak) candidates. cf sweep at the default surge,
# plus surge sensitivity at one promising cf.
CANDIDATES = [(2.0, 1.5),   # current default (the too-hot baseline)
              (2.5, 1.5),
              (3.0, 1.5),
              (4.0, 1.5),
              (3.0, 1.25),
              (3.0, 2.0)]   # Sami's surge peak

SANDY_BAND = (0.25, 0.35)   # target telecom-dead share for geoclaw_2026


def main():
    graph_path = GRAPH_FLOOD if GRAPH_FLOOD.exists() else GRAPH_BASE
    if graph_path is GRAPH_BASE:
        print(f"WARNING: {GRAPH_FLOOD} not found — falling back to base graph "
              f"(fine for mechanics, use flood graph for the real decision)")
    G = nx.read_graphml(graph_path)
    with open(NODES) as f:
        types = {p["node_id"]: p["infra_type"]
                 for p in (feat["properties"]
                           for feat in json.load(f)["features"])}
    n_telecom = sum(1 for t in types.values() if t == "telecom")

    with open(CONFIG) as f:
        base_cfg = yaml.safe_load(f)
    buffer_cfg = load_buffer_config()

    mc_by_scenario = {}
    for sc in SCENARIOS:
        p = SIM / f"monte_carlo_failures_nyc_{sc}.json"
        if not p.exists():
            sys.exit(f"STOP: {p} not found — run the sanity pass first.")
        with open(p) as f:
            runs = json.load(f)[:N_RUNS]
        mc_by_scenario[sc] = [
            set(r.get("failed_nodes", r.get("initial_failures", [])))
            for r in runs]
        print(f"[gate] {sc}: replaying {len(mc_by_scenario[sc])} MC draws")

    print(f"\n{'config':<22} {'scenario':<14} {'tel_seed':>8} {'tel_inter':>9} "
          f"{'tel_crash':>9} {'tel_dead':>8} {'%layer':>7} {'total':>7} "
          f"{'A':>6}  verdict")
    print("-" * 110)

    for cf, peak in CANDIDATES:
        cfg = copy.deepcopy(base_cfg)
        cfg["networks"]["telecom"]["crash_factor"] = cf
        cfg["networks"]["telecom"]["surge_peak"] = peak
        tmp = Path("/tmp/intra_cascade_calib.yaml")
        with open(tmp, "w") as f:
            yaml.dump(cfg, f)
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            ctx = build_intra_context(G, config_path=tmp)

        for sc in SCENARIOS:
            rng = np.random.default_rng(BUFFER_SEED)   # identical per config
            t0 = time.time()
            seeds_l, dead_l, crash_l, inter_l, tseed_l, tdead_l = \
                [], [], [], [], [], []
            for seeds in mc_by_scenario[sc]:
                Gs = sample_stochastic_graph(G, rng=rng, config=buffer_cfg)
                res, ft, cause = simulate_cascade_joint(Gs, seeds, ctx)
                dead = set(res["t96"])
                tel_dead = {n for n in dead if types.get(n) == "telecom"}
                tel_seed = {n for n in seeds if types.get(n) == "telecom"}
                n_crash = sum(1 for n, c in cause.items()
                              if c == "intra_telecom")
                seeds_l.append(len(seeds)); dead_l.append(len(dead))
                crash_l.append(n_crash)
                tseed_l.append(len(tel_seed)); tdead_l.append(len(tel_dead))
                inter_l.append(len(tel_dead) - len(tel_seed) - n_crash)
            share = mean(tdead_l) / n_telecom
            A = mean(dead_l) / mean(seeds_l)
            if sc == "geoclaw_2026":
                verdict = ("IN SANDY BAND" if SANDY_BAND[0] <= share <= SANDY_BAND[1]
                           else ("too hot" if share > SANDY_BAND[1] else "too cold"))
            else:
                verdict = "ok (< regional 25%)" if share < 0.25 else "too hot for rain"
            print(f"cf={cf:<4} peak={peak:<6} {sc:<14} {mean(tseed_l):>8.1f} "
                  f"{mean(inter_l):>9.1f} {mean(crash_l):>9.1f} "
                  f"{mean(tdead_l):>8.1f} {100*share:>6.1f}% {mean(dead_l):>7.1f} "
                  f"{A:>6.2f}  {verdict}   [{time.time()-t0:.0f}s]")

    print(f"\nPick the config whose geoclaw_2026 row is IN SANDY BAND "
          f"({int(100*SANDY_BAND[0])}-{int(100*SANDY_BAND[1])}% of layer, "
          f"crashes a minor addition to inter) AND whose extreme_2080 row "
          f"stays below the regional 25%. Update config/intra_cascade.yaml, "
          f"then re-run the N_MC=100 sanity pass to confirm before the full "
          f"1,000-run rerun.")


if __name__ == "__main__":
    main()
