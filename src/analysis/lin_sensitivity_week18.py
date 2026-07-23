#!/usr/bin/env python3
"""
lin_sensitivity_week18.py — Dr. Lin sensitivity suite (Week 18).

Two experiments on the calibrated intra-telecom mechanism, replaying the
FIRST N production Monte Carlo draws (same seeds, same buffer RNG sequence
per config -> apples-to-apples), exactly the calibrate_telecom_crash.py
replay pattern:

  PART A — kappa bifurcation curve.
      Fine crash-factor grid through the kappa=1.5 crash-contagion cliff
      into the 2.5-4.0 plateau, on geoclaw_2026 (Sandy anchor) and
      extreme_2080 (rain check). Output: % of the 4,150-site layer dead
      and crash counts vs kappa, with the FCC Sandy band shaded and the
      chosen kappa=3.0 marked.

  PART B — MDRI roaming policy experiment.
      roaming_enabled off vs on (FCC "roaming under disaster", mandatory
      since May 2024) at the calibrated kappa=3.0, across extreme_2080 +
      the three GeoClaw scenarios. Metrics: % layer dead, overload
      crashes, and UNSERVED DEMAND share at the surge peak (t=6) from the
      closure's service ledger — deaths tell the infrastructure story,
      unserved demand tells the policy story.

Run from project root (CPU-only; safe alongside an MPS training job):
    python src/analysis/lin_sensitivity_week18.py
    N_RUNS=10 python src/analysis/lin_sensitivity_week18.py      # quick pass
    SKIP_KAPPA=1 python src/analysis/lin_sensitivity_week18.py   # MDRI only
    SKIP_MDRI=1  python src/analysis/lin_sensitivity_week18.py   # kappa only

Outputs:
    reports/figures/fig_kappa_bifurcation.png
    reports/figures/fig_mdri_roaming.png
    data/analysis/lin_sensitivity_week18.json   (all numbers, reproducible)

Expected runtime at N_RUNS=25: ~20-30 min total (Part A ~2/3 of it).
"""

import copy
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "simulation"))
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import yaml

from cascade_joint import build_intra_context, simulate_cascade_joint
from intra_telecom import surge_multiplier
from src.cascade.stochastic_buffer import load_buffer_config, sample_stochastic_graph

GRAPH_FLOOD = ROOT / "data/flood/nyc_infra_graph_all_flood.graphml"
GRAPH_BASE = ROOT / "data/graph/nyc_infra_graph.graphml"
NODES = ROOT / "data/graph/nyc_infra_nodes.geojson"
SIM = ROOT / "data/simulation"
CONFIG = ROOT / "config/intra_cascade.yaml"
FIG_DIR = ROOT / "reports/figures"
OUT_JSON = ROOT / "data/analysis/lin_sensitivity_week18.json"

N_RUNS = int(os.environ.get("N_RUNS", "25"))
BUFFER_SEED = 43                       # same as multi_scenario_runner

KAPPAS = [1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0]
KAPPA_SCENARIOS = ["geoclaw_2026", "extreme_2080"]
MDRI_SCENARIOS = ["extreme_2080", "geoclaw_2026", "geoclaw_2050", "geoclaw_2080"]
CHOSEN_KAPPA = 3.0
SANDY_BAND = (0.25, 0.35)

INDIGO, PURPLE, GRAY = "#4338CA", "#7C3AED", "#6B7280"


# --------------------------------------------------------------------------
# Shared replay machinery (calibrate_telecom_crash.py pattern)
# --------------------------------------------------------------------------

def load_static():
    """Phase-0 gates + static inputs shared by both parts."""
    graph_path = GRAPH_FLOOD if GRAPH_FLOOD.exists() else GRAPH_BASE
    if graph_path is GRAPH_BASE:
        print(f"WARNING: {GRAPH_FLOOD} not found — using base graph")
    if not NODES.exists():
        sys.exit(f"STOP: {NODES} not found")
    if not CONFIG.exists():
        sys.exit(f"STOP: {CONFIG} not found")
    G = nx.read_graphml(graph_path)
    with open(NODES) as f:
        types = {p["node_id"]: p["infra_type"]
                 for p in (feat["properties"]
                           for feat in json.load(f)["features"])}
    n_telecom = sum(1 for t in types.values() if t == "telecom")
    with open(CONFIG) as f:
        base_cfg = yaml.safe_load(f)
    buffer_cfg = load_buffer_config()
    print(f"[gate] graph={graph_path.name}  telecom layer={n_telecom} sites")
    return G, types, n_telecom, base_cfg, buffer_cfg


def load_draws(scenarios):
    out = {}
    for sc in scenarios:
        p = SIM / f"monte_carlo_failures_nyc_{sc}.json"
        if not p.exists():
            sys.exit(f"STOP: {p} not found — production MC files required.")
        with open(p) as f:
            runs = json.load(f)[:N_RUNS]
        out[sc] = [set(r.get("failed_nodes", r.get("initial_failures", [])))
                   for r in runs]
        print(f"[gate] {sc}: replaying {len(out[sc])} production MC draws")
    return out


def build_ctx(G, base_cfg, **telecom_overrides):
    cfg = copy.deepcopy(base_cfg)
    cfg["networks"]["telecom"].update(telecom_overrides)
    tmp = Path("/tmp/intra_cascade_lin_week18.yaml")
    with open(tmp, "w") as f:
        yaml.dump(cfg, f)
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        return build_intra_context(G, config_path=tmp)


def replay(G, ctx, draws, buffer_cfg, types, n_telecom):
    """Replay N draws through the joint engine under one telecom config.

    Returns per-config means:
      share      — telecom dead / layer size (t=96)
      crashes    — deaths with cause 'intra_telecom'
      inter      — telecom deaths minus seeds minus crashes
      seeds      — telecom seeds
      unserved6  — unserved demand share at the t=6 surge peak, from the
                   closure ledger (memo cleared per draw -> honest per-run)
    """
    ent = ctx.get("telecom")
    rng = np.random.default_rng(BUFFER_SEED)           # identical per config
    share_l, crash_l, inter_l, seed_l, uns_l = [], [], [], [], []
    for seeds in draws:
        if ent is not None:
            ent["memo"].clear()
            ent["unserved_log"].clear()
        Gs = sample_stochastic_graph(G, rng=rng, config=buffer_cfg)
        res, ft, cause = simulate_cascade_joint(Gs, seeds, ctx)
        dead = set(res["t96"])
        tel_dead = {n for n in dead if types.get(n) == "telecom"}
        tel_seed = {n for n in seeds if types.get(n) == "telecom"}
        n_crash = sum(1 for n, c in cause.items() if c == "intra_telecom")
        share_l.append(len(tel_dead) / n_telecom)
        crash_l.append(n_crash)
        seed_l.append(len(tel_seed))
        inter_l.append(len(tel_dead) - len(tel_seed) - n_crash)
        # service ledger: per m, the final fixed point = largest dead set
        u6 = 0.0
        if ent is not None and ent["unserved_log"]:
            m6 = round(surge_multiplier(6.0, ent["surge_peak"],
                                        ent["surge_tau_h"]), 6)
            best = None
            for (dk, m), v in ent["unserved_log"].items():
                if m == m6 and (best is None or len(dk) > best[0]):
                    best = (len(dk), v)
            if best is not None and best[1]["demand"] > 0:
                u6 = best[1]["unserved"] / best[1]["demand"]
        uns_l.append(u6)
    return {
        "share": mean(share_l), "crashes": mean(crash_l),
        "inter": mean(inter_l), "seeds": mean(seed_l),
        "unserved6": mean(uns_l), "n_runs": len(draws),
    }


# --------------------------------------------------------------------------
# Part A — kappa bifurcation
# --------------------------------------------------------------------------

def part_a(G, types, n_telecom, base_cfg, buffer_cfg):
    draws = load_draws(KAPPA_SCENARIOS)
    print(f"\n=== PART A — kappa bifurcation "
          f"({len(KAPPAS)} kappas x {len(KAPPA_SCENARIOS)} scenarios x "
          f"{N_RUNS} draws) ===")
    print(f"{'kappa':>6} {'scenario':<14} {'seeds':>6} {'inter':>7} "
          f"{'crash':>8} {'%layer':>7} {'uns@t6':>7}")
    results = {sc: {} for sc in KAPPA_SCENARIOS}
    for kappa in KAPPAS:
        ctx = build_ctx(G, base_cfg, crash_factor=kappa)
        for sc in KAPPA_SCENARIOS:
            t0 = time.time()
            r = replay(G, ctx, draws[sc], buffer_cfg, types, n_telecom)
            results[sc][kappa] = r
            print(f"{kappa:>6} {sc:<14} {r['seeds']:>6.1f} {r['inter']:>7.1f} "
                  f"{r['crashes']:>8.1f} {100*r['share']:>6.1f}% "
                  f"{100*r['unserved6']:>6.1f}%   [{time.time()-t0:.0f}s]")
    return results


def plot_a(results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    for sc, color, label in ((KAPPA_SCENARIOS[0], INDIGO, "geoclaw_2026 (surge)"),
                             (KAPPA_SCENARIOS[1], PURPLE, "extreme_2080 (rain)")):
        ks = sorted(results[sc])
        ax1.plot(ks, [100 * results[sc][k]["share"] for k in ks],
                 "o-", color=color, lw=2, label=label)
        ax2.plot(ks, [max(results[sc][k]["crashes"], 0.5) for k in ks],
                 "o-", color=color, lw=2, label=label)
    ax1.axhspan(100 * SANDY_BAND[0], 100 * SANDY_BAND[1], color="#FBBF24",
                alpha=0.20, label="FCC Sandy band (25\u201335%)")
    for ax in (ax1, ax2):
        ax.axvline(CHOSEN_KAPPA, color="#059669", ls="--", lw=1.5)
        ax.axvline(1.5, color="#DC2626", ls=":", lw=1.2)
        ax.set_xlabel("crash factor \u03ba  (site crashes at \u03ba \u00d7 capacity)")
        ax.grid(alpha=0.25)
    ax1.text(CHOSEN_KAPPA + 0.06, ax1.get_ylim()[1] * 0.93, "\u03ba = 3.0\n(chosen)",
             color="#059669", fontsize=9)
    ax1.text(1.5 + 0.06, ax1.get_ylim()[1] * 0.72, "\u03ba = 1.5\ncrash-contagion\nregime",
             color="#DC2626", fontsize=8.5)
    ax1.set_ylabel("% of telecom layer dead (t=96)")
    ax1.set_title("Telecom losses vs crash threshold", loc="left", fontweight="bold")
    ax1.legend(fontsize=8.5, loc="upper right")
    ax2.set_yscale("log")
    ax2.set_ylabel("overload crashes per run (log)")
    ax2.set_title("The bifurcation: crash counts", loc="left", fontweight="bold")
    ax2.legend(fontsize=8.5)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "fig_kappa_bifurcation.png", dpi=200)
    plt.close(fig)
    print(f"figure -> {FIG_DIR / 'fig_kappa_bifurcation.png'}")


# --------------------------------------------------------------------------
# Part B — MDRI roaming on/off
# --------------------------------------------------------------------------

def part_b(G, types, n_telecom, base_cfg, buffer_cfg):
    draws = load_draws(MDRI_SCENARIOS)
    print(f"\n=== PART B — MDRI roaming policy (kappa={CHOSEN_KAPPA}, "
          f"{len(MDRI_SCENARIOS)} scenarios x 2 settings x {N_RUNS} draws) ===")
    print(f"{'scenario':<14} {'roaming':<8} {'seeds':>6} {'inter':>7} "
          f"{'crash':>8} {'%layer':>7} {'uns@t6':>7}")
    results = {}
    for roaming in (False, True):
        ctx = build_ctx(G, base_cfg, crash_factor=CHOSEN_KAPPA,
                        roaming_enabled=roaming)
        for sc in MDRI_SCENARIOS:
            t0 = time.time()
            r = replay(G, ctx, draws[sc], buffer_cfg, types, n_telecom)
            results[(sc, roaming)] = r
            print(f"{sc:<14} {('ON' if roaming else 'off'):<8} "
                  f"{r['seeds']:>6.1f} {r['inter']:>7.1f} {r['crashes']:>8.1f} "
                  f"{100*r['share']:>6.1f}% {100*r['unserved6']:>6.1f}%   "
                  f"[{time.time()-t0:.0f}s]")
    return results


def plot_b(results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    x = np.arange(len(MDRI_SCENARIOS))
    w = 0.36
    off = [results[(sc, False)] for sc in MDRI_SCENARIOS]
    on = [results[(sc, True)] for sc in MDRI_SCENARIOS]
    ax1.bar(x - w / 2, [100 * r["share"] for r in off], w, color=INDIGO,
            label="single-carrier (roaming off)")
    ax1.bar(x + w / 2, [100 * r["share"] for r in on], w, color=PURPLE,
            label="MDRI roaming on")
    ax1.set_ylabel("% of telecom layer dead (t=96)")
    ax1.set_title("Sites lost", loc="left", fontweight="bold")
    ax2.bar(x - w / 2, [100 * r["unserved6"] for r in off], w, color=INDIGO,
            label="single-carrier (roaming off)")
    ax2.bar(x + w / 2, [100 * r["unserved6"] for r in on], w, color=PURPLE,
            label="MDRI roaming on")
    ax2.set_ylabel("unserved demand at surge peak t=6 (%)")
    ax2.set_title("Service denied", loc="left", fontweight="bold")
    for ax in (ax1, ax2):
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("_", "\n") for s in MDRI_SCENARIOS],
                           fontsize=8.5)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8.5)
    fig.suptitle(f"FCC MDRI roaming-under-disaster as a model switch "
                 f"(\u03ba={CHOSEN_KAPPA}, {N_RUNS} production draws)",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "fig_mdri_roaming.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"figure -> {FIG_DIR / 'fig_mdri_roaming.png'}")


# --------------------------------------------------------------------------

def main():
    G, types, n_telecom, base_cfg, buffer_cfg = load_static()
    payload = {"n_runs": N_RUNS, "buffer_seed": BUFFER_SEED,
               "chosen_kappa": CHOSEN_KAPPA}

    if not os.environ.get("SKIP_KAPPA"):
        res_a = part_a(G, types, n_telecom, base_cfg, buffer_cfg)
        plot_a(res_a)
        payload["kappa_sweep"] = {
            sc: {str(k): v for k, v in res_a[sc].items()} for sc in res_a}

    if not os.environ.get("SKIP_MDRI"):
        res_b = part_b(G, types, n_telecom, base_cfg, buffer_cfg)
        plot_b(res_b)
        payload["mdri"] = {
            f"{sc}|{'on' if r else 'off'}": v
            for (sc, r), v in res_b.items()}

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nresults -> {OUT_JSON}")


if __name__ == "__main__":
    main()
