"""
intra_cascade.py - within-network cascade (Week-14 Step 3, Tier-B).

Models failure spread WITHIN a single infrastructure (distinct from the
cross-type inter-cascade in cascade_sim.py). Two modes, one generic solver,
run only on the networks that have REAL within-type topology in the graph:

  subway -> undirected_overload : Motter-Lai. load = shortest-path betweenness;
            capacity = (1+alpha)*initial_load; remove failed, recompute load on
            the live graph, fail anything now over capacity, iterate. The load
            REDISTRIBUTES (that's why betweenness, not a static attr).
  water  -> directed_flow (against_edges): water_flow is pump->plant; a failed
            plant/trunk (sink) backs up into the pumps feeding it.
  fuel   -> directed_flow (with_edges): fuel_distribution is terminal->station;
            a failed terminal starves the stations downstream.

telecom is DEFERRED (0 within-type edges; see docs/telecom_intra_design_note.md).
power is Jesse's (pandapower). Output schema matches Jesse's per-network
dead-node list so Step 5 can align: [infra_type, t_step, node_id, cause, alpha].

The driver loops over depth(node,t) timesteps - one step (t=0) this week, the
full 6 h range next week with no code change.
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import pandas as pd
import yaml

from src.flood.depth_interface import REPO_ROOT, get_depth_ts, load_config as load_flood_config
from src.simulation.fragility import failure_probability

DEFAULT_CONFIG = REPO_ROOT / "config" / "intra_cascade.yaml"
OUT_DIR = REPO_ROOT / "outputs" / "cascade"
DEAD_COLUMNS = ["infra_type", "t_step", "node_id", "cause", "alpha"]


def load_intra_config(path: str | Path | None = None) -> dict:
    with open(Path(path) if path else DEFAULT_CONFIG) as f:
        return yaml.safe_load(f)


def load_graph(cfg: dict) -> nx.DiGraph:
    G = nx.read_graphml(REPO_ROOT / cfg["graph_path"])
    return G if G.is_directed() else nx.DiGraph(G)


def extract_subgraph(G: nx.DiGraph, infra_type: str) -> nx.DiGraph:
    keep = [n for n, d in G.nodes(data=True) if d.get("infra_type") == infra_type]
    return G.subgraph(keep).copy()


def initial_failures(G: nx.DiGraph, infra_type: str, depth_at: dict, threshold: float) -> set:
    """Nodes of infra_type whose fragility fires at this step's flood depth."""
    fired = set()
    for n, d in G.nodes(data=True):
        if d.get("infra_type") != infra_type:
            continue
        if failure_probability(depth_at.get(n, 0.0), infra_type) >= threshold:
            fired.add(n)
    return fired


# --------------------------------------------------------------------------- #
# generic solver - two modes
# --------------------------------------------------------------------------- #
def intra_cascade_overload(H: nx.Graph, initial_failed: set, alpha: float) -> set:
    """Motter-Lai load-redistribution overload cascade on an (undirected) graph.

    load = raw shortest-path betweenness on the LIVE graph (redistributes when
    nodes are removed). capacity = (1+alpha)*initial_load. Only flow-bearing
    nodes (initial load > 0) can overload, so leaves don't spuriously fail.
    """
    U = H.to_undirected()
    load0 = nx.betweenness_centrality(U, normalized=False)
    capacity = {n: (1.0 + alpha) * load0[n] for n in U}
    failed = set(initial_failed)
    while True:
        alive = [n for n in U if n not in failed]
        live = U.subgraph(alive)
        load = nx.betweenness_centrality(live, normalized=False)
        newly = {n for n in alive if load0[n] > 0 and load.get(n, 0.0) > capacity[n]}
        if not newly:
            return failed
        failed |= newly


def intra_cascade_flow(H: nx.DiGraph, initial_failed: set, propagation: str) -> set:
    """Directed-flow propagation (no capacity test): a failed node drags down
    its dependents. 'with_edges' = successors fail (terminal->station);
    'against_edges' = predecessors fail (plant<-pump backflow)."""
    failed = set(initial_failed)
    frontier = set(initial_failed)
    while frontier:
        nxt = set()
        for u in frontier:
            neigh = H.successors(u) if propagation == "with_edges" else H.predecessors(u)
            nxt.update(v for v in neigh if v not in failed)
        failed |= nxt
        frontier = nxt
    return failed


def run_network(H: nx.DiGraph, initial: set, net_cfg: dict, alpha: float | None = None) -> set:
    mode = net_cfg["mode"]
    if mode == "undirected_overload":
        a = net_cfg["alpha"] if alpha is None else alpha
        return intra_cascade_overload(H, initial, a)
    if mode == "directed_flow":
        return intra_cascade_flow(H, initial, net_cfg["flow_propagation"])
    raise ValueError(f"unknown intra mode '{mode}'")


# --------------------------------------------------------------------------- #
# driver: loop timesteps x networks
# --------------------------------------------------------------------------- #
def run_all(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_intra_config()
    flood_cfg = load_flood_config()
    G = load_graph(cfg)
    depth_ts = get_depth_ts(flood_cfg)
    threshold = float(cfg["fragility_fire_threshold"])

    rows = []
    for t_step, grp in depth_ts.groupby("t_step"):
        depth_at = dict(zip(grp["node_id"].astype(str), grp["depth_m"].astype(float)))
        for infra, net_cfg in cfg["networks"].items():
            H = extract_subgraph(G, infra)
            initial = initial_failures(G, infra, depth_at, threshold) & set(H.nodes)
            a = net_cfg.get("alpha")
            failed = run_network(H, initial, net_cfg, a)
            cause_intra = "intra_overload" if net_cfg["mode"] == "undirected_overload" else "intra_flow"
            for n in failed:
                rows.append((infra, int(t_step), n,
                             "initial_flood" if n in initial else cause_intra, a))
    return pd.DataFrame(rows, columns=DEAD_COLUMNS)


# --------------------------------------------------------------------------- #
# validation: alpha sweep (the Dr. Lin plot) + direct-vs-intra split
# --------------------------------------------------------------------------- #
def _validate_and_save(cfg: dict, out_dir: Path) -> None:
    flood_cfg = load_flood_config()
    G = load_graph(cfg)
    depth_ts = get_depth_ts(flood_cfg)
    threshold = float(cfg["fragility_fire_threshold"])
    t0 = depth_ts[depth_ts["t_step"] == 0]
    depth_at = dict(zip(t0["node_id"].astype(str), t0["depth_m"].astype(float)))

    print("=" * 72)
    print("intra-cascade (Step 3, Tier-B)  |  t_step=0  |  real-topology networks")
    print("=" * 72)

    sweep_rows = []
    for infra, net_cfg in cfg["networks"].items():
        H = extract_subgraph(G, infra)
        U = H.to_undirected()
        ncomp = nx.number_connected_components(U)
        biggest = max((len(c) for c in nx.connected_components(U)), default=0)
        initial = initial_failures(G, infra, depth_at, threshold) & set(H.nodes)
        print(f"\n[{infra}] {H.number_of_nodes()} nodes, {H.number_of_edges()} within-edges, "
              f"{ncomp} components (largest {biggest}); initial flood-fired={len(initial)}")
        if net_cfg["mode"] == "undirected_overload":
            print("  Motter-Lai alpha sweep (smaller slack -> bigger cascade):")
            prev = None
            for a in cfg["alpha_sweep"]:
                failed = intra_cascade_overload(H, initial, a)
                intra = len(failed) - len(initial)
                sweep_rows.append((infra, a, len(initial), len(failed), intra))
                mono = "" if prev is None else ("  monotone-ok" if len(failed) <= prev else "  !! NON-MONOTONE")
                print(f"    alpha={a:>4}: total dead={len(failed):>3}  (initial {len(initial)} + "
                      f"intra-overload {intra}){mono}")
                prev = len(failed)
        else:
            failed = intra_cascade_flow(H, initial, net_cfg["flow_propagation"])
            intra = len(failed) - len(initial)
            print(f"  directed_flow ({net_cfg['flow_propagation']}): total dead={len(failed)}  "
                  f"(initial {len(initial)} + intra-flow {intra})  [real but fragmented topology]")

    dead = run_all(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "intra_dead_nodes.csv"
    dead.to_csv(out_path, index=False)
    sweep_path = out_dir / "intra_alpha_sweep.csv"
    pd.DataFrame(sweep_rows,
                 columns=["infra_type", "alpha", "n_initial", "n_total_dead", "n_intra"]
                 ).to_csv(sweep_path, index=False)
    print("\n" + "-" * 72)
    by = dead.groupby(["infra_type", "cause"]).size().unstack(fill_value=0)
    print("dead nodes by network x cause (default alpha):")
    print(by.to_string())
    print(f"saved: {out_path}  ({len(dead)} rows)")
    print(f"saved: {sweep_path}  (alpha-sweep evidence, {len(sweep_rows)} rows)")
    print("=" * 72)


if __name__ == "__main__":
    _validate_and_save(load_intra_config(), OUT_DIR)
