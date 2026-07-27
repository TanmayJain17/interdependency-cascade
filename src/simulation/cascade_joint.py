#!/usr/bin/env python3
"""
cascade_joint.py — Option C joint inter+intra cascade engine (Week 16).

One loop, one shared dead-node set. Per MC run:
  t=0   : flood-fired seeds only (direct failures; no propagation — keeps the
          v2 direct/cascade separation and the GNN input-mask semantics).
  t>0   : joint fixed point at each timestep:
            (a) inter relaxation: dependency edges fire when the source's
                fail_time + Weibull-sampled buffer has elapsed (clock-advancing)
            (b) intra closure: instant within-network spread on the CURRENT
                shared dead set — subway Motter-Lai overload, water backflow,
                fuel starvation (config/intra_cascade.yaml)
          repeat (a)<->(b) until neither adds a node. Intra kills can trigger
          further inter kills at the same t — the coupling under study.

Scope: subway/water/fuel intra (real within-type topology). telecom intra
deferred (0 within-type edges); power intra pending Jesse's pandapower
parameters. Motter-Lai capacity = (1+alpha) * pristine load, computed ONCE
(capacity is a design property, not a disaster property); live load
redistributes per closure. Subway closures are memoized by frozen dead set —
deterministic given the dead set — for MC tractability.

Week 19: `scheduled_failures` hook — precomputed-mechanism failures injected
when their time arrives. Used by the power coupling: Jesse's sampled final
grid state lands at the t=6 floor with cause 'intra_power' (his library
already contains flood seeding + intra-power cascade, so covered nodes skip
HAZUS seeding upstream in the runner — replace, not union). Floor semantics:
a scheduled node already dead earlier keeps its earlier fail_time/cause, and
scheduled nodes remain reachable by inter/intra kills like any other node
(inter-layer edges INTO power stay live).

Output schema is a superset of cascade_sim.simulate_cascade: same per-timestep
lists, plus per-node fail_time and cause ('flood'|'inter'|'intra_overload'|
'intra_flow').
"""

from pathlib import Path

import networkx as nx
import yaml

from cascade_sim import get_cascade_edges
from intra_telecom import build_telecom_entry, telecom_closure
from intra_hospital import build_hospital_entry, hospital_closure

INTRA_CONFIG = Path("config/intra_cascade.yaml")


# --------------------------------------------------------------------------- #
# Context: built once per scenario (topology is fixed; buffers vary per run)
# --------------------------------------------------------------------------- #

def build_intra_context(G: nx.DiGraph, config_path=INTRA_CONFIG) -> dict:
    """Precompute per-network subgraphs, pristine loads, capacities, memo."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    ctx = {}
    for infra, net_cfg in cfg["networks"].items():
        keep = [n for n, d in G.nodes(data=True) if d.get("infra_type") == infra]
        H = G.subgraph(keep).copy()
        entry = {"mode": net_cfg["mode"], "nodes": set(H.nodes)}
        if net_cfg["mode"] == "undirected_overload":
            U = H.to_undirected()
            load0 = nx.betweenness_centrality(U, normalized=False)
            alpha = float(net_cfg["alpha"])
            entry.update(
                U=U,
                load0=load0,
                capacity={n: (1.0 + alpha) * load0[n] for n in U},
                alpha=alpha,
                memo={},   # frozenset(dead) -> frozenset(closure), shared across runs
            )
            print(f"  [intra:{infra}] {U.number_of_nodes()} nodes, "
                  f"{U.number_of_edges()} within-edges, Motter-Lai alpha={alpha} "
                  f"(pristine loads cached)")
        elif net_cfg["mode"] == "directed_flow":
            entry.update(H=H, direction=net_cfg["flow_propagation"])
            print(f"  [intra:{infra}] {H.number_of_nodes()} nodes, "
                  f"{H.number_of_edges()} within-edges, directed_flow "
                  f"({net_cfg['flow_propagation']})")
        elif net_cfg["mode"] == "capacity_redistribution":
            # telecom: derived Voronoi failover adjacency (no stored edges);
            # entry carries its own nodes/mode keys — replace the stub.
            entry = build_telecom_entry(G, net_cfg)
        elif net_cfg["mode"] == "patient_redistribution":
            # hospital: gravity transfers + diversion strain + collapse
            entry = build_hospital_entry(G, net_cfg)
        else:
            raise ValueError(f"unknown intra mode '{net_cfg['mode']}'")
        ctx[infra] = entry
    return ctx


# --------------------------------------------------------------------------- #
# Intra closures (instant, on the shared dead set)
# --------------------------------------------------------------------------- #

def _overload_closure(entry: dict, dead_local: frozenset) -> frozenset:
    """Motter-Lai fixed point with pristine capacities; memoized."""
    hit = entry["memo"].get(dead_local)
    if hit is not None:
        return hit
    U, load0, cap = entry["U"], entry["load0"], entry["capacity"]
    failed = set(dead_local)
    while True:
        alive = [n for n in U if n not in failed]
        live = U.subgraph(alive)
        load = nx.betweenness_centrality(live, normalized=False)
        newly = {n for n in alive if load0[n] > 0 and load.get(n, 0.0) > cap[n]}
        if not newly:
            out = frozenset(failed)
            entry["memo"][dead_local] = out
            return out
        failed |= newly


def _flow_closure(H: nx.DiGraph, dead_local: set, direction: str) -> set:
    """Directed starvation/backflow closure (BFS)."""
    failed = set(dead_local)
    frontier = set(dead_local)
    while frontier:
        nxt = set()
        for u in frontier:
            neigh = H.successors(u) if direction == "with_edges" else H.predecessors(u)
            nxt.update(v for v in neigh if v not in failed)
        failed |= nxt
        frontier = nxt
    return failed


def intra_closure(dead: set, ctx: dict, t: float = 0.0) -> dict:
    """Given the global dead set, return {node_id: cause} of NEW intra kills.

    `t` (hours since event) is needed only by time-dependent closures
    (telecom demand surge m(t)); flow/overload closures ignore it.
    """
    new = {}
    for infra, entry in ctx.items():
        dead_local = dead & entry["nodes"]
        if not dead_local:
            continue
        if entry["mode"] == "undirected_overload":
            closure = _overload_closure(entry, frozenset(dead_local))
            #cause = "intra_overload"
        elif entry["mode"] == "capacity_redistribution":
            closure = telecom_closure(entry, frozenset(dead_local), t)
        elif entry["mode"] == "patient_redistribution":
            closure = hospital_closure(entry, frozenset(dead_local))
        else:
            closure = _flow_closure(entry["H"], dead_local, entry["direction"])
            #cause = "intra_flow"
        cause = f"intra_{infra}"
        for n in closure:
            if n not in dead:
                new[n] = cause
    return new


# --------------------------------------------------------------------------- #
# Joint engine
# --------------------------------------------------------------------------- #

def simulate_cascade_joint(G: nx.DiGraph, initial_failures: set, intra_ctx: dict,
                           time_steps=None, scheduled_failures=None):
    """Joint inter+intra propagation on one shared dead-node set.

    scheduled_failures : optional {node_id: (hours, cause)} — failures from a
        precomputed mechanism, injected once their time arrives (before the
        inter/intra fixed point at each timestep). Power coupling passes
        Jesse's sampled dead set as {nid: (6.0, 'intra_power')}. A node
        already dead earlier keeps its earlier fail_time/cause (floor, not
        exclusive ownership); fail_time records the SCHEDULED hour, so
        downstream buffer arithmetic is exact even if the first evaluated
        timestep is later.

    Returns (results, fail_time, cause):
      results   : {"t0": [...], "t6": [...], ...} cumulative failed-node lists
      fail_time : {node_id: hours}   (0.0 for flood seeds)
      cause     : {node_id: 'flood'|'inter'|'intra_overload'|'intra_flow'}
    """
    if time_steps is None:
        time_steps = [0, 6, 24, 48, 96]
    if scheduled_failures is None:
        scheduled_failures = {}

    incoming = {}
    for u, v, data in get_cascade_edges(G):
        incoming.setdefault(v, []).append((u, float(data.get("buffer_hours", 0.0))))

    fail_time = {nid: 0.0 for nid in initial_failures if nid in G}
    cause = {nid: "flood" for nid in fail_time}

    results = {}
    if 0 in time_steps:
        results["t0"] = sorted(fail_time)

    for t in time_steps:
        if t == 0:
            continue
        # scheduled precomputed-mechanism failures land first at their hour
        for nid, (t_sched, c) in scheduled_failures.items():
            if t_sched <= t and nid not in fail_time and nid in G:
                fail_time[nid] = float(t_sched)
                cause[nid] = c
        while True:
            # (a) inter relaxation to fixed point at time t
            changed = True
            while changed:
                changed = False
                for node in G.nodes():
                    if node in fail_time:
                        continue
                    for src, buf in incoming.get(node, ()):
                        if src in fail_time and (fail_time[src] + buf) <= t:
                            fail_time[node] = float(t)
                            cause[node] = "inter"
                            changed = True
                            break
            # (b) intra closure on the shared dead set (t drives telecom m(t))
            new_intra = intra_closure(set(fail_time), intra_ctx, t=float(t))
            if not new_intra:
                break
            for nid, c in new_intra.items():
                fail_time[nid] = float(t)
                cause[nid] = c
            # loop: intra kills may enable further inter kills at this same t

        results[f"t{t}"] = sorted(fail_time)

    return results, fail_time, cause
