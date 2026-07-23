#!/usr/bin/env python3
"""
intra_telecom.py — Intra-telecom cascade: capacity-constrained load
redistribution on a per-carrier Voronoi-neighbor graph (Week 17).

Closes the telecom socket left open in Week 14 ("telecom is DEFERRED:
0 within-type edges"). Public data has no tower<->tower edges, so the
within-type topology is DERIVED, not stored: cell sites that are Voronoi
neighbors (computed via Delaunay triangulation, the Voronoi dual) are
failover partners — when a site dies, its users re-camp on adjacent
surviving sites of the SAME CARRIER. Precedent: Sun et al. 2023 (Reliability
Engineering & System Safety) model cellular resilience exactly this way
(Voronoi-based graph, adjacent-base-station compensation).

Mechanism (local load redistribution — Motter-Lai family, but load is user
demand, not betweenness, so redistribution is local/additive rather than a
global shortest-path recompute):

    load_i(t)     = L0_i * m(t) + inherited_i
    capacity_i    = L0_i / rho              (rho = busy-hour utilization;
                                             equivalently alpha = 1/rho - 1)
    crash_i       = crash_factor * capacity_i
    m(t)          = 1 + (surge_peak - 1) * exp(-t / surge_tau_h)

    CAPACITATED SPILL (the physics a naive Motter-Lai transplant misses):
    real sites run ADMISSION CONTROL — a saturated tower serves up to its
    capacity and REJECTS the excess; rejected phones retry on other cells
    reachable at the user's location (1-2 Voronoi tiers, `max_spill_hops`),
    they cannot chain across the city. Demand therefore FILLS capacity and
    spills, it does not pile up as node-killing load. Demand that finds no
    capacity within reach is UNSERVED — the 9/11 / Boston-Marathon mode:
    mass service denial with towers physically fine, which is also what
    Sandy's record shows (site losses were power/flood-driven; overload
    crashes rare). A site physically dies from overload only when its
    OFFERED load (served + rejected attempts) exceeds the crash threshold
    kappa * capacity (signaling-storm regime) — then its users lose signal
    and re-camp like any dead site's. Earlier single-threshold and
    block-accumulate variants were supercritical on this graph (12 seeds
    killed 97% / 93% of a carrier layer); capacity absorption plus the hop
    limit is what keeps cascades local, matching the empirical record.

    Per-closure diagnostics log unserved demand (spill that found no
    capacity + coverage holes) for Sami's zone-impact layer.

Consistency with Sami's downstream framework: his block-cut model uses a
surge factor (peak 2.0, tau 12.0h) and a cut-all threshold of 1/rho. Same
functional forms here, so the two-file handshake stays coherent. Note the
regime boundary: if m(t) > 1/rho the whole layer is congestion-saturated
independent of flooding (the 9/11 / Boston-Marathon jammed-network mode —
Sami's all-purple h=0 map). The flood-driven cascade regime needs
m(t) < 1/rho at evaluated timesteps; defaults are chosen inside it and the
boundary is part of the sensitivity story, not a bug.

Carrier segmentation: phones cannot camp on another operator's tower, so
each carrier is an independent layer (OpenCelliD MNC -> `operator` node
attr; "T-Mobile Metro" aliases to "T-Mobile"). `roaming_enabled: true`
merges all layers — the FCC Mandatory Disaster Response Initiative
("roaming under disaster", mandatory since May 2024) as a model switch.
Known data gap: the OpenCelliD pull used MCC=310 only; Verizon lives mostly
under MCC=311, so Verizon towers are absent (documented limitation; fix =
also ingest 311.csv.gz and rebuild nodes).

Engine contract (cascade_joint.py): build_telecom_entry() is called by
build_intra_context() for mode "capacity_redistribution"; telecom_closure()
is dispatched by intra_closure() with the current timestep t (needed for
m(t)). Closures are deterministic given (dead set, t) and memoized. The
joint engine is MONOTONE (dead stays dead); real congestion is transient,
so a site overloaded at t=6 would in reality recover as the surge decays —
the monotone engine can't express recovery, which makes telecom cascade
counts conservative (upper bound). Documented, swept, honest.
"""

import math
from collections import defaultdict

import numpy as np

try:
    from scipy.spatial import Delaunay
    _HAVE_SCIPY = True
except ImportError:          # pragma: no cover
    _HAVE_SCIPY = False

EARTH_R_M = 6_371_000.0


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #

def _project_local_m(lats, lons):
    """Equirectangular projection to meters around the local mean latitude.

    NYC spans <0.4 deg; distortion <0.3% — plenty for failover distances.
    Avoids a pyproj dependency inside the simulation package.
    """
    lat0 = math.radians(float(np.mean(lats)))
    x = np.radians(lons) * EARTH_R_M * math.cos(lat0)
    y = np.radians(lats) * EARTH_R_M
    return np.column_stack([x, y])


def _delaunay_edges(xy):
    """Voronoi-neighbor pairs via Delaunay triangulation (its dual).

    Falls back to 3-nearest-neighbor for degenerate layers (<4 points or
    collinear sites), which only matters for tiny carrier slices.
    """
    n = len(xy)
    edges = set()
    if _HAVE_SCIPY and n >= 4:
        try:
            tri = Delaunay(xy)
            for simplex in tri.simplices:
                for a in range(3):
                    for b in range(a + 1, 3):
                        i, j = int(simplex[a]), int(simplex[b])
                        edges.add((min(i, j), max(i, j)))
            return edges
        except Exception:
            pass  # QhullError on degenerate input -> kNN fallback
    k = min(3, n - 1)
    for i in range(n):
        d2 = np.sum((xy - xy[i]) ** 2, axis=1)
        for j in np.argsort(d2)[1:k + 1]:
            edges.add((min(i, int(j)), max(i, int(j))))
    return edges


# --------------------------------------------------------------------------- #
# Context builder (called once per scenario by build_intra_context)
# --------------------------------------------------------------------------- #

def build_telecom_entry(G, net_cfg):
    """Build the intra-telecom context entry from node attributes alone.

    Needs per-node: lat, lon, operator, tower_count (all present in the
    citywide graph since the Week-? telecom clustering). No graph edges are
    read or written — the failover adjacency is derived here.
    """
    rho = float(net_cfg.get("rho", 0.6))
    crash_factor = float(net_cfg.get("crash_factor", 3.0))
    surge_peak = float(net_cfg.get("surge_peak", 1.5))
    surge_tau = float(net_cfg.get("surge_tau_h", 12.0))
    max_km = float(net_cfg.get("max_reassign_km", 3.0))
    max_hops = int(net_cfg.get("max_spill_hops", 2))
    roaming = bool(net_cfg.get("roaming_enabled", False))
    demand_mode = net_cfg.get("demand", "tower_count")
    aliases = net_cfg.get("operator_aliases", {}) or {}

    nodes, lats, lons, L0, layer_of = [], [], [], {}, {}
    for nid, d in G.nodes(data=True):
        if d.get("infra_type") != "telecom":
            continue
        lat, lon = d.get("lat"), d.get("lon")
        if lat is None or lon is None:
            continue
        op = str(d.get("operator") or "UNKNOWN")
        op = aliases.get(op, op)
        nodes.append(nid)
        lats.append(float(lat))
        lons.append(float(lon))
        layer_of[nid] = "ALL" if roaming else op
        if demand_mode == "uniform":
            L0[nid] = 1.0
        else:  # tower_count: deployment density as revealed demand
            L0[nid] = max(float(d.get("tower_count") or 1.0), 1.0)

    xy_all = _project_local_m(np.array(lats), np.array(lons))
    pos = {nid: xy_all[i] for i, nid in enumerate(nodes)}

    # Per-carrier-layer Delaunay adjacency, pruned at max failover range.
    by_layer = defaultdict(list)
    for nid in nodes:
        by_layer[layer_of[nid]].append(nid)

    adj = {nid: [] for nid in nodes}      # nid -> [(neighbor, dist_m), ...]
    n_edges, n_pruned = 0, 0
    for layer, members in by_layer.items():
        if len(members) < 2:
            continue
        xy = np.array([pos[nid] for nid in members])
        for i, j in _delaunay_edges(xy):
            u, v = members[i], members[j]
            dist = float(np.linalg.norm(pos[u] - pos[v]))
            if dist > max_km * 1000.0:
                n_pruned += 1
                continue
            adj[u].append((v, dist))
            adj[v].append((u, dist))
            n_edges += 1

    capacity = {nid: L0[nid] / rho for nid in nodes}
    entry = {
        "mode": "capacity_redistribution",
        "nodes": set(nodes),
        "adj": adj,
        "L0": L0,
        "capacity": capacity,
        "crash": {nid: crash_factor * capacity[nid] for nid in nodes},
        "rho": rho,
        "crash_factor": crash_factor,
        "max_spill_hops": max_hops,
        "surge_peak": surge_peak,
        "surge_tau_h": surge_tau,
        "layer_of": layer_of,
        "memo": {},            # (frozenset(dead_local), m_rounded) -> frozenset
        "unserved_log": {},    # (dead_key, m) -> {"blocked":, "holes":} units
    }
    iso = sum(1 for nid in nodes if not adj[nid])
    layers = {k: len(v) for k, v in sorted(by_layer.items())}
    print(f"  [intra:telecom] {len(nodes)} sites, {n_edges} Voronoi failover "
          f"edges ({n_pruned} pruned >{max_km}km), {iso} isolated | "
          f"layers={layers} roaming={roaming} | rho={rho} "
          f"(alpha={1/rho - 1:.2f}), spill hops={max_hops}, crash at {crash_factor}x capacity, "
          f"surge peak={surge_peak} tau={surge_tau}h | mass-blocking if "
          f"m(t) > {1/rho:.2f}, mass-crash only if m(t) > "
          f"{crash_factor/rho:.2f}")
    return entry


# --------------------------------------------------------------------------- #
# Closure (called by intra_closure at each timestep, memoized)
# --------------------------------------------------------------------------- #

def surge_multiplier(t, peak, tau):
    """m(t) = 1 + (peak-1)*exp(-t/tau); demand surge decaying after the event."""
    return 1.0 + (peak - 1.0) * math.exp(-float(t) / tau)


def telecom_closure(entry, dead_local, t):
    """Fixed point of capacitated spill given dead sites at time t.

    Deterministic in (dead_local, m(t)); round semantics: all spills in a
    round are aggregated per receiver, THEN absorption and crash checks run,
    so within-round order does not matter.

    Bookkeeping per alive site v:
      served[v]  <= capacity[v]      (admission control cap)
      offered[v] = served + rejected attempts; crash if > kappa*capacity
    Spill packets (u, amount, hops): amount splits among u's surviving
    same-layer partners by inverse distance; each receiver absorbs up to
    remaining headroom; the rejected remainder re-spills from the receiver
    with hops-1, or becomes unserved at hops 0 (a phone can only reach
    towers whose signal covers it). Crashed sites re-spill their served
    load with a fresh hop budget (their users lose signal outright).
    Terminates: packet hops strictly decrease; crashes are monotone and
    bounded by the layer size.
    """
    m = surge_multiplier(t, entry["surge_peak"], entry["surge_tau_h"])
    key = (frozenset(dead_local), round(m, 6))
    hit = entry["memo"].get(key)
    if hit is not None:
        return hit

    adj, L0, cap, crash = entry["adj"], entry["L0"], entry["capacity"], entry["crash"]
    H = entry["max_spill_hops"]
    failed = set(dead_local)
    served, offered, unserved = {}, {}, 0.0

    # Own demand: admission control applies even to a site's own users when
    # m(t) pushes own load past capacity (Sami's mass-blocking regime) —
    # the excess spills like any rejected attempt.
    pending = []                # list of (node, amount, hops_left, prev)
    for nid in entry["nodes"]:
        own = L0[nid] * m
        if nid in failed:
            pending.append((nid, own, H, None))
            continue
        served[nid] = min(own, cap[nid])
        offered[nid] = own
        excess = own - served[nid]
        if excess > 0.0:
            pending.append((nid, excess, H, None))

    while pending:
        arrivals = defaultdict(float)  # (receiver, hops_left, sender) -> amt
        for u, amount, hops, prev in pending:
            partners = [(v, d) for v, d in adj[u]
                        if v not in failed and v != prev]
            if not partners:
                unserved += amount      # coverage hole — no failover
                continue
            wts = np.array([1.0 / max(d, 1.0) for _, d in partners])
            wts /= wts.sum()
            for (v, _), w in zip(partners, wts):
                arrivals[(v, hops, u)] += amount * float(w)

        pending = []
        crashed_now = set()
        for (v, hops, sender), amt in arrivals.items():
            if v in failed:             # crashed earlier this closure —
                if hops > 0:            # users retry further afield
                    pending.append((v, amt, hops - 1, sender))
                else:
                    unserved += amt
                continue
            offered[v] += amt
            headroom = cap[v] - served[v]
            absorbed = min(headroom, amt)
            served[v] += absorbed
            rejected = amt - absorbed
            if rejected > 0.0:
                if hops > 1:            # no straight backtrack (v != sender):
                    pending.append((v, rejected, hops - 1, sender))
                else:
                    unserved += rejected
            if offered[v] > crash[v]:
                crashed_now.add(v)
        for v in crashed_now:           # signaling-storm overload: site dies
            failed.add(v)
            pending.append((v, served.pop(v), H, None))
            offered.pop(v, None)

    out = frozenset(failed)
    # Memo is shared across MC runs (ctx built once per scenario); stochastic
    # buffers make dead sets diverse, so cap growth to bound memory.
    if len(entry["memo"]) < 50_000:
        entry["memo"][key] = out
        entry["unserved_log"][key] = {
            "unserved": unserved,
            "served": sum(served.values()),
            "demand": sum(L0[n] * m for n in entry["nodes"]),
        }
    return out
