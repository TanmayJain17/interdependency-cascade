#!/usr/bin/env python3
"""
intra_hospital.py — Intra-hospital cascade: gravity patient redistribution
with system-affiliation routing, diversion strain, and surge collapse
(Week 17, closes the second intra socket after telecom).

Empirical basis (Hurricane Sandy, NYC): Lee et al. 2015 (Disaster Med
Public Health Prep 9:256-264) found proximity strongly predicted where
displaced patients went, but public-hospital (Bellevue) patients flowed to
other PUBLIC hospitals (Metropolitan, Woodhull) at rates above what
distance alone predicts — so routing = gravity (beds x distance decay)
times an affiliation bonus when origin and receiver share a system.
Receiving-side strain is real and bounded: Beth Israel, the only open
lower-Manhattan hospital, ran ~115% of BASELINE inpatient census (750 vs ~650; i.e. ~0.88x
certified beds) with ED +20% / ambulance +64% (Tran et al. 2014) —
massive strain, kept accepting, no collapse.

Mechanism (two thresholds, mirroring the telecom block/crash split):

    census_i(0)   = baseline_occupancy * beds_i
    divert_cap_i  = divert_frac   * beds_i   (~0.95: stops ACCEPTING)
    collapse_cap_i= collapse_frac * beds_i   (~1.20: surge capacity blown)

  A dead hospital's census must relocate. Flow from origin j to open,
  non-diverting hospital k within max_transfer_km:

      w_jk  ∝  beds_k * exp(-d_jk / distance_decay_km) * bonus_jk
      bonus_jk = affiliation_bonus if system(j) == system(k) else 1

  Receivers absorb up to divert_cap; refusals re-route from the refusing
  hospital (patients keep their ORIGIN system for affinity). A hospital
  hitting divert_cap goes on DIVERSION — strain, logged, NOT death.
  EMTALA fallback: when no non-diverting hospital is reachable, diversion
  is suspended (real NYC EMS policy when too many EDs divert) and load
  force-distributes across diverting-but-alive hospitals — the only path
  past divert_cap. Census beyond collapse_cap = functional COLLAPSE: the
  hospital enters the shared dead set (cause 'intra_hospital') and its
  whole census re-pends. Patients placeable nowhere in range are UNPLACED
  (boarded / field hospital / out-of-city), logged for reporting.

  With capped forced absorption, patient volume alone cannot collapse a
  receiving hospital — matching the record (Sandy and COVID: zero
  receiving hospitals functionally failed from volume; overwhelmed
  systems produce crisis-standards care and out-of-system overflow, not
  building-death cascades). The mechanism's outputs are therefore STRAIN
  (census ratios up to surge capacity), DIVERSION counts, and UNPLACED
  demand — not dead nodes. An uncapped EMTALA variant was rejected after
  producing 34 collapse deaths/run (58% of the layer) in a rain scenario;
  same lesson as the telecom supercriticality, hospital flavor.

Data:
  beds        data/healthcare/hospital_beds.csv (NYS DOH HFIS certified
              beds via join_hospital_beds.py; official, never guessed).
              Nodes with NULL beds are EXCLUDED from the mechanism
              (loudly) until resolved via MANUAL_OVERRIDES.
  affiliation data/healthcare/hospitals_nyc.geojson OPNAME (NYC FacDB,
              official operator names), normalized to systems by the
              OPNAME_SYSTEM_RULES below; unrecognized operators become
              singleton systems (no bonus with anyone — the safe default).

Engine contract: mode "patient_redistribution" in cascade_joint. Closure
is deterministic in the dead set (no time dependence in v1 — disaster ED
demand surge is future work), memoized, monotone-engine caveats as for
telecom.
"""

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

EARTH_R_M = 6_371_000.0

# OPNAME (FacDB official operator) -> hospital system, by substring match
# (case-insensitive, first hit wins). Public knowledge; unrecognized
# OPNAMEs become singleton systems and get no affiliation bonus.
OPNAME_SYSTEM_RULES = [
    ("nyc health and hospitals", "NYC Health + Hospitals"),
    ("new york and presbyterian", "NewYork-Presbyterian"),
    ("newyork-presbyterian", "NewYork-Presbyterian"),
    ("montefiore", "Montefiore"),
    ("nyu langone", "NYU Langone"),
    ("mount sinai", "Mount Sinai"),
    ("beth israel medical center", "Mount Sinai"),
    ("st lukes roosevelt", "Mount Sinai"),
    ("ny eye and ear", "Mount Sinai"),
    ("lenox hill", "Northwell"),
    ("staten island university", "Northwell"),
    ("long island jewish", "Northwell"),
    ("northwell", "Northwell"),
    ("brookdale", "One Brooklyn Health"),
    ("interfaith", "One Brooklyn Health"),
    ("kingsbrook", "One Brooklyn Health"),
    ("jamaica hospital", "MediSys"),
    ("flushing hospital", "MediSys"),
]


def _system_of(opname):
    low = (opname or "").lower()
    for needle, system in OPNAME_SYSTEM_RULES:
        if needle in low:
            return system
    return opname or "UNKNOWN"        # singleton system


def _dist_m(lat1, lon1, lat2, lon2):
    lat0 = math.radians((lat1 + lat2) / 2.0)
    dx = math.radians(lon2 - lon1) * EARTH_R_M * math.cos(lat0)
    dy = math.radians(lat2 - lat1) * EARTH_R_M
    return math.hypot(dx, dy)


# --------------------------------------------------------------------------- #
# Context builder
# --------------------------------------------------------------------------- #

def build_hospital_entry(G, net_cfg, root=None):
    root = Path(root) if root else Path(".")
    beds_csv = root / net_cfg.get("beds_csv", "data/healthcare/hospital_beds.csv")
    facdb = root / net_cfg.get("facdb_geojson",
                               "data/healthcare/hospitals_nyc.geojson")
    occ = float(net_cfg.get("baseline_occupancy", 0.75))
    divert_frac = float(net_cfg.get("divert_frac", 0.95))
    collapse_frac = float(net_cfg.get("collapse_frac", 1.20))
    d0_km = float(net_cfg.get("distance_decay_km", 5.0))
    bonus = float(net_cfg.get("affiliation_bonus", 2.0))
    max_km = float(net_cfg.get("max_transfer_km", 40.0))

    import csv
    if not beds_csv.exists():
        raise FileNotFoundError(
            f"{beds_csv} missing — run src/data_acquisition/"
            f"join_hospital_beds.py first (official NYS DOH beds required; "
            f"this module never guesses capacity).")
    beds = {}
    with open(beds_csv, newline="") as f:
        for r in csv.DictReader(f):
            v = r.get("beds_total", "").strip()
            beds[r["node_id"]] = float(v) if v else None

    op_by_name = {}
    if facdb.exists():
        for feat in json.load(open(facdb))["features"]:
            p = feat["properties"]
            op_by_name[p["FACNAME"].strip().lower()] = p.get("OPNAME")

    nodes, pos, system, excluded, zero_bed = [], {}, {}, [], []
    for nid, d in G.nodes(data=True):
        if d.get("infra_type") != "hospital":
            continue
        b = beds.get(nid)
        if b is None:
            excluded.append(nid)
            continue
        if b <= 0:
            zero_bed.append(nid)   # official 0 certified beds: no inpatient
            continue               # census to displace, no receiving capacity
        nodes.append(nid)
        pos[nid] = (float(d["lat"]), float(d["lon"]))
        system[nid] = _system_of(op_by_name.get(str(d.get("name", "")).strip().lower()))

    dist = {}
    for i, a in enumerate(nodes):
        for b_ in nodes[i + 1:]:
            m = _dist_m(*pos[a], *pos[b_])
            if m <= max_km * 1000.0:
                dist[(a, b_)] = dist[(b_, a)] = m

    entry = {
        "mode": "patient_redistribution",
        "nodes": set(nodes),
        "beds": {n: beds[n] for n in nodes},
        "census0": {n: occ * beds[n] for n in nodes},
        "divert_cap": {n: divert_frac * beds[n] for n in nodes},
        "collapse_cap": {n: collapse_frac * beds[n] for n in nodes},
        "system": system,
        "dist": dist,
        "d0_m": d0_km * 1000.0,
        "bonus": bonus,
        "excluded": excluded,
        "zero_bed": zero_bed,
        "memo": {},
        "strain_log": {},
    }
    from collections import Counter
    sys_counts = Counter(system.values())
    multi = {s: c for s, c in sys_counts.items() if c > 1}
    print(f"  [intra:hospital] {len(nodes)} hospitals, "
          f"{sum(beds[n] for n in nodes):.0f} certified beds (NYS DOH), "
          f"occupancy={occ}, divert@{divert_frac}, collapse@{collapse_frac} | "
          f"gravity d0={d0_km}km bonus={bonus} | "
          f"multi-facility systems: {multi}")
    if excluded:
        print(f"  [intra:hospital] EXCLUDED (beds NULL — resolve via "
              f"MANUAL_OVERRIDES in join_hospital_beds.py): {excluded}")
    if zero_bed:
        print(f"  [intra:hospital] zero certified beds (official; no "
              f"inpatient role in redistribution): {zero_bed}")
    return entry


# --------------------------------------------------------------------------- #
# Closure
# --------------------------------------------------------------------------- #

def hospital_closure(entry, dead_local):
    """Fixed point of patient redistribution given dead hospitals.

    Deterministic in the dead set; round semantics (aggregate arrivals,
    then apply, then collapse-check). Diversion set grows monotonically ->
    guaranteed termination; max_rounds is a safety net only.
    """
    key = frozenset(dead_local)
    hit = entry["memo"].get(key)
    if hit is not None:
        return hit

    nodes, beds = entry["nodes"], entry["beds"]
    census0, divert_cap = entry["census0"], entry["divert_cap"]
    collapse_cap, system = entry["collapse_cap"], entry["system"]
    dist, d0, bonus = entry["dist"], entry["d0_m"], entry["bonus"]

    failed = set(dead_local) & nodes
    census = {n: census0[n] for n in nodes if n not in failed}
    diverting = set()
    unplaced = 0.0
    # packets: (routing_location, origin_system, amount)
    pending = [(j, system[j], census0[j]) for j in failed]

    def weights(src, origin_sys, pool):
        out = []
        for k in pool:
            d = dist.get((src, k))
            if d is None:
                continue
            w = beds[k] * math.exp(-d / d0)
            if system[k] == origin_sys:
                w *= bonus
            out.append((k, w))
        return out

    rounds = 0
    while pending and rounds < 200:
        rounds += 1
        normal = defaultdict(float)    # (receiver, origin_sys) -> amount
        forced = defaultdict(float)    # EMTALA flow, absorbs past divert_cap
        for src, osys, amount in pending:
            open_pool = [k for k in census if k not in diverting]
            cands = weights(src, osys, open_pool)
            if not cands:              # everyone reachable is diverting:
                cands = weights(src, osys, list(census))   # EMTALA suspend
                if not cands:
                    unplaced += amount                     # nothing alive
                    continue
                tot = sum(w for _, w in cands)
                for k, w in cands:
                    forced[(k, osys)] += amount * w / tot
                continue
            tot = sum(w for _, w in cands)
            for k, w in cands:
                normal[(k, osys)] += amount * w / tot

        pending = []
        for (k, osys), amt in normal.items():
            if k not in census:
                pending.append((k, osys, amt))     # collapsed this closure
                continue
            room = divert_cap[k] - census[k]
            absorbed = max(min(room, amt), 0.0)
            census[k] += absorbed
            if census[k] >= divert_cap[k] - 1e-9:
                diverting.add(k)
            if amt - absorbed > 1e-9:
                pending.append((k, osys, amt - absorbed))  # refused, re-route
        for (k, osys), amt in forced.items():
            if k not in census:
                pending.append((k, osys, amt))
                continue
            # diversion suspended, but forced absorption caps at SURGE
            # capacity; beyond that, patients go out of system (unplaced:
            # out-of-city transfer / field hospital / DMAT). Sandy & COVID
            # record: zero receiving hospitals collapsed from volume.
            room = collapse_cap[k] - census[k]
            absorbed = max(min(room, amt), 0.0)
            census[k] += absorbed
            diverting.add(k)
            if amt - absorbed > 1e-9:
                unplaced += amt - absorbed
        # collapse check: defensive invariant — with capped forced
        # absorption, patient volume alone can no longer exceed surge
        # capacity, so this should never fire (kept as a guard for
        # future mechanisms, e.g. staff loss).
        for k in [k for k, c in census.items() if c > collapse_cap[k]]:
            failed.add(k)
            pending.append((k, system[k], census.pop(k)))
            diverting.discard(k)

    out = frozenset(failed)
    if len(entry["memo"]) < 50_000:
        entry["memo"][key] = out
    if len(entry["strain_log"]) < 5_000:
        entry["strain_log"][key] = {
            "n_diverting": len(diverting),
            "unplaced": round(unplaced, 1),
            "census_ratio": {n: round(census[n] / beds[n], 3)
                             for n in census},
        }
    return out
