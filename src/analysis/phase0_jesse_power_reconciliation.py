#!/usr/bin/env python3
"""
phase0_jesse_power_reconciliation.py — Week 19, Phase 0 (READ-ONLY diagnostic)

Reconciles Jesse's intra-power Monte Carlo library (data/external/jesse_power)
against our World-1 power layer, BEFORE any integration touches production.

What it answers:
  1. Can our power nodes be joined to Jesse's OSI substations (name tier,
     spatial tier), and do any of our node names match his WARD bus names
     directly (shared-ancestry diagnostic — UNKNOWN124169 appears on both sides)?
  2. Per gc scenario: our per-node P(seed@t0), P(dead@t6), P(dead@t96) vs
     Jesse's pct_failed — the discrepancy table for Miura/Lin.
  3. DEP adjudication: do OUR DEP scenarios produce power seeds where Jesse's
     HAZUS translation found zero?

Integration design this diagnostic supports (locked Week 19, do not relitigate
in code review):
  * intra_power.py REPLACES flood seeding + intra-power cascade for matched
    nodes (replace-not-union: Jesse's final state already contains the flood).
  * Whole-run sampling, index-paired (our run i <-> Jesse run i mod 1000).
    Valid because the hazard field is deterministic per scenario and fragility
    draws are conditionally independent -> the joint factorizes exactly.
  * Jesse's dead set is a t=6 FLOOR; inter-layer edges INTO power stay live.

Usage (project root, conda flood):
    python src/analysis/phase0_jesse_power_reconciliation.py
Env overrides:
    JESSE_DIR (default data/external/jesse_power)
    NODES_GEOJSON (default data/graph/nyc_infra_nodes.geojson)
    SIM_DIR (default data/simulation)
    OUT_DIR (default data/analysis)
Writes: OUT_DIR/phase0_jesse_power_reconciliation.json  (+ printed report)
"""

import csv
import json
import math
import os
import re
import sys
from collections import defaultdict

JESSE_DIR = os.environ.get("JESSE_DIR", "data/external/jesse_power")
NODES_GEOJSON = os.environ.get("NODES_GEOJSON", "data/graph/nyc_infra_nodes.geojson")
SIM_DIR = os.environ.get("SIM_DIR", "data/simulation")
OUT_DIR = os.environ.get("OUT_DIR", "data/analysis")

GC_SCENARIOS = {"gc_2026": "geoclaw_2026", "gc_2050": "geoclaw_2050", "gc_2080": "geoclaw_2080"}
DEP_SCENARIOS = {"dep_extreme_2080": "extreme_2080",
                 "dep_moderate_2050": "moderate_2050",
                 "dep_moderate_current": "moderate_current"}
SPATIAL_THRESH_M = 500.0  # nearest-OSI fallback acceptance radius

FAIL = []


def gate(ok, label, detail=""):
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {label}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAIL.append(label)
    return ok


def slug(s):
    s = re.sub(r"[^a-z0-9]+", "_", str(s).lower())
    return s.strip("_")


def haversine_m(lat1, lon1, lat2, lon2):
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


# --------------------------------------------------------------------------
# G1 — Jesse inventory
# --------------------------------------------------------------------------
def load_jesse():
    print("\n=== G1: Jesse handoff inventory ===")
    cw_path = os.path.join(JESSE_DIR, "osi_node_ward_assignment.csv")
    gate(os.path.exists(cw_path), "crosswalk present", cw_path)
    with open(cw_path, newline="", encoding="utf-8-sig") as fh:
        crosswalk = list(csv.DictReader(fh))
    gate(len(crosswalk) == 175, "crosswalk rows == 175", f"got {len(crosswalk)}")

    jesse = {"crosswalk": crosswalk, "pct": {}, "ward_bus_names": set()}
    for scen in GC_SCENARIOS:
        p = os.path.join(JESSE_DIR, scen, f"osi_node_failure_pct_{scen}.csv")
        if not gate(os.path.exists(p), f"{scen} pct file present"):
            continue
        with open(p, newline="", encoding="utf-8-sig") as fh:
            rows = list(csv.DictReader(fh))
        gate(len(rows) == 175, f"{scen} pct rows == 175", f"got {len(rows)}")
        conv = {int(r["n_converged_runs"]) for r in rows}
        gate(conv == {1000}, f"{scen} converged runs == 1000", f"got {sorted(conv)}")
        # osi name -> pct; flag same-name rows with DIFFERENT pct (join ambiguity)
        by_name = defaultdict(set)
        for r in rows:
            by_name[slug(r["osi_name"])].add(float(r["pct_failed"]))
        collide = {k: sorted(v) for k, v in by_name.items() if len(v) > 1}
        if collide:
            print(f"  [WARN] {scen}: osi names with differing pct (using MAX): {collide}")
        jesse["pct"][scen] = {k: max(v) for k, v in by_name.items()}
        el = os.path.join(JESSE_DIR, scen, f"mc_element_failure_pct_{scen}.csv")
        with open(el, newline="", encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                if r["element_type"] == "bus":
                    jesse["ward_bus_names"].add(slug(r["name"]))
    print(f"  distinct ward bus name slugs: {len(jesse['ward_bus_names'])}")
    return jesse


# --------------------------------------------------------------------------
# G2 — our power nodes (discovery-first: schema is probed, not assumed)
# --------------------------------------------------------------------------
def load_our_power_nodes():
    print("\n=== G2: our power nodes ===")
    gate(os.path.exists(NODES_GEOJSON), "nodes geojson present", NODES_GEOJSON)
    with open(NODES_GEOJSON) as fh:
        geo = json.load(fh)
    feats = geo.get("features", [])
    gate(len(feats) > 0, "geojson has features", f"{len(feats)} features")
    props0 = feats[0].get("properties", {})
    print(f"  property keys observed: {sorted(props0.keys())}")

    id_keys = [k for k in ("node_id", "id", "nid", "name") if k in props0]
    type_keys = [k for k in ("node_type", "type", "infra_type", "category") if k in props0]

    power = {}
    for ft in feats:
        pr = ft.get("properties", {})
        nid = next((pr[k] for k in id_keys if pr.get(k)), None)
        ntype = next((str(pr[k]).lower() for k in type_keys if pr.get(k)), "")
        if nid is None:
            continue
        is_power = ntype == "power" or str(nid).startswith("power_")
        if not is_power:
            continue
        lon = lat = None
        geom = ft.get("geometry") or {}
        if geom.get("type") == "Point":
            lon, lat = geom["coordinates"][0], geom["coordinates"][1]
        power[str(nid)] = {"lat": lat, "lon": lon,
                           "name": pr.get("name", str(nid)),
                           "slug": slug(re.sub(r"^power_", "", str(nid)))}
    gate(len(power) > 0, "power nodes found", f"{len(power)} power nodes")
    return power


# --------------------------------------------------------------------------
# G3 — our MC marginals from cascade_results (schema confirmed Week 19)
# --------------------------------------------------------------------------
def our_marginals(sim_name):
    path = os.path.join(SIM_DIR, f"cascade_results_nyc_{sim_name}.json")
    if not os.path.exists(path):
        print(f"  [WARN] missing {path} — skipping")
        return None
    sz = os.path.getsize(path) / 1e6
    print(f"  loading {path} ({sz:.0f} MB) ...")
    with open(path) as fh:
        runs = json.load(fh)
    n = len(runs)
    seed = defaultdict(int)
    t6 = defaultdict(int)
    total = defaultdict(int)
    for run in runs:
        for nid, t in run.get("fail_time_per_node", {}).items():
            if not nid.startswith("power_"):
                continue
            total[nid] += 1
            if t == 0:
                seed[nid] += 1
            if t <= 6:
                t6[nid] += 1
    del runs
    return {"n_runs": n,
            "seed": {k: v / n for k, v in seed.items()},
            "t6": {k: v / n for k, v in t6.items()},
            "total": {k: v / n for k, v in total.items()}}


# --------------------------------------------------------------------------
# P1 — join tiers
# --------------------------------------------------------------------------
def build_join(power, jesse):
    print("\n=== P1: join construction ===")
    osi_by_slug = defaultdict(list)
    for r in jesse["crosswalk"]:
        osi_by_slug[slug(r["osi_name"])].append(r)

    join = {}
    tier_counts = defaultdict(int)
    for nid, info in power.items():
        s = info["slug"]
        if s in osi_by_slug:
            join[nid] = {"tier": "name", "osi_slug": s, "distance_m": None}
            tier_counts["name"] += 1
            continue
        if s in jesse["ward_bus_names"]:
            join[nid] = {"tier": "ward_name", "osi_slug": None, "ward_slug": s, "distance_m": None}
            tier_counts["ward_name"] += 1
            continue
        # spatial fallback
        best = (None, float("inf"))
        if info["lat"] is not None:
            for r in jesse["crosswalk"]:
                d = haversine_m(info["lat"], info["lon"],
                                float(r["osi_lat"]), float(r["osi_lon"]))
                if d < best[1]:
                    best = (slug(r["osi_name"]), d)
        if best[0] is not None and best[1] <= SPATIAL_THRESH_M:
            join[nid] = {"tier": "spatial", "osi_slug": best[0], "distance_m": round(best[1], 1)}
            tier_counts["spatial"] += 1
        else:
            join[nid] = {"tier": "unmatched", "osi_slug": None,
                         "distance_m": round(best[1], 1) if best[0] else None}
            tier_counts["unmatched"] += 1

    npow = len(power)
    print(f"  name-tier matches:      {tier_counts['name']}/{npow}")
    print(f"  WARD-name matches:      {tier_counts['ward_name']}/{npow}  "
          "(shared-ancestry diagnostic — these skip the OSI layer entirely)")
    print(f"  spatial matches <=500m: {tier_counts['spatial']}/{npow}")
    print(f"  unmatched:              {tier_counts['unmatched']}/{npow}")
    for nid, j in sorted(join.items()):
        if j["tier"] in ("spatial", "unmatched"):
            print(f"    {j['tier']:9s} {nid}  (nearest OSI {j.get('osi_slug')} @ {j.get('distance_m')} m)")
    gate(tier_counts["unmatched"] < npow, "at least one node matched")
    return join, dict(tier_counts)


# --------------------------------------------------------------------------
# P2 — reconciliation tables per gc scenario
# --------------------------------------------------------------------------
def reconcile(power, jesse, join):
    print("\n=== P2: reconciliation — ours vs Jesse (matched nodes) ===")
    out = {}
    for scen, sim_name in GC_SCENARIOS.items():
        ours = our_marginals(sim_name)
        if ours is None:
            continue
        rows = []
        for nid, j in join.items():
            if j["tier"] not in ("name", "spatial"):
                continue
            jp = jesse["pct"][scen].get(j["osi_slug"])
            if jp is None:
                continue
            rows.append({"node": nid, "tier": j["tier"],
                         "ours_seed": round(ours["seed"].get(nid, 0.0), 4),
                         "ours_t6": round(ours["t6"].get(nid, 0.0), 4),
                         "ours_t96": round(ours["total"].get(nid, 0.0), 4),
                         "jesse_pct": round(jp / 100.0, 4)})
        rows.sort(key=lambda r: r["ours_t96"] - r["jesse_pct"], reverse=True)
        n = len(rows)
        mean = lambda k: sum(r[k] for r in rows) / n if n else 0.0
        overkill = sum(1 for r in rows if r["ours_t96"] > 0.9 and r["jesse_pct"] < 0.2)
        print(f"\n  {scen} ({ours['n_runs']} of our runs, {n} matched nodes):")
        print(f"    mean ours_seed={mean('ours_seed'):.3f}  ours_t6={mean('ours_t6'):.3f}  "
              f"ours_t96={mean('ours_t96'):.3f}  vs jesse={mean('jesse_pct'):.3f}")
        print(f"    nodes ours>0.9 while jesse<0.2: {overkill}/{n}")
        print(f"    {'node':38s}{'tier':9s}{'seed':>7s}{'t6':>7s}{'t96':>7s}{'jesse':>7s}")
        for r in rows[:15]:
            print(f"    {r['node']:38s}{r['tier']:9s}{r['ours_seed']:7.3f}{r['ours_t6']:7.3f}"
                  f"{r['ours_t96']:7.3f}{r['jesse_pct']:7.3f}")
        out[scen] = {"n_matched": n, "n_runs": ours["n_runs"],
                     "mean_ours_seed": mean("ours_seed"), "mean_ours_t6": mean("ours_t6"),
                     "mean_ours_t96": mean("ours_t96"), "mean_jesse": mean("jesse_pct"),
                     "overkill_count": overkill, "rows": rows}
    return out


# --------------------------------------------------------------------------
# P3 — DEP adjudication
# --------------------------------------------------------------------------
def dep_check():
    print("\n=== P3: DEP power-seed adjudication (Jesse reports zero) ===")
    out = {}
    for label, sim_name in DEP_SCENARIOS.items():
        ours = our_marginals(sim_name)
        if ours is None:
            out[label] = {"status": "file_missing"}
            continue
        seeded = {k: v for k, v in ours["seed"].items() if v > 0}
        agrees = len(seeded) == 0
        print(f"  {label}: power nodes with nonzero seed prob = {len(seeded)} "
              f"-> {'AGREES with Jesse zero' if agrees else 'CONTRADICTS Jesse zero'}")
        for k, v in sorted(seeded.items(), key=lambda kv: -kv[1])[:10]:
            print(f"      {k}: {v:.3f}")
        out[label] = {"status": "ok", "n_seeded_power_nodes": len(seeded),
                      "agrees_with_jesse_zero": agrees,
                      "seeded": dict(sorted(seeded.items(), key=lambda kv: -kv[1]))}
    return out


def main():
    print("PHASE 0 — Jesse intra-power reconciliation (read-only)")
    jesse = load_jesse()
    power = load_our_power_nodes()
    if FAIL:
        print(f"\nHARD STOP — gate failures: {FAIL}")
        sys.exit(1)
    join, tiers = build_join(power, jesse)
    recon = reconcile(power, jesse, join)
    dep = dep_check()

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "phase0_jesse_power_reconciliation.json")
    with open(out_path, "w") as fh:
        json.dump({"join_tiers": tiers,
                   "join": join,
                   "reconciliation": recon,
                   "dep_adjudication": dep,
                   "spatial_threshold_m": SPATIAL_THRESH_M,
                   "n_power_nodes": len(power)}, fh, indent=1)
    print(f"\nWrote {out_path}")
    if FAIL:
        print(f"Completed WITH gate failures: {FAIL}")
        sys.exit(1)
    print("Phase 0 complete — all gates passed.")


if __name__ == "__main__":
    main()
