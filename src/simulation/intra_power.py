#!/usr/bin/env python3
"""
intra_power.py — intra-power-network cascade via Jesse's Monte Carlo library.

Unlike intra_telecom / intra_hospital (mechanistic models), the power domain is
served by an EMPIRICAL RESAMPLER over Jesse's pandapower Ward-network library
(data/external/jesse_power): per World-1 MC run, one complete Jesse run is
selected (index-paired) and its final out-of-service state is mapped onto our
power nodes.

Semantics (locked Week 19, validated by phase0_jesse_power_reconciliation):
  * REPLACE, not union: for LIBRARY-COVERED nodes, Jesse's sampled final state
    replaces flood seeding + intra-power cascade entirely (his state already
    contains the flood). Nodes outside his coverage keep legacy HAZUS seeding.
  * Index pairing: our run i uses Jesse run (i mod 1000). Valid because the
    hazard field is deterministic per scenario and fragility draws are
    conditionally independent, so the joint distribution factorizes exactly;
    pairing eliminates resampling noise and reproduces his marginals by
    construction.
  * STATE FLOOR at the first timestep (t=6): Jesse-dead nodes are dead from t=6
    onward; inter-layer dependency edges INTO power remain live at every
    timestep (the seed-0/t96~1 pile-on archetype in Phase 0 is real mechanism).
  * Scenario coverage is per-library: scenarios absent from the library fall
    back to legacy behavior wholesale (currently gc_2026/2050/2080;
    dep_extreme_2080 pending Jesse's rerun on the fixed raster).

Handoff-3 upgrade (Week 20):
  * Run summaries now ship `osi_node_failed_final` — Jesse's own per-run
    OSI-level death list, already including own-flood draws and the GOETHALS
    rescue. When present it is used DIRECTLY: no bus decoding, no tied_rule.
    Exact by construction (handoff validated 175/175 across all 23 scenarios).
  * Scenario coverage 3 -> 23: gc trio via the static map below, plus all 20
    synthetics via the syn_ts_<A>_<B>_* -> gwyn_<A>_<B> resolver.
  * Bus-level decoding below is retained ONLY as the fallback for the older
    gc-trio library (reproducibility of Week-19 results).

Per-run bus-death decoding:
  * If the run summary has `final_bus_oos_idx` (requested from Jesse), pp bus
    indices are used directly — exact.
  * Otherwise NAME-LEVEL fallback: a crosswalk bus is dead iff its NAME appears
    in `final_bus_oos`. Twelve of the 50 Ward bus names are shared by two pp
    idxs (multi-voltage substations), so name-level death is slightly
    conservative (kills both voltages when either dies). Quantified by
    validate_intra_power.py.
  * `tied_rule`: 'matched' = node dies when its matched bus dies;
    'any_tied' = node dies when any bus in ward_tied_pp_bus_idxs dies.
    Default 'matched' (adjudicated empirically by the validator against
    Jesse's own pct_failed).

Public API (for cascade_joint wiring):
    lib  = load_power_library(jesse_dir)
    join = build_power_join(nodes_geojson_path, lib)     # node_id -> osi_node_id
    st   = sample_power_state(lib, join, scenario, run_idx, tied_rule='matched')
    st.covered_nodes : set of our node_ids governed by the library this scenario
                       (these SKIP legacy flood seeding)
    st.dead_nodes    : subset dead at the t=6 floor (cause tag: 'intra_power')
    Scenarios not in lib.scenarios -> sample_power_state returns empty coverage
    (caller falls back to legacy for the whole power layer).
"""

import csv
import json
import math
import os
import re
from collections import namedtuple

# our scenario name -> Jesse folder name
SCENARIO_MAP = {
    "geoclaw_2026": "gc_2026",
    "geoclaw_2050": "gc_2050",
    "geoclaw_2080": "gc_2080",
    "extreme_2080": "dep_extreme_2080",   # pending Jesse's 4th-scenario rerun
}
SPATIAL_THRESH_M = 500.0

PowerState = namedtuple("PowerState", ["covered_nodes", "dead_nodes", "jesse_run_idx"])


def _slug(s):
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


def _haversine_m(lat1, lon1, lat2, lon2):
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin(math.radians(lat2 - lat1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 2 * r * math.asin(math.sqrt(a))


class PowerLibrary:
    """Jesse's handoff, loaded once: crosswalk, idx/name maps, per-run dead sets."""

    def __init__(self, jesse_dir):
        self.jesse_dir = jesse_dir
        cw_path = os.path.join(jesse_dir, "osi_node_ward_assignment.csv")
        with open(cw_path, newline="", encoding="utf-8-sig") as fh:
            self.crosswalk = list(csv.DictReader(fh))
        self.osi = {r["osi_node_id"]: r for r in self.crosswalk}

        # discover scenario folders present on disk
        self.scenarios = {}   # our-name -> jesse-name
        for ours, theirs in SCENARIO_MAP.items():
            if os.path.isdir(os.path.join(jesse_dir, theirs)):
                self.scenarios[ours] = theirs

        # bus idx -> name (from any element file; identical across scenarios).
        # Only needed by the LEGACY bus-decode fallback; guarded so an
        # osi-column-era library never fails here.
        self.bus_name_by_idx = {}
        try:
            any_scen = next(iter(self.scenarios.values()), None)
            if any_scen is None:
                gwyns = [d for d in os.listdir(jesse_dir) if d.startswith("gwyn_")]
                any_scen = gwyns[0] if gwyns else None
            if any_scen:
                el_path = os.path.join(jesse_dir, any_scen,
                                       f"mc_element_failure_pct_{any_scen}.csv")
                with open(el_path, newline="", encoding="utf-8-sig") as fh:
                    for r in csv.DictReader(fh):
                        if r["element_type"] == "bus":
                            self.bus_name_by_idx[int(r["element_id"])] = r["name"]
        except (OSError, KeyError, ValueError):
            pass

        # per-scenario per-run dead sets, decoded lazily
        self._runs = {}          # jesse-name -> list of {"idx": set|None, "names": set}
        self.used_idx_column = {}  # jesse-name -> bool (True once Jesse ships B1)

    def runs(self, jesse_scen):
        if jesse_scen in self._runs:
            return self._runs[jesse_scen]
        path = os.path.join(self.jesse_dir, jesse_scen, f"mc_run_summary_{jesse_scen}.csv")
        decoded = []
        with open(path, newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            has_idx = "final_bus_oos_idx" in (reader.fieldnames or [])
            has_osi = "osi_node_failed_final" in (reader.fieldnames or [])
            for row in reader:
                names = {x for x in (row.get("final_bus_oos") or "").split(";") if x}
                idxs = None
                if has_idx:
                    idxs = {int(x) for x in (row.get("final_bus_oos_idx") or "").split(";") if x}
                osi = None
                if has_osi:
                    osi = {x.strip() for x in (row.get("osi_node_failed_final") or "").split(";")
                           if x.strip()}
                decoded.append({"idx": idxs, "names": names, "osi": osi})
        self.used_idx_column[jesse_scen] = has_idx
        self.used_osi_column = getattr(self, "used_osi_column", {})
        self.used_osi_column[jesse_scen] = has_osi
        self._runs[jesse_scen] = decoded
        return decoded

    def bus_dead(self, run, pp_idx):
        """Is pp bus `pp_idx` out of service in this decoded run?"""
        if run["idx"] is not None:                      # exact path (post-B1)
            return pp_idx in run["idx"]
        name = self.bus_name_by_idx.get(pp_idx)         # name-level fallback
        return name is not None and name in run["names"]

    def osi_dead(self, run, osi_row, tied_rule="matched"):
        if run.get("osi") is not None:            # handoff-3 EXACT path
            return osi_row["osi_node_id"] in run["osi"]
        if self.bus_dead(run, int(osi_row["ward_pp_bus_idx"])):
            return True
        if tied_rule == "any_tied":
            for x in (osi_row.get("ward_tied_pp_bus_idxs") or "").split(";"):
                if x and self.bus_dead(run, int(x)):
                    return True
        return False

    def resolve(self, scenario):
        """Our scenario name -> Jesse folder name, or None (-> legacy fallback).

        Static map first (gc trio + dep_extreme), then the synthetic pattern:
        syn_ts_<A>_<B>_<peak> -> gwyn_<A>_<B> if that folder exists on disk.
        """
        js = self.scenarios.get(scenario)
        if js:
            return js
        m = re.match(r"^syn_ts_(\d+)_(\d+)(?:_|$)", str(scenario))
        if m:
            cand = f"gwyn_{m.group(1)}_{m.group(2)}"
            if os.path.isdir(os.path.join(self.jesse_dir, cand)):
                self.scenarios[scenario] = cand   # cache
                return cand
        return None


def load_power_library(jesse_dir="data/external/jesse_power"):
    return PowerLibrary(jesse_dir)


def build_power_join(nodes_geojson_path, lib):
    """our power node_id -> osi_node_id.

    Tier 1: name slug match; same-name OSI collisions (e.g. three Farragut
    sites) disambiguated by nearest coordinates. Tier 2: nearest OSI node
    within SPATIAL_THRESH_M. Everything else (NJ / out-of-footprint) is
    excluded -> keeps legacy behavior.
    """
    with open(nodes_geojson_path) as fh:
        geo = json.load(fh)
    osi_by_slug = {}
    for r in lib.crosswalk:
        osi_by_slug.setdefault(_slug(r["osi_name"]), []).append(r)

    join = {}
    for ft in geo.get("features", []):
        pr = ft.get("properties", {})
        nid = str(pr.get("node_id") or pr.get("id") or "")
        ntype = str(pr.get("infra_type") or pr.get("node_type") or "").lower()
        if not (ntype == "power" or nid.startswith("power_")):
            continue
        lat = pr.get("lat")
        lon = pr.get("lon")
        if (lat is None or lon is None) and (ft.get("geometry") or {}).get("type") == "Point":
            lon, lat = ft["geometry"]["coordinates"][:2]
        s = _slug(re.sub(r"^power_", "", nid))
        cands = osi_by_slug.get(s)
        if cands:
            if len(cands) == 1 or lat is None:
                join[nid] = cands[0]["osi_node_id"]
            else:  # name collision -> nearest coordinates
                best = min(cands, key=lambda r: _haversine_m(
                    lat, lon, float(r["osi_lat"]), float(r["osi_lon"])))
                join[nid] = best["osi_node_id"]
            continue
        if lat is None:
            continue
        best, bd = None, SPATIAL_THRESH_M
        for r in lib.crosswalk:
            d = _haversine_m(lat, lon, float(r["osi_lat"]), float(r["osi_lon"]))
            if d <= bd:
                best, bd = r, d
        if best is not None:
            join[nid] = best["osi_node_id"]
    return join


def sample_power_state(lib, join, scenario, run_idx, tied_rule="matched"):
    """Power-layer state for World-1 run `run_idx` under `scenario`.

    Returns PowerState(covered_nodes, dead_nodes, jesse_run_idx).
    Empty coverage when the scenario is not in the library (caller falls back
    to legacy seeding for the entire power layer).
    """
    jesse_scen = lib.resolve(scenario)
    if jesse_scen is None:
        return PowerState(covered_nodes=set(), dead_nodes=set(), jesse_run_idx=None)
    runs = lib.runs(jesse_scen)
    j = run_idx % len(runs)                 # index pairing
    run = runs[j]
    covered, dead = set(), set()
    for nid, osi_id in join.items():
        covered.add(nid)
        if lib.osi_dead(run, lib.osi[osi_id], tied_rule=tied_rule):
            dead.add(nid)
    return PowerState(covered_nodes=covered, dead_nodes=dead, jesse_run_idx=j)
