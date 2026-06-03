"""
Boston graph builder — schema-identical to NYC for zero-shot transfer.

Produces a PyG HeteroData with the EXACT NYC schema:
  - 6 node types (power, telecom, hospital, subway, water, fuel), 8 features each
  - 19 edge relation types
so the NYC-trained CascadeGNN can run on Boston without any architecture change.

Design:
  * Feature vectors + edge-type mapping + normalization are REUSED BY IMPORT
    from src/graph/convert_to_pyg.py (graph_to_heterodata). That guarantees the
    8-feature schema and the 19 relation triplets match NYC bit-for-bit — no
    re-implementation, no drift.
  * The NetworkX node/edge construction RULES (distance caps, nearest-neighbor
    pairing, buffer hours, edge_type strings) are PORTED VERBATIM from NYC's
    src/graph/build_graph.py. Edge COUNTS differ (Boston is smaller/sparser);
    edge LOGIC is identical, which is what preserves transfer validity.
  * UTM zone: Boston is UTM 19N (EPSG:32619), not NYC's 18N. Using Boston's
    correct zone gives real-meter distances; the cap semantics (5 km, 50 km)
    therefore mean the same physical thing in both cities. edge_attr is
    min-max normalized per graph anyway, so absolute scale never transfers.

Data-gap handling (water_flow):
  Boston has no pumping-station data (the Week-12 BWSC gap), so the
  pump->treatment rule yields zero edges. To keep the 19-relation contract,
  (water, water_flow, water) is force-created as an EMPTY relation
  (edge_index [2,0]); the model's water_flow conv then contributes nothing,
  faithfully representing "no pump->treatment topology in our Boston data."

Flood is NOT baked into the static graph (flood_depth stays 0.0, exactly like
NYC). The per-scenario severity-ladder depths are applied at inference time by
the Phase 3 wrapper.

Output: data/boston/graph/boston_infra_heterodata.pt
"""
from __future__ import annotations

import logging
import zipfile

import geopandas as gpd
import networkx as nx
import pandas as pd
from pyproj import Transformer

from src.cities import boston
from src.graph.convert_to_pyg import graph_to_heterodata

log = logging.getLogger(__name__)

OUT_PATH = boston.GRAPH_DIR / "boston_infra_heterodata.pt"

# ── Coordinate helpers — ported from build_graph.py, Boston UTM zone 19N ──────
_to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32619", always_xy=True)


def utm(lon, lat):
    return _to_utm.transform(lon, lat)


def dist_m(lon1, lat1, lon2, lat2):
    e1, n1 = utm(lon1, lat1)
    e2, n2 = utm(lon2, lat2)
    return ((e1 - e2) ** 2 + (n1 - n2) ** 2) ** 0.5


def slug(s):
    s = str(s)
    for ch in [" ", "-", "/", ",", "(", ")", "'", ".", "&"]:
        s = s.replace(ch, "_")
    return s.lower().strip("_")


def nearest_within_cap(src_nodes, target_node, max_m):
    if not src_nodes:
        return None, None
    dists = [
        (dist_m(s["lon"], s["lat"], target_node["lon"], target_node["lat"]), s)
        for s in src_nodes
    ]
    d, best = min(dists, key=lambda x: x[0])
    if d > max_m:
        return None, None
    return best, d


# ── NYC construction constants (verbatim from build_graph.py) ────────────────
MAX_POWER_DEPENDENCY_M = 5_000
MAX_WATER_SUPPLY_M = 5_000
MAX_FUEL_SUPPLY_M = 5_000
MAX_FUEL_DISTRIBUTION_M = 50_000
MAX_SCADA_M = 5_000
MAX_REPAIR_ACCESS_M = 5_000

BUFFER_HOURS = {"telecom": 6.0, "hospital": 96.0, "subway": 0.0, "water": 3.0, "fuel": 0.0}
SCADA_BUFFER_HOURS = 2.0
REPAIR_BUFFER_HOURS = 0.0
WATER_SUPPLY_BUFFER = 24.0
FUEL_TERMINAL_POWER_BUFFER = 4.0
FUEL_DISTRIBUTION_BUFFER = 0.0
FUEL_TO_HOSPITAL_BUFFER = 96.0
FUEL_TO_TELECOM_BUFFER = 48.0
FUEL_TO_WATER_BUFFER = 48.0
VOLT_WEIGHT = {"345": 3.0, "220-287": 2.0, "100-161": 1.5, "UNDER 100": 1.0,
               "500": 5.0, "NOT AVAILABLE": 1.0}

RAW = boston.RAW_DIR


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD BOSTON LAYERS → NYC-schema node dicts
# ══════════════════════════════════════════════════════════════════════════════

def _subway_routes_by_parent_station() -> dict[str, set]:
    """Derive route membership per parent station from the GTFS zip.

    GTFS join: routes←trips←stop_times←stops(parent_station). Mirrors the
    semantic of NYC's per-station `routes` field so the subway_line chaining
    rule has the same input.
    """
    zip_path = RAW / "subway" / "mbta_gtfs.zip"
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open("trips.txt") as f:
            trips = pd.read_csv(f, dtype=str)
        with zf.open("stop_times.txt") as f:
            stop_times = pd.read_csv(f, dtype=str, usecols=["trip_id", "stop_id"])
        with zf.open("stops.txt") as f:
            stops = pd.read_csv(f, dtype=str, usecols=["stop_id", "parent_station"])

    trip_route = dict(zip(trips["trip_id"], trips["route_id"]))
    stop_parent = dict(zip(stops["stop_id"], stops["parent_station"]))

    routes_by_parent: dict[str, set] = {}
    for trip_id, stop_id in zip(stop_times["trip_id"], stop_times["stop_id"]):
        route_id = trip_route.get(trip_id)
        if route_id is None:
            continue
        parent = stop_parent.get(stop_id)
        # A platform's own stop_id is the key in our subway nodes if it has no
        # parent; otherwise the parent station is the node.
        key = parent if parent and str(parent) != "nan" else stop_id
        routes_by_parent.setdefault(key, set()).add(route_id)
    return routes_by_parent


def load_boston_nodes() -> list[dict]:
    nodes: list[dict] = []

    # ── Power substations ──
    subs = gpd.read_file(RAW / "power" / "hifld_substations.geojson")
    for _, r in subs.iterrows():
        name = str(r["NAME"]).upper().strip()
        nodes.append(dict(
            node_id=f"power_{slug(name)}",
            name=name.title(), infra_type="power", subtype=str(r.get("TYPE", "")),
            lat=float(r["LATITUDE"]), lon=float(r["LONGITUDE"]),
            status=str(r.get("STATUS", "")), external=False,
        ))
    log.info("  power: %d substations", sum(1 for n in nodes if n["infra_type"] == "power"))

    # ── Telecom (FCC ASR structures — no clustering needed, already sparse) ──
    tel = gpd.read_file(RAW / "telecom" / "fcc_asr_towers.geojson")
    for i, r in tel.iterrows():
        nodes.append(dict(
            node_id=f"telecom_asr_{i:05d}",
            name=f"{str(r.get('ENTITY', 'ASR'))[:40]} structure",
            infra_type="telecom", subtype="ANTENNA_STRUCTURE",
            lat=float(r.geometry.y), lon=float(r.geometry.x),
            tower_count=1, radio_types="",  # ASR carries no radio type → has_lte/has_5g = 0
            external=False,
        ))
    log.info("  telecom: %d ASR structures", len(tel))

    # ── Hospitals (HIFLD; bed_count NOT fed — extractor uses NYC's constant) ──
    hosp = gpd.read_file(RAW / "healthcare" / "hospitals.geojson")
    for _, r in hosp.iterrows():
        nodes.append(dict(
            node_id=f"hospital_{slug(str(r['NAME'])[:40])}",
            name=str(r["NAME"]).title(), infra_type="hospital",
            subtype=str(r.get("TYPE", "")),
            lat=float(r["LATITUDE"]), lon=float(r["LONGITUDE"]),
            external=False,
        ))
    log.info("  hospital: %d", len(hosp))

    # ── Subway (parent stations, routes from GTFS) ──
    stops = gpd.read_file(RAW / "subway" / "stops.geojson")
    parent_stations = stops[stops["location_type"].astype(str) == "1"].copy()
    routes_by_parent = _subway_routes_by_parent_station()
    for _, r in parent_stations.iterrows():
        sid = str(r["stop_id"])
        routes = " ".join(sorted(routes_by_parent.get(sid, set())))
        wb = r.get("wheelchair_boarding")
        ada = 1 if str(wb) == "1" else 0
        nodes.append(dict(
            node_id=f"subway_{slug(str(r['stop_name']))}_{slug(sid)}",
            name=str(r["stop_name"]), infra_type="subway", subtype="STATION",
            lat=float(r["stop_lat"]), lon=float(r["stop_lon"]),
            routes=routes, ada=ada, external=False,
        ))
    n_routed = sum(1 for n in nodes if n["infra_type"] == "subway" and n.get("routes"))
    log.info("  subway: %d parent stations (%d with routes)",
             len(parent_stations), n_routed)

    # ── Water ──
    # Primary set: MassDEP CSO outfalls (canonical, cross-validated with EPA R1).
    # NOTE: these are DISCHARGE points, not supply infrastructure. NYC's water
    # nodes are pumps/treatment plants that hospitals depend on; Boston's are the
    # opposite end of the system. The (water, supplies, hospital) edges are
    # therefore topological placeholders, not physical supply — the deepest data
    # caveat in this pilot, and a direct consequence of the Week-12 BWSC gap.
    water = gpd.read_file(RAW / "water" / "massdep_cso_outfalls.geojson")
    for i, r in water.iterrows():
        nm = str(r.get("DEP_OUTFL_", r.get("OUTFALL_ID", f"outfall_{i}")))
        nodes.append(dict(
            node_id=f"water_cso_{slug(nm)}_{i:04d}",
            name=f"CSO {nm}", infra_type="water", subtype="CSO_OUTFALL",
            lat=float(r.geometry.y), lon=float(r.geometry.x),
            external=False,
        ))
    n_cso = len(water)

    # Also fold in the one genuine water-infrastructure facility from the NPDES
    # layer: MWRA Deer Island Treatment Plant (the only FACILITY_TYPE_DESC ==
    # "Municipal or Water District"). It is a real treatment node (is_treatment=1)
    # and the only treatment plant in our Boston data. The other NPDES records are
    # industrial dischargers (Logan, Gillette, Citgo, Sunoco [already a fuel node],
    # North Station) — NOT water infrastructure — so they are deliberately excluded
    # to avoid cross-node-type double-counting.
    npdes = gpd.read_file(RAW / "water" / "npdes_facilities_outfalls.geojson")
    treat = npdes[npdes["FACILITY_TYPE_DESC"] == "Municipal or Water District"]
    treat = treat.drop_duplicates(subset=["FACILITY_NAME"])
    n_treat = 0
    for _, r in treat.iterrows():
        nm = str(r["FACILITY_NAME"])
        nodes.append(dict(
            node_id=f"water_treatment_{slug(nm[:40])}",
            name=nm.title(), infra_type="water",
            subtype="WASTEWATER TREATMENT PLANT",  # → is_treatment=1 in the extractor
            lat=float(r.geometry.y), lon=float(r.geometry.x),
            external=False,
        ))
        n_treat += 1
    log.info("  water: %d CSO outfalls + %d treatment plant (MWRA Deer Island)",
             n_cso, n_treat)

    # ── Fuel: terminals (5) + OSM gas stations (amenity=fuel only) ──
    terminals = gpd.read_file(RAW / "fuel" / "major_terminals.geojson")
    for _, r in terminals.iterrows():
        nodes.append(dict(
            node_id=f"fuel_terminal_{slug(str(r['name'])[:40])}",
            name=str(r["name"]), infra_type="fuel", subtype="PETROLEUM_TERMINAL",
            lat=float(r["lat"]), lon=float(r["lon"]),
            capacity_bbl=0, external=True,
        ))
    osm = gpd.read_file(RAW / "fuel" / "osm_stations.geojson")
    stations = osm[osm["osm_type"] == "fuel"] if "osm_type" in osm.columns else osm
    for i, r in stations.iterrows():
        nm = str(r.get("name", "Gas Station"))
        nodes.append(dict(
            node_id=f"fuel_station_{slug(nm)[:30]}_{i:04d}",
            name=nm, infra_type="fuel", subtype="GAS_STATION",
            lat=float(r.geometry.y), lon=float(r.geometry.x),
            brand=str(r.get("brand", "")), external=False,
        ))
    log.info("  fuel: %d terminals + %d stations", len(terminals), len(stations))

    # Dedup node_ids
    seen, uniq = set(), []
    for n in nodes:
        if n["node_id"] not in seen:
            seen.add(n["node_id"])
            uniq.append(n)
    return uniq


# ══════════════════════════════════════════════════════════════════════════════
# 2. BUILD DIRECTED GRAPH — NYC's exact edge rules (ported verbatim)
# ══════════════════════════════════════════════════════════════════════════════

def build_graph(nodes: list[dict]) -> nx.DiGraph:
    G = nx.DiGraph()
    for n in nodes:
        G.add_node(n["node_id"], **{k: v for k, v in n.items() if k != "node_id" and v is not None})
    node_lookup = {n["node_id"]: n for n in nodes}

    def add_edge(u, v, edge_type, weight=1.0, **attrs):
        if G.has_node(u) and G.has_node(v) and u != v:
            G.add_edge(u, v, edge_type=edge_type, weight=weight, **attrs)

    def add_bidir(u, v, edge_type, weight=1.0, **attrs):
        add_edge(u, v, edge_type, weight, **attrs)
        add_edge(v, u, edge_type, weight, **attrs)

    by_type = lambda t: [n for n in nodes if n["infra_type"] == t]

    # ── 4a. Power transmission lines → power_line (bidir) ──
    lines = gpd.read_file(RAW / "power" / "hifld_transmission_lines.geojson")
    line_added, line_miss = 0, 0
    for _, row in lines.iterrows():
        u = f"power_{slug(str(row['SUB_1']).upper().strip())}"
        v = f"power_{slug(str(row['SUB_2']).upper().strip())}"
        if not (G.has_node(u) and G.has_node(v)):
            line_miss += 1
            continue
        volt = str(row.get("VOLT_CLASS", "NOT AVAILABLE"))
        add_bidir(u, v, "power_line", weight=VOLT_WEIGHT.get(volt, 1.0),
                  volt_class=volt, buffer_hours=0.0)
        line_added += 1
    log.info("  power_line: %d lines matched (%d dropped, endpoint not a node) → %d directed",
             line_added, line_miss, line_added * 2)

    # ── 4b. Subway lines → subway_line (bidir, chain along route axis) ──
    line_stations: dict[str, list] = {}
    for n in nodes:
        if n["infra_type"] == "subway" and n.get("routes"):
            for ln in str(n["routes"]).split():
                line_stations.setdefault(ln, []).append(n["node_id"])
    subway_added = 0
    for ln, sids in line_stations.items():
        coords = [(nid, node_lookup[nid]["lon"], node_lookup[nid]["lat"]) for nid in sids]
        lons = [c[1] for c in coords]
        lats = [c[2] for c in coords]
        if (max(lons) - min(lons)) > (max(lats) - min(lats)):
            coords.sort(key=lambda x: x[1])
        else:
            coords.sort(key=lambda x: x[2])
        for i in range(len(coords) - 1):
            u, ulon, ulat = coords[i]
            v, vlon, vlat = coords[i + 1]
            d = dist_m(ulon, ulat, vlon, vlat)
            if d > 10_000:
                continue
            add_bidir(u, v, "subway_line", weight=round(d / 1000, 3),
                      line=ln, distance_m=round(d), buffer_hours=0.0)
            subway_added += 1
    log.info("  subway_line: %d links → %d directed", subway_added, subway_added * 2)

    # ── 4c. Power → dependents → power_dependency ──
    all_power = by_type("power")
    dep_targets = [n for n in nodes
                   if n["infra_type"] in ("hospital", "water", "telecom", "subway", "fuel")
                   and not n.get("external")]
    pdc = 0
    for tn in dep_targets:
        nearest, d = nearest_within_cap(all_power, tn, MAX_POWER_DEPENDENCY_M)
        if nearest is None:
            continue
        add_edge(nearest["node_id"], tn["node_id"], "power_dependency",
                 weight=round(1 / (1 + d / 1000), 3), distance_m=round(d),
                 buffer_hours=BUFFER_HOURS.get(tn["infra_type"], 0.0),
                 dependency_class="physical")
        pdc += 1
    # external fuel terminals: 20km cap
    for tn in [n for n in nodes if n["infra_type"] == "fuel" and n.get("subtype") == "PETROLEUM_TERMINAL"]:
        nearest, d = nearest_within_cap(all_power, tn, 20_000)
        if nearest is None:
            continue
        add_edge(nearest["node_id"], tn["node_id"], "power_dependency",
                 weight=round(1 / (1 + d / 1000), 3), distance_m=round(d),
                 buffer_hours=FUEL_TERMINAL_POWER_BUFFER, dependency_class="physical")
        pdc += 1
    log.info("  power_dependency (→feeds): %d", pdc)

    # ── 4d. Water flow: pump → treatment (Boston has neither → 0; placeholder later) ──
    pump_types = {"PUMPING STATION", "WASTEWATER PUMPING STATION", "STORMWATER PUMPING STATION"}
    treatment_types = {"WATER POLLUTION CONTROL PLANT", "WASTEWATER TREATMENT PLANT"}
    pumps = [n for n in nodes if n["infra_type"] == "water" and n["subtype"] in pump_types]
    treatments = [n for n in nodes if n["infra_type"] == "water" and n["subtype"] in treatment_types]
    wfc = 0
    for pump in pumps:
        for plant in treatments:
            d = dist_m(pump["lon"], pump["lat"], plant["lon"], plant["lat"])
            if d <= MAX_WATER_SUPPLY_M:
                add_edge(pump["node_id"], plant["node_id"], "water_flow",
                         weight=round(d / 1000, 3), distance_m=round(d),
                         buffer_hours=0.0, dependency_class="physical")
                wfc += 1
    log.info("  water_flow: %d (pumps=%d, treatment=%d) — empty placeholder if 0",
             wfc, len(pumps), len(treatments))

    # ── 4e. Water → hospital → water_supplies ──
    waters = by_type("water")
    hosps = by_type("hospital")
    wsc = 0
    for hn in hosps:
        nearest, d = nearest_within_cap(waters, hn, MAX_WATER_SUPPLY_M)
        if nearest is None:
            continue
        add_edge(nearest["node_id"], hn["node_id"], "water_supplies",
                 weight=round(1 / (1 + d / 1000), 3), distance_m=round(d),
                 buffer_hours=WATER_SUPPLY_BUFFER, dependency_class="physical")
        wsc += 1
    log.info("  water_supplies (→supplies hospital): %d", wsc)

    # ── 4f. Telecom → power → scada_monitoring ──
    tels = by_type("telecom")
    sc = 0
    for pn in [n for n in all_power if not n.get("external")]:
        nearest, d = nearest_within_cap(tels, pn, MAX_SCADA_M)
        if nearest is None:
            continue
        add_edge(nearest["node_id"], pn["node_id"], "scada_monitoring",
                 weight=round(1 / (1 + d / 1000), 3), distance_m=round(d),
                 buffer_hours=SCADA_BUFFER_HOURS, dependency_class="cyber")
        sc += 1
    log.info("  scada_monitoring (telecom→power): %d", sc)

    # ── 4g. Subway → infra → repair_access (recovery) ──
    subways = by_type("subway")
    repair_targets = [n for n in nodes
                      if n["infra_type"] in ("power", "hospital", "water", "telecom", "fuel")
                      and not n.get("external")]
    rc = 0
    for tn in repair_targets:
        nearest, d = nearest_within_cap(subways, tn, MAX_REPAIR_ACCESS_M)
        if nearest is None:
            continue
        add_edge(nearest["node_id"], tn["node_id"], "repair_access",
                 weight=round(1 / (1 + d / 1000), 3), distance_m=round(d),
                 buffer_hours=REPAIR_BUFFER_HOURS, dependency_class="logical", layer="recovery")
        rc += 1
    log.info("  repair_access (subway→*): %d", rc)

    # ── 4h. Fuel distribution: terminal → station → fuel_distribution ──
    stations = [n for n in nodes if n["infra_type"] == "fuel" and n.get("subtype") == "GAS_STATION"]
    terms = [n for n in nodes if n["infra_type"] == "fuel" and n.get("subtype") == "PETROLEUM_TERMINAL"]
    fdc = 0
    for sn in stations:
        nearest, d = nearest_within_cap(terms, sn, MAX_FUEL_DISTRIBUTION_M)
        if nearest is None:
            continue
        add_edge(nearest["node_id"], sn["node_id"], "fuel_distribution",
                 weight=round(1 / (1 + d / 1000), 3), distance_m=round(d),
                 buffer_hours=FUEL_DISTRIBUTION_BUFFER, dependency_class="physical")
        fdc += 1
    log.info("  fuel_distribution (terminal→station): %d", fdc)

    # ── 4i. Fuel → generators (hospital, telecom≥5 towers, water) → fuel_supplies ──
    fsc = 0
    for hn in hosps:
        nearest, d = nearest_within_cap(stations, hn, MAX_FUEL_SUPPLY_M)
        if nearest:
            add_edge(nearest["node_id"], hn["node_id"], "fuel_supplies",
                     weight=round(1 / (1 + d / 1000), 3), distance_m=round(d),
                     buffer_hours=FUEL_TO_HOSPITAL_BUFFER, dependency_class="physical")
            fsc += 1
    for tn in [n for n in tels if float(n.get("tower_count", 0)) >= 5]:
        nearest, d = nearest_within_cap(stations, tn, MAX_FUEL_SUPPLY_M)
        if nearest:
            add_edge(nearest["node_id"], tn["node_id"], "fuel_supplies",
                     weight=round(1 / (1 + d / 1000), 3), distance_m=round(d),
                     buffer_hours=FUEL_TO_TELECOM_BUFFER, dependency_class="physical")
            fsc += 1
    for wn in [n for n in waters if not n.get("external")]:
        nearest, d = nearest_within_cap(stations, wn, MAX_FUEL_SUPPLY_M)
        if nearest:
            add_edge(nearest["node_id"], wn["node_id"], "fuel_supplies",
                     weight=round(1 / (1 + d / 1000), 3), distance_m=round(d),
                     buffer_hours=FUEL_TO_WATER_BUFFER, dependency_class="physical")
            fsc += 1
    log.info("  fuel_supplies (fuel→generators): %d", fsc)

    return G


# ══════════════════════════════════════════════════════════════════════════════
# 3. DRIVER
# ══════════════════════════════════════════════════════════════════════════════

# Two relations have no qualifying Boston edges under NYC's EXACT rules — both
# are force-created as empty so the 19-relation schema contract with NYC holds.
# Neither is a rule change; both are honest data-modeling gaps:
#   * water_flow      — Boston has no pumping-station data (Week-12 BWSC gap), so
#                       NYC's pump->treatment rule has zero qualifying nodes.
#   * fuel_supplies→telecom — NYC connects fuel only to telecom clusters with
#                       tower_count>=5 (a "big enough to have a backup generator"
#                       proxy). Boston's telecom is FCC ASR macro structures,
#                       each an individual record (tower_count=1), so none meet
#                       the >=5 threshold. We keep NYC's threshold unchanged
#                       (no retuning) and represent the absence faithfully.
FORCE_EMPTY_RELATIONS = [
    ("water", "water_flow", "water"),
    ("fuel", "fuel_supplies", "telecom"),
]


def build(save: bool = True):
    log.info("Loading Boston layers → NYC-schema nodes...")
    nodes = load_boston_nodes()
    log.info("Total Boston nodes: %d", len(nodes))

    log.info("Building directed graph with NYC edge rules...")
    G = build_graph(nodes)
    log.info("Graph: %d nodes, %d directed edges", G.number_of_nodes(), G.number_of_edges())

    log.info("Converting to PyG HeteroData (reusing NYC graph_to_heterodata)...")
    data = graph_to_heterodata(G, verbose=True, force_edge_types=FORCE_EMPTY_RELATIONS)

    if save:
        boston.GRAPH_DIR.mkdir(parents=True, exist_ok=True)
        import torch
        torch.save(data, OUT_PATH)
        log.info("Saved Boston HeteroData → %s", OUT_PATH)
    return data


if __name__ == "__main__":
    from src.cities.boston_ingest._logging import configure as configure_logging
    configure_logging()
    build()
