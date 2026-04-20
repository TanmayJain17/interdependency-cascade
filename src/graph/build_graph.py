"""
build_graph_nyc.py  (v1 — citywide expansion)
=============================================
Builds a DIRECTED heterogeneous infrastructure graph for all of New York City.

This is the citywide counterpart to build_graph.py (which covers Lower Manhattan
only). Same edge logic, same dependency semantics, same buffer durations. The
LM build_graph.py remains untouched so both pipelines can coexist.

Differences from LM build_graph.py:
  [IN]    Reads *_nyc.geojson files instead of *_lm.geojson
  [IN]    Telecom raw from cell_towers_nyc.geojson (20,190 towers)
  [IN]    Gas stations from gas_stations_nyc.geojson (1,181 stations)
  [SCALE] Hospital -> power dependency edges capped at 5 km (citywide nonsense
          guard: a Staten Island hospital must not depend on a Bronx substation)
  [SCALE] Same 5 km cap applied to water -> hospital, fuel -> *, etc. to keep
          all inter-infrastructure edges spatially sane
  [NOTE]  NJ substations (west of Hudson) stay in the graph marked external=True.
          They physically connect to NYC via the transmission network so they
          cannot be dropped without breaking cascade realism.
  [OUT]   Writes nyc_infra_graph.graphml / nyc_infra_nodes.geojson /
          nyc_infra_edges.geojson in data/graph/
"""

import os
import json
import geopandas as gpd
import pandas as pd
import networkx as nx
import numpy as np
from shapely.geometry import Point, LineString, box
from pyproj import Transformer

os.makedirs("data/graph", exist_ok=True)

# ── Coordinate helpers ───────────────────────────────────────────────────────
to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32618", always_xy=True)


def utm(lon, lat):
    return to_utm.transform(lon, lat)


def dist_m(lon1, lat1, lon2, lat2):
    e1, n1 = utm(lon1, lat1)
    e2, n2 = utm(lon2, lat2)
    return ((e1 - e2) ** 2 + (n1 - n2) ** 2) ** 0.5


# ══════════════════════════════════════════════════════════════════════════════
# Configuration constants
# ══════════════════════════════════════════════════════════════════════════════

# NYC bounding box — used for marking NJ substations as external
NYC_BBOX = {
    "min_lat": 40.495, "max_lat": 40.920,
    "min_lon": -74.260, "max_lon": -73.700,
}

# Rough Hudson River longitude boundary (east of this = NYC, west = NJ)
# Manhattan's west edge is about -74.018; we use -74.03 to keep Hoboken/Jersey
# City substations flagged as external.
NJ_BOUNDARY_LON = -74.03

# Telecom grid cell size in decimal degrees. 0.005 ~= 500 m.
# At citywide scale: ~20k towers -> ~1,600 clusters at 500 m grid.
TELECOM_GRID = 0.005

# Distance caps for inter-infrastructure proximity edges (meters).
# Citywide, unbounded nearest-neighbor produces cross-borough nonsense.
MAX_POWER_DEPENDENCY_M = 5_000     # hospital/water/telecom/fuel/subway <- power
MAX_WATER_SUPPLY_M = 5_000          # hospital <- water
MAX_FUEL_SUPPLY_M = 5_000           # hospital/telecom/water <- fuel
MAX_FUEL_DISTRIBUTION_M = 50_000    # gas station <- terminal (supply chains are long)
MAX_SCADA_M = 5_000                 # power <- telecom
MAX_REPAIR_ACCESS_M = 5_000         # everything <- subway

# Buffer durations (hours)
BUFFER_HOURS = {
    "telecom": 6.0,
    "hospital": 96.0,
    "subway": 0.0,
    "water": 3.0,
    "fuel": 0.0,
}
SCADA_BUFFER_HOURS = 2.0
REPAIR_BUFFER_HOURS = 0.0
WATER_SUPPLY_BUFFER = 24.0

FUEL_TERMINAL_POWER_BUFFER = 4.0
FUEL_DISTRIBUTION_BUFFER = 0.0
FUEL_TO_HOSPITAL_BUFFER = 96.0
FUEL_TO_TELECOM_BUFFER = 48.0
FUEL_TO_WATER_BUFFER = 48.0


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD ALL DATASETS (citywide versions)
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 72)
print("build_graph_nyc.py — citywide heterogeneous infrastructure graph")
print("=" * 72)
print("\nLoading citywide datasets...")

# ── Power ────────────────────────────────────────────────────────────────────
power_sub_all = gpd.read_file("data/power/substations_nyc.geojson")
power_lines = gpd.read_file("data/power/transmission_lines_nyc.geojson")
print(f"  Power: {len(power_sub_all)} substations, {len(power_lines)} transmission lines")

# ── Telecom: cluster by 500m grid citywide ───────────────────────────────────
telecom_raw = gpd.read_file("data/telecom/cell_towers_nyc.geojson")
OP_MAP = {(310, 260): "T-Mobile", (310, 410): "AT&T", (310, 240): "T-Mobile Metro"}
telecom_raw["operator"] = telecom_raw.apply(
    lambda r: OP_MAP.get((int(r["mcc"]), int(r["net"])), f'{int(r["mcc"])}/{int(r["net"])}'),
    axis=1,
)

telecom_raw["grid_lat"] = (telecom_raw["lat"] / TELECOM_GRID).round() * TELECOM_GRID
telecom_raw["grid_lon"] = (telecom_raw["lon"] / TELECOM_GRID).round() * TELECOM_GRID
telecom = (
    telecom_raw.groupby(["operator", "grid_lat", "grid_lon"])
    .agg(
        lat=("lat", "mean"),
        lon=("lon", "mean"),
        tower_count=("lat", "count"),
        radio_types=("radio", lambda x: ",".join(sorted(x.unique()))),
    )
    .reset_index()
    .drop(columns=["grid_lat", "grid_lon"])
)
print(
    f"  Telecom: {len(telecom_raw):,} towers → {len(telecom):,} grid clusters "
    f"(~{int(TELECOM_GRID * 111_000)}m grid)"
)

# ── Hospitals ────────────────────────────────────────────────────────────────
hospitals_raw = gpd.read_file("data/healthcare/hospitals_nyc.geojson")
hospitals = (
    hospitals_raw
    .sort_values("FACTYPE")
    .drop_duplicates(subset=["LATITUDE", "LONGITUDE"], keep="first")
    .reset_index(drop=True)
)
print(f"  Hospitals: {len(hospitals_raw)} → {len(hospitals)} after dedup")

# ── Subway ───────────────────────────────────────────────────────────────────
subway_raw = gpd.read_file("data/transit/subway_stations_nyc.geojson")
subway = (
    subway_raw.groupby(["GTFS Latitude", "GTFS Longitude"], as_index=False)
    .agg(
        stop_name=("Stop Name", "first"),
        routes=("Daytime Routes", lambda x: " ".join(sorted(set(" ".join(x.dropna()).split())))),
        ada=("ADA", "max"),
        division=("Division", "first"),
    )
    .rename(columns={"GTFS Latitude": "lat", "GTFS Longitude": "lon"})
)
print(f"  Subway: {len(subway_raw)} → {len(subway)} after dedup")

# ── Water ────────────────────────────────────────────────────────────────────
water_raw = gpd.read_file("data/water/water_infra_nyc.geojson")
EXCLUDE_WATER = ["STEPHEN A SCHWARZMAN BLDG-NYPL"]
water = (
    water_raw[~water_raw["name"].isin(EXCLUDE_WATER)]
    .drop_duplicates(subset=["lat", "lon"])
    .reset_index(drop=True)
)
# Mark DEP WWTPs as external (they're major regional assets, not borough-specific)
# Everything else gets external=False
if "source" in water.columns:
    water["external"] = water["source"] == "nyc_dep"
else:
    water["external"] = False
print(f"  Water: {len(water)} facilities")

# ── Fuel ─────────────────────────────────────────────────────────────────────
fuel_stations = gpd.read_file("data/fuel/gas_stations_nyc.geojson")
fuel_stations["external"] = False  # all NYC gas stations are internal
fuel_terminals = gpd.read_file("data/fuel/petroleum_terminals_nyc.geojson")
fuel_terminals["external"] = True  # terminals are upstream/regional
print(
    f"  Fuel: {len(fuel_stations)} gas stations + {len(fuel_terminals)} terminals "
    f"= {len(fuel_stations) + len(fuel_terminals)} total"
)


# ══════════════════════════════════════════════════════════════════════════════
# 2. BUILD UNIFIED NODE TABLE
# ══════════════════════════════════════════════════════════════════════════════

print("\nBuilding node table...")


def slug(s):
    s = str(s)
    for ch in [" ", "-", "/", ",", "(", ")", "'", ".", "&"]:
        s = s.replace(ch, "_")
    return s.lower().strip("_")


def in_nyc(lat, lon):
    return (
        NYC_BBOX["min_lat"] <= lat <= NYC_BBOX["max_lat"]
        and NYC_BBOX["min_lon"] <= lon <= NYC_BBOX["max_lon"]
    )


nodes = []

# ── Power substations ────────────────────────────────────────────────────────
# All 208 from HIFLD go in. NJ substations (west of Hudson) are flagged external.
substation_in_graph = set()  # track names we've added to avoid dupes with line endpoints
for _, r in power_sub_all.iterrows():
    name = str(r["NAME"]).upper().strip()
    lat = float(r["LATITUDE"])
    lon = float(r["LONGITUDE"])
    is_external = lon < NJ_BOUNDARY_LON  # NJ side of Hudson
    nodes.append(
        dict(
            node_id=f"power_{slug(name)}",
            name=name.title(),
            infra_type="power",
            subtype=str(r["TYPE"]),
            lat=lat,
            lon=lon,
            status=str(r["STATUS"]),
            external=is_external,
        )
    )
    substation_in_graph.add(name)

# Add any extra substations referenced by transmission lines but not in the
# substations file (some lines reference SUB_1/SUB_2 names that don't appear as
# point features)
extra_subs = set()
for _, row in power_lines.iterrows():
    for sub_col in ["SUB_1", "SUB_2"]:
        name = str(row[sub_col]).upper().strip()
        if name and name != "NOT AVAILABLE" and name not in substation_in_graph:
            extra_subs.add(name)

if extra_subs:
    # Try to find coords in the full substations file by name; otherwise skip
    sub_lookup = {str(r["NAME"]).upper().strip(): r for _, r in power_sub_all.iterrows()}
    for name in extra_subs:
        r = sub_lookup.get(name)
        if r is not None:
            # Already handled above — defensive
            continue
        # We don't have coords; these are external and will be referenced by edges
        # only, not become point-based nodes. Skip.
        pass

n_power_ext = sum(1 for n in nodes if n["infra_type"] == "power" and n["external"])
n_power_int = sum(1 for n in nodes if n["infra_type"] == "power" and not n["external"])
print(f"  Power: {n_power_int} NYC + {n_power_ext} NJ/external")

# ── Telecom clusters ─────────────────────────────────────────────────────────
for i, r in telecom.iterrows():
    nodes.append(
        dict(
            node_id=f"telecom_cluster_{i:05d}",
            name=f"{r['operator']} cluster",
            infra_type="telecom",
            subtype="CELL_CLUSTER",
            lat=float(r["lat"]),
            lon=float(r["lon"]),
            tower_count=int(r["tower_count"]),
            radio_types=str(r["radio_types"]),
            operator=str(r["operator"]),
            external=False,
        )
    )

# ── Hospitals ────────────────────────────────────────────────────────────────
for _, r in hospitals.iterrows():
    nodes.append(
        dict(
            node_id=f"hospital_{slug(str(r['FACNAME'])[:40])}",
            name=str(r["FACNAME"]).title(),
            infra_type="hospital",
            subtype=str(r["FACTYPE"]),
            lat=float(r["LATITUDE"]),
            lon=float(r["LONGITUDE"]),
            address=str(r.get("ADDRESS", "")),
            external=False,
        )
    )

# ── Subway ───────────────────────────────────────────────────────────────────
for _, r in subway.iterrows():
    routes_clean = str(r["routes"]).replace(" ", "") if r["routes"] else ""
    nodes.append(
        dict(
            node_id=f"subway_{slug(r['stop_name'])}_{slug(routes_clean)}",
            name=str(r["stop_name"]),
            infra_type="subway",
            subtype="STATION",
            lat=float(r["lat"]),
            lon=float(r["lon"]),
            routes=str(r["routes"]),
            ada=int(r["ada"]),
            external=False,
        )
    )

# ── Water ────────────────────────────────────────────────────────────────────
for _, r in water.iterrows():
    nodes.append(
        dict(
            node_id=f"water_{slug(str(r['name'])[:40])}",
            name=str(r["name"]).title(),
            infra_type="water",
            subtype=str(r["type"]),
            lat=float(r["lat"]),
            lon=float(r["lon"]),
            operator=str(r.get("operator", "")),
            external=bool(r["external"]),
        )
    )

# ── Fuel — Gas Stations ──────────────────────────────────────────────────────
for i, r in fuel_stations.iterrows():
    name = str(r.get("name", "Gas Station"))
    brand = str(r.get("brand", ""))
    display_name = f"{name} ({brand})" if brand and brand != "nan" else name
    nodes.append(
        dict(
            node_id=f"fuel_station_{slug(name)[:30]}_{i:04d}",
            name=display_name,
            infra_type="fuel",
            subtype="GAS_STATION",
            lat=float(r["lat"]),
            lon=float(r["lon"]),
            brand=brand,
            external=False,
        )
    )

# ── Fuel — Petroleum Terminals ───────────────────────────────────────────────
for _, r in fuel_terminals.iterrows():
    nodes.append(
        dict(
            node_id=f"fuel_terminal_{slug(str(r['name'])[:40])}",
            name=str(r["name"]),
            infra_type="fuel",
            subtype="PETROLEUM_TERMINAL",
            lat=float(r["lat"]),
            lon=float(r["lon"]),
            capacity_bbl=int(r.get("capacity_bbl", 0)) if pd.notna(r.get("capacity_bbl")) else 0,
            external=True,
        )
    )

# ── Deduplicate node_ids ─────────────────────────────────────────────────────
seen = set()
unique_nodes = []
for n in nodes:
    if n["node_id"] not in seen:
        seen.add(n["node_id"])
        unique_nodes.append(n)
nodes = unique_nodes

print(f"\n  Total nodes: {len(nodes):,}")
for it in ["power", "telecom", "hospital", "subway", "water", "fuel"]:
    cnt = sum(1 for n in nodes if n["infra_type"] == it)
    ext = sum(1 for n in nodes if n["infra_type"] == it and n.get("external"))
    print(f"    {it:10s}: {cnt:5d}  ({ext} external)")


# ══════════════════════════════════════════════════════════════════════════════
# 3. INITIALISE DIRECTED GRAPH
# ══════════════════════════════════════════════════════════════════════════════

G = nx.DiGraph()
for n in nodes:
    G.add_node(
        n["node_id"], **{k: v for k, v in n.items() if k != "node_id" and v is not None}
    )

node_lookup = {n["node_id"]: n for n in nodes}


# ══════════════════════════════════════════════════════════════════════════════
# 4. EDGES
# ══════════════════════════════════════════════════════════════════════════════

edges = []


def add_edge(u, v, edge_type, weight=1.0, **attrs):
    if G.has_node(u) and G.has_node(v) and u != v:
        G.add_edge(u, v, edge_type=edge_type, weight=weight, **attrs)
        edges.append(dict(u=u, v=v, edge_type=edge_type, weight=weight, **attrs))


def add_bidir_edge(u, v, edge_type, weight=1.0, **attrs):
    add_edge(u, v, edge_type, weight, **attrs)
    add_edge(v, u, edge_type, weight, **attrs)


def nearest_within_cap(src_nodes, target_node, max_m):
    """
    Find the nearest node in src_nodes to target_node, within max_m meters.
    Returns (node, distance_m) or (None, None) if no node is within cap.
    """
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


# ── 4a. Power transmission lines (bidirectional) ─────────────────────────────
VOLT_WEIGHT = {"345": 3.0, "220-287": 2.0, "100-161": 1.5, "UNDER 100": 1.0,
               "500": 5.0, "NOT AVAILABLE": 1.0}

line_edge_count = 0
line_endpoint_misses = 0
for _, row in power_lines.iterrows():
    u_name = slug(str(row["SUB_1"]).upper().strip())
    v_name = slug(str(row["SUB_2"]).upper().strip())
    u = f"power_{u_name}"
    v = f"power_{v_name}"
    if not (G.has_node(u) and G.has_node(v)):
        line_endpoint_misses += 1
        continue
    volt = str(row["VOLT_CLASS"])
    w = VOLT_WEIGHT.get(volt, 1.0)
    add_bidir_edge(
        u, v, "power_line",
        weight=w, volt_class=volt, line_type=str(row["TYPE"]),
        buffer_hours=0.0,
    )
    line_edge_count += 1

print(f"\nPower transmission lines:")
print(f"  Physical lines added:    {line_edge_count}")
print(f"  Dropped (endpoint miss): {line_endpoint_misses}")
print(f"  Directed edges:          {line_edge_count * 2}")


# ── 4b. Subway lines (bidirectional) ─────────────────────────────────────────
line_stations = {}
for n in nodes:
    if n["infra_type"] == "subway" and n.get("routes"):
        for line in str(n["routes"]).split():
            line_stations.setdefault(line, []).append(n["node_id"])

subway_edge_count = 0
for line, station_ids in line_stations.items():
    coords = [(nid, node_lookup[nid]["lon"], node_lookup[nid]["lat"]) for nid in station_ids]
    lons = [c[1] for c in coords]
    lats = [c[2] for c in coords]
    # Sort stations along the line's primary axis
    if (max(lons) - min(lons)) > (max(lats) - min(lats)):
        coords.sort(key=lambda x: x[1])
    else:
        coords.sort(key=lambda x: x[2])
    for i in range(len(coords) - 1):
        u, u_lon, u_lat = coords[i]
        v, v_lon, v_lat = coords[i + 1]
        d = dist_m(u_lon, u_lat, v_lon, v_lat)
        # Guard against crazy jumps: if consecutive stations on a "line" are > 10km
        # apart, they're probably the line splitting into branches — skip.
        if d > 10_000:
            continue
        add_bidir_edge(
            u, v, "subway_line",
            weight=round(d / 1000, 3), line=line,
            distance_m=round(d), buffer_hours=0.0,
        )
        subway_edge_count += 1

print(f"Subway physical links: {subway_edge_count} → {subway_edge_count * 2} directed")


# ── 4c. Power → dependent infrastructure ─────────────────────────────────────
# NYC internal power nodes serve as the source pool
local_power = [n for n in nodes if n["infra_type"] == "power" and not n.get("external")]
# External power (NJ etc.) can also feed NYC nodes — some NJ substations do feed NYC
# via transmission. Keep them in the candidate pool.
all_power_candidates = [n for n in nodes if n["infra_type"] == "power"]

dep_targets = [
    n for n in nodes
    if n["infra_type"] in ("hospital", "water", "telecom", "subway", "fuel")
    and not n.get("external")
]

power_dep_count = 0
power_dep_dropped = 0
for tn in dep_targets:
    nearest, d = nearest_within_cap(all_power_candidates, tn, MAX_POWER_DEPENDENCY_M)
    if nearest is None:
        power_dep_dropped += 1
        continue
    buf = BUFFER_HOURS.get(tn["infra_type"], 0.0)
    add_edge(
        nearest["node_id"], tn["node_id"], "power_dependency",
        weight=round(1 / (1 + d / 1000), 3),
        distance_m=round(d), buffer_hours=buf,
        dependency_class="physical",
    )
    power_dep_count += 1

# External fuel terminals — give them power dependency with a longer cap since
# they're meant to be out in the harbor. Use 20 km so a terminal on the Bayonne
# side can still pair with its nearest NYC substation.
ext_fuel_terminals = [
    n for n in nodes
    if n["infra_type"] == "fuel" and n.get("subtype") == "PETROLEUM_TERMINAL"
]
for tn in ext_fuel_terminals:
    nearest, d = nearest_within_cap(all_power_candidates, tn, 20_000)
    if nearest is None:
        continue
    add_edge(
        nearest["node_id"], tn["node_id"], "power_dependency",
        weight=round(1 / (1 + d / 1000), 3),
        distance_m=round(d),
        buffer_hours=FUEL_TERMINAL_POWER_BUFFER,
        dependency_class="physical",
    )
    power_dep_count += 1

print(f"\nPower dependency edges (cap {MAX_POWER_DEPENDENCY_M}m): {power_dep_count}")
print(f"  Dropped (no substation within cap): {power_dep_dropped}")
for itype in ["telecom", "hospital", "subway", "water", "fuel"]:
    cnt = sum(
        1 for e in edges
        if e["edge_type"] == "power_dependency"
        and node_lookup.get(e["v"], {}).get("infra_type") == itype
    )
    buf = BUFFER_HOURS.get(itype, 0)
    print(f"  {itype:10s}: {cnt:5d} edges, buffer = {buf:.1f} h")


# ── 4d. Water flow: pump → treatment plant (≤ 5 km) ─────────────────────────
pump_types = {
    "PUMPING STATION", "WASTEWATER PUMPING STATION", "STORMWATER PUMPING STATION",
}
treatment_types = {
    "WATER POLLUTION CONTROL PLANT", "WASTEWATER TREATMENT PLANT",
}

pumps = [n for n in nodes if n["infra_type"] == "water" and n["subtype"] in pump_types]
treatments = [n for n in nodes if n["infra_type"] == "water" and n["subtype"] in treatment_types]

water_flow_count = 0
for pump in pumps:
    for plant in treatments:
        d = dist_m(pump["lon"], pump["lat"], plant["lon"], plant["lat"])
        if d <= MAX_WATER_SUPPLY_M:
            add_edge(
                pump["node_id"], plant["node_id"], "water_flow",
                weight=round(d / 1000, 3), distance_m=round(d),
                buffer_hours=0.0, dependency_class="physical",
            )
            water_flow_count += 1

print(f"Water flow edges (pump→plant, cap {MAX_WATER_SUPPLY_M}m): {water_flow_count}")


# ── 4e. Water → hospital (with buffer) ───────────────────────────────────────
all_water_nodes = [n for n in nodes if n["infra_type"] == "water"]
hosp_nodes = [n for n in nodes if n["infra_type"] == "hospital"]

water_supply_count = 0
water_supply_dropped = 0
for hn in hosp_nodes:
    nearest, d = nearest_within_cap(all_water_nodes, hn, MAX_WATER_SUPPLY_M)
    if nearest is None:
        water_supply_dropped += 1
        continue
    add_edge(
        nearest["node_id"], hn["node_id"], "water_supplies",
        weight=round(1 / (1 + d / 1000), 3), distance_m=round(d),
        buffer_hours=WATER_SUPPLY_BUFFER, dependency_class="physical",
    )
    water_supply_count += 1

print(f"Water → hospital edges (cap {MAX_WATER_SUPPLY_M}m): {water_supply_count}")
print(f"  Dropped: {water_supply_dropped}")


# ── 4f. Telecom → power (SCADA reverse dependency) ───────────────────────────
telecom_nodes = [n for n in nodes if n["infra_type"] == "telecom"]

scada_count = 0
scada_dropped = 0
for pn in local_power:
    nearest, d = nearest_within_cap(telecom_nodes, pn, MAX_SCADA_M)
    if nearest is None:
        scada_dropped += 1
        continue
    add_edge(
        nearest["node_id"], pn["node_id"], "scada_monitoring",
        weight=round(1 / (1 + d / 1000), 3), distance_m=round(d),
        buffer_hours=SCADA_BUFFER_HOURS, dependency_class="cyber",
    )
    scada_count += 1

print(f"SCADA (telecom→power) edges (cap {MAX_SCADA_M}m): {scada_count}, dropped {scada_dropped}")


# ── 4g. Subway → infrastructure (repair access, recovery layer) ──────────────
subway_nodes = [n for n in nodes if n["infra_type"] == "subway"]
repair_targets = [
    n for n in nodes
    if n["infra_type"] in ("power", "hospital", "water", "telecom", "fuel")
    and not n.get("external")
]

repair_count = 0
repair_dropped = 0
for tn in repair_targets:
    nearest, d = nearest_within_cap(subway_nodes, tn, MAX_REPAIR_ACCESS_M)
    if nearest is None:
        repair_dropped += 1
        continue
    add_edge(
        nearest["node_id"], tn["node_id"], "repair_access",
        weight=round(1 / (1 + d / 1000), 3), distance_m=round(d),
        buffer_hours=REPAIR_BUFFER_HOURS, dependency_class="logical",
        layer="recovery",
    )
    repair_count += 1

print(f"Repair access edges (cap {MAX_REPAIR_ACCESS_M}m): {repair_count}, dropped {repair_dropped}")


# ── 4h. Fuel distribution: terminal → gas station (cap 50 km) ────────────────
fuel_station_nodes = [
    n for n in nodes if n["infra_type"] == "fuel" and n.get("subtype") == "GAS_STATION"
]
fuel_terminal_nodes = [
    n for n in nodes if n["infra_type"] == "fuel" and n.get("subtype") == "PETROLEUM_TERMINAL"
]

fuel_dist_count = 0
for sn in fuel_station_nodes:
    nearest, d = nearest_within_cap(fuel_terminal_nodes, sn, MAX_FUEL_DISTRIBUTION_M)
    if nearest is None:
        continue
    add_edge(
        nearest["node_id"], sn["node_id"], "fuel_distribution",
        weight=round(1 / (1 + d / 1000), 3), distance_m=round(d),
        buffer_hours=FUEL_DISTRIBUTION_BUFFER, dependency_class="physical",
    )
    fuel_dist_count += 1

print(f"Fuel distribution (terminal→station, cap {MAX_FUEL_DISTRIBUTION_M}m): {fuel_dist_count}")


# ── 4i. Fuel → generators (hospital, telecom, water) ─────────────────────────
fuel_supply_count = 0

# Fuel → Hospital
for hn in hosp_nodes:
    nearest, d = nearest_within_cap(fuel_station_nodes, hn, MAX_FUEL_SUPPLY_M)
    if nearest is None:
        continue
    add_edge(
        nearest["node_id"], hn["node_id"], "fuel_supplies",
        weight=round(1 / (1 + d / 1000), 3), distance_m=round(d),
        buffer_hours=FUEL_TO_HOSPITAL_BUFFER, dependency_class="physical",
    )
    fuel_supply_count += 1

# Fuel → Telecom (only clusters with >= 5 towers, likely to have generators)
telecom_with_generators = [
    n for n in telecom_nodes if float(n.get("tower_count", 0)) >= 5
]
for tn in telecom_with_generators:
    nearest, d = nearest_within_cap(fuel_station_nodes, tn, MAX_FUEL_SUPPLY_M)
    if nearest is None:
        continue
    add_edge(
        nearest["node_id"], tn["node_id"], "fuel_supplies",
        weight=round(1 / (1 + d / 1000), 3), distance_m=round(d),
        buffer_hours=FUEL_TO_TELECOM_BUFFER, dependency_class="physical",
    )
    fuel_supply_count += 1

# Fuel → Water (non-external water nodes only)
water_internal = [n for n in nodes if n["infra_type"] == "water" and not n.get("external")]
for wn in water_internal:
    nearest, d = nearest_within_cap(fuel_station_nodes, wn, MAX_FUEL_SUPPLY_M)
    if nearest is None:
        continue
    add_edge(
        nearest["node_id"], wn["node_id"], "fuel_supplies",
        weight=round(1 / (1 + d / 1000), 3), distance_m=round(d),
        buffer_hours=FUEL_TO_WATER_BUFFER, dependency_class="physical",
    )
    fuel_supply_count += 1

print(f"Fuel supply edges (fuel → generators, cap {MAX_FUEL_SUPPLY_M}m): {fuel_supply_count}")
print(f"  fuel → hospital: {sum(1 for e in edges if e['edge_type']=='fuel_supplies' and node_lookup.get(e['v'],{}).get('infra_type')=='hospital')}")
print(f"  fuel → telecom:  {sum(1 for e in edges if e['edge_type']=='fuel_supplies' and node_lookup.get(e['v'],{}).get('infra_type')=='telecom')}")
print(f"  fuel → water:    {sum(1 for e in edges if e['edge_type']=='fuel_supplies' and node_lookup.get(e['v'],{}).get('infra_type')=='water')}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. SUMMARY & DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════════════

from collections import Counter

print(f"\n{'=' * 72}")
print(f"Graph summary: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} directed edges")
print(f"{'=' * 72}")

etype_counts = Counter(d["edge_type"] for _, _, d in G.edges(data=True))
for et, cnt in sorted(etype_counts.items()):
    print(f"  {et:25s}: {cnt:6d}")

cascade_edges = sum(
    1 for _, _, d in G.edges(data=True) if d.get("layer") != "recovery"
)
recovery_edges = sum(
    1 for _, _, d in G.edges(data=True) if d.get("layer") == "recovery"
)
print(f"\n  Cascade layer edges:  {cascade_edges:,}")
print(f"  Recovery layer edges: {recovery_edges:,}")

print("\n  Buffer duration distribution:")
buf_counter = Counter(d.get("buffer_hours", 0) for _, _, d in G.edges(data=True))
for buf, cnt in sorted(buf_counter.items()):
    print(f"    {buf:6.1f} h : {cnt:6d} edges")


# ══════════════════════════════════════════════════════════════════════════════
# 6. SAVE OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════

print("\nSaving outputs...")

# GraphML
nx.write_graphml(G, "data/graph/nyc_infra_graph.graphml")
print("  Saved → data/graph/nyc_infra_graph.graphml")

# Nodes GeoJSON
node_records = []
for nid, attrs in G.nodes(data=True):
    if attrs.get("lat") is not None and attrs.get("lon") is not None:
        node_records.append({**attrs, "node_id": nid})

nodes_gdf = gpd.GeoDataFrame(
    node_records,
    geometry=[Point(r["lon"], r["lat"]) for r in node_records],
    crs="EPSG:4326",
)
nodes_gdf.to_file("data/graph/nyc_infra_nodes.geojson", driver="GeoJSON")
print(f"  Saved → data/graph/nyc_infra_nodes.geojson  ({len(nodes_gdf):,} nodes)")

# Edges GeoJSON
edge_records = []
for u, v, attrs in G.edges(data=True):
    un = node_lookup.get(u, {})
    vn = node_lookup.get(v, {})
    if un.get("lat") is not None and vn.get("lat") is not None:
        geom = LineString([(un["lon"], un["lat"]), (vn["lon"], vn["lat"])])
        edge_records.append({**attrs, "u": u, "v": v, "geometry": geom})

edges_gdf = gpd.GeoDataFrame(edge_records, crs="EPSG:4326")
edges_gdf.to_file("data/graph/nyc_infra_edges.geojson", driver="GeoJSON")
print(f"  Saved → data/graph/nyc_infra_edges.geojson  ({len(edges_gdf):,} edges)")


# ══════════════════════════════════════════════════════════════════════════════
# 7. FINAL DIAGNOSTIC
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("GRAPH DIAGNOSTICS")
print("=" * 72)

wcc = nx.number_weakly_connected_components(G)
print(f"  Weakly connected components: {wcc}")
if wcc > 1:
    sizes = sorted(
        (len(c) for c in nx.weakly_connected_components(G)), reverse=True
    )
    print(f"    Top-5 component sizes: {sizes[:5]}")
    print(f"    Smallest component size: {sizes[-1]}")

print(f"\n  Node count by infrastructure type and scope:")
for itype in ["power", "telecom", "hospital", "subway", "water", "fuel"]:
    internal = sum(1 for n in nodes if n["infra_type"] == itype and not n.get("external"))
    external = sum(1 for n in nodes if n["infra_type"] == itype and n.get("external"))
    total = internal + external
    print(f"    {itype:10s}: {total:5d}  ({internal} NYC / {external} external)")

in_deg = dict(G.in_degree())
out_deg = dict(G.out_degree())

print(f"\n  Top-5 by OUT-degree (sources of cascade):")
for nid, deg in sorted(out_deg.items(), key=lambda x: -x[1])[:5]:
    itype = G.nodes[nid].get("infra_type", "?")
    name = G.nodes[nid].get("name", nid)
    print(f"    [{itype:8s}]  {name[:45]:<45s}  out-degree={deg}")

print(f"\n  Top-5 by IN-degree (cascade sinks):")
for nid, deg in sorted(in_deg.items(), key=lambda x: -x[1])[:5]:
    itype = G.nodes[nid].get("infra_type", "?")
    name = G.nodes[nid].get("name", nid)
    print(f"    [{itype:8s}]  {name[:45]:<45s}  in-degree={deg}")

print(f"\n  Isolated nodes (no edges): "
      f"{sum(1 for n in G.nodes if G.degree(n) == 0)}")

print("\nDone. Next step: flood_overlay_v3.py with NODES_IN=nyc_infra_nodes.geojson")