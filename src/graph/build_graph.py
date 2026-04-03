"""
build_graph.py  (v4 — adds fuel/petroleum as 6th infrastructure type)
==============
Builds a DIRECTED heterogeneous infrastructure graph for Lower Manhattan.

Changes from v3:
  [NEW]  Fuel infrastructure as 6th node type with two subtypes:
         - GAS_STATION:        local retail fuel outlets (need power to pump)
         - PETROLEUM_TERMINAL: external bulk storage (NY Harbor supply nodes)

  [NEW]  Fuel-related edge types:
         - fuel_distribution:  terminal → gas_station (supply chain)
         - power → fuel:       power_dependency (station=0h, terminal=4h buffer)
         - fuel → hospital:    fuel_supplies (models generator refueling dependency)
         - fuel → telecom:     fuel_supplies (models generator refueling dependency)
         - fuel → water:       fuel_supplies (models diesel pump refueling)

  Motivation (from Dr. Miura's Week 3 discussion):
         During Sandy, 67% of NYC gas stations had no fuel (EIA survey).
         The cascade: port closed → terminals lost power → trucks couldn't load
         → stations ran dry → hospital generators couldn't refuel.
         This creates a TIME-DELAYED FEEDBACK LOOP that makes the 96h hospital
         generator buffer finite: without fuel resupply, hospitals WILL fail.

Prior changes (v3):
  [FIX 1] nx.Graph → nx.DiGraph — cascade propagation is directional
  [FIX 2] Every inter-infrastructure edge gets buffer_duration_hours
  [FIX 3] Telecom grid 250m → 500m to reduce node imbalance
"""

import os, json
import geopandas as gpd
import pandas as pd
import networkx as nx
import numpy as np
from shapely.geometry import Point, LineString
from pyproj import Transformer

os.makedirs("data/graph", exist_ok=True)

# ── coordinate helpers ────────────────────────────────────────────────────────
to_utm   = Transformer.from_crs("EPSG:4326", "EPSG:32618", always_xy=True)
to_wgs84 = Transformer.from_crs("EPSG:32618", "EPSG:4326", always_xy=True)

def utm(lon, lat):
    return to_utm.transform(lon, lat)

def dist_m(lon1, lat1, lon2, lat2):
    e1, n1 = utm(lon1, lat1)
    e2, n2 = utm(lon2, lat2)
    return ((e1-e2)**2 + (n1-n2)**2)**0.5


# ══════════════════════════════════════════════════════════════════════════════
# Buffer duration constants (hours)
# ══════════════════════════════════════════════════════════════════════════════

BUFFER_HOURS = {
    "telecom":  6.0,    # battery backup: 4-8 hr range, median 6
    "hospital": 96.0,   # NFPA 110 generator fuel requirement
    "subway":   0.0,    # traction power loss is immediate
    "water":    3.0,    # pump reserves: 2-4 hr range, median 3
    "fuel":     0.0,    # gas stations: no backup — can't pump without grid power
}

SCADA_BUFFER_HOURS  = 2.0   # local SCADA cache before operators go blind
REPAIR_BUFFER_HOURS = 0.0   # access constraint — not a failure propagation delay
WATER_SUPPLY_BUFFER = 24.0  # hospitals have ~1 day of stored water

# ── Fuel-specific buffer durations ────────────────────────────────────────────
FUEL_TERMINAL_POWER_BUFFER = 4.0    # terminals have some backup generators (~4h)
FUEL_DISTRIBUTION_BUFFER   = 0.0    # supply chain: if terminal down, no truck loading
FUEL_TO_HOSPITAL_BUFFER    = 96.0   # hospital generators have 96h fuel reserve
FUEL_TO_TELECOM_BUFFER     = 48.0   # telecom generators: ~24-72h, median 48h
FUEL_TO_WATER_BUFFER       = 48.0   # diesel pump backup: ~48h


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD & CLEAN ALL DATASETS
# ══════════════════════════════════════════════════════════════════════════════

print("Loading datasets…")

# ── Power ─────────────────────────────────────────────────────────────────────
power_sub   = gpd.read_file("data/power/substations_lower_manhattan.geojson")
power_lines = gpd.read_file("data/power/transmission_lines_lower_manhattan.geojson")

# ── Telecom: cluster by ~500m grid ────────────────────────────────────────────
telecom_raw = gpd.read_file("data/telecom/cell_towers_lower_manhattan.geojson")
OP_MAP = {(310, 260): "T-Mobile", (310, 410): "AT&T", (310, 240): "T-Mobile Metro"}
telecom_raw["operator"] = telecom_raw.apply(
    lambda r: OP_MAP.get((int(r["mcc"]), int(r["net"])), f'{r["mcc"]}/{r["net"]}'), axis=1
)

GRID = 0.005  # ~500m effective grid
telecom_raw["grid_lat"] = (telecom_raw["lat"] / GRID).round() * GRID
telecom_raw["grid_lon"] = (telecom_raw["lon"] / GRID).round() * GRID
telecom = (
    telecom_raw.groupby(["operator", "grid_lat", "grid_lon"])
    .agg(
        lat          = ("lat",      "mean"),
        lon          = ("lon",      "mean"),
        tower_count  = ("lat",      "count"),
        radio_types  = ("radio",    lambda x: ",".join(sorted(x.unique()))),
    )
    .reset_index()
    .drop(columns=["grid_lat", "grid_lon"])
)
print(f"  Telecom: {len(telecom_raw)} towers → {len(telecom)} grid clusters (~500m)")

# ── Hospitals ─────────────────────────────────────────────────────────────────
hospitals_raw = gpd.read_file("data/healthcare/hospitals_lm.geojson")
hospitals = (
    hospitals_raw
    .sort_values("FACTYPE")
    .drop_duplicates(subset=["LATITUDE", "LONGITUDE"], keep="first")
    .reset_index(drop=True)
)
print(f"  Hospitals: {len(hospitals_raw)} → {len(hospitals)} after dedup")

# ── Subway ────────────────────────────────────────────────────────────────────
subway_raw = gpd.read_file("data/transit/subway_stations_lm.geojson")
subway = (
    subway_raw
    .groupby(["GTFS Latitude", "GTFS Longitude"], as_index=False)
    .agg(
        stop_name   =("Stop Name",      "first"),
        routes      =("Daytime Routes", lambda x: " ".join(sorted(set(" ".join(x.dropna()).split())))),
        ada         =("ADA",            "max"),
        division    =("Division",       "first"),
    )
    .rename(columns={"GTFS Latitude": "lat", "GTFS Longitude": "lon"})
)
print(f"  Subway: {len(subway_raw)} → {len(subway)} after dedup")

# ── Water ─────────────────────────────────────────────────────────────────────
water_raw  = gpd.read_file("data/water/water_infra_lm.geojson")
water_srv  = gpd.read_file("data/water/water_treatment_serving_lm.geojson")

EXCLUDE_WATER = ["STEPHEN A SCHWARZMAN BLDG-NYPL"]

water_bbox = (
    water_raw[~water_raw["name"].isin(EXCLUDE_WATER)]
    .drop_duplicates(subset=["lat", "lon"])
    .reset_index(drop=True)
)
water_serving = (
    water_srv
    .drop_duplicates(subset=["lat", "lon"])
    .reset_index(drop=True)
)
water_bbox["external"]   = False
water_serving["external"] = True
srv_new = water_serving[
    ~water_serving.apply(lambda r: any(
        dist_m(r.lon, r.lat, b.lon, b.lat) < 50 for _, b in water_bbox.iterrows()
    ), axis=1)
]
water = pd.concat([water_bbox, srv_new], ignore_index=True)
print(f"  Water: {len(water_bbox)} in-bbox + {len(srv_new)} serving-LM = {len(water)} total")

# ── Fuel (NEW in v4) ─────────────────────────────────────────────────────────
# Use expanded gas stations (covers LM + immediate surroundings)
FUEL_STATIONS_PATH  = "data/fuel/gas_stations_expanded.geojson"
FUEL_TERMINALS_PATH = "data/fuel/petroleum_terminals_nyc.geojson"

LM_BBOX = {
    'min_lat': 40.700, 'max_lat': 40.755,
    'min_lon': -74.020, 'max_lon': -73.970
}

fuel_stations = gpd.GeoDataFrame()
fuel_terminals = gpd.GeoDataFrame()

if os.path.exists(FUEL_STATIONS_PATH):
    fuel_stations = gpd.read_file(FUEL_STATIONS_PATH)
    # Mark stations inside strict LM bbox as local, others as semi-external
    fuel_stations["external"] = ~(
        (fuel_stations["lat"] >= LM_BBOX['min_lat']) &
        (fuel_stations["lat"] <= LM_BBOX['max_lat']) &
        (fuel_stations["lon"] >= LM_BBOX['min_lon']) &
        (fuel_stations["lon"] <= LM_BBOX['max_lon'])
    )
    print(f"  Fuel stations: {len(fuel_stations)} "
          f"({(~fuel_stations['external']).sum()} in LM, "
          f"{fuel_stations['external'].sum()} expanded area)")
else:
    print(f"  WARNING: {FUEL_STATIONS_PATH} not found — run download_fuel.py first")

if os.path.exists(FUEL_TERMINALS_PATH):
    fuel_terminals = gpd.read_file(FUEL_TERMINALS_PATH)
    fuel_terminals["external"] = True  # all terminals are external to LM
    print(f"  Fuel terminals: {len(fuel_terminals)} (all external)")
else:
    print(f"  WARNING: {FUEL_TERMINALS_PATH} not found — run download_fuel.py first")


# ══════════════════════════════════════════════════════════════════════════════
# 2. BUILD UNIFIED NODE TABLE
# ══════════════════════════════════════════════════════════════════════════════

print("\nBuilding node table…")

def slug(s):
    return s.lower().replace(" ", "_").replace("-", "_").replace("/", "_").replace(",", "").replace("(", "").replace(")", "").replace("'", "")

nodes = []

# Power substations (LM)
for _, r in power_sub.iterrows():
    nodes.append(dict(
        node_id    = f"power_{slug(r['NAME'])}",
        name       = r["NAME"].title(),
        infra_type = "power",
        subtype    = r["TYPE"],
        lat        = float(r["LATITUDE"]),
        lon        = float(r["LONGITUDE"]),
        status     = r["STATUS"],
        external   = False,
    ))

# External power substations — look up coords from NYC-wide file
sub_nyc = gpd.read_file("data/power/substations_nyc.geojson")
sub_nyc_lookup = {r["NAME"].upper(): r for _, r in sub_nyc.iterrows()}

lm_sub_names = set(power_sub["NAME"].str.upper())
external_subs = set()
for _, row in power_lines.iterrows():
    for sub_col in ["SUB_1", "SUB_2"]:
        name = str(row[sub_col]).upper().strip()
        if name not in lm_sub_names and name not in external_subs:
            external_subs.add(name)
            nyc_match = sub_nyc_lookup.get(name)
            ext_lat = float(nyc_match["LATITUDE"])  if nyc_match is not None else None
            ext_lon = float(nyc_match["LONGITUDE"]) if nyc_match is not None else None
            if nyc_match is not None:
                print(f"    External sub coords found: {name.title()} → {ext_lat:.4f}, {ext_lon:.4f}")
            else:
                print(f"    External sub NOT in NYC file: {name.title()}")
            nodes.append(dict(
                node_id    = f"power_{slug(name)}",
                name       = name.title(),
                infra_type = "power",
                subtype    = "SUBSTATION",
                lat        = ext_lat,
                lon        = ext_lon,
                status     = "EXTERNAL",
                external   = True,
            ))
print(f"  Power: {len(power_sub)} local + {len(external_subs)} external substations")

# Telecom (geographic grid clusters)
for i, r in telecom.iterrows():
    nodes.append(dict(
        node_id     = f"telecom_cluster_{i:04d}",
        name        = f"{r['operator']} cluster",
        infra_type  = "telecom",
        subtype     = "CELL_CLUSTER",
        lat         = float(r["lat"]),
        lon         = float(r["lon"]),
        tower_count = int(r["tower_count"]),
        radio_types = str(r["radio_types"]),
        operator    = str(r["operator"]),
        external    = False,
    ))

# Hospitals
for _, r in hospitals.iterrows():
    nodes.append(dict(
        node_id    = f"hospital_{slug(r['FACNAME'][:30])}",
        name       = r["FACNAME"].title(),
        infra_type = "hospital",
        subtype    = r["FACTYPE"],
        lat        = float(r["LATITUDE"]),
        lon        = float(r["LONGITUDE"]),
        address    = str(r["ADDRESS"]),
        external   = False,
    ))

# Subway stations
for _, r in subway.iterrows():
    nodes.append(dict(
        node_id    = f"subway_{slug(r['stop_name'])}_{slug(r['routes'].replace(' ',''))}",
        name       = r["stop_name"],
        infra_type = "subway",
        subtype    = "STATION",
        lat        = float(r["lat"]),
        lon        = float(r["lon"]),
        routes     = r["routes"],
        ada        = int(r["ada"]),
        external   = False,
    ))

# Water
for _, r in water.iterrows():
    nodes.append(dict(
        node_id    = f"water_{slug(r['name'][:35])}",
        name       = r["name"].title(),
        infra_type = "water",
        subtype    = r["type"],
        lat        = float(r["lat"]),
        lon        = float(r["lon"]),
        operator   = str(r.get("operator", "")),
        external   = bool(r["external"]),
    ))

# ── Fuel — Gas Stations (NEW in v4) ──────────────────────────────────────────
for i, r in fuel_stations.iterrows():
    name = str(r.get("name", "Gas Station"))
    brand = str(r.get("brand", ""))
    display_name = f"{name} ({brand})" if brand and brand != "nan" else name
    nodes.append(dict(
        node_id    = f"fuel_station_{slug(name)}_{i:03d}",
        name       = display_name,
        infra_type = "fuel",
        subtype    = "GAS_STATION",
        lat        = float(r["lat"]),
        lon        = float(r["lon"]),
        brand      = brand,
        external   = bool(r.get("external", False)),
    ))

# ── Fuel — Petroleum Terminals (NEW in v4) ────────────────────────────────────
for _, r in fuel_terminals.iterrows():
    nodes.append(dict(
        node_id       = f"fuel_terminal_{slug(r['name'][:35])}",
        name          = str(r["name"]),
        infra_type    = "fuel",
        subtype       = "PETROLEUM_TERMINAL",
        lat           = float(r["lat"]),
        lon           = float(r["lon"]),
        capacity_bbl  = int(r.get("capacity_bbl", 0)),
        external      = True,
    ))

n_fuel_stations = len(fuel_stations)
n_fuel_terminals = len(fuel_terminals)
print(f"  Fuel: {n_fuel_stations} gas stations + {n_fuel_terminals} terminals = {n_fuel_stations + n_fuel_terminals} total")

# Deduplicate node_ids
seen, unique_nodes = set(), []
for n in nodes:
    if n["node_id"] not in seen:
        seen.add(n["node_id"])
        unique_nodes.append(n)
nodes = unique_nodes

print(f"\n  Total nodes: {len(nodes)}")
for it in ["power", "telecom", "hospital", "subway", "water", "fuel"]:
    cnt = sum(1 for n in nodes if n["infra_type"] == it)
    ext = sum(1 for n in nodes if n["infra_type"] == it and n.get("external"))
    print(f"    {it:10s}: {cnt:3d}  ({ext} external)")


# ══════════════════════════════════════════════════════════════════════════════
# 3. INITIALISE DIRECTED GRAPH
# ══════════════════════════════════════════════════════════════════════════════

G = nx.DiGraph()

for n in nodes:
    G.add_node(n["node_id"], **{k: v for k, v in n.items() if k != "node_id" and v is not None})

node_lookup = {n["node_id"]: n for n in nodes}


# ══════════════════════════════════════════════════════════════════════════════
# 4. EDGES
# ══════════════════════════════════════════════════════════════════════════════

edges = []

def add_edge(u, v, edge_type, weight=1.0, **attrs):
    """Add a directed edge u → v."""
    if G.has_node(u) and G.has_node(v) and u != v:
        G.add_edge(u, v, edge_type=edge_type, weight=weight, **attrs)
        edges.append(dict(u=u, v=v, edge_type=edge_type, weight=weight, **attrs))

def add_bidir_edge(u, v, edge_type, weight=1.0, **attrs):
    """Add edges in both directions u ↔ v (for intra-infrastructure links)."""
    add_edge(u, v, edge_type, weight, **attrs)
    add_edge(v, u, edge_type, weight, **attrs)


# ── 4a. Power lines (bidirectional) ──────────────────────────────────────────
VOLT_WEIGHT = {"345": 3.0, "100-161": 1.5, "NOT AVAILABLE": 1.0}

for _, row in power_lines.iterrows():
    u = f"power_{slug(str(row['SUB_1']).upper())}"
    v = f"power_{slug(str(row['SUB_2']).upper())}"
    volt = str(row["VOLT_CLASS"])
    w    = VOLT_WEIGHT.get(volt, 1.0)
    add_bidir_edge(u, v, "power_line", weight=w, volt_class=volt,
                   line_type=row["TYPE"], buffer_hours=0.0)

print(f"\nPower line edges: {sum(1 for e in edges if e['edge_type']=='power_line')}"
      f"  (bidirectional → {sum(1 for e in edges if e['edge_type']=='power_line')//2} physical lines)")


# ── 4b. Subway lines (bidirectional) ─────────────────────────────────────────
line_stations = {}
for n in nodes:
    if n["infra_type"] == "subway" and n.get("routes"):
        for line in n["routes"].split():
            line_stations.setdefault(line, []).append(n["node_id"])

subway_edge_count = 0
for line, station_ids in line_stations.items():
    coords = [(nid, node_lookup[nid]["lon"], node_lookup[nid]["lat"]) for nid in station_ids]
    lons = [c[1] for c in coords]
    lats = [c[2] for c in coords]
    if (max(lons)-min(lons)) > (max(lats)-min(lats)):
        coords.sort(key=lambda x: x[1])
    else:
        coords.sort(key=lambda x: x[2])
    for i in range(len(coords)-1):
        u, u_lon, u_lat = coords[i]
        v, v_lon, v_lat = coords[i+1]
        d = dist_m(u_lon, u_lat, v_lon, v_lat)
        add_bidir_edge(u, v, "subway_line", weight=round(d/1000, 3),
                       line=line, distance_m=round(d), buffer_hours=0.0)
        subway_edge_count += 1

print(f"Subway line edges: {subway_edge_count} physical links"
      f" → {subway_edge_count*2} directed edges")


# ── 4c. Power → dependent infrastructure (directed, with buffer durations) ───
local_power  = [n for n in nodes if n["infra_type"] == "power"
                and not n.get("external") and n.get("lat")]
dep_targets  = [n for n in nodes if n["infra_type"] in ("hospital", "water", "telecom", "subway", "fuel")
                and not n.get("external") and n.get("lat")]

power_dep_count = 0
for tn in dep_targets:
    dists = [(dist_m(pn["lon"], pn["lat"], tn["lon"], tn["lat"]), pn) for pn in local_power]
    d, nearest = min(dists, key=lambda x: x[0])

    # Look up buffer duration based on target infrastructure type
    # For fuel: gas stations have no backup power (buffer=0h)
    buf = BUFFER_HOURS.get(tn["infra_type"], 0.0)

    add_edge(nearest["node_id"], tn["node_id"], "power_dependency",
             weight=round(1 / (1 + d / 1000), 3),
             distance_m=round(d),
             buffer_hours=buf,
             dependency_class="physical")
    power_dep_count += 1

# Also add power → external fuel terminals (they need power for truck loading)
ext_fuel_terminals = [n for n in nodes if n["infra_type"] == "fuel"
                      and n.get("subtype") == "PETROLEUM_TERMINAL" and n.get("lat")]
# Terminals are external — connect to nearest local substation conceptually
# (they draw from regional grid, but for cascade: if NYC grid fails, terminals may too)
for tn in ext_fuel_terminals:
    if local_power:
        dists = [(dist_m(pn["lon"], pn["lat"], tn["lon"], tn["lat"]), pn) for pn in local_power]
        d, nearest = min(dists, key=lambda x: x[0])
        add_edge(nearest["node_id"], tn["node_id"], "power_dependency",
                 weight=round(1 / (1 + d / 1000), 3),
                 distance_m=round(d),
                 buffer_hours=FUEL_TERMINAL_POWER_BUFFER,
                 dependency_class="physical")
        power_dep_count += 1

print(f"Power dependency edges (directed, nearest-substation): {power_dep_count}")
print(f"  Buffer hours by target type:")
for itype in ["telecom", "hospital", "subway", "water", "fuel"]:
    cnt = sum(1 for e in edges
              if e["edge_type"] == "power_dependency"
              and node_lookup.get(e["v"], {}).get("infra_type") == itype)
    buf = BUFFER_HOURS.get(itype, 0)
    print(f"    {itype:10s}: {cnt:3d} edges, buffer = {buf:.1f} h")


# ── 4d. Water flow: pump → treatment plant (directed, ≤ 5 km) ────────────────
WATER_FLOW_RADIUS_M = 5000

pump_types     = {"PUMPING STATION", "WASTEWATER PUMPING STATION", "STORMWATER PUMPING STATION"}
treatment_types = {"WATER POLLUTION CONTROL PLANT", "WASTEWATER TREATMENT PLANT"}

pumps      = [n for n in nodes if n["infra_type"] == "water" and n["subtype"] in pump_types and n.get("lat")]
treatments = [n for n in nodes if n["infra_type"] == "water" and n["subtype"] in treatment_types and n.get("lat")]

water_flow_count = 0
for pump in pumps:
    for plant in treatments:
        d = dist_m(pump["lon"], pump["lat"], plant["lon"], plant["lat"])
        if d <= WATER_FLOW_RADIUS_M:
            add_edge(pump["node_id"], plant["node_id"], "water_flow",
                     weight=round(d/1000, 3), distance_m=round(d),
                     buffer_hours=0.0, dependency_class="physical")
            water_flow_count += 1

print(f"Water flow edges (directed, ≤{WATER_FLOW_RADIUS_M/1000:.0f}km): {water_flow_count}")


# ── 4e. Water → hospital (directed, with buffer) ─────────────────────────────
all_water_nodes = [n for n in nodes if n["infra_type"] == "water" and n.get("lat")]
hosp_nodes      = [n for n in nodes if n["infra_type"] == "hospital" and n.get("lat")]

water_supply_count = 0
for hn in hosp_nodes:
    dists = [(dist_m(wn["lon"], wn["lat"], hn["lon"], hn["lat"]), wn) for wn in all_water_nodes]
    d, nearest_water = min(dists, key=lambda x: x[0])
    add_edge(nearest_water["node_id"], hn["node_id"], "water_supplies",
             weight=round(1 / (1 + d / 1000), 3), distance_m=round(d),
             buffer_hours=WATER_SUPPLY_BUFFER, dependency_class="physical")
    water_supply_count += 1

print(f"Water supply edges (directed, hospital←water, buffer={WATER_SUPPLY_BUFFER}h): {water_supply_count}")


# ── 4f. Telecom → power (SCADA, directed reverse dependency) ─────────────────
telecom_nodes = [n for n in nodes if n["infra_type"] == "telecom" and n.get("lat")]

scada_count = 0
for pn in local_power:
    dists = [(dist_m(tn["lon"], tn["lat"], pn["lon"], pn["lat"]), tn) for tn in telecom_nodes]
    if not dists:
        continue
    d, nearest_tel = min(dists, key=lambda x: x[0])
    add_edge(nearest_tel["node_id"], pn["node_id"], "scada_monitoring",
             weight=round(1 / (1 + d / 1000), 3), distance_m=round(d),
             buffer_hours=SCADA_BUFFER_HOURS, dependency_class="cyber")
    scada_count += 1

print(f"SCADA monitoring edges (directed, power←telecom, buffer={SCADA_BUFFER_HOURS}h): {scada_count}")


# ── 4g. Subway → infrastructure (repair access, directed, recovery layer) ────
subway_nodes   = [n for n in nodes if n["infra_type"] == "subway" and n.get("lat")]
repair_targets = [n for n in nodes
                  if n["infra_type"] in ("power", "hospital", "water", "telecom", "fuel")
                  and not n.get("external") and n.get("lat")]

repair_count = 0
for tn in repair_targets:
    dists = [(dist_m(sn["lon"], sn["lat"], tn["lon"], tn["lat"]), sn) for sn in subway_nodes]
    d, nearest_sub = min(dists, key=lambda x: x[0])
    add_edge(nearest_sub["node_id"], tn["node_id"], "repair_access",
             weight=round(1 / (1 + d / 1000), 3), distance_m=round(d),
             buffer_hours=REPAIR_BUFFER_HOURS, dependency_class="logical",
             layer="recovery")
    repair_count += 1

print(f"Repair access edges (directed, recovery layer): {repair_count}")


# ── 4h. Fuel distribution: terminal → gas station (NEW in v4) ────────────────
fuel_station_nodes  = [n for n in nodes if n["infra_type"] == "fuel"
                       and n.get("subtype") == "GAS_STATION" and n.get("lat")]
fuel_terminal_nodes = [n for n in nodes if n["infra_type"] == "fuel"
                       and n.get("subtype") == "PETROLEUM_TERMINAL" and n.get("lat")]

fuel_dist_count = 0
for sn in fuel_station_nodes:
    if not fuel_terminal_nodes:
        break
    # Connect each gas station to its nearest petroleum terminal
    dists = [(dist_m(tn["lon"], tn["lat"], sn["lon"], sn["lat"]), tn) for tn in fuel_terminal_nodes]
    d, nearest_term = min(dists, key=lambda x: x[0])
    add_edge(nearest_term["node_id"], sn["node_id"], "fuel_distribution",
             weight=round(1 / (1 + d / 1000), 3), distance_m=round(d),
             buffer_hours=FUEL_DISTRIBUTION_BUFFER,
             dependency_class="physical")
    fuel_dist_count += 1

print(f"Fuel distribution edges (terminal → station): {fuel_dist_count}")


# ── 4i. Fuel → generators: hospital, telecom, water (NEW in v4) ──────────────
# Gas stations supply fuel for backup generators at hospitals, telecom, water
# Connect each fuel-dependent node to its nearest gas station

fuel_supply_count = 0

# Fuel → Hospital (generator refueling)
for hn in hosp_nodes:
    if not fuel_station_nodes:
        break
    dists = [(dist_m(fn["lon"], fn["lat"], hn["lon"], hn["lat"]), fn) for fn in fuel_station_nodes]
    d, nearest_fuel = min(dists, key=lambda x: x[0])
    add_edge(nearest_fuel["node_id"], hn["node_id"], "fuel_supplies",
             weight=round(1 / (1 + d / 1000), 3), distance_m=round(d),
             buffer_hours=FUEL_TO_HOSPITAL_BUFFER,
             dependency_class="physical")
    fuel_supply_count += 1

# Fuel → Telecom (generator refueling)
# Only connect to telecom clusters with high tower count (likely have generators)
telecom_with_generators = [n for n in telecom_nodes
                           if float(n.get("tower_count", 0)) >= 5]
for tn in telecom_with_generators:
    if not fuel_station_nodes:
        break
    dists = [(dist_m(fn["lon"], fn["lat"], tn["lon"], tn["lat"]), fn) for fn in fuel_station_nodes]
    d, nearest_fuel = min(dists, key=lambda x: x[0])
    add_edge(nearest_fuel["node_id"], tn["node_id"], "fuel_supplies",
             weight=round(1 / (1 + d / 1000), 3), distance_m=round(d),
             buffer_hours=FUEL_TO_TELECOM_BUFFER,
             dependency_class="physical")
    fuel_supply_count += 1

# Fuel → Water (diesel pump backup)
water_pumps_local = [n for n in nodes if n["infra_type"] == "water"
                     and not n.get("external") and n.get("lat")]
for wn in water_pumps_local:
    if not fuel_station_nodes:
        break
    dists = [(dist_m(fn["lon"], fn["lat"], wn["lon"], wn["lat"]), fn) for fn in fuel_station_nodes]
    d, nearest_fuel = min(dists, key=lambda x: x[0])
    add_edge(nearest_fuel["node_id"], wn["node_id"], "fuel_supplies",
             weight=round(1 / (1 + d / 1000), 3), distance_m=round(d),
             buffer_hours=FUEL_TO_WATER_BUFFER,
             dependency_class="physical")
    fuel_supply_count += 1

print(f"Fuel supply edges (fuel → generators): {fuel_supply_count}")
print(f"  fuel → hospital: {sum(1 for e in edges if e['edge_type']=='fuel_supplies' and node_lookup.get(e['v'],{}).get('infra_type')=='hospital')}")
print(f"  fuel → telecom:  {sum(1 for e in edges if e['edge_type']=='fuel_supplies' and node_lookup.get(e['v'],{}).get('infra_type')=='telecom')}")
print(f"  fuel → water:    {sum(1 for e in edges if e['edge_type']=='fuel_supplies' and node_lookup.get(e['v'],{}).get('infra_type')=='water')}")


# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\nGraph summary: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges (directed)")
from collections import Counter
etype_counts = Counter(d["edge_type"] for _, _, d in G.edges(data=True))
for et, cnt in sorted(etype_counts.items()):
    print(f"  {et:25s}: {cnt:5d}")

# Cascade vs recovery edge counts
cascade_edges = sum(1 for _, _, d in G.edges(data=True) if d.get("layer") != "recovery")
recovery_edges = sum(1 for _, _, d in G.edges(data=True) if d.get("layer") == "recovery")
print(f"\n  Cascade layer edges:  {cascade_edges}")
print(f"  Recovery layer edges: {recovery_edges}")

# Buffer duration summary
print(f"\n  Buffer duration distribution:")
buf_counter = Counter(d.get("buffer_hours", 0) for _, _, d in G.edges(data=True))
for buf, cnt in sorted(buf_counter.items()):
    print(f"    {buf:6.1f} h : {cnt:5d} edges")


# ══════════════════════════════════════════════════════════════════════════════
# 5. SAVE OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════

# GraphML
nx.write_graphml(G, "data/graph/lm_infra_graph.graphml")
print("\nSaved → data/graph/lm_infra_graph.graphml")

# Nodes GeoJSON
node_records = []
for nid, attrs in G.nodes(data=True):
    if attrs.get("lat") and attrs.get("lon"):
        node_records.append({**attrs, "node_id": nid})

nodes_gdf = gpd.GeoDataFrame(
    node_records,
    geometry=[Point(r["lon"], r["lat"]) for r in node_records],
    crs="EPSG:4326",
)
nodes_gdf.to_file("data/graph/lm_infra_nodes.geojson", driver="GeoJSON")
print(f"Saved → data/graph/lm_infra_nodes.geojson  ({len(nodes_gdf)} nodes)")

# Edges GeoJSON
edge_records = []
for u, v, attrs in G.edges(data=True):
    un = node_lookup.get(u, {})
    vn = node_lookup.get(v, {})
    if un.get("lat") and vn.get("lat"):
        geom = LineString([(un["lon"], un["lat"]), (vn["lon"], vn["lat"])])
        edge_records.append({**attrs, "u": u, "v": v, "geometry": geom})

edges_gdf = gpd.GeoDataFrame(edge_records, crs="EPSG:4326")
edges_gdf.to_file("data/graph/lm_infra_edges.geojson", driver="GeoJSON")
print(f"Saved → data/graph/lm_infra_edges.geojson  ({len(edges_gdf)} edges)")

# ── Final diagnostic ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("GRAPH DIAGNOSTICS")
print("=" * 60)

wcc = nx.number_weakly_connected_components(G)
print(f"  Weakly connected components: {wcc}  (should be 1)")

scc = nx.number_strongly_connected_components(G)
print(f"  Strongly connected components: {scc}  (will be >1 — expected for directed)")

in_deg  = dict(G.in_degree())
out_deg = dict(G.out_degree())

print(f"\n  Top-5 by OUT-degree (nodes that cascade TO others):")
for nid, deg in sorted(out_deg.items(), key=lambda x: -x[1])[:5]:
    itype = G.nodes[nid].get("infra_type", "?")
    name  = G.nodes[nid].get("name", nid)
    print(f"    [{itype:8s}]  {name:<35s}  out-degree={deg}")

print(f"\n  Top-5 by IN-degree (nodes that RECEIVE cascading failures):")
for nid, deg in sorted(in_deg.items(), key=lambda x: -x[1])[:5]:
    itype = G.nodes[nid].get("infra_type", "?")
    name  = G.nodes[nid].get("name", nid)
    print(f"    [{itype:8s}]  {name:<35s}  in-degree={deg}")

print(f"\n  Average out-degree by infra type:")
for itype in ["power", "telecom", "hospital", "subway", "water", "fuel"]:
    type_nodes = [n for n in G.nodes if G.nodes[n].get("infra_type") == itype]
    if type_nodes:
        avg = np.mean([out_deg[n] for n in type_nodes])
        print(f"    {itype:10s}: {avg:.1f}")

# ── Fuel cascade pathway diagnostic ──────────────────────────────────────────
print(f"\n  Fuel cascade pathway check:")
fuel_nodes_in_graph = [n for n in G.nodes if G.nodes[n].get("infra_type") == "fuel"]
print(f"    Fuel nodes in graph: {len(fuel_nodes_in_graph)}")
print(f"    Fuel stations:  {sum(1 for n in fuel_nodes_in_graph if G.nodes[n].get('subtype')=='GAS_STATION')}")
print(f"    Fuel terminals: {sum(1 for n in fuel_nodes_in_graph if G.nodes[n].get('subtype')=='PETROLEUM_TERMINAL')}")
print(f"    fuel_distribution edges:  {etype_counts.get('fuel_distribution', 0)}")
print(f"    fuel_supplies edges:      {etype_counts.get('fuel_supplies', 0)}")
print(f"    power → fuel edges:       {sum(1 for e in edges if e['edge_type']=='power_dependency' and node_lookup.get(e['v'],{}).get('infra_type')=='fuel')}")
