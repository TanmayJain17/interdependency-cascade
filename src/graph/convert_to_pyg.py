"""
convert_to_pyg_nyc.py  (v1 — citywide expansion)
================================================
Converts the citywide NetworkX DiGraph to PyTorch Geometric HeteroData.

This is the citywide counterpart to convert_to_pyg.py (which covers Lower
Manhattan only). Same feature extractors, same edge type mapping, same
normalization. Different input/output paths.

Input:  data/graph/nyc_infra_graph.graphml
Output: data/graph/nyc_infra_heterodata.pt
"""

import os
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import HeteroData

os.makedirs("data/graph", exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD NETWORKX GRAPH
# ══════════════════════════════════════════════════════════════════════════════

print("Loading NetworkX graph (citywide)...")
G = nx.read_graphml("data/graph/nyc_infra_graph.graphml")
print(f"  Nodes: {G.number_of_nodes():,}, Edges: {G.number_of_edges():,}")

nodes_by_type = {}
for nid, attrs in G.nodes(data=True):
    itype = attrs.get("infra_type", "unknown")
    nodes_by_type.setdefault(itype, []).append((nid, attrs))

for itype, nlist in sorted(nodes_by_type.items()):
    print(f"  {itype:10s}: {len(nlist):,} nodes")


# ══════════════════════════════════════════════════════════════════════════════
# 2. DEFAULT VALUES FOR MISSING FEATURES
# ══════════════════════════════════════════════════════════════════════════════

DEFAULTS = {
    "elevation": 3.0,
    "flood_depth": 0.0,
    "max_volt_connected": 138.0,
    "battery_backup_hrs": 6.0,
    "bed_count": 200.0,
    "generator_fuel_hrs": 96.0,
    "water_storage_days": 1.0,
    "depth_below_surface": 15.0,
    "has_flood_gates": 0.0,
    "pump_capacity": 50.0,
    "fuel_capacity": 10000.0,
    "terminal_capacity": 1000000.0,
}

VOLT_CLASS_KV = {
    "345": 345.0, "220-287": 250.0, "100-161": 138.0,
    "UNDER 100": 69.0, "NOT AVAILABLE": 138.0, "500": 500.0,
}


# ══════════════════════════════════════════════════════════════════════════════
# 3. COMPUTE DERIVED FEATURES FROM GRAPH STRUCTURE
# ══════════════════════════════════════════════════════════════════════════════

print("\nComputing derived features...")

# Degree centrality is O(V+E), fine for 6k nodes
degree_centrality = nx.degree_centrality(G)
print(f"  Computed degree centrality for {len(degree_centrality):,} nodes")

# Max voltage of connected transmission lines per power node
power_max_volt = {}
for u, v, d in G.edges(data=True):
    if d.get("edge_type") == "power_line":
        volt_class = str(d.get("volt_class", "NOT AVAILABLE"))
        kv = VOLT_CLASS_KV.get(volt_class, 138.0)
        for node in [u, v]:
            power_max_volt[node] = max(power_max_volt.get(node, 0), kv)


# ══════════════════════════════════════════════════════════════════════════════
# 4. FEATURE EXTRACTORS
# ══════════════════════════════════════════════════════════════════════════════

print("\nBuilding feature vectors...")


def safe_float(val, default=0.0):
    if val is None:
        return default
    try:
        f = float(val)
        return default if np.isnan(f) else f
    except (ValueError, TypeError):
        return default


def power_features(nid, attrs):
    return [
        safe_float(attrs.get("lat"), 40.72),
        safe_float(attrs.get("lon"), -73.99),
        DEFAULTS["elevation"],
        DEFAULTS["flood_depth"],
        1.0 if attrs.get("status") == "IN SERVICE" else 0.0,
        power_max_volt.get(nid, DEFAULTS["max_volt_connected"]),
        degree_centrality.get(nid, 0.0),
        1.0 if str(attrs.get("external", "false")).lower() == "true" else 0.0,
    ]


def telecom_features(nid, attrs):
    radio = str(attrs.get("radio_types", ""))
    return [
        safe_float(attrs.get("lat"), 40.72),
        safe_float(attrs.get("lon"), -73.99),
        DEFAULTS["elevation"],
        DEFAULTS["flood_depth"],
        safe_float(attrs.get("tower_count"), 1.0),
        1.0 if "LTE" in radio else 0.0,
        1.0 if "NR" in radio or "5G" in radio else 0.0,
        DEFAULTS["battery_backup_hrs"],
    ]


def hospital_features(nid, attrs):
    return [
        safe_float(attrs.get("lat"), 40.72),
        safe_float(attrs.get("lon"), -73.99),
        DEFAULTS["elevation"],
        DEFAULTS["flood_depth"],
        DEFAULTS["bed_count"],
        DEFAULTS["generator_fuel_hrs"],
        DEFAULTS["water_storage_days"],
        1.0,
    ]


def subway_features(nid, attrs):
    routes = str(attrs.get("routes", ""))
    num_routes = len(routes.split()) if routes else 0
    ada = safe_float(attrs.get("ada"), 0.0)
    return [
        safe_float(attrs.get("lat"), 40.72),
        safe_float(attrs.get("lon"), -73.99),
        DEFAULTS["elevation"],
        DEFAULTS["flood_depth"],
        float(num_routes),
        1.0 if ada >= 1 else 0.0,
        DEFAULTS["depth_below_surface"],
        DEFAULTS["has_flood_gates"],
    ]


def water_features(nid, attrs):
    subtype = str(attrs.get("subtype", "")).upper()
    is_treatment = 1.0 if "TREATMENT" in subtype or "CONTROL" in subtype else 0.0
    is_pump = 1.0 if "PUMP" in subtype else 0.0
    return [
        safe_float(attrs.get("lat"), 40.72),
        safe_float(attrs.get("lon"), -73.99),
        DEFAULTS["elevation"],
        DEFAULTS["flood_depth"],
        is_treatment,
        is_pump,
        DEFAULTS["pump_capacity"],
        1.0 if str(attrs.get("external", "false")).lower() == "true" else 0.0,
    ]


def fuel_features(nid, attrs):
    subtype = str(attrs.get("subtype", "")).upper()
    is_terminal = 1.0 if "TERMINAL" in subtype else 0.0

    if is_terminal > 0:
        capacity = safe_float(attrs.get("capacity_bbl"), DEFAULTS["terminal_capacity"])
    else:
        capacity = DEFAULTS["fuel_capacity"]
    has_backup = 1.0 if is_terminal > 0 else 0.0

    return [
        safe_float(attrs.get("lat"), 40.72),
        safe_float(attrs.get("lon"), -73.99),
        DEFAULTS["elevation"],
        DEFAULTS["flood_depth"],
        is_terminal,
        capacity,
        has_backup,
        1.0 if str(attrs.get("external", "false")).lower() == "true" else 0.0,
    ]


FEATURE_EXTRACTORS = {
    "power": power_features,
    "telecom": telecom_features,
    "hospital": hospital_features,
    "subway": subway_features,
    "water": water_features,
    "fuel": fuel_features,
}

FEATURE_NAMES = {
    "power": ["lat", "lon", "elevation", "flood_depth", "is_active", "max_volt_kv", "degree_centrality", "is_external"],
    "telecom": ["lat", "lon", "elevation", "flood_depth", "tower_count", "has_lte", "has_5g", "battery_backup_hrs"],
    "hospital": ["lat", "lon", "elevation", "flood_depth", "bed_count", "generator_fuel_hrs", "water_storage_days", "is_critical"],
    "subway": ["lat", "lon", "elevation", "flood_depth", "num_routes", "ada_accessible", "depth_below_surface", "has_flood_gates"],
    "water": ["lat", "lon", "elevation", "flood_depth", "is_treatment_plant", "is_pump", "capacity_proxy", "is_external"],
    "fuel": ["lat", "lon", "elevation", "flood_depth", "is_terminal", "capacity_proxy", "has_backup_power", "is_external"],
}


# ══════════════════════════════════════════════════════════════════════════════
# 5. BUILD PyG HeteroData
# ══════════════════════════════════════════════════════════════════════════════

print("\nBuilding PyG HeteroData...")
data = HeteroData()
node_id_to_idx = {}

for itype, nlist in nodes_by_type.items():
    if itype == "unknown":
        continue
    extractor = FEATURE_EXTRACTORS.get(itype)
    if extractor is None:
        print(f"  WARNING: No feature extractor for type '{itype}', skipping")
        continue

    features = []
    node_ids_ordered = []
    for local_idx, (nid, attrs) in enumerate(nlist):
        feat = extractor(nid, attrs)
        features.append(feat)
        node_id_to_idx[nid] = (itype, local_idx)
        node_ids_ordered.append(nid)

    feat_tensor = torch.tensor(features, dtype=torch.float32)
    data[itype].x = feat_tensor
    data[itype].node_ids = node_ids_ordered
    data[itype].num_nodes = len(nlist)

    print(
        f"  {itype:10s}: x.shape = {str(list(feat_tensor.shape)):>12}   "
        f"features = {FEATURE_NAMES.get(itype, [])}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# 6. EDGE CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

EDGE_TYPE_MAP = {
    "power_line":        ("power",   "power_line",        "power"),
    "subway_line":       ("subway",  "subway_line",       "subway"),
    "power_dependency":  ("power",   "feeds",             None),
    "water_flow":        ("water",   "water_flow",        "water"),
    "water_supplies":    ("water",   "supplies",          "hospital"),
    "scada_monitoring":  ("telecom", "scada_monitors",    "power"),
    "repair_access":     ("subway",  "repair_access",     None),
    "fuel_distribution": ("fuel",    "fuel_distribution", "fuel"),
    "fuel_supplies":     ("fuel",    "fuel_supplies",     None),
}

pyg_edges = {}

for u, v, d in G.edges(data=True):
    et = d.get("edge_type", "unknown")
    if u not in node_id_to_idx or v not in node_id_to_idx:
        continue

    src_type, src_idx = node_id_to_idx[u]
    dst_type, dst_idx = node_id_to_idx[v]

    mapping = EDGE_TYPE_MAP.get(et)
    rel_name = mapping[1] if mapping else et
    triplet = (src_type, rel_name, dst_type)

    if triplet not in pyg_edges:
        pyg_edges[triplet] = {"src": [], "dst": [], "features": []}

    pyg_edges[triplet]["src"].append(src_idx)
    pyg_edges[triplet]["dst"].append(dst_idx)

    edge_feat = [
        safe_float(d.get("weight"), 1.0),
        safe_float(d.get("distance_m"), 0.0),
        safe_float(d.get("buffer_hours"), 0.0),
        1.0 / (1.0 + safe_float(d.get("buffer_hours"), 0.0)),
    ]
    pyg_edges[triplet]["features"].append(edge_feat)

print(f"\n  Edge types in HeteroData:")
for triplet, edge_data in sorted(pyg_edges.items(), key=lambda x: str(x[0])):
    src_indices = torch.tensor(edge_data["src"], dtype=torch.long)
    dst_indices = torch.tensor(edge_data["dst"], dtype=torch.long)

    data[triplet].edge_index = torch.stack([src_indices, dst_indices], dim=0)
    data[triplet].edge_attr = torch.tensor(edge_data["features"], dtype=torch.float32)

    is_recovery = triplet[1] == "repair_access"
    data[triplet].is_recovery = is_recovery
    layer_tag = " [RECOVERY]" if is_recovery else " [CASCADE]"
    print(
        f"    {str(triplet):<55}  edges={len(edge_data['src']):>6,}  "
        f"edge_attr={list(data[triplet].edge_attr.shape)}{layer_tag}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# 7. NORMALIZATION
# ══════════════════════════════════════════════════════════════════════════════

print("\nNormalizing features...")

for itype in nodes_by_type:
    if itype == "unknown" or not hasattr(data[itype], "x"):
        continue
    x = data[itype].x
    x_min = x.min(dim=0).values
    x_max = x.max(dim=0).values
    x_range = x_max - x_min
    x_range[x_range == 0] = 1.0
    data[itype].x_raw = x.clone()
    data[itype].x = (x - x_min) / x_range
    print(f"  {itype:10s}: normalized {x.shape[1]} features to [0,1]")

for triplet in pyg_edges:
    if hasattr(data[triplet], "edge_attr"):
        ea = data[triplet].edge_attr
        ea_min = ea.min(dim=0).values
        ea_max = ea.max(dim=0).values
        ea_range = ea_max - ea_min
        ea_range[ea_range == 0] = 1.0
        data[triplet].edge_attr_raw = ea.clone()
        data[triplet].edge_attr = (ea - ea_min) / ea_range


# ══════════════════════════════════════════════════════════════════════════════
# 8. SAVE
# ══════════════════════════════════════════════════════════════════════════════

outpath = "data/graph/nyc_infra_heterodata.pt"
torch.save(data, outpath)
size_mb = os.path.getsize(outpath) / (1024 * 1024)
print(f"\nSaved → {outpath}  ({size_mb:.1f} MB)")


# ══════════════════════════════════════════════════════════════════════════════
# 9. SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("PyG HeteroData SCHEMA SUMMARY")
print("=" * 72)

print(f"\nNode types: {data.node_types}")
print(f"Edge types: {len(data.edge_types)}")

print(f"\n{'Type':<12} {'Nodes':>8} {'Features':>8}")
print("-" * 35)
for ntype in data.node_types:
    n = data[ntype].num_nodes
    f = data[ntype].x.shape[1]
    print(f"{ntype:<12} {n:>8,} {f:>8}")

print(f"\n{'Edge Type':<55} {'Edges':>8} {'Layer':>10}")
print("-" * 80)
for etype in data.edge_types:
    ne = data[etype].edge_index.shape[1]
    is_rec = getattr(data[etype], "is_recovery", False)
    layer = "RECOVERY" if is_rec else "CASCADE"
    print(f"{str(etype):<55} {ne:>8,} {layer:>10}")

total_nodes = sum(data[nt].num_nodes for nt in data.node_types)
total_edges = sum(data[et].edge_index.shape[1] for et in data.edge_types)
print(f"\nTotal: {total_nodes:,} nodes, {total_edges:,} directed edges")

# Verification
print(f"\nVerification: loading saved HeteroData...")
data_loaded = torch.load(outpath, weights_only=False)
print(f"  Loaded successfully: {len(data_loaded.node_types)} node types, "
      f"{len(data_loaded.edge_types)} edge types")
print(f"  Power features shape:    {list(data_loaded['power'].x.shape)}")
print(f"  Telecom features shape:  {list(data_loaded['telecom'].x.shape)}")
print(f"  Fuel features shape:     {list(data_loaded['fuel'].x.shape)}")

print("\nDone. Next: run flood_overlay_v3.py with NODES_IN=nyc_infra_nodes.geojson")