"""
convert_to_pyg_nyc.py  (v1 — citywide expansion)
================================================
Converts the citywide NetworkX DiGraph to PyTorch Geometric HeteroData.

This is the citywide counterpart to convert_to_pyg.py (which covers Lower
Manhattan only). Same feature extractors, same edge type mapping, same
normalization. Different input/output paths.

Input:  data/graph/nyc_infra_graph.graphml
Output: data/graph/nyc_infra_heterodata.pt

------------------------------------------------------------------------------
Week 12-cont. additive refactor (no behavior change for NYC):
  The conversion logic that previously ran as a top-level script body is now
  wrapped in importable functions — `graph_to_heterodata(G)` plus the pure
  helpers/constants it uses — and the NYC script orchestration moved under a
  `main()` / `if __name__ == "__main__"` guard. This lets the Boston pilot
  (src/cities/boston_graph.py) reuse the EXACT same feature extractors,
  edge-type mapping, and normalization by import, guaranteeing schema parity.
  Verified byte-identical: re-running this script reproduces the same
  nyc_infra_heterodata.pt (node x tensors + edge_index hashes unchanged).
------------------------------------------------------------------------------
"""

import os
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import HeteroData


# ══════════════════════════════════════════════════════════════════════════════
# DEFAULT VALUES FOR MISSING FEATURES  (importable constants)
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

FEATURE_NAMES = {
    "power": ["lat", "lon", "elevation", "flood_depth", "is_active", "max_volt_kv", "degree_centrality", "is_external"],
    "telecom": ["lat", "lon", "elevation", "flood_depth", "tower_count", "has_lte", "has_5g", "battery_backup_hrs"],
    "hospital": ["lat", "lon", "elevation", "flood_depth", "bed_count", "generator_fuel_hrs", "water_storage_days", "is_critical"],
    "subway": ["lat", "lon", "elevation", "flood_depth", "num_routes", "ada_accessible", "depth_below_surface", "has_flood_gates"],
    "water": ["lat", "lon", "elevation", "flood_depth", "is_treatment_plant", "is_pump", "capacity_proxy", "is_external"],
    "fuel": ["lat", "lon", "elevation", "flood_depth", "is_terminal", "capacity_proxy", "has_backup_power", "is_external"],
}

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


def safe_float(val, default=0.0):
    if val is None:
        return default
    try:
        f = float(val)
        return default if np.isnan(f) else f
    except (ValueError, TypeError):
        return default


# ══════════════════════════════════════════════════════════════════════════════
# Feature extractor factory
#
# The extractors close over two graph-derived lookups (degree_centrality,
# power_max_volt), so they are produced by a factory rather than defined at
# module scope. Logic is byte-for-byte the original.
# ══════════════════════════════════════════════════════════════════════════════

def make_feature_extractors(degree_centrality, power_max_volt):
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

    return {
        "power": power_features,
        "telecom": telecom_features,
        "hospital": hospital_features,
        "subway": subway_features,
        "water": water_features,
        "fuel": fuel_features,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Core conversion (importable — reused by the Boston pilot)
# ══════════════════════════════════════════════════════════════════════════════

def graph_to_heterodata(G, *, verbose=True, force_edge_types=None):
    """Convert a NetworkX DiGraph (infra_type-tagged nodes, edge_type-tagged
    edges) to a normalized PyG HeteroData with the NYC schema.

    Args:
        G: NetworkX DiGraph with node attr 'infra_type' and edge attr 'edge_type'.
        verbose: print build progress.
        force_edge_types: optional iterable of canonical (src, rel, dst) triplets
            to guarantee present in the output even if G has zero such edges.
            Used by the Boston pilot to keep the 19-relation schema intact when
            a relation (e.g. water_flow) has no edges due to a data gap — the
            relation is created with an empty [2, 0] edge_index. NYC passes None.
    """
    nodes_by_type = {}
    for nid, attrs in G.nodes(data=True):
        itype = attrs.get("infra_type", "unknown")
        nodes_by_type.setdefault(itype, []).append((nid, attrs))

    if verbose:
        for itype, nlist in sorted(nodes_by_type.items()):
            print(f"  {itype:10s}: {len(nlist):,} nodes")

    # ── Derived features from graph structure ──
    degree_centrality = nx.degree_centrality(G)
    power_max_volt = {}
    for u, v, d in G.edges(data=True):
        if d.get("edge_type") == "power_line":
            volt_class = str(d.get("volt_class", "NOT AVAILABLE"))
            kv = VOLT_CLASS_KV.get(volt_class, 138.0)
            for node in [u, v]:
                power_max_volt[node] = max(power_max_volt.get(node, 0), kv)

    extractors = make_feature_extractors(degree_centrality, power_max_volt)

    # ── Build HeteroData node tensors ──
    data = HeteroData()
    node_id_to_idx = {}

    for itype, nlist in nodes_by_type.items():
        if itype == "unknown":
            continue
        extractor = extractors.get(itype)
        if extractor is None:
            if verbose:
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
        if verbose:
            print(f"  {itype:10s}: x.shape = {str(list(feat_tensor.shape)):>12}   "
                  f"features = {FEATURE_NAMES.get(itype, [])}")

    # ── Edge construction ──
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

    for triplet, edge_data in sorted(pyg_edges.items(), key=lambda x: str(x[0])):
        src_indices = torch.tensor(edge_data["src"], dtype=torch.long)
        dst_indices = torch.tensor(edge_data["dst"], dtype=torch.long)
        data[triplet].edge_index = torch.stack([src_indices, dst_indices], dim=0)
        data[triplet].edge_attr = torch.tensor(edge_data["features"], dtype=torch.float32)
        is_recovery = triplet[1] == "repair_access"
        data[triplet].is_recovery = is_recovery
        if verbose:
            layer_tag = " [RECOVERY]" if is_recovery else " [CASCADE]"
            print(f"    {str(triplet):<55}  edges={len(edge_data['src']):>6,}  "
                  f"edge_attr={list(data[triplet].edge_attr.shape)}{layer_tag}")

    # ── Force-create empty relations to preserve the schema contract ──
    if force_edge_types:
        for triplet in force_edge_types:
            if triplet in data.edge_types:
                continue
            data[triplet].edge_index = torch.zeros((2, 0), dtype=torch.long)
            data[triplet].edge_attr = torch.zeros((0, 4), dtype=torch.float32)
            data[triplet].is_recovery = triplet[1] == "repair_access"
            if verbose:
                print(f"    {str(triplet):<55}  edges=     0  [EMPTY — schema placeholder]")

    # ── Normalization (per-feature min-max to [0,1]) ──
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

    for triplet in list(data.edge_types):
        if hasattr(data[triplet], "edge_attr") and data[triplet].edge_attr.shape[0] > 0:
            ea = data[triplet].edge_attr
            ea_min = ea.min(dim=0).values
            ea_max = ea.max(dim=0).values
            ea_range = ea_max - ea_min
            ea_range[ea_range == 0] = 1.0
            data[triplet].edge_attr_raw = ea.clone()
            data[triplet].edge_attr = (ea - ea_min) / ea_range

    return data


# ══════════════════════════════════════════════════════════════════════════════
# NYC script orchestration (unchanged behavior; now under main guard)
# ══════════════════════════════════════════════════════════════════════════════

def main(graphml_path="data/graph/nyc_infra_graph.graphml",
         out_path="data/graph/nyc_infra_heterodata.pt"):
    os.makedirs("data/graph", exist_ok=True)

    print("Loading NetworkX graph (citywide)...")
    G = nx.read_graphml(graphml_path)
    print(f"  Nodes: {G.number_of_nodes():,}, Edges: {G.number_of_edges():,}")

    print("\nBuilding PyG HeteroData...")
    data = graph_to_heterodata(G, verbose=True)

    torch.save(data, out_path)
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"\nSaved → {out_path}  ({size_mb:.1f} MB)")

    print("\n" + "=" * 72)
    print("PyG HeteroData SCHEMA SUMMARY")
    print("=" * 72)
    print(f"\nNode types: {data.node_types}")
    print(f"Edge types: {len(data.edge_types)}")
    print(f"\n{'Type':<12} {'Nodes':>8} {'Features':>8}")
    print("-" * 35)
    for ntype in data.node_types:
        print(f"{ntype:<12} {data[ntype].num_nodes:>8,} {data[ntype].x.shape[1]:>8}")
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

    print(f"\nVerification: loading saved HeteroData...")
    data_loaded = torch.load(out_path, weights_only=False)
    print(f"  Loaded successfully: {len(data_loaded.node_types)} node types, "
          f"{len(data_loaded.edge_types)} edge types")
    print("\nDone. Next: run flood_overlay_v3.py with NODES_IN=nyc_infra_nodes.geojson")


if __name__ == "__main__":
    main()
