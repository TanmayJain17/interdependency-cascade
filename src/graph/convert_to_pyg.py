"""
convert_to_pyg.py
=================
Converts the Lower Manhattan NetworkX DiGraph to PyTorch Geometric HeteroData.

Solves two issues:
  [ISSUE 4] NetworkX → PyG HeteroData with typed nodes and edges
  [ISSUE 5] Numerical feature vectors for each node type
            (uses available data + reasonable defaults for missing fields)

Node feature vectors per type:
  power    (dim=8):  lat, lon, elevation, flood_depth, is_active, max_volt_connected, 
                     degree_centrality, is_external
  telecom  (dim=8):  lat, lon, elevation, flood_depth, tower_count, has_lte, has_5g,
                     battery_backup_hrs
  hospital (dim=8):  lat, lon, elevation, flood_depth, bed_count, generator_fuel_hrs,
                     water_storage_days, is_critical_facility
  subway   (dim=8):  lat, lon, elevation, flood_depth, num_routes, ada_accessible,
                     depth_below_surface, has_flood_gates  
  water    (dim=8):  lat, lon, elevation, flood_depth, is_treatment_plant, is_pump,
                     capacity_proxy, is_external

Edge feature vectors (all types, dim=4):
  weight, distance_m, buffer_hours, coupling_strength

All features normalized to [0,1] range for GNN training.

Outputs:
  data/graph/lm_infra_heterodata.pt   (PyG HeteroData, torch.save)
  Prints full schema summary
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

print("Loading NetworkX graph...")
G = nx.read_graphml("data/graph/lm_infra_graph.graphml")
print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

# Separate nodes by infrastructure type
nodes_by_type = {}
for nid, attrs in G.nodes(data=True):
    itype = attrs.get("infra_type", "unknown")
    nodes_by_type.setdefault(itype, []).append((nid, attrs))

for itype, nlist in sorted(nodes_by_type.items()):
    print(f"  {itype:10s}: {len(nlist)} nodes")


# ══════════════════════════════════════════════════════════════════════════════
# 2. DEFAULT VALUES FOR MISSING FEATURES
# ══════════════════════════════════════════════════════════════════════════════

# These are reasonable engineering defaults — flag them as assumptions
# in your presentation. They'll be replaced with real data as it becomes available.

DEFAULTS = {
    "elevation":            3.0,    # meters above sea level (Lower Manhattan avg)
    "flood_depth":          0.0,    # will be filled by GISSR overlay
    
    # Power
    "max_volt_connected":   138.0,  # kV, typical NYC transmission voltage
    
    # Telecom  
    "battery_backup_hrs":   6.0,    # FCC recommendation median
    
    # Hospital
    "bed_count":            200.0,  # NYC average acute care hospital
    "generator_fuel_hrs":   96.0,   # NFPA 110 minimum
    "water_storage_days":   1.0,    # ~1 day typical hospital reserve
    
    # Subway
    "depth_below_surface":  15.0,   # meters, NYC subway average depth
    "has_flood_gates":      0.0,    # most stations don't have them (binary)
    
    # Water
    "pump_capacity":        50.0,   # MGD proxy
}

# Voltage class mapping (from transmission lines VOLT_CLASS → kV)
VOLT_CLASS_KV = {
    "345": 345.0,
    "220-287": 250.0,
    "100-161": 138.0,
    "UNDER 100": 69.0,
    "NOT AVAILABLE": 138.0,  # assume typical
}


# ══════════════════════════════════════════════════════════════════════════════
# 3. COMPUTE DERIVED FEATURES FROM GRAPH STRUCTURE
# ══════════════════════════════════════════════════════════════════════════════

print("\nComputing derived features...")

# Degree centrality (normalized 0-1)
degree_centrality = nx.degree_centrality(G)

# For power substations: find max voltage of connected transmission lines
power_max_volt = {}
for u, v, d in G.edges(data=True):
    if d.get("edge_type") == "power_line":
        volt_class = str(d.get("volt_class", "NOT AVAILABLE"))
        kv = VOLT_CLASS_KV.get(volt_class, 138.0)
        for node in [u, v]:
            power_max_volt[node] = max(power_max_volt.get(node, 0), kv)


# ══════════════════════════════════════════════════════════════════════════════
# 4. BUILD FEATURE VECTORS PER NODE TYPE
# ══════════════════════════════════════════════════════════════════════════════

print("\nBuilding feature vectors...")

def safe_float(val, default=0.0):
    """Safely convert to float, returning default for None/nan/non-numeric."""
    if val is None:
        return default
    try:
        f = float(val)
        return default if np.isnan(f) else f
    except (ValueError, TypeError):
        return default


# ── Feature extractors per type ───────────────────────────────────────────────

def power_features(nid, attrs):
    """8-dim feature vector for power substations."""
    return [
        safe_float(attrs.get("lat"), 40.72),
        safe_float(attrs.get("lon"), -73.99),
        DEFAULTS["elevation"],                              # placeholder
        DEFAULTS["flood_depth"],                            # placeholder for GISSR
        1.0 if attrs.get("status") == "IN SERVICE" else 0.0,
        power_max_volt.get(nid, DEFAULTS["max_volt_connected"]),
        degree_centrality.get(nid, 0.0),
        1.0 if str(attrs.get("external", "false")).lower() == "true" else 0.0,
    ]

def telecom_features(nid, attrs):
    """8-dim feature vector for telecom clusters."""
    radio = str(attrs.get("radio_types", ""))
    return [
        safe_float(attrs.get("lat"), 40.72),
        safe_float(attrs.get("lon"), -73.99),
        DEFAULTS["elevation"],
        DEFAULTS["flood_depth"],
        safe_float(attrs.get("tower_count"), 1.0),
        1.0 if "LTE" in radio else 0.0,
        1.0 if "NR" in radio or "5G" in radio else 0.0,    # NR = 5G New Radio
        DEFAULTS["battery_backup_hrs"],
    ]

def hospital_features(nid, attrs):
    """8-dim feature vector for hospitals."""
    return [
        safe_float(attrs.get("lat"), 40.72),
        safe_float(attrs.get("lon"), -73.99),
        DEFAULTS["elevation"],
        DEFAULTS["flood_depth"],
        DEFAULTS["bed_count"],                              # not in FacDB — use default
        DEFAULTS["generator_fuel_hrs"],
        DEFAULTS["water_storage_days"],
        1.0,                                                # all hospitals are critical facilities
    ]

def subway_features(nid, attrs):
    """8-dim feature vector for subway stations."""
    routes = str(attrs.get("routes", ""))
    num_routes = len(routes.split()) if routes else 0
    ada = safe_float(attrs.get("ada"), 0.0)
    return [
        safe_float(attrs.get("lat"), 40.72),
        safe_float(attrs.get("lon"), -73.99),
        DEFAULTS["elevation"],
        DEFAULTS["flood_depth"],
        float(num_routes),
        1.0 if ada >= 1 else 0.0,                          # binary: accessible or not
        DEFAULTS["depth_below_surface"],
        DEFAULTS["has_flood_gates"],
    ]

def water_features(nid, attrs):
    """8-dim feature vector for water infrastructure."""
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

FEATURE_EXTRACTORS = {
    "power":    power_features,
    "telecom":  telecom_features,
    "hospital": hospital_features,
    "subway":   subway_features,
    "water":    water_features,
}

FEATURE_NAMES = {
    "power":    ["lat", "lon", "elevation", "flood_depth", "is_active", "max_volt_kv", "degree_centrality", "is_external"],
    "telecom":  ["lat", "lon", "elevation", "flood_depth", "tower_count", "has_lte", "has_5g", "battery_backup_hrs"],
    "hospital": ["lat", "lon", "elevation", "flood_depth", "bed_count", "generator_fuel_hrs", "water_storage_days", "is_critical"],
    "subway":   ["lat", "lon", "elevation", "flood_depth", "num_routes", "ada_accessible", "depth_below_surface", "has_flood_gates"],
    "water":    ["lat", "lon", "elevation", "flood_depth", "is_treatment_plant", "is_pump", "capacity_proxy", "is_external"],
}


# ══════════════════════════════════════════════════════════════════════════════
# 5. BUILD PyG HeteroData
# ══════════════════════════════════════════════════════════════════════════════

print("\nBuilding PyG HeteroData...")

data = HeteroData()

# ── Node ID mappings ──────────────────────────────────────────────────────────
# PyG uses integer indices. We need to map string node_ids to per-type indices.
node_id_to_idx = {}   # global: node_id_str → (type, local_idx)

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
    
    # Store as tensor
    feat_tensor = torch.tensor(features, dtype=torch.float32)
    data[itype].x = feat_tensor
    data[itype].node_ids = node_ids_ordered    # keep string IDs for debugging
    data[itype].num_nodes = len(nlist)
    
    print(f"  {itype:10s}: x.shape = {list(feat_tensor.shape)}  "
          f"features = {FEATURE_NAMES.get(itype, [])}")


# ── Edge construction ─────────────────────────────────────────────────────────

# Map edge_type strings to (src_type, relation, dst_type) triplets for PyG
EDGE_TYPE_MAP = {
    "power_line":       ("power",    "power_line",       "power"),
    "subway_line":      ("subway",   "subway_line",      "subway"),
    "power_dependency": ("power",    "feeds",            None),       # dst varies
    "water_flow":       ("water",    "water_flow",       "water"),
    "water_supplies":   ("water",    "supplies",         "hospital"),
    "scada_monitoring": ("telecom",  "scada_monitors",   "power"),
    "repair_access":    ("subway",   "repair_access",    None),       # dst varies
}

# Collect edges by PyG triplet type
pyg_edges = {}       # (src_type, rel, dst_type) → {'src': [], 'dst': [], 'features': []}

for u, v, d in G.edges(data=True):
    et = d.get("edge_type", "unknown")
    
    # Look up source and destination types from node mapping
    if u not in node_id_to_idx or v not in node_id_to_idx:
        continue
    
    src_type, src_idx = node_id_to_idx[u]
    dst_type, dst_idx = node_id_to_idx[v]
    
    # Determine relation name
    mapping = EDGE_TYPE_MAP.get(et)
    if mapping:
        _, rel_name, _ = mapping
    else:
        rel_name = et
    
    triplet = (src_type, rel_name, dst_type)
    
    if triplet not in pyg_edges:
        pyg_edges[triplet] = {'src': [], 'dst': [], 'features': []}
    
    pyg_edges[triplet]['src'].append(src_idx)
    pyg_edges[triplet]['dst'].append(dst_idx)
    
    # Edge features: weight, distance_m, buffer_hours, coupling_strength
    edge_feat = [
        safe_float(d.get("weight"), 1.0),
        safe_float(d.get("distance_m"), 0.0),
        safe_float(d.get("buffer_hours"), 0.0),
        # coupling_strength: inverse of buffer hours (immediate = 1.0, 96h = low)
        1.0 / (1.0 + safe_float(d.get("buffer_hours"), 0.0)),
    ]
    pyg_edges[triplet]['features'].append(edge_feat)


# Store in HeteroData
print(f"\n  Edge types in HeteroData:")
for triplet, edge_data in sorted(pyg_edges.items(), key=lambda x: str(x[0])):
    src_indices = torch.tensor(edge_data['src'], dtype=torch.long)
    dst_indices = torch.tensor(edge_data['dst'], dtype=torch.long)
    
    data[triplet].edge_index = torch.stack([src_indices, dst_indices], dim=0)
    data[triplet].edge_attr = torch.tensor(edge_data['features'], dtype=torch.float32)
    
    # Tag recovery layer edges
    is_recovery = triplet[1] == "repair_access"
    data[triplet].is_recovery = is_recovery
    
    layer_tag = " [RECOVERY]" if is_recovery else " [CASCADE]"
    print(f"    {str(triplet):55s}  edges={len(edge_data['src']):5d}  "
          f"edge_attr={list(data[triplet].edge_attr.shape)}{layer_tag}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. NORMALIZATION
# ══════════════════════════════════════════════════════════════════════════════

print("\nNormalizing features...")

# Per-type min-max normalization for node features
for itype in nodes_by_type:
    if itype == "unknown" or not hasattr(data[itype], 'x'):
        continue
    
    x = data[itype].x
    x_min = x.min(dim=0).values
    x_max = x.max(dim=0).values
    x_range = x_max - x_min
    x_range[x_range == 0] = 1.0    # avoid division by zero for constant features
    
    data[itype].x_raw = x.clone()  # keep raw values for debugging
    data[itype].x = (x - x_min) / x_range
    
    print(f"  {itype:10s}: normalized {x.shape[1]} features to [0,1]")

# Normalize edge features per edge type
for triplet in pyg_edges:
    if hasattr(data[triplet], 'edge_attr'):
        ea = data[triplet].edge_attr
        ea_min = ea.min(dim=0).values
        ea_max = ea.max(dim=0).values
        ea_range = ea_max - ea_min
        ea_range[ea_range == 0] = 1.0
        
        data[triplet].edge_attr_raw = ea.clone()
        data[triplet].edge_attr = (ea - ea_min) / ea_range


# ══════════════════════════════════════════════════════════════════════════════
# 7. SAVE
# ══════════════════════════════════════════════════════════════════════════════

outpath = "data/graph/lm_infra_heterodata.pt"
torch.save(data, outpath)
print(f"\nSaved → {outpath}")


# ══════════════════════════════════════════════════════════════════════════════
# 8. SUMMARY & VERIFICATION
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PyG HeteroData SCHEMA SUMMARY")
print("=" * 60)

print(f"\nNode types: {data.node_types}")
print(f"Edge types: {data.edge_types}")

print(f"\n{'Type':<12} {'Nodes':>6} {'Features':>8} {'Feature Names'}")
print("-" * 80)
for ntype in data.node_types:
    n = data[ntype].num_nodes
    f = data[ntype].x.shape[1]
    fnames = FEATURE_NAMES.get(ntype, ["?"])
    print(f"{ntype:<12} {n:>6} {f:>8}   {fnames}")

print(f"\n{'Edge Type':<55} {'Edges':>6} {'Edge Feats':>10} {'Layer'}")
print("-" * 90)
for etype in data.edge_types:
    ne = data[etype].edge_index.shape[1]
    nf = data[etype].edge_attr.shape[1] if hasattr(data[etype], 'edge_attr') else 0
    is_rec = getattr(data[etype], 'is_recovery', False)
    layer = "RECOVERY" if is_rec else "CASCADE"
    print(f"{str(etype):<55} {ne:>6} {nf:>10}   {layer}")

total_nodes = sum(data[nt].num_nodes for nt in data.node_types)
total_edges = sum(data[et].edge_index.shape[1] for et in data.edge_types)
print(f"\nTotal: {total_nodes} nodes, {total_edges} directed edges")

# ── Feature completeness audit ────────────────────────────────────────────────
print(f"\n{'='*60}")
print("FEATURE COMPLETENESS AUDIT")
print(f"{'='*60}")
print("""
Features from REAL DATA (available in downloaded datasets):
  ✓ lat, lon                    — all node types (from GIS data)
  ✓ tower_count                 — telecom (from OpenCelliD)
  ✓ has_lte, has_5g             — telecom (from OpenCelliD radio field)
  ✓ num_routes, ada_accessible  — subway (from MTA data)
  ✓ is_active / status          — power (from HIFLD)
  ✓ is_treatment_plant, is_pump — water (from FacDB/OSM subtype)
  ✓ max_volt_connected          — power (derived from transmission line VOLT_CLASS)
  ✓ degree_centrality           — power (computed from graph structure)
  ✓ is_external                 — power, water (from graph construction)

Features using DEFAULTS (need real data — flag for advisors):
  ⚠ elevation = 3.0m            — need USGS DEM overlay
  ⚠ flood_depth = 0.0           — need GISSR overlay (NEXT STEP)
  ⚠ bed_count = 200             — need NYS DOH hospital capacity data
  ⚠ generator_fuel_hrs = 96     — need facility-level survey data
  ⚠ water_storage_days = 1.0    — need facility-level survey data
  ⚠ battery_backup_hrs = 6.0    — need carrier-specific data or FCC filings
  ⚠ depth_below_surface = 15m   — need MTA station depth data
  ⚠ has_flood_gates = 0         — need MTA capital projects data
  ⚠ pump_capacity = 50 MGD      — need NYC DEP facility data

Edge features:
  ✓ weight                      — computed from distance/voltage
  ✓ distance_m                  — computed from coordinates
  ✓ buffer_hours                — from engineering standards (NFPA 110, FCC)
  ✓ coupling_strength           — derived from buffer_hours (1/(1+buffer))
""")

# ── Quick test: can we load it back? ──────────────────────────────────────────
print("Verification: loading saved HeteroData...")
data_loaded = torch.load(outpath, weights_only=False)
print(f"  Loaded successfully: {data_loaded.node_types}, {len(data_loaded.edge_types)} edge types")
print(f"  Power features shape: {data_loaded['power'].x.shape}")
print(f"  Sample power node (raw): {data_loaded['power'].x_raw[0].tolist()}")
print(f"  Sample power node (normalized): {data_loaded['power'].x[0].tolist()}")
print("\nDone. Ready for GISSR flood overlay → update flood_depth feature → cascade simulation.")