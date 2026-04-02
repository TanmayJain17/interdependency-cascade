"""
flood_overlay_v2.py
===================
Overlays GISSR Sandy flood depths onto the LM infrastructure graph.

KEY FIX vs v1:  v1 used the sparse binary inundation_area.tif raster (only 4,409
pixels marked as flooded → 3/328 nodes tagged).  This version uses the GISSR
model's PRIMARY output: per-division flood heights from the CSV files.  Each
infrastructure node is mapped to its GISSR division via the DEM tile bounding
boxes, then assigned that division's computed flood depth.

Inputs:
  - data/graph/lm_infra_nodes.geojson              (328 nodes, v3 graph)
  - data/graph/lm_infra_graph.graphml               (v3 directed graph)
  - GIS_FloodSimulation/Sample_Output_Data/
      single_storm_flood_heights_c.csv               (cold storm, 1×18 depths)
      single_storm_flood_heights_w.csv               (warm storm, 1×18 depths)
  - GIS_FloodSimulation/Data/LM_div18/dem_lm_z35_*.TIF  (18 division DEMs)

Outputs:
  - data/flood/lm_infra_nodes_flood.geojson         (nodes + flood attributes)
  - data/flood/lm_infra_graph_flood.graphml         (graph + flood node attrs)
  - data/flood/flood_summary.txt                    (printable summary)
"""

import os, json, glob
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import rasterio
from pyproj import Transformer
from collections import Counter

# ── Paths ──────────────────────────────────────────────────────────────────────
GRAPH_DIR = "data/graph"
FLOOD_DIR = "data/flood"
GISSR     = "GIS_FloodSimulation"

NODES_IN  = os.path.join(GRAPH_DIR, "lm_infra_nodes.geojson")
GRAPH_IN  = os.path.join(GRAPH_DIR, "lm_infra_graph.graphml")
NODES_OUT = os.path.join(FLOOD_DIR, "lm_infra_nodes_flood.geojson")
GRAPH_OUT = os.path.join(FLOOD_DIR, "lm_infra_graph_flood.graphml")
SUMMARY   = os.path.join(FLOOD_DIR, "flood_summary.txt")

COLD_CSV  = os.path.join(GISSR, "Sample_Output_Data", "single_storm_flood_heights_c.csv")
WARM_CSV  = os.path.join(GISSR, "Sample_Output_Data", "single_storm_flood_heights_w.csv")
DIV_DIR   = os.path.join(GISSR, "Data", "LM_div18")

os.makedirs(FLOOD_DIR, exist_ok=True)

# ── CRS transformer: WGS84 → NY State Plane Long Island (US ft) ──────────────
T_to2263 = Transformer.from_crs("EPSG:4326", "EPSG:2263", always_xy=True)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  LOAD GISSR DIVISION BOUNDS AND FLOOD HEIGHTS
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("GISSR FLOOD OVERLAY v2 — Division-Based Depth Assignment")
print("=" * 60)

# 1a. Division bounding boxes from DEM tile extents (EPSG:2263, US survey feet)
print("\n[1] Loading 18 GISSR division boundaries from DEM tiles…")
div_tifs = sorted(glob.glob(os.path.join(DIV_DIR, "dem_lm_z35_*.TIF")))
assert len(div_tifs) == 18, f"Expected 18 division DEMs, found {len(div_tifs)}"

div_bounds = []  # list of (left, bottom, right, top) in EPSG:2263 feet
for path in div_tifs:
    with rasterio.open(path) as src:
        b = src.bounds
        div_bounds.append((b.left, b.bottom, b.right, b.top))
        
print(f"    Loaded {len(div_bounds)} division bounding boxes")

# 1b. Load flood heights per division
print("\n[2] Loading GISSR flood heights…")
cold_depths = pd.read_csv(COLD_CSV, header=None).iloc[0].values.astype(float)
warm_depths = pd.read_csv(WARM_CSV, header=None).iloc[0].values.astype(float)

print(f"    Cold storm (Sandy-type): {np.sum(cold_depths > 0)}/18 divisions flooded")
print(f"      Range: {cold_depths.min():.2f} – {cold_depths.max():.2f} m")
print(f"      Depths: {np.round(cold_depths, 2).tolist()}")
print(f"    Warm storm:              {np.sum(warm_depths > 0)}/18 divisions flooded")
print(f"      Range: {warm_depths.min():.2f} – {warm_depths.max():.2f} m")


# ══════════════════════════════════════════════════════════════════════════════
# 2.  MAP EACH NODE TO ITS GISSR DIVISION
# ══════════════════════════════════════════════════════════════════════════════

def node_to_division(lon, lat):
    """Map a WGS84 point to its GISSR division index (0–17), or -1 if outside."""
    x_ft, y_ft = T_to2263.transform(lon, lat)
    for i, (l, b, r, t) in enumerate(div_bounds):
        if l <= x_ft <= r and b <= y_ft <= t:
            return i
    return -1


print("\n[3] Mapping 328 infrastructure nodes to GISSR divisions…")
nodes_gdf = gpd.read_file(NODES_IN)

divisions       = []
cold_flood_m    = []
warm_flood_m    = []
sandy_inundated = []

for _, row in nodes_gdf.iterrows():
    lon, lat = row["lon"], row["lat"]
    div = node_to_division(lon, lat)
    divisions.append(div)
    
    if 0 <= div < 18:
        c_depth = float(cold_depths[div])
        w_depth = float(warm_depths[div])
    else:
        c_depth = 0.0
        w_depth = 0.0
    
    cold_flood_m.append(round(c_depth, 3))
    warm_flood_m.append(round(w_depth, 3))
    sandy_inundated.append(c_depth > 0.0)

# Add columns to GeoDataFrame
nodes_gdf["gissr_division"]  = divisions
nodes_gdf["flood_depth_m"]   = cold_flood_m       # primary: cold storm (Sandy scenario)
nodes_gdf["warm_flood_m"]    = warm_flood_m        # secondary: warm storm scenario
nodes_gdf["sandy_inundated"] = sandy_inundated     # boolean: flooded under cold storm


# ══════════════════════════════════════════════════════════════════════════════
# 3.  PRINT SUMMARY STATISTICS
# ══════════════════════════════════════════════════════════════════════════════

total = len(nodes_gdf)
n_mapped   = sum(d >= 0 for d in divisions)
n_unmapped = sum(d == -1 for d in divisions)
n_flooded  = sum(sandy_inundated)
n_dry_div  = n_mapped - n_flooded

summary_lines = []
def log(line=""):
    print(line)
    summary_lines.append(line)

log(f"\n{'=' * 60}")
log(f"FLOOD OVERLAY RESULTS — GISSR Sandy Cold Storm")
log(f"{'=' * 60}")
log(f"Total nodes:             {total}")
log(f"Mapped to GISSR div:     {n_mapped}  ({100*n_mapped/total:.1f}%)")
log(f"Unmapped (external/OOB): {n_unmapped}  ({100*n_unmapped/total:.1f}%)")
log(f"In flooded divisions:    {n_flooded}  ({100*n_flooded/total:.1f}%)")
log(f"In dry divisions:        {n_dry_div}  ({100*n_dry_div/total:.1f}%)")

log(f"\n{'Type':<12} {'Total':>6} {'Mapped':>7} {'Flooded':>8} {'%Flood':>7}  {'Avg depth(m)':>13}")
log("-" * 60)
for itype in ["power", "telecom", "hospital", "subway", "water"]:
    sub = nodes_gdf[nodes_gdf["infra_type"] == itype]
    n_total   = len(sub)
    n_map     = (sub["gissr_division"] >= 0).sum()
    n_flood   = sub["sandy_inundated"].sum()
    avg_depth = sub[sub["sandy_inundated"]]["flood_depth_m"].mean() if n_flood > 0 else 0.0
    pct       = 100 * n_flood / n_total if n_total > 0 else 0
    log(f"{itype:<12} {n_total:>6} {n_map:>7} {n_flood:>8} {pct:>6.1f}%  {avg_depth:>13.2f}")

log(f"\nDivision breakdown:")
log(f"{'Div':>4s}  {'Nodes':>5s}  {'Depth(m)':>9s}  {'Status':>8s}")
log("-" * 35)
div_counter = Counter(divisions)
for div in sorted(div_counter.keys()):
    d = cold_depths[div] if 0 <= div < 18 else 0.0
    tag = "FLOODED" if d > 0 else ("external" if div == -1 else "dry")
    log(f"{div:>4d}  {div_counter[div]:>5d}  {d:>9.2f}  {tag:>8s}")

# Named infrastructure in flooded zones
log(f"\nKey infrastructure in flooded divisions:")
for itype in ["power", "hospital", "water"]:
    sub = nodes_gdf[(nodes_gdf["infra_type"] == itype) & (nodes_gdf["sandy_inundated"])]
    if not sub.empty:
        log(f"  {itype.upper()}:")
        for _, r in sub.iterrows():
            log(f"    {r['name']:40s}  div={r['gissr_division']:>2d}  depth={r['flood_depth_m']:.2f}m")


# ══════════════════════════════════════════════════════════════════════════════
# 4.  SAVE UPDATED NODES GEOJSON
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n[4] Saving outputs…")
nodes_gdf.to_file(NODES_OUT, driver="GeoJSON")
print(f"    Nodes → {NODES_OUT}")


# ══════════════════════════════════════════════════════════════════════════════
# 5.  UPDATE GRAPH WITH FLOOD ATTRIBUTES
# ══════════════════════════════════════════════════════════════════════════════

G = nx.read_graphml(GRAPH_IN)
flood_lookup = nodes_gdf.set_index("node_id")[
    ["sandy_inundated", "flood_depth_m", "warm_flood_m", "gissr_division"]
]

for nid in G.nodes():
    if nid in flood_lookup.index:
        r = flood_lookup.loc[nid]
        G.nodes[nid]["sandy_inundated"] = str(bool(r["sandy_inundated"]))
        G.nodes[nid]["flood_depth_m"]   = float(r["flood_depth_m"])
        G.nodes[nid]["warm_flood_m"]    = float(r["warm_flood_m"])
        G.nodes[nid]["gissr_division"]  = int(r["gissr_division"])

nx.write_graphml(G, GRAPH_OUT)
print(f"    Graph → {GRAPH_OUT}")

# Save text summary
with open(SUMMARY, "w") as f:
    f.write("\n".join(summary_lines))
print(f"    Summary → {SUMMARY}")


# ══════════════════════════════════════════════════════════════════════════════
# 6.  VALIDATION NOTES
# ══════════════════════════════════════════════════════════════════════════════

print(f"""
{'=' * 60}
METHODOLOGY NOTES
{'=' * 60}
Flood depth source:  GISSR cold-storm Sandy simulation
                     (Miura et al., 2021 — validated against actual Sandy extent)

Assignment method:   Each node mapped to its GISSR division via DEM tile
                     bounding boxes (18 divisions covering LM south of 34th St).
                     Division flood height assigned directly from GISSR output.

Nodes with div=-1:   External substations (Gowanus, Jamaica, Rainey, Marion,
                     West 49th), external water plants, and telecom/subway nodes
                     outside the GISSR study area. Assigned flood_depth=0.
                     NOTE: Some of these (e.g., Gowanus) DID flood during Sandy
                     but are outside the GISSR LM model domain.

Cold vs Warm storm:  Cold storm = Sandy-like (asymmetric, strong surge on west).
                     Warm storm = uniform ~3.48m across all divisions (worst case).
                     Both stored; cold storm used as primary scenario.

Next step:           Use flood_depth_m to compute failure probabilities via
                     fragility curves, then run cascade simulation.
""")