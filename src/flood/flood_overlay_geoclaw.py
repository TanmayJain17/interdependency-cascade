#!/usr/bin/env python3
"""
flood_overlay_geoclaw.py

Overlay GeoClaw "Sandy-like" peak flood scenarios (2026 / 2050 / 2080) onto the
infrastructure node set, adding 3 new flood-depth columns per node.

This is a parallel to flood_overlay_v3.py (NYC DEP polygon scenarios). After
running both, each node has 6 scenario columns: 3 DEP + 3 GeoClaw.

INPUTS
  data/flood/<scope>_infra_nodes_dep_flood.geojson    (DEP overlay output)
  data/flood/<scope>_infra_graph_dep_flood.graphml
  data/raw/flood_geoclaw/fgout_eta_{2026,2050,2080}_peak.tif

OUTPUTS
  data/flood/<scope>_infra_nodes_all_flood.geojson    (DEP + GeoClaw columns)
  data/flood/<scope>_infra_graph_all_flood.graphml
  data/flood/geoclaw_flood_summary.txt

USAGE
  python src/flood/flood_overlay_geoclaw.py --scope lm
  python src/flood/flood_overlay_geoclaw.py --scope nyc

NOTE ON RASTER FORMAT (2026-04-29)
  The TIFs received from Yuki's PhD student are uint8 binary inundation masks
  (1 = flooded, 0 = dry), not Float32 depth rasters. In BINARY_MODE the script
  assigns a single REPRESENTATIVE_DEPTH_M to every flooded node. When real
  depth rasters arrive, set BINARY_MODE = False and re-run — values are then
  sampled directly as depth in metres.
"""

import argparse
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import rasterio
from rasterio.warp import transform as warp_transform


# ─── Config ────────────────────────────────────────────────────────────────────
# Flip BINARY_MODE to False once the real depth rasters arrive. No other changes.
BINARY_MODE = False
REPRESENTATIVE_DEPTH_M = 1.0
# Rationale for 1.0 m default:
#   • Matches the median GISSR Sandy cold-storm depth across flooded LM divisions
#     (flooded divisions ranged 0.19-1.96 m, median ≈ 1.0 m).
#   • Sits above the fragility median for all 6 infra types (subway 0.1, water 0.3,
#     hospital 0.6, power 0.6) so flooded nodes are meaningfully exposed.
#   • Stays below the saturation plateau for the most sensitive curves (~2 m),
#     leaving headroom — important once real depths arrive and we don't want
#     a uniform-1.0m baseline to look identical to a depth-resolved 2m+ scenario.

SCENARIOS = [
    ("gc_2026", "data/raw/flood_geoclaw/fgout_eta_2026_360.tif"),
    ("gc_2050", "data/raw/flood_geoclaw/fgout_eta_2050_360.tif"),
    ("gc_2080", "data/raw/flood_geoclaw/fgout_eta_2080_360.tif"),
]


# ─── CLI ───────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--scope", choices=["lm", "nyc"], default="nyc",
                help="Lower Manhattan (lm) or full NYC (nyc)")
ap.add_argument("--nodes-in", default=None,
                help="Override input nodes geojson path")
ap.add_argument("--graph-in", default=None,
                help="Override input graphml path")
args = ap.parse_args()

scope = args.scope
NODES_IN = Path(args.nodes_in or f"data/flood/{scope}_infra_nodes_dep_flood.geojson")
GRAPH_IN = Path(args.graph_in or f"data/flood/{scope}_infra_graph_dep_flood.graphml")
NODES_OUT = Path(f"data/flood/{scope}_infra_nodes_all_flood.geojson")
GRAPH_OUT = Path(f"data/flood/{scope}_infra_graph_all_flood.graphml")
SUMMARY = Path("data/flood/geoclaw_flood_summary.txt")


# ─── Load nodes ────────────────────────────────────────────────────────────────
print("=" * 70)
print(f"GeoClaw flood overlay  —  scope={scope.upper()}")
mode_str = (f"BINARY (representative depth = {REPRESENTATIVE_DEPTH_M} m)"
            if BINARY_MODE else "DEPTH (continuous, sampled from raster)")
print(f"Mode: {mode_str}")
print("=" * 70)

if not NODES_IN.exists():
    raise FileNotFoundError(
        f"{NODES_IN} not found. Run flood_overlay_v3.py first to produce it, "
        f"or pass --nodes-in to override."
    )

nodes = gpd.read_file(NODES_IN)
print(f"\nLoaded {len(nodes):,} nodes from {NODES_IN}")
print(f"Node CRS: {nodes.crs}")

# Ensure node coords in WGS84 for sampling (GeoClaw TIFs are EPSG:4326)
nodes_wgs = nodes if nodes.crs and nodes.crs.to_epsg() == 4326 else nodes.to_crs(4326)
xs = nodes_wgs.geometry.x.values
ys = nodes_wgs.geometry.y.values


# ─── Sample each scenario ──────────────────────────────────────────────────────
summary_lines = [
    f"GeoClaw flood overlay — scope={scope.upper()}",
    f"Mode: {'BINARY' if BINARY_MODE else 'DEPTH'}",
    (f"Representative depth: {REPRESENTATIVE_DEPTH_M} m" if BINARY_MODE
     else "Depth sampled directly from raster"),
    f"Input: {NODES_IN}",
    f"Total nodes: {len(nodes):,}",
    "",
]

for col, path_str in SCENARIOS:
    path = Path(path_str)
    if not path.exists():
        msg = f"[WARN] {path} not found — skipping {col}"
        print(msg)
        summary_lines.append(msg)
        continue

    print(f"\n--- {col}  ({path.name}) ---")
    with rasterio.open(path) as src:
        # Reproject node coords into raster CRS only if it differs
        crs_str = src.crs.to_wkt() if src.crs else ""
        is_wgs84 = ("WGS 84" in crs_str or "WGS_1984" in crs_str
                    or "EPSG:4326" in crs_str)
        if is_wgs84:
            sx, sy = xs, ys
        else:
            sx, sy = warp_transform("EPSG:4326", src.crs, xs.tolist(), ys.tolist())
            sx, sy = np.array(sx), np.array(sy)
        coords = list(zip(sx, sy))

        # Sample band 1 at every node coordinate
        raw = np.array([v[0] for v in src.sample(coords)], dtype=float)

        # Treat raster nodata as 0 (dry)
        if src.nodata is not None and not np.isnan(src.nodata):
            raw[raw == src.nodata] = 0.0
        raw = np.where(np.isnan(raw), 0.0, raw)
        # Off-grid samples come back as 0 already; nothing extra needed

        # Convert raster value → depth in metres
        if BINARY_MODE:
            depth = np.where(raw > 0, REPRESENTATIVE_DEPTH_M, 0.0)
        else:
            depth = np.maximum(raw, 0.0)  # clip any negative eta-below-ground

    nodes[f"{col}_depth_m"] = depth

    n_flooded = int((depth > 0).sum())
    pct = 100 * n_flooded / len(nodes)
    print(f"  Flooded nodes: {n_flooded:,} / {len(nodes):,}  ({pct:.1f}%)")
    if not BINARY_MODE and n_flooded > 0:
        flooded_depths = depth[depth > 0]
        print(f"  Depth range:   {flooded_depths.min():.2f} – {flooded_depths.max():.2f} m")
        print(f"  Depth median:  {np.median(flooded_depths):.2f} m  (flooded only)")

    # Per-type breakdown for the summary file
    summary_lines.append(f"--- {col} ---")
    summary_lines.append(f"Flooded: {n_flooded:,} / {len(nodes):,} ({pct:.1f}%)")
    if "type" in nodes.columns:
        flooded_mask = depth > 0
        per_type = nodes.loc[flooded_mask, "type"].value_counts().sort_index()
        for t, n in per_type.items():
            total = int((nodes["type"] == t).sum())
            summary_lines.append(f"  {t:<14} {n:>5} / {total:<5}  ({100 * n / total:.0f}%)")
    summary_lines.append("")


# ─── Save updated nodes ────────────────────────────────────────────────────────
NODES_OUT.parent.mkdir(parents=True, exist_ok=True)
nodes.to_file(NODES_OUT, driver="GeoJSON")
print(f"\n✓ Saved nodes  → {NODES_OUT}")


# ─── Propagate to graph ────────────────────────────────────────────────────────
print("\nPropagating flood attributes to graph nodes...")
G = nx.read_graphml(GRAPH_IN)

# Find which column links GeoJSON rows to graph node IDs
id_col = next((c for c in ("node_id", "id", "osm_id") if c in nodes.columns), None)
if id_col is None:
    raise RuntimeError(
        f"Could not find node-id column in {NODES_IN}. "
        f"Available columns: {list(nodes.columns)}"
    )

id_to_idx = {str(nid): i for i, nid in enumerate(nodes[id_col])}
added_cols = [f"{col}_depth_m" for col, _ in SCENARIOS if f"{col}_depth_m" in nodes.columns]

n_updated, n_missing = 0, 0
for n_id in G.nodes:
    idx = id_to_idx.get(str(n_id))
    if idx is None:
        n_missing += 1
        continue
    for c in added_cols:
        G.nodes[n_id][c] = float(nodes.iloc[idx][c])
    n_updated += 1

print(f"  Updated:  {n_updated:,} / {G.number_of_nodes():,} graph nodes")
if n_missing:
    print(f"  Missing:  {n_missing:,} graph nodes had no match in nodes geojson "
          f"(check id_col='{id_col}')")

nx.write_graphml(G, GRAPH_OUT)
print(f"✓ Saved graph  → {GRAPH_OUT}")


# ─── Summary file ──────────────────────────────────────────────────────────────
SUMMARY.parent.mkdir(parents=True, exist_ok=True)
SUMMARY.write_text("\n".join(summary_lines))
print(f"✓ Saved summary → {SUMMARY}")

print("\nNext step: extend the SCENARIOS list in src/simulation/multi_scenario_runner.py "
      f"to include {', '.join(added_cols)}.")