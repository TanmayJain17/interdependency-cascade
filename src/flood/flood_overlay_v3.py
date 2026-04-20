#!/usr/bin/env python3
"""
flood_overlay_v3.py

Overlay the three NYC DEP Stormwater Flood Map scenarios against an infrastructure
node set, and propagate flood attributes to the corresponding graph file.

Run from project root (~/Desktop/RA/):
    python3 src/flood/flood_overlay_v3.py           # default: NYC citywide
    python3 src/flood/flood_overlay_v3.py nyc       # explicit citywide
    python3 src/flood/flood_overlay_v3.py lm        # Lower Manhattan only

Added columns per node (6):
    flood_moderate_current_category   integer 0-2
    flood_moderate_current_depth_m
    flood_moderate_2050_category      integer 0-3
    flood_moderate_2050_depth_m
    flood_extreme_2080_category       integer 0-3
    flood_extreme_2080_depth_m

Category-to-depth mapping (for HAZUS-style fragility curves):
    0 = Not flooded                             -> 0.00 m
    1 = Nuisance Flooding (4-12 in)             -> 0.20 m
    2 = Deep and Contiguous Flooding (>= 1 ft)  -> 0.60 m
    3 = Future High Tides (tidal inundation)    -> 0.80 m
"""

from __future__ import annotations

import os
# Must be set BEFORE geopandas import -- GDAL reads this at import time.
os.environ.setdefault("OGR_GEOJSON_MAX_OBJ_SIZE", "0")

import sys
from pathlib import Path

import geopandas as gpd
import networkx as nx
import pandas as pd
from shapely.validation import make_valid


# -----------------------------------------------------------------------------
# Scope configuration (LM vs NYC)
# -----------------------------------------------------------------------------

SCOPE_CONFIG = {
    "lm": {
        "nodes_in":  Path("data/flood/lm_infra_nodes_flood.geojson"),
        "graph_in":  Path("data/flood/lm_infra_graph_flood.graphml"),
        "nodes_out": Path("data/flood/lm_infra_nodes_dep_flood.geojson"),
        "graph_out": Path("data/flood/lm_infra_graph_dep_flood.graphml"),
        "summary":   Path("data/flood/dep_flood_summary_lm.txt"),
        "label":     "Lower Manhattan",
    },
    "nyc": {
        "nodes_in":  Path("data/graph/nyc_infra_nodes.geojson"),
        "graph_in":  Path("data/graph/nyc_infra_graph.graphml"),
        "nodes_out": Path("data/flood/nyc_infra_nodes_dep_flood.geojson"),
        "graph_out": Path("data/flood/nyc_infra_graph_dep_flood.graphml"),
        "summary":   Path("data/flood/dep_flood_summary_nyc.txt"),
        "label":     "NYC citywide",
    },
}

SCENARIOS: dict[str, Path] = {
    "moderate_current": Path("data/raw/flood/moderate_current.geojson"),
    "moderate_2050":    Path("data/raw/flood/moderate_2050.geojson"),
    "extreme_2080":     Path("data/raw/flood/extreme_2080.geojson"),
}

CATEGORY_DEPTH_M: dict[int, float] = {
    0: 0.00, 1: 0.20, 2: 0.60, 3: 0.80,
}

CATEGORY_LABEL: dict[int, str] = {
    0: "not flooded",
    1: "Nuisance (4-12 in)",
    2: "Deep (>= 1 ft)",
    3: "Future High Tides",
}


# -----------------------------------------------------------------------------
# Geometry hygiene
# -----------------------------------------------------------------------------

def fix_invalid_geometries(flood_gdf: gpd.GeoDataFrame, layer_label: str) -> gpd.GeoDataFrame:
    invalid_mask = ~flood_gdf.geometry.is_valid
    n_invalid = int(invalid_mask.sum())
    if n_invalid == 0:
        print(f"    [{layer_label}] all {len(flood_gdf)} geometries are valid")
        return flood_gdf

    print(f"    [{layer_label}] fixing {n_invalid} invalid geometries with make_valid()...")
    fixed = flood_gdf.copy()
    fixed.loc[invalid_mask, "geometry"] = fixed.loc[invalid_mask, "geometry"].apply(make_valid)

    def _polygonal_only(g):
        if g is None or g.is_empty:
            return g
        if g.geom_type in ("Polygon", "MultiPolygon"):
            return g
        if g.geom_type == "GeometryCollection":
            polys = [p for p in g.geoms if p.geom_type in ("Polygon", "MultiPolygon")]
            if not polys:
                return None
            if len(polys) == 1:
                return polys[0]
            from shapely.ops import unary_union
            return unary_union(polys)
        return None

    fixed["geometry"] = fixed.geometry.apply(_polygonal_only)
    fixed = fixed[fixed.geometry.notna() & ~fixed.geometry.is_empty].copy()

    still_invalid = int((~fixed.geometry.is_valid).sum())
    print(f"    [{layer_label}] after fix: {len(fixed)} rows, {still_invalid} still invalid")
    return fixed


# -----------------------------------------------------------------------------
# Overlay core
# -----------------------------------------------------------------------------

def overlay_scenario(
    nodes_gdf: gpd.GeoDataFrame,
    flood_gdf: gpd.GeoDataFrame,
    scenario_name: str,
) -> gpd.GeoDataFrame:
    cat_col   = f"flood_{scenario_name}_category"
    depth_col = f"flood_{scenario_name}_depth_m"

    if flood_gdf.crs != nodes_gdf.crs:
        flood_gdf = flood_gdf.to_crs(nodes_gdf.crs)

    flood_gdf = fix_invalid_geometries(flood_gdf, scenario_name)

    joined = gpd.sjoin(
        nodes_gdf[["geometry"]],
        flood_gdf[["Flooding_Category", "geometry"]],
        how="left",
        predicate="within",
    )

    max_cat = (
        joined.groupby(level=0)["Flooding_Category"]
        .max()
        .reindex(nodes_gdf.index)
        .fillna(0)
        .astype(int)
    )

    result = nodes_gdf.copy()
    result[cat_col]   = max_cat
    result[depth_col] = result[cat_col].map(CATEGORY_DEPTH_M).astype(float)
    return result


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------

def assign_borough(lat: float, lon: float) -> str:
    """Approximate borough from lat/lon. Good enough for summary reporting.

    Based on NYC Department of City Planning borough bounding boxes:
      Staten Island: lat 40.496 - 40.651, lon -74.256 - -74.052
      Manhattan:     lat 40.680 - 40.880, lon -74.020 - -73.910
      Bronx:         lat 40.785 - 40.920, lon -73.935 - -73.760
      Queens:        lat 40.541 - 40.800, lon -73.962 - -73.700
      Brooklyn:      lat 40.570 - 40.740, lon -74.042 - -73.833

    Anything outside NYC (mostly NJ substations + NJ telecom towers west of
    the Hudson) gets tagged 'NJ (external)' so it doesn't contaminate borough
    stats.
    """
    # Staten Island: south of Manhattan AND west of Hudson
    if lat < 40.65 and lon < -74.03:
        return "Staten Island"
    # NJ: anything else west of Hudson
    if lon < -74.03:
        return "NJ (external)"
    # Bronx: north of Harlem River
    if lat > 40.80 and lon > -73.93:
        return "Bronx"
    # Manhattan: the long strip between Hudson and East River
    if -74.02 <= lon <= -73.93 and 40.70 <= lat <= 40.88:
        return "Manhattan"
    # Queens: east side of NYC
    if lon > -73.90 or (lon > -73.93 and lat > 40.70):
        return "Queens"
    # Default: Brooklyn (south-central NYC)
    return "Brooklyn"


def per_scenario_block(nodes_gdf: gpd.GeoDataFrame, scenario_name: str) -> str:
    cat_col = f"flood_{scenario_name}_category"
    lines = [f"\n-- {scenario_name} --"]

    total = len(nodes_gdf)
    flooded = int((nodes_gdf[cat_col] > 0).sum())
    lines.append(f"  nodes flooded: {flooded:,}/{total:,} ({100 * flooded / total:.1f}%)")

    for cat in sorted(nodes_gdf[cat_col].unique()):
        n = int((nodes_gdf[cat_col] == cat).sum())
        label = CATEGORY_LABEL.get(int(cat), f"Category {cat}")
        lines.append(f"  category {int(cat)} ({label}): {n:,}")

    if "infra_type" in nodes_gdf.columns:
        lines.append("  by infrastructure type:")
        pivot = (
            nodes_gdf.groupby("infra_type")[cat_col]
            .apply(lambda s: int((s > 0).sum()))
        )
        totals = nodes_gdf["infra_type"].value_counts()
        for infra_type, n_flooded in pivot.sort_values(ascending=False).items():
            n_total = int(totals[infra_type])
            pct = 100 * n_flooded / n_total if n_total else 0
            lines.append(f"    {infra_type:<10}: {n_flooded:>5,}/{n_total:<5,} flooded ({pct:.1f}%)")

    # Borough breakdown if this is citywide (>500 nodes)
    if "lat" in nodes_gdf.columns and "lon" in nodes_gdf.columns and len(nodes_gdf) > 500:
        lines.append("  by borough (approximate):")
        with_boro = nodes_gdf.assign(
            _boro=nodes_gdf.apply(lambda r: assign_borough(r["lat"], r["lon"]), axis=1)
        )
        for boro, sub in with_boro.groupby("_boro"):
            flooded_b = int((sub[cat_col] > 0).sum())
            total_b = len(sub)
            pct = 100 * flooded_b / total_b if total_b else 0
            lines.append(f"    {boro:<15}: {flooded_b:>5,}/{total_b:<5,} flooded ({pct:.1f}%)")

    return "\n".join(lines)


def sandy_cross_check_block(nodes_gdf: gpd.GeoDataFrame) -> str:
    if "sandy_inundated" not in nodes_gdf.columns:
        return ""

    sandy = nodes_gdf["sandy_inundated"].astype(str).str.lower() == "true"
    dep   = nodes_gdf["flood_moderate_current_category"] >= 1

    both       = int((sandy & dep).sum())
    only_sandy = int((sandy & ~dep).sum())
    only_dep   = int((~sandy & dep).sum())
    neither    = int((~sandy & ~dep).sum())

    return (
        "\n\n-- Sandy (GISSR) vs DEP Moderate-Current cross-check --\n"
        "  Quadrant                      count\n"
        f"  Both Sandy and DEP:           {both}\n"
        f"  Sandy only (not in DEP):      {only_sandy}\n"
        f"  DEP only (dry in Sandy):      {only_dep}   <-- pluvial-prone, not in GISSR\n"
        f"  Neither:                      {neither}\n"
    )


# -----------------------------------------------------------------------------
# Graph propagation
# -----------------------------------------------------------------------------

def propagate_to_graph(
    nodes_gdf: gpd.GeoDataFrame,
    graph_in: Path,
    graph_out: Path,
) -> None:
    if not graph_in.exists():
        print(f"[WARN] Graph file not found: {graph_in} -- skipping graph update")
        return

    if "node_id" not in nodes_gdf.columns:
        print("[WARN] 'node_id' column missing on nodes -- skipping graph update")
        return

    print(f"\n[GRAPH] Reading {graph_in}")
    G = nx.read_graphml(graph_in)

    new_cols = [
        c for c in nodes_gdf.columns
        if c.startswith("flood_moderate_") or c.startswith("flood_extreme_")
    ]
    if not new_cols:
        print("[WARN] No new flood columns to propagate")
        return

    lookup = nodes_gdf.set_index("node_id")[new_cols]
    updated = 0
    for nid in G.nodes():
        if nid in lookup.index:
            row = lookup.loc[nid]
            for col in new_cols:
                val = row[col]
                if "category" in col:
                    G.nodes[nid][col] = int(val)
                else:
                    G.nodes[nid][col] = float(val)
            updated += 1

    graph_out.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(G, graph_out)
    print(f"[GRAPH] Updated {updated:,}/{len(G.nodes()):,} nodes with {len(new_cols)} attrs")
    print(f"[GRAPH] Wrote {graph_out}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> int:
    # Parse scope argument (default to nyc)
    scope = "nyc"
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in SCOPE_CONFIG:
            scope = arg
        else:
            print(f"ERROR: unknown scope '{arg}'. Use 'lm' or 'nyc'.")
            return 1

    cfg = SCOPE_CONFIG[scope]

    print("=" * 72)
    print(f"NYC DEP Stormwater Flood Overlay (v3) -- scope: {cfg['label']}")
    print("=" * 72)

    if not cfg["nodes_in"].exists():
        print(f"ERROR: input nodes file not found: {cfg['nodes_in']}")
        return 1

    print(f"Input nodes: {cfg['nodes_in']}")
    nodes_gdf = gpd.read_file(cfg["nodes_in"])
    print(f"  Loaded {len(nodes_gdf):,} nodes   CRS: {nodes_gdf.crs}")

    summary_blocks: list[str] = [
        "=" * 72,
        f"DEP STORMWATER FLOOD OVERLAY -- {cfg['label']}",
        "=" * 72,
        f"Input nodes: {cfg['nodes_in']} ({len(nodes_gdf):,} nodes)",
        "",
        "Category-to-depth mapping:",
        "  0 -> 0.00 m (not flooded)",
        "  1 -> 0.20 m (Nuisance: 4-12 in)",
        "  2 -> 0.60 m (Deep and Contiguous: >= 1 ft)",
        "  3 -> 0.80 m (Future High Tides)",
    ]

    for scenario_name, flood_path in SCENARIOS.items():
        print(f"\n[OVERLAY] {scenario_name}")
        if not flood_path.exists():
            print(f"  ERROR: flood file not found: {flood_path}")
            continue
        print(f"  Loading {flood_path}...")
        flood_gdf = gpd.read_file(flood_path)
        cats = sorted(flood_gdf["Flooding_Category"].unique().tolist())
        print(f"    {len(flood_gdf)} polygons, categories {cats}")

        print("  Running spatial join...")
        nodes_gdf = overlay_scenario(nodes_gdf, flood_gdf, scenario_name)

        block = per_scenario_block(nodes_gdf, scenario_name)
        print(block)
        summary_blocks.append(block)

    xcheck = sandy_cross_check_block(nodes_gdf)
    if xcheck:
        print(xcheck)
        summary_blocks.append(xcheck)

    cfg["nodes_out"].parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[SAVE] Writing nodes: {cfg['nodes_out']}")
    nodes_gdf.to_file(cfg["nodes_out"], driver="GeoJSON")

    propagate_to_graph(nodes_gdf, cfg["graph_in"], cfg["graph_out"])

    cfg["summary"].write_text("\n".join(summary_blocks))
    print(f"[SAVE] Summary: {cfg['summary']}")

    print("\n" + "=" * 72)
    print("DONE")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())