#!/usr/bin/env python3
"""
flood_overlay_geoclaw_1.py

Overlay GeoClaw "Sandy-like" peak flood scenarios (2026 / 2050 / 2080) onto the
infrastructure node set, adding 3 flood-depth columns per node.

DEPTH CORRECTION (2026-07-06)
  The fgout rasters store eta = WATER-SURFACE ELEVATION (m, NAVD88), not depth.
  Diagnostic evidence: at the Battery, gc_2026 eta = 3.16 m where ground is
  ~2-3 m NAVD88 (NOAA Sandy peak at the Battery gauge: 3.44 m NAVD88); dry
  cells are NaN; scenario medians 3.09/3.53/4.00 m are water-surface levels,
  not plausible standing-water depths. Prior behaviour sampled eta directly
  as depth, saturating HAZUS fragility curves (~6x over-flooding).

  Corrected conversion, per node:
      dry  (eta = NaN)  ->  depth = 0
      wet  (eta valid)  ->  depth = max(eta - ground, 0)
  where ground = USGS 3DEP 1/3 arc-second DEM (m, NAVD88) sampled at the node.
  A wet-footprint node whose ground sits above the local water surface is
  therefore correctly reclassified as DRY.

INPUTS
  data/flood/<scope>_infra_nodes_dep_flood.geojson    (DEP overlay output)
  data/flood/<scope>_infra_graph_dep_flood.graphml
  data/raw/flood_geoclaw/fgout_eta_{2026,2050,2080}_360.tif   (eta, NAVD88)
  data/raw/dem/USGS_13_n41w074.tif                     (ground, NAVD88)
  data/raw/dem/USGS_13_n41w075.tif

OUTPUTS
  data/flood/<scope>_infra_nodes_all_flood.geojson    (DEP + GeoClaw columns)
  data/flood/<scope>_infra_graph_all_flood.graphml
  data/flood/geoclaw_flood_summary.txt

USAGE
  python src/flood/flood_overlay_geoclaw.py --scope nyc
"""

import argparse
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import rasterio
from rasterio.warp import transform as warp_transform


# ─── Config ────────────────────────────────────────────────────────────────────
SCENARIOS = [
    ("gc_2026", "data/raw/flood_geoclaw/fgout_eta_2026_360.tif"),
    ("gc_2050", "data/raw/flood_geoclaw/fgout_eta_2050_360.tif"),
    ("gc_2080", "data/raw/flood_geoclaw/fgout_eta_2080_360.tif"),
]

DEM_PATHS = [
    "data/raw/dem/USGS_13_n41w074.tif",
    "data/raw/dem/USGS_13_n41w075.tif",
]

# Startup datum sanity check: (lon, lat, label, plausible NAVD88 range in m)
DEM_REF_POINTS = [
    (-73.9665, 40.7812, "Central Park Great Lawn", (20.0, 45.0)),
    (-73.9855, 40.7580, "Times Square",            (10.0, 25.0)),
    (-74.0170, 40.7033, "The Battery",             (1.0, 5.0)),
]


# ─── CLI ───────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--scope", choices=["lm", "nyc"], default="nyc")
ap.add_argument("--nodes-in", default=None, help="Override input nodes geojson")
ap.add_argument("--graph-in", default=None, help="Override input graphml")
ap.add_argument("--dem", nargs="+", default=None,
                help="Override DEM tile path(s) (m, NAVD88)")
args = ap.parse_args()

scope = args.scope
NODES_IN = Path(args.nodes_in or f"data/flood/{scope}_infra_nodes_dep_flood.geojson")
GRAPH_IN = Path(args.graph_in or f"data/flood/{scope}_infra_graph_dep_flood.graphml")
NODES_OUT = Path(f"data/flood/{scope}_infra_nodes_all_flood.geojson")
GRAPH_OUT = Path(f"data/flood/{scope}_infra_graph_all_flood.graphml")
SUMMARY = Path("data/flood/geoclaw_flood_summary.txt")
dem_paths = [Path(p) for p in (args.dem or DEM_PATHS)]


# ─── DEM sampling ──────────────────────────────────────────────────────────────
def sample_raster(path, lons, lats):
    """Sample band 1 at WGS84 lon/lat points; nodata/off-grid -> NaN."""
    out = np.full(len(lons), np.nan)
    with rasterio.open(path) as src:
        if src.crs and src.crs.to_epsg() in (4326, 4269):
            sx, sy = np.asarray(lons), np.asarray(lats)
        else:
            sx, sy = warp_transform("EPSG:4326", src.crs,
                                    list(lons), list(lats))
            sx, sy = np.asarray(sx), np.asarray(sy)
        left, bottom, right, top = src.bounds
        inside = (sx >= left) & (sx <= right) & (sy >= bottom) & (sy <= top)
        if not inside.any():
            return out
        vals = np.array(
            [v[0] for v in src.sample(zip(sx[inside], sy[inside]))],
            dtype=float,
        )
        if src.nodata is not None and not np.isnan(src.nodata):
            vals[vals == src.nodata] = np.nan
        # USGS tiles use huge negative fill values; anything wildly below
        # bathymetric range is fill, not ground.
        vals[vals < -100] = np.nan
        out[inside] = vals
    return out


def sample_dem(lons, lats, paths):
    """Ground elevation (m NAVD88) per point, first covering tile wins."""
    ground = np.full(len(lons), np.nan)
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(
                f"DEM tile missing: {p} — download the USGS 3DEP 1/3\" tiles "
                f"(see module docstring) or pass --dem."
            )
        need = np.isnan(ground)
        if not need.any():
            break
        vals = sample_raster(p, np.asarray(lons)[need], np.asarray(lats)[need])
        ground[np.flatnonzero(need)] = vals
    return ground


# ─── Load nodes ────────────────────────────────────────────────────────────────
print("=" * 70)
print(f"GeoClaw flood overlay  —  scope={scope.upper()}")
print("Mode: DEPTH = eta − DEM  (eta: GeoClaw water surface; DEM: USGS 3DEP,")
print("      both m NAVD88; dry cells are NaN in the eta rasters)")
print("=" * 70)

if not NODES_IN.exists():
    raise FileNotFoundError(f"{NODES_IN} not found. Run flood_overlay_v3.py first.")

nodes = gpd.read_file(NODES_IN)
print(f"\nLoaded {len(nodes):,} nodes from {NODES_IN}  (CRS: {nodes.crs})")

nodes_wgs = nodes if nodes.crs and nodes.crs.to_epsg() == 4326 else nodes.to_crs(4326)
xs = nodes_wgs.geometry.x.values
ys = nodes_wgs.geometry.y.values


# ─── Ground elevation per node + datum sanity check ────────────────────────────
print("\nSampling ground elevation (USGS 3DEP 1/3\", NAVD88) at all nodes...")
ground = sample_dem(xs, ys, dem_paths)
n_no_ground = int(np.isnan(ground).sum())
g_valid = ground[~np.isnan(ground)]
print(f"  Ground range: {g_valid.min():.1f} – {g_valid.max():.1f} m "
      f"(median {np.median(g_valid):.1f} m)")
if n_no_ground:
    print(f"  [WARN] {n_no_ground} nodes have no DEM coverage — "
          f"treated as DRY in all GeoClaw scenarios (listed in summary).")

print("  Datum sanity check (DEM at reference points):")
ref_vals = sample_dem([p[0] for p in DEM_REF_POINTS],
                      [p[1] for p in DEM_REF_POINTS], dem_paths)
datum_ok = True
for (lo, la, label, (rng_lo, rng_hi)), v in zip(DEM_REF_POINTS, ref_vals):
    status = "OK" if (not np.isnan(v) and rng_lo <= v <= rng_hi) else "SUSPECT"
    datum_ok &= status == "OK"
    print(f"    {label:<24} {v:>7.1f} m   expected {rng_lo:.0f}–{rng_hi:.0f}  [{status}]")
if not datum_ok:
    raise RuntimeError("DEM reference values out of plausible NAVD88 range — "
                       "wrong tiles or wrong datum. Not proceeding.")


# ─── Sample each scenario: depth = max(eta − ground, 0) ────────────────────────
summary_lines = [
    f"GeoClaw flood overlay — scope={scope.upper()}",
    "Depth model: depth = max(eta − ground, 0); eta NaN => dry",
    "eta: GeoClaw fgout water-surface elevation (m, NAVD88)",
    f"ground: USGS 3DEP 1/3 arc-second DEM (m, NAVD88): "
    f"{', '.join(p.name for p in dem_paths)}",
    f"Input: {NODES_IN}",
    f"Total nodes: {len(nodes):,}   (no DEM coverage: {n_no_ground})",
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
    eta = sample_raster(path, xs, ys)

    in_footprint = ~np.isnan(eta)                       # old code's "wet"
    depth = np.where(in_footprint & ~np.isnan(ground),
                     np.maximum(eta - ground, 0.0), 0.0)
    wet = depth > 0
    reclassified_dry = in_footprint & ~wet              # ground above water

    nodes[f"{col}_depth_m"] = depth

    n_fp, n_wet, n_re = int(in_footprint.sum()), int(wet.sum()), int(reclassified_dry.sum())
    print(f"  In eta footprint (old 'flooded'): {n_fp:,}")
    print(f"  Wet after eta−DEM:                {n_wet:,}   "
          f"({n_re:,} reclassified dry: ground above water surface)")
    if n_wet:
        d = depth[wet]
        print(f"  Depth range:  {d.min():.2f} – {d.max():.2f} m   "
              f"median {np.median(d):.2f} m   mean {d.mean():.2f} m")

    summary_lines.append(f"--- {col} ---")
    summary_lines.append(
        f"In eta footprint: {n_fp:,}   wet after eta−DEM: {n_wet:,}   "
        f"reclassified dry: {n_re:,}"
    )
    if n_wet:
        d = depth[wet]
        summary_lines.append(
            f"Depth (wet only): min {d.min():.2f}  median {np.median(d):.2f}  "
            f"mean {d.mean():.2f}  max {d.max():.2f} m"
        )
    if "type" in nodes.columns and n_wet:
        per_type = nodes.loc[wet, "type"].value_counts().sort_index()
        for t, n in per_type.items():
            total = int((nodes["type"] == t).sum())
            summary_lines.append(f"  {t:<14} {n:>5} / {total:<5}  ({100 * n / total:.0f}%)")
    summary_lines.append("")

if n_no_ground:
    no_dem_ids = nodes.loc[np.isnan(ground), :]
    id_col_tmp = next((c for c in ("node_id", "id", "osm_id")
                       if c in nodes.columns), None)
    if id_col_tmp:
        summary_lines.append("Nodes without DEM coverage (forced dry):")
        summary_lines += [f"  {v}" for v in no_dem_ids[id_col_tmp].tolist()]
        summary_lines.append("")


# ─── Save updated nodes ────────────────────────────────────────────────────────
NODES_OUT.parent.mkdir(parents=True, exist_ok=True)
nodes.to_file(NODES_OUT, driver="GeoJSON")
print(f"\n✓ Saved nodes  → {NODES_OUT}")


# ─── Propagate to graph ────────────────────────────────────────────────────────
print("\nPropagating flood attributes to graph nodes...")
G = nx.read_graphml(GRAPH_IN)

id_col = next((c for c in ("node_id", "id", "osm_id") if c in nodes.columns), None)
if id_col is None:
    raise RuntimeError(f"No node-id column in {NODES_IN}; columns: {list(nodes.columns)}")

id_to_idx = {str(nid): i for i, nid in enumerate(nodes[id_col])}
added_cols = [f"{c}_depth_m" for c, _ in SCENARIOS if f"{c}_depth_m" in nodes.columns]

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
    print(f"  Missing:  {n_missing:,} graph nodes had no match (id_col='{id_col}')")

nx.write_graphml(G, GRAPH_OUT)
print(f"✓ Saved graph  → {GRAPH_OUT}")

SUMMARY.parent.mkdir(parents=True, exist_ok=True)
SUMMARY.write_text("\n".join(summary_lines))
print(f"✓ Saved summary → {SUMMARY}")