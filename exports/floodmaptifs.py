#!/usr/bin/env python3
"""
export_flood_rasters.py — six flood-depth GeoTIFFs for external consumers (Jesse).

One common grid (the GeoClaw fgout grid, EPSG:4326, ~7 m), one convention:
  pixel value = water depth in meters; 0 = dry; NaN = no data (outside DEM).

GeoClaw scenarios: depth = max(eta − ground, 0), ground = USGS 3DEP 1/3" DEM
(both m NAVD88), reprojected onto the eta grid. NEVER ship raw fgout eta —
those are water-SURFACE elevations, not depths.

DEP scenarios: NYC DEP publishes flood-class POLYGONS, not depths. We burn the
class-representative depths used across this project (Nuisance=0.20 m,
Deep=0.60 m, FutureHighTides=0.80 m) onto the same grid. These are class
depths, not continuous depths — stated in the README.

Run:  python src/flood/export_flood_rasters.py     (~5–10 min, ~2 GB RAM)
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features
from rasterio.warp import reproject, Resampling

GC = {
    "gc_2026": "data/raw/flood_geoclaw/fgout_eta_2026_360.tif",
    "gc_2050": "data/raw/flood_geoclaw/fgout_eta_2050_360.tif",
    "gc_2080": "data/raw/flood_geoclaw/fgout_eta_2080_360.tif",
}
DEP = {
    "dep_moderate_current": "data/raw/flood/moderate_current.geojson",
    "dep_moderate_2050":    "data/raw/flood/moderate_2050.geojson",
    "dep_extreme_2080":     "data/raw/flood/extreme_2080.geojson",
}
DEM_TILES = ["data/raw/dem/USGS_13_n41w074.tif", "data/raw/dem/USGS_13_n41w075.tif"]
CATEGORY_DEPTH_M = {0: 0.00, 1: 0.20, 2: 0.60, 3: 0.80}
OUT = Path("data/export/flood_rasters_for_jesse")
OUT.mkdir(parents=True, exist_ok=True)

# ---- reference grid + profile from the first eta raster -----------------
with rasterio.open(next(iter(GC.values()))) as ref:
    profile = ref.profile.copy()
    grid_transform, grid_crs = ref.transform, ref.crs
    H, W = ref.height, ref.width
profile.update(dtype="float32", count=1, nodata=np.nan,
               compress="deflate", predictor=3,
               tiled=True, blockxsize=256, blockysize=256)

# ---- DEM mosaic resampled onto the eta grid (once) ----------------------
print("Reprojecting DEM tiles onto the GeoClaw grid...")
ground = np.full((H, W), np.nan, dtype="float32")
for tile in DEM_TILES:
    with rasterio.open(tile) as src:
        tmp = np.full((H, W), np.nan, dtype="float32")
        reproject(rasterio.band(src, 1), tmp,
                  dst_transform=grid_transform, dst_crs=grid_crs,
                  resampling=Resampling.bilinear,
                  src_nodata=src.nodata, dst_nodata=np.nan)
        take = np.isnan(ground) & ~np.isnan(tmp)
        ground[take] = tmp[take]
ground[ground < -100] = np.nan          # USGS fill values
g = ground[~np.isnan(ground)]
print(f"  ground: {g.min():.1f}–{g.max():.1f} m (median {np.median(g):.1f}) "
      f"— expect Central Park ~30 m territory")

def write(name, arr, note):
    path = OUT / f"{name}_depth_m.tif"
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype("float32"), 1)
        dst.update_tags(units="meters", convention="0=dry, NaN=no-data",
                        crs="EPSG:4326", note=note)
    wet = arr[np.nan_to_num(arr) > 0]
    print(f"  {path.name:<34} wet px {wet.size:>10,}  "
          f"median {np.median(wet):.2f} m  max {wet.max():.2f} m")

# ---- GeoClaw: depth = max(eta − ground, 0) -------------------------------
print("\nGeoClaw scenarios (eta − DEM):")
for name, p in GC.items():
    with rasterio.open(p) as src:
        eta = src.read(1).astype("float32")
    depth = np.where(np.isnan(eta), 0.0,
                     np.where(np.isnan(ground), np.nan,
                              np.maximum(eta - ground, 0.0)))
    write(name, depth, "GeoClaw Sandy-like surge; depth = eta(NAVD88) - USGS 3DEP DEM(NAVD88)")
    del eta, depth

# ---- DEP: rasterize class polygons ---------------------------------------
print("\nDEP scenarios (class polygons -> representative depths):")
for name, p in DEP.items():
    gdf = gpd.read_file(p).to_crs(grid_crs)
    cat_col = next(c for c in gdf.columns if gdf[c].dtype.kind in "if"
                   and set(gdf[c].dropna().unique()) <= set(CATEGORY_DEPTH_M))
    shapes = ((geom, CATEGORY_DEPTH_M[int(c)])
              for geom, c in zip(gdf.geometry, gdf[cat_col]) if geom is not None)
    arr = features.rasterize(shapes, out_shape=(H, W), transform=grid_transform,
                             fill=0.0, dtype="float32")
    write(name, arr, "NYC DEP stormwater class polygons; values are CLASS-REPRESENTATIVE "
                     "depths (0.20/0.60/0.80 m), not continuous; excludes storm surge")

# ---- README ---------------------------------------------------------------
(OUT / "README.txt").write_text(f"""FLOOD DEPTH RASTERS — 6 scenarios, one grid
Grid: EPSG:4326, {W}x{H}, ~7 m px (GeoClaw fgout grid). Units: meters of water depth.
Convention: 0 = dry land, NaN = no data (outside DEM coverage / open water).

gc_2026 / gc_2050 / gc_2080:
  Sandy-like storm surge at 2026/2050/2080 sea level (GeoClaw).
  depth = max(eta - ground, 0); eta = water-surface elevation (m NAVD88),
  ground = USGS 3DEP 1/3" DEM (m NAVD88). Validated vs NOAA Battery Sandy peak
  (3.44 m NAVD88) and GISSR Sandy LM median (~1.0 m).
  IMPORTANT: do NOT use the raw fgout_eta_*.tif files as depth — they are
  water-SURFACE elevations and overstate depth ~3x.

dep_moderate_current / dep_moderate_2050 / dep_extreme_2080:
  NYC DEP stormwater (rainfall) scenarios. DEP publishes flood-CLASS polygons,
  not depths; values here are class-representative depths (Nuisance=0.20,
  Deep=0.60, FutureHighTides=0.80 m). DEP maps exclude storm surge by design —
  do not sum the two families.
""")
print(f"\nDone -> {OUT}/  (6 tifs + README)")