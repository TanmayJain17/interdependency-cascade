#!/usr/bin/env python3
"""
diagnose_eta_dem.py — Phase-0 discovery for the eta-vs-depth fix.

Answers two questions before we touch flood_overlay_geoclaw.py:
  Q1. What do DRY cells contain in the fgout eta rasters?
      (ground elevation? zero? nodata?)  -> determines the masking rule.
  Q2. Is there a DEM/topo raster anywhere in data/ we can subtract,
      and do its values line up with eta over known dry ground?
      -> determines the DEM source + datum offset.

Run:  python src/flood/diagnose_eta_dem.py
"""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform as warp_transform

GEOCLAW_DIR = Path("data/raw/flood_geoclaw")
ETA_TIFS = sorted(GEOCLAW_DIR.glob("fgout_eta_*_360.tif"))

# Reference points (lon, lat, label, expected ground ~NAVD88)
REF_POINTS = [
    (-73.9665, 40.7812, "Central Park Great Lawn (high, dry)", "~30 m"),
    (-73.9855, 40.7580, "Times Square (mid, dry)",             "~15 m"),
    (-74.0170, 40.7033, "The Battery (low, Sandy-flooded)",    "~2-3 m"),
    (-74.0450, 40.6660, "NY Harbor open water",                "< 0 m (bathymetry)"),
]


def sample_at(src, lons, lats):
    """Sample band 1 at WGS84 points, reprojecting into the raster CRS."""
    crs_str = src.crs.to_wkt() if src.crs else ""
    if "WGS 84" in crs_str or "WGS_1984" in crs_str or "4326" in str(src.crs):
        xs, ys = lons, lats
    else:
        xs, ys = warp_transform("EPSG:4326", src.crs, list(lons), list(lats))
    return np.array([v[0] for v in src.sample(zip(xs, ys))], dtype=float)


print("=" * 72)
print("PART 1 — everything in data/raw/flood_geoclaw/")
print("=" * 72)
for p in sorted(GEOCLAW_DIR.iterdir()):
    print(f"  {p.name:<45} {p.stat().st_size:>12,} bytes")

print()
print("=" * 72)
print("PART 2 — eta raster anatomy + reference-point samples")
print("=" * 72)
lons = [p[0] for p in REF_POINTS]
lats = [p[1] for p in REF_POINTS]

for tif in ETA_TIFS:
    with rasterio.open(tif) as src:
        band = src.read(1).astype(float)
        nodata = src.nodata
        print(f"\n--- {tif.name} ---")
        print(f"  CRS: {src.crs}   size: {src.width}x{src.height}   "
              f"res: {src.res}   dtype: {src.dtypes[0]}   nodata: {nodata}")
        valid = band[~np.isnan(band)]
        if nodata is not None and not np.isnan(nodata):
            n_nodata = int((band == nodata).sum())
            valid = valid[valid != nodata]
        else:
            n_nodata = int(np.isnan(band).sum())
        print(f"  pixels: total={band.size:,}  nodata/nan={n_nodata:,}  "
              f"==0: {int((valid == 0).sum()):,}  <0: {int((valid < 0).sum()):,}  "
              f">0: {int((valid > 0).sum()):,}")
        print(f"  valid value range: {valid.min():.2f} .. {valid.max():.2f}   "
              f"median: {np.median(valid):.2f}")
        vals = sample_at(src, lons, lats)
        print("  reference points (eta value @ point):")
        for (lo, la, label, exp), v in zip(REF_POINTS, vals):
            print(f"    {label:<38} eta={v:>8.2f}   (ground {exp})")

print()
print("=" * 72)
print("PART 3 — DEM candidates anywhere under data/")
print("=" * 72)
KEYWORDS = ("dem", "topo", "elev", "lidar", "bathy", "terrain")
candidates = [
    p for p in Path("data").rglob("*")
    if p.is_file()
    and p.suffix.lower() in (".tif", ".tiff", ".asc", ".img", ".nc", ".vrt")
    and any(k in p.name.lower() for k in KEYWORDS)
]
if not candidates:
    print("  (none found by keyword — also listing ALL rasters under data/raw/)")
    candidates = [p for p in Path("data/raw").rglob("*")
                  if p.suffix.lower() in (".tif", ".tiff", ".asc", ".img", ".vrt")]
for p in candidates:
    print(f"\n  {p}  ({p.stat().st_size:,} bytes)")
    try:
        with rasterio.open(p) as src:
            print(f"    CRS: {src.crs}  size: {src.width}x{src.height}  "
                  f"res: {src.res}  dtype: {src.dtypes[0]}")
            vals = sample_at(src, lons, lats)
            for (lo, la, label, exp), v in zip(REF_POINTS, vals):
                print(f"    {label:<38} value={v:>8.2f}   (ground {exp})")
    except Exception as e:
        print(f"    [could not open as raster: {e}]")

print("\nDone. Paste this full output back.")