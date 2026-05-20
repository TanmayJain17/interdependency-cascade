import rasterio
from pathlib import Path

for name in ["fgout_eta_2026_360.tif", "fgout_eta_2050_360.tif", "fgout_eta_2080_360.tif"]:
    p = Path("data/raw/flood_geoclaw") / name
    if not p.exists():
        print(f"  not found: {p}")
        continue
    with rasterio.open(p) as src:
        print(f"{name}")
        print(f"  CRS:    {src.crs}")
        print(f"  dtype:  {src.dtypes[0]}")
        print(f"  bounds: {src.bounds}")
        print(f"  size:   {src.width} x {src.height}")
        print()
