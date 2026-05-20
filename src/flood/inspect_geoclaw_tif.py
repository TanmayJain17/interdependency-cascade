# src/flood/inspect_geoclaw_tif.py
import rasterio
import numpy as np
from pathlib import Path

TIF = Path("data/raw/flood_geoclaw/fgout_eta_2026_360.tif")  # adjust if needed

with rasterio.open(TIF) as src:
    print(f"=== {TIF.name} ===")
    print(f"CRS:        {src.crs}")
    print(f"Size:       {src.width} x {src.height} = {src.width*src.height:,} cells")
    print(f"Resolution: {src.res} (units of CRS)")
    print(f"Bounds:     {src.bounds}")
    print(f"Bands:      {src.count}, dtype: {src.dtypes[0]}")
    print(f"Nodata:     {src.nodata}")

    arr = src.read(1)

    # treat nodata + nan as invalid
    mask = np.isfinite(arr)
    if src.nodata is not None:
        mask &= (arr != src.nodata)
    valid = arr[mask]

    print(f"\nValid cells: {valid.size:,} / {arr.size:,} "
          f"({100*valid.size/arr.size:.1f}%)")
    if valid.size:
        print(f"Min:    {valid.min():.3f}")
        print(f"Max:    {valid.max():.3f}")
        print(f"Mean:   {valid.mean():.3f}")
        print(f"Median: {np.median(valid):.3f}")
        print(f"\nDistribution:")
        for q in [10, 25, 50, 75, 90, 95, 99]:
            print(f"  p{q:>2}: {np.percentile(valid, q):.3f}")
        print(f"\n  cells > 0:    {(valid > 0).sum():,}")
        print(f"  cells > 0.1m: {(valid > 0.1).sum():,}")
        print(f"  cells > 1.0m: {(valid > 1.0).sum():,}")
        print(f"  cells > 3.0m: {(valid > 3.0).sum():,}")
        print(f"  cells negative: {(valid < 0).sum():,}")