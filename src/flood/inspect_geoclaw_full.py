# src/flood/inspect_geoclaw_full.py
import rasterio
from pathlib import Path

TIF = Path("data/raw/flood_geoclaw/fgout_eta_2026_peak.tif")

with rasterio.open(TIF) as src:
    print(f"=== {TIF.name} ===\n")

    # Driver & basic
    print(f"Driver: {src.driver}")
    print(f"Compression: {src.compression}")
    print(f"Tags (dataset): {src.tags()}")
    print()

    # Per-band info — this is where scale/offset live
    for i in range(1, src.count + 1):
        print(f"--- Band {i} ---")
        print(f"  dtype:       {src.dtypes[i-1]}")
        print(f"  nodata:      {src.nodatavals[i-1]}")
        print(f"  scale:       {src.scales[i-1]}")     # << key: not 1.0 means values are scaled
        print(f"  offset:      {src.offsets[i-1]}")    # << key: not 0.0 means values are offset
        print(f"  units:       {src.units[i-1]}")
        print(f"  description: {src.descriptions[i-1]}")
        print(f"  tags:        {src.tags(i)}")
        # Color table check (would mean it's a categorical/visualization raster)
        try:
            cmap = src.colormap(i)
            print(f"  colormap:    {len(cmap)} entries  (first 5: "
                  f"{ {k: cmap[k] for k in list(cmap)[:5]} })")
        except ValueError:
            print(f"  colormap:    none")
        print()

# Sidecar files in the same directory
print("=== Sidecar files ===")
for p in sorted(TIF.parent.iterdir()):
    if TIF.stem in p.name and p != TIF:
        print(f"  {p.name}  ({p.stat().st_size:,} bytes)")