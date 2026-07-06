# NYC flood maps — 6 scenarios (pandapower cascade)

Six mappable flood scenarios sampled onto the same 6,231 infrastructure nodes, in two families. Each subfolder is a self-contained bundle (all-nodes GeoJSON + power-only CSV + preview PNG + README).

## Summary

| # | Folder | Family | Scenario | Total flooded | Power flooded | Max depth |
|---|--------|--------|----------|--------------:|--------------:|----------:|
| 1 | `01_dep_moderate_current` | DEP | Present-day (DEP Moderate, current) | 61 | 0/203 | 0.60 m |
| 2 | `02_dep_moderate_2050` | DEP | Mid-Century (DEP Moderate, 2050s) | 97 | 0/203 | 0.80 m |
| 3 | `03_dep_extreme_2080` | DEP | Late-Century (DEP Extreme, 2080s) | 458 | 10/203 | 0.80 m |
| 4 | `04_geoclaw_gc_2026` | GeoClaw | GeoClaw storm-surge, 2026 horizon (PROVISIONAL) | 373 | 25/203 | 3.72 m |
| 5 | `05_geoclaw_gc_2050` | GeoClaw | GeoClaw storm-surge, 2050 horizon (PROVISIONAL) | 504 | 31/203 | 4.48 m |
| 6 | `06_geoclaw_gc_2080` | GeoClaw | GeoClaw storm-surge, 2080 horizon (PROVISIONAL) | 640 | 32/203 | 6.31 m |

## ⚠️ Read this first — DEP is analysis-grade, GeoClaw is PROVISIONAL

- **DEP** scenarios (`moderate_current`, `moderate_2050`, `extreme_2080`) are the clean, analysis-grade maps — NYC DEP Stormwater Flood Maps (pluvial/tidal compound), categorical depths (0 / 0.20 / 0.60 / 0.80 m).
- **GeoClaw** scenarios (`gc_2026`, `gc_2050`, `gc_2080`) are storm-surge model outputs with a **known, unresolved over-flood bug (~6x)**: water-surface elevation (`eta`) is used directly as depth instead of subtracting ground elevation (DEM). Depths are inflated (up to 6.3 m vs DEP's 0.80 m cap) and far too many nodes read as flooded. **Do not use the GeoClaw trio for cascade analysis** until the DEM subtraction is fixed; they are included only because all six scenarios were requested, and each carries a warning banner in its own README.

## Mapping the data yourself

A ready-to-run script, `map_flood_scenarios.py`, is included in this folder. It
reads the `*_nodes.geojson` files with only the Python standard library (no
geopandas needed) and renders maps with matplotlib.

```bash
pip install matplotlib          # required
pip install folium              # optional — only for interactive HTML maps

# run from inside this folder:
python map_flood_scenarios.py                 # static PNGs for all 6 -> ./maps/
python map_flood_scenarios.py --interactive   # also folium HTML maps (real basemap)
python map_flood_scenarios.py --all-infra     # color every flooded node, not just power
python map_flood_scenarios.py --scenario 04_geoclaw_gc_2026   # just one
```

It writes per-scenario maps plus a combined 6-panel comparison grid into a
`maps/` subfolder. GeoClaw scenarios are labeled in red as PROVISIONAL. You can
of course ignore the script and load the GeoJSON/CSV straight into
geopandas/QGIS/pandapower — the columns are `node_id`, `infra_type`, geometry,
`flood_depth_m` (meters), plus `lon`/`lat`/`flooded` in the CSVs.

## Coordinates

- Original CRS: EPSG:4326
- Output CRS: EPSG:4326 (lon/lat decimal degrees)

