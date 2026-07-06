# DEP Mid-Century flood scenario — export for Jesse (pandapower cascade)

## Scenario
- **Plain label:** Mid-Century
- **Full label:** Mid-Century (DEP Moderate, 2050s)
- **Source column:** `flood_moderate_2050_depth_m` (depth in **meters**)
- **Source dataset:** NYC DEP Stormwater Flood Maps — pluvial/tidal compound
  (citywide stormwater + tidal flooding overlay), Moderate scenario,
  2050s horizon.
- **Source file:** `data/flood/nyc_infra_nodes_dep_flood.geojson` (DEP-only node overlay)

### Naming note
The original request expected a column literally named `slr21_aep10`. That column does **not** exist in this dataset. The DEP scenarios are named `moderate_current` / `moderate_2050` / `extreme_2080`. "Mid-Century" was mapped (after confirmation) to **`flood_moderate_2050_depth_m`**, and the output files are named after that real scenario.

## Coordinates
- **Original CRS:** EPSG:4326
- **Output CRS:** EPSG:4326 (reprojected for portability; lon/lat decimal degrees)

## Units & depth encoding
Depths are in meters. DEP publishes flood categories that map to representative depths:
`0 → 0.00 m` (not flooded), `1 → 0.20 m` (nuisance, 4–12 in),
`2 → 0.60 m` (deep & contiguous, ≥ 1 ft), `3 → 0.80 m` (future high tides).

## Coverage (this scenario)
- Total nodes: **6231**
- Nodes flooded citywide (depth > 0): **97**
- Power nodes: **203**
- Power nodes flooded (depth > 0): **0**

> Companion scenario: **Late-Century** (Late-Century (DEP Extreme, 2080s)) flags **10**
> of 203 power nodes as flooded — exported separately at `exports/flood_map_for_jesse_2080/`.

## Files
- `dep_moderate_2050_nodes.geojson` — all 6231 infrastructure nodes
  (`node_id`, `infra_type`, `geometry`, `flood_depth_m`).
- `power_nodes_flood_depth.csv` — power nodes only
  (`node_id`, `infra_type`, `lon`, `lat`, `flood_depth_m`, `flooded`).
- `preview.png` — quick-look map: all nodes light grey, power nodes highlighted
  (dry = grey, flooded = colored by depth).
- `README.md` — this file.

## Caveat — GeoClaw surge scenarios intentionally excluded
This bundle uses **DEP stormwater/tidal** depths only. GeoClaw storm-surge
scenarios (`gc_2026` / `gc_2050` / `gc_2080`) are **deliberately excluded** due to
an unresolved eta-vs-depth bug that over-floods by roughly **6×** (water-surface
elevation `eta` is being used directly instead of subtracting ground elevation /
DEM). Those scenarios should not be used for cascade analysis pending DEM
subtraction. The DEP-only source file used here contains no GeoClaw columns.
