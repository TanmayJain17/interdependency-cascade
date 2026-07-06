# Late-Century (DEP Extreme, 2080s) — export for Jesse (pandapower cascade)

## Scenario
- **Family:** DEP
- **Plain label:** Late-Century
- **Full label:** Late-Century (DEP Extreme, 2080s)
- **Source column:** `flood_extreme_2080_depth_m` (depth in **meters**)
- **Source dataset:** NYC DEP Stormwater Flood Maps — pluvial/tidal compound, Extreme scenario, 2080s horizon.
- **Source file:** `data/flood/nyc_infra_nodes_all_flood.geojson`

## Coordinates
- **Original CRS:** EPSG:4326
- **Output CRS:** EPSG:4326 (lon/lat decimal degrees)

## Units & depth encoding
Depths are in meters. DEP publishes flood *categories* that map to representative depths: `0 -> 0.00 m` (not flooded), `1 -> 0.20 m` (nuisance, 4-12 in), `2 -> 0.60 m` (deep & contiguous, >= 1 ft), `3 -> 0.80 m` (future high tides).

## Coverage (this scenario)
- Total nodes: **6231**
- Nodes flooded citywide (depth > 0): **458**
- Power nodes: **203**
- Power nodes flooded (depth > 0): **10**
- Max depth: **0.80 m**

## Files
- `dep_extreme_2080_nodes.geojson` — all 6231 infrastructure nodes
  (`node_id`, `infra_type`, `geometry`, `flood_depth_m`).
- `power_nodes_flood_depth.csv` — power nodes only
  (`node_id`, `infra_type`, `lon`, `lat`, `flood_depth_m`, `flooded`).
- `preview.png` — quick-look map: all nodes light grey, power nodes highlighted
  (dry = grey, flooded = colored by depth).
- `README.md` — this file.

See the parent `README.md` for the full 6-scenario summary table and the
DEP-vs-GeoClaw caveat.
