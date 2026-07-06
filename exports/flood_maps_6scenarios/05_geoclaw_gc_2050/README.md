# GeoClaw storm-surge, 2050 horizon (PROVISIONAL) — export for Jesse (pandapower cascade)

> # ⚠️ PROVISIONAL — DO NOT USE FOR CASCADE ANALYSIS ⚠️
>
> This is a **GeoClaw storm-surge** scenario with a **known, unresolved
> over-flood bug (~6x)**. The water-surface elevation (`eta`) is being used
> directly as depth instead of subtracting ground elevation (DEM), so:
>
> - depths are inflated (this scenario reaches **4.48 m**; DEP caps at 0.80 m),
> - far too many nodes read as flooded (**504** citywide / **31** power
>   nodes here, vs DEP present-day's 61 / 0).
>
> Treat these numbers as a **provisional upper bound only**. The three DEP
> scenarios in this bundle (`moderate_current`, `moderate_2050`,
> `extreme_2080`) are the analysis-grade maps; use those for cascade work
> until the DEM-subtraction fix lands.


## Scenario
- **Family:** GeoClaw
- **Plain label:** GeoClaw surge 2050
- **Full label:** GeoClaw storm-surge, 2050 horizon (PROVISIONAL)
- **Source column:** `gc_2050_depth_m` (depth in **meters**)
- **Source dataset:** GeoClaw storm-surge model, 2050 horizon, depth sampled from raster (PROVISIONAL — see warning).
- **Source file:** `data/flood/nyc_infra_nodes_all_flood.geojson`

## Coordinates
- **Original CRS:** EPSG:4326
- **Output CRS:** EPSG:4326 (lon/lat decimal degrees)

## Units & depth encoding
Depths are continuous meters sampled directly from the GeoClaw surge raster (no categorical binning).

## Coverage (this scenario)
- Total nodes: **6231**
- Nodes flooded citywide (depth > 0): **504**
- Power nodes: **203**
- Power nodes flooded (depth > 0): **31**
- Max depth: **4.48 m**

## Files
- `geoclaw_gc_2050_nodes.geojson` — all 6231 infrastructure nodes
  (`node_id`, `infra_type`, `geometry`, `flood_depth_m`).
- `power_nodes_flood_depth.csv` — power nodes only
  (`node_id`, `infra_type`, `lon`, `lat`, `flood_depth_m`, `flooded`).
- `preview.png` — quick-look map: all nodes light grey, power nodes highlighted
  (dry = grey, flooded = colored by depth).
- `README.md` — this file.

See the parent `README.md` for the full 6-scenario summary table and the
DEP-vs-GeoClaw caveat.
