"""
Export Drive-ready bundles of ALL SIX mappable flood scenarios for a
power-systems collaborator (Jesse) doing pandapower cascade work.

Two scenario families (all sampled onto the same 6,231 infra nodes):

  DEP — NYC DEP Stormwater Flood Maps (pluvial/tidal compound). Clean,
  analysis-grade. Categorical depths (0 / 0.20 / 0.60 / 0.80 m).
    moderate_current -> flood_moderate_current_depth_m ("Present-day", Moderate)
    moderate_2050    -> flood_moderate_2050_depth_m    ("Mid-Century",  Moderate, 2050s)
    extreme_2080     -> flood_extreme_2080_depth_m     ("Late-Century", Extreme,  2080s)

  GeoClaw — storm-surge model, continuous depths sampled from raster.
  *** PROVISIONAL — KNOWN OVER-FLOOD BUG (~6x). *** Water-surface elevation
  (eta) is being used directly instead of subtracting ground elevation (DEM),
  so depths are inflated (max 3.7-6.3 m vs DEP's 0.80 m cap) and far too many
  nodes read as flooded. gc_2026 flags 373 nodes / 25 power nodes vs DEP's 61 / 0.
  DO NOT USE for cascade analysis until the DEM subtraction is fixed. Included
  here only because the collaborator explicitly requested all six scenarios;
  each GeoClaw README carries a loud warning banner.
    gc_2026 -> gc_2026_depth_m   (2026 horizon)
    gc_2050 -> gc_2050_depth_m   (2050 horizon)
    gc_2080 -> gc_2080_depth_m   (2080 horizon)

Source file: data/flood/nyc_infra_nodes_all_flood.geojson (all 6 depth columns).

Each per-scenario bundle contains:
  <scenario>_nodes.geojson      all 6,231 infra nodes (node_id, infra_type,
                                geometry, flood_depth_m)
  power_nodes_flood_depth.csv   power nodes only (node_id, lon, lat, depth, flooded)
  preview.png                   quick-look map (power nodes highlighted by depth)
  README.md                     scenario-specific notes + caveats

A master README.md and a single flood_maps_6scenarios.zip are written at the
parent for easy Drive sharing.

Standalone, read-only w.r.t. the pipeline. Run from project root:
    conda run -n flood python src/export/export_flood_map_for_jesse.py
    conda run -n flood python src/export/export_flood_map_for_jesse.py --scenario gc_2080
"""

import argparse
import shutil
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

SRC_FILE = Path("data/flood/nyc_infra_nodes_all_flood.geojson")
OUT_CRS = "EPSG:4326"
PARENT = Path("exports/flood_maps_6scenarios")

DEP_DEPTH_NOTE = (
    "Depths are in meters. DEP publishes flood *categories* that map to "
    "representative depths: `0 -> 0.00 m` (not flooded), `1 -> 0.20 m` "
    "(nuisance, 4-12 in), `2 -> 0.60 m` (deep & contiguous, >= 1 ft), "
    "`3 -> 0.80 m` (future high tides)."
)
GC_DEPTH_NOTE = (
    "Depths are continuous meters sampled directly from the GeoClaw surge "
    "raster (no categorical binning)."
)

GC_WARNING = """\
> # ⚠️ PROVISIONAL — DO NOT USE FOR CASCADE ANALYSIS ⚠️
>
> This is a **GeoClaw storm-surge** scenario with a **known, unresolved
> over-flood bug (~6x)**. The water-surface elevation (`eta`) is being used
> directly as depth instead of subtracting ground elevation (DEM), so:
>
> - depths are inflated (this scenario reaches **{maxd:.2f} m**; DEP caps at 0.80 m),
> - far too many nodes read as flooded (**{nflood}** citywide / **{pflood}** power
>   nodes here, vs DEP present-day's 61 / 0).
>
> Treat these numbers as a **provisional upper bound only**. The three DEP
> scenarios in this bundle (`moderate_current`, `moderate_2050`,
> `extreme_2080`) are the analysis-grade maps; use those for cascade work
> until the DEM-subtraction fix lands.
"""

SCENARIOS = {
    # ---- DEP family (clean, analysis-grade) --------------------------------
    "moderate_current": {
        "depth_col": "flood_moderate_current_depth_m",
        "family": "DEP",
        "plain": "Present-day",
        "full": "Present-day (DEP Moderate, current)",
        "subdir": "01_dep_moderate_current",
        "geojson_name": "dep_moderate_current_nodes.geojson",
        "source_desc": "NYC DEP Stormwater Flood Maps — pluvial/tidal compound, "
                       "Moderate scenario, present-day horizon.",
        "depth_note": DEP_DEPTH_NOTE,
    },
    "moderate_2050": {
        "depth_col": "flood_moderate_2050_depth_m",
        "family": "DEP",
        "plain": "Mid-Century",
        "full": "Mid-Century (DEP Moderate, 2050s)",
        "subdir": "02_dep_moderate_2050",
        "geojson_name": "dep_moderate_2050_nodes.geojson",
        "source_desc": "NYC DEP Stormwater Flood Maps — pluvial/tidal compound, "
                       "Moderate scenario, 2050s horizon.",
        "depth_note": DEP_DEPTH_NOTE,
    },
    "extreme_2080": {
        "depth_col": "flood_extreme_2080_depth_m",
        "family": "DEP",
        "plain": "Late-Century",
        "full": "Late-Century (DEP Extreme, 2080s)",
        "subdir": "03_dep_extreme_2080",
        "geojson_name": "dep_extreme_2080_nodes.geojson",
        "source_desc": "NYC DEP Stormwater Flood Maps — pluvial/tidal compound, "
                       "Extreme scenario, 2080s horizon.",
        "depth_note": DEP_DEPTH_NOTE,
    },
    # ---- GeoClaw family (PROVISIONAL, over-floods ~6x) ---------------------
    "gc_2026": {
        "depth_col": "gc_2026_depth_m",
        "family": "GeoClaw",
        "plain": "GeoClaw surge 2026",
        "full": "GeoClaw storm-surge, 2026 horizon (PROVISIONAL)",
        "subdir": "04_geoclaw_gc_2026",
        "geojson_name": "geoclaw_gc_2026_nodes.geojson",
        "source_desc": "GeoClaw storm-surge model, 2026 horizon, depth sampled "
                       "from raster (PROVISIONAL — see warning).",
        "depth_note": GC_DEPTH_NOTE,
    },
    "gc_2050": {
        "depth_col": "gc_2050_depth_m",
        "family": "GeoClaw",
        "plain": "GeoClaw surge 2050",
        "full": "GeoClaw storm-surge, 2050 horizon (PROVISIONAL)",
        "subdir": "05_geoclaw_gc_2050",
        "geojson_name": "geoclaw_gc_2050_nodes.geojson",
        "source_desc": "GeoClaw storm-surge model, 2050 horizon, depth sampled "
                       "from raster (PROVISIONAL — see warning).",
        "depth_note": GC_DEPTH_NOTE,
    },
    "gc_2080": {
        "depth_col": "gc_2080_depth_m",
        "family": "GeoClaw",
        "plain": "GeoClaw surge 2080",
        "full": "GeoClaw storm-surge, 2080 horizon (PROVISIONAL)",
        "subdir": "06_geoclaw_gc_2080",
        "geojson_name": "geoclaw_gc_2080_nodes.geojson",
        "source_desc": "GeoClaw storm-surge model, 2080 horizon, depth sampled "
                       "from raster (PROVISIONAL — see warning).",
        "depth_note": GC_DEPTH_NOTE,
    },
}


def build(key, gdf, original_crs):
    cfg = SCENARIOS[key]
    depth_col = cfg["depth_col"]
    out_dir = PARENT / cfg["subdir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    geojson_out = out_dir / cfg["geojson_name"]
    csv_out = out_dir / "power_nodes_flood_depth.csv"
    png_out = out_dir / "preview.png"
    readme_out = out_dir / "README.md"

    if depth_col not in gdf.columns:
        raise SystemExit(f"ABORT: depth column {depth_col!r} not found in {SRC_FILE}")

    # --- (1) all-nodes GeoJSON ----------------------------------------------
    nodes = gdf[["node_id", "infra_type", "geometry"]].copy()
    nodes["flood_depth_m"] = gdf[depth_col].astype(float)
    nodes = nodes[["node_id", "infra_type", "geometry", "flood_depth_m"]]
    if geojson_out.exists():
        geojson_out.unlink()
    nodes.to_file(geojson_out, driver="GeoJSON")

    # --- (2) power-only CSV --------------------------------------------------
    power = gdf[gdf["infra_type"] == "power"].copy()
    power["lon"] = power.geometry.x
    power["lat"] = power.geometry.y
    power["flood_depth_m"] = power[depth_col].astype(float)
    power["flooded"] = power["flood_depth_m"] > 0
    power_csv = power[["node_id", "infra_type", "lon", "lat",
                       "flood_depth_m", "flooded"]]
    power_csv.to_csv(csv_out, index=False)

    n_power = len(power_csv)
    n_power_flooded = int(power_csv["flooded"].sum())
    n_total = len(gdf)
    n_flooded_total = int((nodes["flood_depth_m"] > 0).sum())
    dmax = float(nodes["flood_depth_m"].max())

    print(f"[{key:16s}] {cfg['family']:8s} total flooded {n_flooded_total:4d}"
          f"  power flooded {n_power_flooded:3d}/{n_power}  max {dmax:.2f}m")

    # --- (3) preview map -----------------------------------------------------
    vmax = dmax if dmax > 0 else 1.0
    norm = Normalize(vmin=0.0, vmax=vmax)
    cmap = plt.get_cmap("YlGnBu")

    fig, ax = plt.subplots(figsize=(9, 9))
    gdf.plot(ax=ax, color="#cccccc", markersize=2, alpha=0.45, zorder=1)

    pdry = power[power["flood_depth_m"] <= 0]
    pwet = power[power["flood_depth_m"] > 0]
    if len(pdry):
        ax.scatter(pdry.geometry.x, pdry.geometry.y, s=20, c="#7a7a7a",
                   edgecolors="black", linewidths=0.3, zorder=2,
                   label=f"power, dry ({len(pdry)})")
    if len(pwet):
        ax.scatter(pwet.geometry.x, pwet.geometry.y, s=34,
                   c=pwet["flood_depth_m"], cmap=cmap, norm=norm,
                   edgecolors="black", linewidths=0.5, zorder=3,
                   label=f"power, flooded ({len(pwet)})")

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("power-node flood depth (m)")

    banner = "  [PROVISIONAL — over-floods ~6x]" if cfg["family"] == "GeoClaw" else ""
    ax.set_title(f"{cfg['full']} — flood depth (m){banner}")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(png_out, dpi=150)
    plt.close(fig)

    # --- (4) README ----------------------------------------------------------
    warning = ""
    if cfg["family"] == "GeoClaw":
        warning = "\n" + GC_WARNING.format(
            maxd=dmax, nflood=n_flooded_total, pflood=n_power_flooded) + "\n"

    readme = f"""# {cfg['full']} — export for Jesse (pandapower cascade)
{warning}
## Scenario
- **Family:** {cfg['family']}
- **Plain label:** {cfg['plain']}
- **Full label:** {cfg['full']}
- **Source column:** `{depth_col}` (depth in **meters**)
- **Source dataset:** {cfg['source_desc']}
- **Source file:** `data/flood/nyc_infra_nodes_all_flood.geojson`

## Coordinates
- **Original CRS:** {original_crs}
- **Output CRS:** {OUT_CRS} (lon/lat decimal degrees)

## Units & depth encoding
{cfg['depth_note']}

## Coverage (this scenario)
- Total nodes: **{n_total}**
- Nodes flooded citywide (depth > 0): **{n_flooded_total}**
- Power nodes: **{n_power}**
- Power nodes flooded (depth > 0): **{n_power_flooded}**
- Max depth: **{dmax:.2f} m**

## Files
- `{cfg['geojson_name']}` — all {n_total} infrastructure nodes
  (`node_id`, `infra_type`, `geometry`, `flood_depth_m`).
- `power_nodes_flood_depth.csv` — power nodes only
  (`node_id`, `infra_type`, `lon`, `lat`, `flood_depth_m`, `flooded`).
- `preview.png` — quick-look map: all nodes light grey, power nodes highlighted
  (dry = grey, flooded = colored by depth).
- `README.md` — this file.

See the parent `README.md` for the full 6-scenario summary table and the
DEP-vs-GeoClaw caveat.
"""
    readme_out.write_text(readme)

    return {
        "key": key, "cfg": cfg, "n_total": n_total,
        "n_flooded_total": n_flooded_total, "n_power": n_power,
        "n_power_flooded": n_power_flooded, "dmax": dmax,
    }


def write_master_readme(rows, original_crs):
    lines = []
    lines.append("# NYC flood maps — 6 scenarios for Jesse (pandapower cascade)\n")
    lines.append(
        "Six mappable flood scenarios sampled onto the same 6,231 infrastructure "
        "nodes, in two families. Each subfolder is a self-contained bundle "
        "(all-nodes GeoJSON + power-only CSV + preview PNG + README).\n")
    lines.append("## Summary\n")
    lines.append("| # | Folder | Family | Scenario | Total flooded | Power flooded | Max depth |")
    lines.append("|---|--------|--------|----------|--------------:|--------------:|----------:|")
    for i, r in enumerate(rows, 1):
        c = r["cfg"]
        lines.append(
            f"| {i} | `{c['subdir']}` | {c['family']} | {c['full']} | "
            f"{r['n_flooded_total']} | {r['n_power_flooded']}/{r['n_power']} | "
            f"{r['dmax']:.2f} m |")
    lines.append("")
    lines.append("## ⚠️ Read this first — DEP is analysis-grade, GeoClaw is PROVISIONAL\n")
    lines.append(
        "- **DEP** scenarios (`moderate_current`, `moderate_2050`, `extreme_2080`) "
        "are the clean, analysis-grade maps — NYC DEP Stormwater Flood Maps "
        "(pluvial/tidal compound), categorical depths (0 / 0.20 / 0.60 / 0.80 m).\n"
        "- **GeoClaw** scenarios (`gc_2026`, `gc_2050`, `gc_2080`) are storm-surge "
        "model outputs with a **known, unresolved over-flood bug (~6x)**: "
        "water-surface elevation (`eta`) is used directly as depth instead of "
        "subtracting ground elevation (DEM). Depths are inflated (up to 6.3 m vs "
        "DEP's 0.80 m cap) and far too many nodes read as flooded. **Do not use "
        "the GeoClaw trio for cascade analysis** until the DEM subtraction is "
        "fixed; they are included only because all six scenarios were requested, "
        "and each carries a warning banner in its own README.\n")
    lines.append("## Coordinates\n")
    lines.append(f"- Original CRS: {original_crs}")
    lines.append(f"- Output CRS: {OUT_CRS} (lon/lat decimal degrees)\n")
    (PARENT / "README.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="all",
                    choices=list(SCENARIOS) + ["all"])
    args = ap.parse_args()

    gdf = gpd.read_file(SRC_FILE)
    original_crs = gdf.crs
    gdf = gdf.to_crs(OUT_CRS)

    PARENT.mkdir(parents=True, exist_ok=True)

    keys = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    rows = [build(key, gdf, original_crs) for key in keys]

    if args.scenario == "all":
        write_master_readme(rows, original_crs)
        zip_path = shutil.make_archive(str(PARENT), "zip", root_dir=PARENT.parent,
                                       base_dir=PARENT.name)
        print(f"\n=== MASTER ===")
        print(f"parent (absolute): {PARENT.resolve()}")
        print(f"zip: {Path(zip_path).resolve()} "
              f"({Path(zip_path).stat().st_size / (1024 * 1024):.2f} MB)")


if __name__ == "__main__":
    main()
