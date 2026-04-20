#!/usr/bin/env python3
"""
diagnose_borough_assignment.py

Verify whether the 2,031 nodes classified as 'Staten Island' are actually on SI,
or whether NJ substations are being mislabeled. Also check flood distribution
for genuine SI infrastructure.

Run from project root:
    python3 src/flood/diagnose_borough_assignment.py
"""

from pathlib import Path
import geopandas as gpd
import pandas as pd

NODES_PATH = Path("data/flood/nyc_infra_nodes_dep_flood.geojson")


def assign_borough_current(lat: float, lon: float) -> str:
    """The rule used in flood_overlay_v3.py — under investigation."""
    if lon < -74.05:
        return "Staten Island"
    if lat > 40.88 and lon > -73.92:
        return "Bronx"
    if -74.02 <= lon <= -73.90 and 40.70 <= lat <= 40.88:
        return "Manhattan"
    if lon > -73.86:
        return "Queens"
    return "Brooklyn"


# Actual Staten Island bounding box (from NYC Dept of City Planning):
# lat 40.496 - 40.651, lon -74.256 - -74.052
# NJ Bergen/Hudson county extends roughly lat 40.67 - 40.93, lon -74.27 - -74.03

def assign_borough_corrected(lat: float, lon: float) -> str:
    """Stricter bounds based on actual NYC borough geography."""
    # Staten Island: south of 40.65 AND west of -74.03
    if lat < 40.65 and lon < -74.03:
        return "Staten Island"
    # NJ (anything west of Hudson that isn't Staten Island)
    if lon < -74.03:
        return "NJ (external)"
    # Bronx: lat > 40.79 (roughly Harlem River boundary)
    if lat > 40.80 and lon > -73.93:
        return "Bronx"
    # Manhattan: the strip
    if -74.02 <= lon <= -73.93 and 40.70 <= lat <= 40.88:
        return "Manhattan"
    # Queens: eastern NYC
    if lon > -73.90 or (lon > -73.93 and lat > 40.70):
        return "Queens"
    # Default to Brooklyn
    return "Brooklyn"


def main():
    gdf = gpd.read_file(NODES_PATH)
    print(f"Loaded {len(gdf):,} nodes")

    # Apply both rules
    gdf["boro_current"] = gdf.apply(
        lambda r: assign_borough_current(r["lat"], r["lon"]), axis=1
    )
    gdf["boro_corrected"] = gdf.apply(
        lambda r: assign_borough_corrected(r["lat"], r["lon"]), axis=1
    )

    print("\n" + "=" * 72)
    print("CURRENT RULE vs CORRECTED RULE — total node counts")
    print("=" * 72)
    comparison = pd.DataFrame({
        "current": gdf["boro_current"].value_counts(),
        "corrected": gdf["boro_corrected"].value_counts(),
    }).fillna(0).astype(int)
    print(comparison)

    # Sanity check: look at the 'Staten Island' bucket under current rule
    # and see what lat/lon range they're actually in
    print("\n" + "=" * 72)
    print("NODES LABELED 'Staten Island' BY CURRENT RULE")
    print("=" * 72)
    si_current = gdf[gdf["boro_current"] == "Staten Island"]
    print(f"  Count: {len(si_current):,}")
    print(f"  lat range: [{si_current['lat'].min():.4f}, {si_current['lat'].max():.4f}]")
    print(f"  lon range: [{si_current['lon'].min():.4f}, {si_current['lon'].max():.4f}]")
    print(f"\n  If lat_max > 40.72, these include non-SI nodes (Hoboken/Jersey City/etc).")

    # Split these into 'actually SI' and 'actually NJ'
    actually_si = si_current[si_current["lat"] < 40.65]
    actually_nj = si_current[si_current["lat"] >= 40.65]
    print(f"\n  Actually Staten Island (lat < 40.65): {len(actually_si):,}")
    print(f"  Actually NJ (lat >= 40.65):             {len(actually_nj):,}")

    # Infra type breakdown of mislabeled NJ nodes
    if len(actually_nj) > 0:
        print(f"\n  NJ nodes currently mislabeled as SI, by infra_type:")
        print(actually_nj["infra_type"].value_counts().to_string())

    # Flood coverage on *actually Staten Island* nodes
    print("\n" + "=" * 72)
    print("FLOOD COVERAGE ON ACTUAL STATEN ISLAND NODES (lat < 40.65, lon < -74.03)")
    print("=" * 72)
    for scenario in ["moderate_current", "moderate_2050", "extreme_2080"]:
        cat_col = f"flood_{scenario}_category"
        if cat_col not in actually_si.columns:
            continue
        flooded = int((actually_si[cat_col] > 0).sum())
        total = len(actually_si)
        pct = 100 * flooded / total if total else 0
        print(f"  {scenario:<20}: {flooded:>3}/{total:<4} flooded ({pct:.1f}%)")

    # Flood coverage on actually-NJ nodes for comparison
    print("\n" + "=" * 72)
    print("FLOOD COVERAGE ON ACTUAL NJ NODES (currently mislabeled as SI)")
    print("=" * 72)
    for scenario in ["moderate_current", "moderate_2050", "extreme_2080"]:
        cat_col = f"flood_{scenario}_category"
        if cat_col not in actually_nj.columns:
            continue
        flooded = int((actually_nj[cat_col] > 0).sum())
        total = len(actually_nj)
        pct = 100 * flooded / total if total else 0
        print(f"  {scenario:<20}: {flooded:>3}/{total:<4} flooded ({pct:.1f}%)")
    print("\n  (NJ nodes SHOULD be 0% flooded — DEP maps cover NYC only)")


if __name__ == "__main__":
    main()