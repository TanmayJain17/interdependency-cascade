"""
Download NYC Fuel Infrastructure Data (Gas Stations + Petroleum Terminals)
Sources:
1. NYS Gas Station ArcGIS MapServer (Fuel NY Initiative)
2. OpenStreetMap Overpass API (amenity=fuel)
3. EIA Petroleum Product Terminals (ArcGIS FeatureServer)
4. Hardcoded NY Harbor terminals (known major facilities)

This script:
1. Downloads gas stations from NYS + OSM, deduplicates
2. Downloads petroleum storage terminals from EIA + hardcoded
3. Filters to Lower Manhattan + NY Harbor supply area
4. Saves as GeoJSON for integration into build_graph.py
"""

import requests
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import os
import io
import time

# ============================================================
# Configuration
# ============================================================

LM_BOUNDS = {
    "min_lat": 40.700, "max_lat": 40.755,
    "min_lon": -74.020, "max_lon": -73.970,
}

NYC_HARBOR_BOUNDS = {
    "min_lat": 40.49, "max_lat": 40.92,
    "min_lon": -74.27, "max_lon": -73.70,
}

OUTPUT_DIR = "data/fuel"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Geometry safety helper
# ============================================================

def safe_lonlat(gdf):
    """
    Extract (lon, lat) arrays from ANY geometry type. Handles Point, MultiPoint,
    Polygon, LineString, etc. by using geometry.centroid for non-Point geometries.
    Returns two pandas Series (lon, lat).
    """
    # For Points, .centroid returns the point itself, so .x/.y work universally
    centroids = gdf.geometry.centroid
    lon = centroids.x
    lat = centroids.y
    return lon, lat


# ============================================================
# METHOD 1: NYS Gas Station ArcGIS MapServer
# ============================================================

def download_nys_gas_stations():
    print("=" * 60)
    print("METHOD 1: NYS Gas Station ArcGIS MapServer")
    print("=" * 60)

    BASE_URL = "https://gisservices.its.ny.gov/arcgis/rest/services/gas_station/MapServer"
    NYC_BBOX_WGS84 = (
        f"{NYC_HARBOR_BOUNDS['min_lon']},{NYC_HARBOR_BOUNDS['min_lat']},"
        f"{NYC_HARBOR_BOUNDS['max_lon']},{NYC_HARBOR_BOUNDS['max_lat']}"
    )

    all_stations = []
    for layer_id in [1, 2, 3]:
        url = f"{BASE_URL}/{layer_id}/query"
        params = {
            "where": "1=1",
            "geometry": NYC_BBOX_WGS84,
            "geometryType": "esriGeometryEnvelope",
            "inSR": "4326",
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": "*",
            "outSR": "4326",
            "f": "geojson",
            "resultRecordCount": 2000,
        }
        try:
            print(f"  Querying layer {layer_id}...")
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                features = data.get("features", [])
                if features:
                    gdf = gpd.read_file(io.StringIO(response.text))
                    # Report geometry types so we know what we're dealing with
                    geom_types = gdf.geometry.type.value_counts().to_dict()
                    all_stations.append(gdf)
                    print(f"    Layer {layer_id}: {len(gdf)} stations, geometries: {geom_types}")
                else:
                    print(f"    Layer {layer_id}: 0 features in bbox")
            else:
                print(f"    Layer {layer_id}: HTTP {response.status_code}")
        except Exception as e:
            print(f"    Layer {layer_id} failed: {e}")
        time.sleep(0.5)

    if all_stations:
        combined = pd.concat(all_stations, ignore_index=True)
        combined = gpd.GeoDataFrame(combined, geometry="geometry", crs="EPSG:4326")
        print(f"\n  Total from NYS MapServer: {len(combined)}")
        return combined
    print("  No data from NYS MapServer")
    return None


# ============================================================
# METHOD 2: OpenStreetMap Overpass API
# ============================================================

def download_osm_gas_stations():
    print("\n" + "=" * 60)
    print("METHOD 2: OpenStreetMap Gas Stations")
    print("=" * 60)

    # Try multiple Overpass endpoints; some reject User-Agent defaults with 406
    overpass_endpoints = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://overpass.openstreetmap.ru/api/interpreter",
    ]

    query = f"""
    [out:json][timeout:180];
    (
      node["amenity"="fuel"]({NYC_HARBOR_BOUNDS['min_lat']},{NYC_HARBOR_BOUNDS['min_lon']},{NYC_HARBOR_BOUNDS['max_lat']},{NYC_HARBOR_BOUNDS['max_lon']});
      way["amenity"="fuel"]({NYC_HARBOR_BOUNDS['min_lat']},{NYC_HARBOR_BOUNDS['min_lon']},{NYC_HARBOR_BOUNDS['max_lat']},{NYC_HARBOR_BOUNDS['max_lon']});
    );
    out center;
    """
    headers = {
        "User-Agent": "NYU-CUSP-flood-cascade-research/1.0 (tanmay.jain@nyu.edu)",
        "Accept": "application/json",
    }

    for endpoint in overpass_endpoints:
        print(f"  Querying {endpoint}...")
        try:
            response = requests.post(
                endpoint, data={"data": query}, headers=headers, timeout=180
            )
            if response.status_code != 200:
                print(f"    HTTP {response.status_code} — trying next endpoint")
                continue

            elements = response.json().get("elements", [])
            print(f"  Found {len(elements)} gas station features")

            rows = []
            for e in elements:
                lat = e.get("lat") or e.get("center", {}).get("lat")
                lon = e.get("lon") or e.get("center", {}).get("lon")
                tags = e.get("tags", {})
                if lat and lon:
                    rows.append({
                        "name": tags.get("name", tags.get("brand", "Unknown Gas Station")),
                        "lat": float(lat),
                        "lon": float(lon),
                        "brand": tags.get("brand", ""),
                        "operator": tags.get("operator", ""),
                        "has_diesel": "yes" if tags.get("fuel:diesel") == "yes" else "no",
                        "source": "osm",
                    })

            if rows:
                df = pd.DataFrame(rows)
                gdf = gpd.GeoDataFrame(
                    df,
                    geometry=[Point(r["lon"], r["lat"]) for _, r in df.iterrows()],
                    crs="EPSG:4326",
                )
                print(f"  Parsed {len(gdf)} gas stations from OSM")
                if "brand" in gdf.columns:
                    top = gdf["brand"].replace("", pd.NA).dropna().value_counts().head(5)
                    print(f"  Top brands: {top.to_dict()}")
                return gdf
            else:
                print("  No gas stations parsed from this endpoint")
        except Exception as e:
            print(f"  Query failed on {endpoint}: {e}")

    print("  All OSM endpoints failed")
    return None


# ============================================================
# METHOD 3: EIA Petroleum Product Terminals (ArcGIS)
# ============================================================

def download_eia_terminals():
    print("\n" + "=" * 60)
    print("METHOD 3: EIA Petroleum Product Terminals")
    print("=" * 60)

    url = "https://services7.arcgis.com/FGr1D95XCGALKXqM/arcgis/rest/services/Petroleum_Product_Terminals/FeatureServer/0/query"
    params = {
        "where": "1=1",
        "geometry": f"{NYC_HARBOR_BOUNDS['min_lon']},{NYC_HARBOR_BOUNDS['min_lat']},{NYC_HARBOR_BOUNDS['max_lon']},{NYC_HARBOR_BOUNDS['max_lat']}",
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "*",
        "outSR": "4326",
        "f": "geojson",
        "resultRecordCount": 500,
    }
    try:
        print("  Querying EIA FeatureServer...")
        response = requests.get(url, params=params, timeout=30)
        if response.status_code == 200:
            data = response.json()
            features = data.get("features", [])
            if features:
                gdf = gpd.read_file(io.StringIO(response.text))
                print(f"  Found {len(gdf)} terminals in NYC/NJ Harbor area")
                return gdf
            print("  No terminals in bounding box from EIA")
            return None
        print(f"  EIA query failed: HTTP {response.status_code}")
        return None
    except Exception as e:
        print(f"  EIA query failed: {e}")
        return None


# ============================================================
# METHOD 4: Hardcoded NY Harbor Petroleum Terminals
# ============================================================

def get_hardcoded_terminals():
    print("\n" + "=" * 60)
    print("METHOD 4: Hardcoded NY Harbor Petroleum Terminals")
    print("=" * 60)

    terminals = pd.DataFrame([
        {"name": "Bayonne Terminal Complex", "lat": 40.6620, "lon": -74.0960,
         "capacity_bbl": 5_000_000, "products": "gasoline,diesel,jet_fuel",
         "port_access": True, "notes": "Largest terminal cluster in NY Harbor"},
        {"name": "Tremley Point Terminal (Linden, NJ)", "lat": 40.6280, "lon": -74.2200,
         "capacity_bbl": 3_000_000, "products": "gasoline,diesel",
         "port_access": True, "notes": "Phillips 66 / Buckeye Partners terminal"},
        {"name": "Perth Amboy Terminal", "lat": 40.5080, "lon": -74.2640,
         "capacity_bbl": 2_000_000, "products": "gasoline,diesel,heating_oil",
         "port_access": True, "notes": "NuStar / Buckeye terminal"},
        {"name": "Bronx Terminal (Hunts Point)", "lat": 40.8100, "lon": -73.8800,
         "capacity_bbl": 500_000, "products": "gasoline,diesel,heating_oil",
         "port_access": True, "notes": "Serves Bronx and Upper Manhattan"},
        {"name": "Brooklyn Terminal (Gowanus)", "lat": 40.6730, "lon": -73.9900,
         "capacity_bbl": 1_000_000, "products": "heating_oil,diesel",
         "port_access": True, "notes": "Near Gowanus Canal; flooded during Sandy"},
        {"name": "Staten Island Terminal (Howland Hook)", "lat": 40.6420, "lon": -74.1700,
         "capacity_bbl": 800_000, "products": "gasoline,diesel",
         "port_access": True, "notes": "Global Partners terminal"},
    ])
    terminals["type"] = "PETROLEUM_TERMINAL"
    terminals["source"] = "hardcoded_nyserda"
    terminals["external"] = True
    print(f"  {len(terminals)} major terminals defined")
    print(terminals[["name", "lat", "lon", "capacity_bbl"]].to_string())
    return terminals


# ============================================================
# Main execution
# ============================================================

if __name__ == "__main__":

    # ── 1. GAS STATIONS ──
    print("\n" + "#" * 60)
    print("# PART 1: GAS STATIONS (Tier 2 — retail fuel)")
    print("#" * 60)

    nys_stations = download_nys_gas_stations()
    osm_stations = download_osm_gas_stations()

    print("\n" + "=" * 60)
    print("COMBINING GAS STATION SOURCES")
    print("=" * 60)

    station_dfs = []

    if nys_stations is not None and len(nys_stations) > 0:
        # Use safe_lonlat to handle any geometry type (Point/MultiPoint/Polygon)
        lon, lat = safe_lonlat(nys_stations)

        # Build normalized dataframe — pull name from whatever column exists
        name_col = None
        for candidate in ["FULLSTNAME", "Name", "FACNAME", "STATION", "NAME"]:
            if candidate in nys_stations.columns:
                name_col = candidate
                break
        names = nys_stations[name_col] if name_col else pd.Series(["Unknown"] * len(nys_stations))

        brand_col = None
        for candidate in ["Brand", "BRAND"]:
            if candidate in nys_stations.columns:
                brand_col = candidate
                break
        brands = nys_stations[brand_col] if brand_col else pd.Series([""] * len(nys_stations))

        operator_col = None
        for candidate in ["Owner", "OWNER", "OPERATOR"]:
            if candidate in nys_stations.columns:
                operator_col = candidate
                break
        operators = nys_stations[operator_col] if operator_col else pd.Series([""] * len(nys_stations))

        nys_norm = pd.DataFrame({
            "name":     names.values,
            "lat":      lat.values,
            "lon":      lon.values,
            "brand":    brands.values,
            "operator": operators.values,
            "source":   "nys_mapserver",
        })
        station_dfs.append(nys_norm)
        print(f"  NYS MapServer: {len(nys_norm)} stations")

    if osm_stations is not None and len(osm_stations) > 0:
        osm_norm = pd.DataFrame({
            "name":     osm_stations["name"].values,
            "lat":      osm_stations["lat"].values,
            "lon":      osm_stations["lon"].values,
            "brand":    osm_stations.get("brand", pd.Series([""] * len(osm_stations))).values,
            "operator": osm_stations.get("operator", pd.Series([""] * len(osm_stations))).values,
            "source":   "osm",
        })
        station_dfs.append(osm_norm)
        print(f"  OSM: {len(osm_norm)} stations")

    if not station_dfs:
        print("  WARNING: No gas station data from any source!")
        print("  Continuing with petroleum terminals only...")
        all_stations = pd.DataFrame()
    else:
        all_stations = pd.concat(station_dfs, ignore_index=True)
        # Drop rows with NaN coords (can happen with weird geometries)
        all_stations = all_stations.dropna(subset=["lat", "lon"]).copy()

        # Deduplicate: ~11m rounding across sources
        all_stations["_lat_r"] = all_stations["lat"].round(4)
        all_stations["_lon_r"] = all_stations["lon"].round(4)
        all_stations = (
            all_stations
            .sort_values("source", key=lambda s: s.map({"nys_mapserver": 0, "osm": 1}))
            .drop_duplicates(subset=["_lat_r", "_lon_r"], keep="first")
            .drop(columns=["_lat_r", "_lon_r"])
        )
        print(f"\n  Total unique gas stations: {len(all_stations)}")

        # Save citywide file
        all_stations_gdf = gpd.GeoDataFrame(
            all_stations,
            geometry=[Point(r.lon, r.lat) for r in all_stations.itertuples()],
            crs="EPSG:4326",
        )
        all_stations_gdf["type"] = "GAS_STATION"
        all_stations_gdf["external"] = False
        outpath = os.path.join(OUTPUT_DIR, "gas_stations_nyc.geojson")
        all_stations_gdf.to_file(outpath, driver="GeoJSON")
        print(f"  Saved → {outpath}  ({len(all_stations_gdf)} NYC gas stations)")

    # Filter to Lower Manhattan
    if len(all_stations) > 0:
        lm_mask = (
            (all_stations["lat"] >= LM_BOUNDS["min_lat"]) &
            (all_stations["lat"] <= LM_BOUNDS["max_lat"]) &
            (all_stations["lon"] >= LM_BOUNDS["min_lon"]) &
            (all_stations["lon"] <= LM_BOUNDS["max_lon"])
        )
        stations_lm = all_stations[lm_mask].copy()
        print(f"  Gas stations in Lower Manhattan: {len(stations_lm)}")

        if len(stations_lm) > 0:
            print(stations_lm[["name", "lat", "lon", "brand", "source"]].to_string())

        expanded_mask = (
            (all_stations["lat"] >= LM_BOUNDS["min_lat"] - 0.01) &
            (all_stations["lat"] <= LM_BOUNDS["max_lat"] + 0.01) &
            (all_stations["lon"] >= LM_BOUNDS["min_lon"] - 0.01) &
            (all_stations["lon"] <= LM_BOUNDS["max_lon"] + 0.01)
        )
        stations_expanded = all_stations[expanded_mask].copy()
        print(f"  Gas stations in expanded LM area: {len(stations_expanded)}")
    else:
        stations_lm = pd.DataFrame()
        stations_expanded = pd.DataFrame()

    # ── 2. PETROLEUM TERMINALS ──
    print("\n" + "#" * 60)
    print("# PART 2: PETROLEUM TERMINALS (Tier 1 — bulk storage)")
    print("#" * 60)

    eia_terminals = download_eia_terminals()
    hardcoded_terminals = get_hardcoded_terminals()

    print("\n" + "=" * 60)
    print("COMBINING TERMINAL SOURCES")
    print("=" * 60)

    terminal_dfs = []

    if eia_terminals is not None and len(eia_terminals) > 0:
        lon, lat = safe_lonlat(eia_terminals)
        name_col = None
        for candidate in ["Terminal_N", "Name", "TERMINAL", "NAME"]:
            if candidate in eia_terminals.columns:
                name_col = candidate
                break
        names = eia_terminals[name_col] if name_col else pd.Series(["Unknown Terminal"] * len(eia_terminals))

        cap_col = None
        for candidate in ["Total_She", "CAPACITY"]:
            if candidate in eia_terminals.columns:
                cap_col = candidate
                break
        caps = eia_terminals[cap_col] if cap_col else pd.Series([0] * len(eia_terminals))

        prod_col = None
        for candidate in ["Products", "PRODUCTS"]:
            if candidate in eia_terminals.columns:
                prod_col = candidate
                break
        prods = eia_terminals[prod_col] if prod_col else pd.Series([""] * len(eia_terminals))

        eia_norm = pd.DataFrame({
            "name":         names.values,
            "lat":          lat.values,
            "lon":          lon.values,
            "capacity_bbl": caps.values,
            "products":     prods.values,
            "source":       "eia",
        })
        terminal_dfs.append(eia_norm)
        print(f"  EIA: {len(eia_norm)} terminals")

    hc_norm = pd.DataFrame({
        "name":         hardcoded_terminals["name"].values,
        "lat":          hardcoded_terminals["lat"].values,
        "lon":          hardcoded_terminals["lon"].values,
        "capacity_bbl": hardcoded_terminals["capacity_bbl"].values,
        "products":     hardcoded_terminals["products"].values,
        "source":       "hardcoded_nyserda",
    })
    terminal_dfs.append(hc_norm)
    print(f"  Hardcoded: {len(hc_norm)} terminals")

    all_terminals = pd.concat(terminal_dfs, ignore_index=True)
    all_terminals = all_terminals.dropna(subset=["lat", "lon"]).copy()

    all_terminals["_lat_r"] = all_terminals["lat"].round(2)
    all_terminals["_lon_r"] = all_terminals["lon"].round(2)
    all_terminals = (
        all_terminals
        .sort_values("source", key=lambda s: s.map({"eia": 0, "hardcoded_nyserda": 1}))
        .drop_duplicates(subset=["_lat_r", "_lon_r"], keep="first")
        .drop(columns=["_lat_r", "_lon_r"])
    )
    all_terminals["external"] = True
    all_terminals["type"] = "PETROLEUM_TERMINAL"

    print(f"\n  Total unique terminals: {len(all_terminals)}")
    print(all_terminals[["name", "lat", "lon", "capacity_bbl", "source"]].to_string())

    # ── 3. SAVE OUTPUTS ──
    print("\n" + "=" * 60)
    print("SAVING OUTPUTS")
    print("=" * 60)

    if len(stations_lm) > 0:
        stations_gdf = gpd.GeoDataFrame(
            stations_lm,
            geometry=[Point(r.lon, r.lat) for r in stations_lm.itertuples()],
            crs="EPSG:4326",
        )
        stations_gdf["type"] = "GAS_STATION"
        stations_gdf["external"] = False
        outpath = os.path.join(OUTPUT_DIR, "gas_stations_lm.geojson")
        stations_gdf.to_file(outpath, driver="GeoJSON")
        print(f"  Saved → {outpath}  ({len(stations_gdf)} stations)")

    if len(stations_expanded) > 0:
        exp_gdf = gpd.GeoDataFrame(
            stations_expanded,
            geometry=[Point(r.lon, r.lat) for r in stations_expanded.itertuples()],
            crs="EPSG:4326",
        )
        outpath = os.path.join(OUTPUT_DIR, "gas_stations_expanded.geojson")
        exp_gdf.to_file(outpath, driver="GeoJSON")
        print(f"  Saved → {outpath}  ({len(exp_gdf)} stations)")

    terminals_gdf = gpd.GeoDataFrame(
        all_terminals,
        geometry=[Point(r.lon, r.lat) for r in all_terminals.itertuples()],
        crs="EPSG:4326",
    )
    outpath = os.path.join(OUTPUT_DIR, "petroleum_terminals_nyc.geojson")
    terminals_gdf.to_file(outpath, driver="GeoJSON")
    print(f"  Saved → {outpath}  ({len(terminals_gdf)} terminals)")

    # Combined LM fuel
    combined_rows = []
    if len(stations_lm) > 0:
        for _, r in stations_lm.iterrows():
            combined_rows.append({
                "name": r["name"], "lat": r["lat"], "lon": r["lon"],
                "type": "GAS_STATION",
                "subtype": r.get("brand", "RETAIL"),
                "source": r["source"], "external": False,
            })
    for _, r in all_terminals.iterrows():
        combined_rows.append({
            "name": r["name"], "lat": r["lat"], "lon": r["lon"],
            "type": "PETROLEUM_TERMINAL",
            "subtype": "BULK_STORAGE",
            "source": r["source"], "external": True,
            "capacity_bbl": r.get("capacity_bbl", 0),
        })
    combined_df = pd.DataFrame(combined_rows)
    combined_gdf = gpd.GeoDataFrame(
        combined_df,
        geometry=[Point(r["lon"], r["lat"]) for _, r in combined_df.iterrows()],
        crs="EPSG:4326",
    )
    outpath = os.path.join(OUTPUT_DIR, "fuel_infra_all.geojson")
    combined_gdf.to_file(outpath, driver="GeoJSON")
    print(f"  Saved → {outpath}  ({len(combined_gdf)} total fuel nodes)")

    # ── 4. SUMMARY ──
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    n_stations_nyc = len(all_stations) if len(all_stations) > 0 else 0
    n_stations_lm  = len(stations_lm) if len(stations_lm) > 0 else 0
    n_terminals    = len(all_terminals)
    print(f"  Gas stations (NYC-wide):    {n_stations_nyc}")
    print(f"  Gas stations in LM:         {n_stations_lm}")
    print(f"  Petroleum terminals (ext):  {n_terminals}")
    print(f"  Total fuel nodes citywide:  {n_stations_nyc + n_terminals}")
    print(f"\n  Files saved to: {os.path.abspath(OUTPUT_DIR)}/")