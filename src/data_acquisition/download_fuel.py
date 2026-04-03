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

Fuel supply chain for our model:
  Petroleum Terminal (Tier 1, external) → Gas Station (Tier 2, local)
  Gas Station → Hospital generators, Telecom generators, Water pumps
  Power → Gas Station (needs electricity to pump fuel)
  Power → Petroleum Terminal (needs electricity for truck loading)
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

# Lower Manhattan bounding box (matches other scripts)
LM_BOUNDS = {
    'min_lat': 40.700, 'max_lat': 40.755,
    'min_lon': -74.020, 'max_lon': -73.970
}

# Wider NYC area for petroleum terminals (NY Harbor extends into NJ)
NYC_HARBOR_BOUNDS = {
    'min_lat': 40.49, 'max_lat': 40.92,
    'min_lon': -74.27, 'max_lon': -73.70
}

OUTPUT_DIR = 'data/fuel'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# METHOD 1: NYS Gas Station ArcGIS MapServer
# ============================================================

def download_nys_gas_stations():
    """
    Query NYS gas station MapServer (Fuel NY Initiative).
    Layers 1-3 contain registered gas station data.
    Server published by NYS Office of Information Technology Services.
    """
    print("=" * 60)
    print("METHOD 1: NYS Gas Station ArcGIS MapServer")
    print("=" * 60)

    BASE_URL = "https://gisservices.its.ny.gov/arcgis/rest/services/gas_station/MapServer"

    # NYC bounding box in Web Mercator (EPSG:3857) — the MapServer uses this CRS
    # Convert from WGS84: NYC area roughly -74.05,40.68,-73.90,40.80
    NYC_BBOX_3857 = '-8240000,4960000,-8225000,4980000'  # approximate LM + surrounding
    # Wider NYC bbox
    NYC_BBOX_WGS84 = f"{LM_BOUNDS['min_lon']-0.05},{LM_BOUNDS['min_lat']-0.02},{LM_BOUNDS['max_lon']+0.05},{LM_BOUNDS['max_lat']+0.02}"

    all_stations = []

    # Try layers 1, 2, 3 (layer 0 is evacuation ramps, 4-5 are flood zones)
    for layer_id in [1, 2, 3]:
        url = f"{BASE_URL}/{layer_id}/query"
        params = {
            'where': '1=1',
            'geometry': NYC_BBOX_WGS84,
            'geometryType': 'esriGeometryEnvelope',
            'inSR': '4326',
            'spatialRel': 'esriSpatialRelIntersects',
            'outFields': '*',
            'outSR': '4326',
            'f': 'geojson',
            'resultRecordCount': 2000,
        }

        try:
            print(f"  Querying layer {layer_id}...")
            response = requests.get(url, params=params, timeout=30)

            if response.status_code == 200:
                data = response.json()
                features = data.get('features', [])
                if features:
                    gdf = gpd.read_file(io.StringIO(response.text))
                    all_stations.append(gdf)
                    print(f"    Layer {layer_id}: {len(gdf)} stations")
                    if len(gdf) > 0:
                        print(f"    Columns: {list(gdf.columns)[:10]}")
                else:
                    print(f"    Layer {layer_id}: 0 features in bbox")
            else:
                print(f"    Layer {layer_id}: HTTP {response.status_code}")
        except Exception as e:
            print(f"    Layer {layer_id} failed: {e}")

        time.sleep(0.5)

    if all_stations:
        combined = pd.concat(all_stations, ignore_index=True)
        combined = gpd.GeoDataFrame(combined, geometry='geometry', crs='EPSG:4326')
        print(f"\n  Total from NYS MapServer: {len(combined)}")
        return combined
    else:
        print("  No data from NYS MapServer")
        return None


# ============================================================
# METHOD 2: OpenStreetMap Overpass API
# ============================================================

def download_osm_gas_stations():
    """
    Query OSM for gas stations (amenity=fuel) in the NYC/LM area.
    Most reliable method — crowd-sourced but comprehensive for NYC.
    """
    print("\n" + "=" * 60)
    print("METHOD 2: OpenStreetMap Gas Stations")
    print("=" * 60)

    overpass_url = "http://overpass-api.de/api/interpreter"

    # Query gas stations in a wider area around Lower Manhattan
    # Expand bbox to catch stations that serve LM
    query = f"""
    [out:json][timeout:60];
    (
      node["amenity"="fuel"]({LM_BOUNDS['min_lat']-0.02},{LM_BOUNDS['min_lon']-0.02},{LM_BOUNDS['max_lat']+0.02},{LM_BOUNDS['max_lon']+0.02});
      way["amenity"="fuel"]({LM_BOUNDS['min_lat']-0.02},{LM_BOUNDS['min_lon']-0.02},{LM_BOUNDS['max_lat']+0.02},{LM_BOUNDS['max_lon']+0.02});
    );
    out center;
    """

    print("  Querying OSM Overpass API...")
    try:
        response = requests.post(overpass_url, data={'data': query}, timeout=60)
        if response.status_code != 200:
            print(f"  Failed: HTTP {response.status_code}")
            return None

        elements = response.json().get('elements', [])
        print(f"  Found {len(elements)} gas station features")

        rows = []
        for e in elements:
            lat = e.get('lat') or e.get('center', {}).get('lat')
            lon = e.get('lon') or e.get('center', {}).get('lon')
            tags = e.get('tags', {})
            if lat and lon:
                rows.append({
                    'name':     tags.get('name', tags.get('brand', 'Unknown Gas Station')),
                    'lat':      float(lat),
                    'lon':      float(lon),
                    'brand':    tags.get('brand', ''),
                    'operator': tags.get('operator', ''),
                    'fuel_types': ','.join(
                        k.replace('fuel:', '') for k in tags
                        if k.startswith('fuel:') and tags[k] == 'yes'
                    ),
                    'has_diesel': 'yes' if tags.get('fuel:diesel') == 'yes' else 'no',
                    'source':   'osm',
                })

        if rows:
            df = pd.DataFrame(rows)
            gdf = gpd.GeoDataFrame(
                df, geometry=[Point(r['lon'], r['lat']) for _, r in df.iterrows()],
                crs='EPSG:4326'
            )
            print(f"  Parsed {len(gdf)} gas stations from OSM")
            if 'brand' in gdf.columns:
                print(f"  Top brands: {gdf['brand'].value_counts().head(5).to_dict()}")
            return gdf
        else:
            print("  No gas stations parsed")
            return None

    except Exception as e:
        print(f"  OSM query failed: {e}")
        return None


# ============================================================
# METHOD 3: EIA Petroleum Product Terminals (ArcGIS)
# ============================================================

def download_eia_terminals():
    """
    Query EIA US Energy Atlas for petroleum product terminals.
    These are the bulk storage/distribution terminals in NY Harbor.
    Feature layer ID 36 in the EIA Energy Atlas FeatureServer.
    """
    print("\n" + "=" * 60)
    print("METHOD 3: EIA Petroleum Product Terminals")
    print("=" * 60)

    # EIA US Energy Atlas FeatureServer — Petroleum Product Terminals = layer 36
    url = "https://services7.arcgis.com/FGr1D95XCGALKXqM/arcgis/rest/services/Petroleum_Product_Terminals/FeatureServer/0/query"

    params = {
        'where': '1=1',
        'geometry': f"{NYC_HARBOR_BOUNDS['min_lon']},{NYC_HARBOR_BOUNDS['min_lat']},{NYC_HARBOR_BOUNDS['max_lon']},{NYC_HARBOR_BOUNDS['max_lat']}",
        'geometryType': 'esriGeometryEnvelope',
        'inSR': '4326',
        'spatialRel': 'esriSpatialRelIntersects',
        'outFields': '*',
        'outSR': '4326',
        'f': 'geojson',
        'resultRecordCount': 500,
    }

    try:
        print("  Querying EIA FeatureServer...")
        response = requests.get(url, params=params, timeout=30)

        if response.status_code == 200:
            data = response.json()
            features = data.get('features', [])
            if features:
                gdf = gpd.read_file(io.StringIO(response.text))
                print(f"  Found {len(gdf)} terminals in NYC/NJ Harbor area")
                if len(gdf) > 0:
                    print(f"  Columns: {list(gdf.columns)[:10]}")
                    for _, r in gdf.iterrows():
                        name = r.get('Terminal_N') or r.get('Name') or r.get('TERMINAL') or 'Unknown'
                        print(f"    {name}")
                return gdf
            else:
                print("  No terminals in bounding box from EIA")
                return None
        else:
            print(f"  EIA query failed: HTTP {response.status_code}")
            return None

    except Exception as e:
        print(f"  EIA query failed: {e}")
        return None


# ============================================================
# METHOD 4: Hardcoded NY Harbor Petroleum Terminals
# ============================================================

def get_hardcoded_terminals():
    """
    Major petroleum terminals serving NYC.
    Data from NYSERDA Terminal Resiliency Assessment + EIA reports.
    These are external nodes (outside LM) that supply fuel to LM.

    During Sandy:
    - 67% of NYC gas stations had no fuel (EIA emergency survey)
    - Terminal power outages disrupted truck loading operations
    - Port of New York was closed, blocking tanker deliveries
    """
    print("\n" + "=" * 60)
    print("METHOD 4: Hardcoded NY Harbor Petroleum Terminals")
    print("=" * 60)

    terminals = pd.DataFrame([
        # Major NY Harbor petroleum terminals (NJ side — serve NYC)
        {'name': 'Bayonne Terminal Complex',
         'lat': 40.6620, 'lon': -74.0960,
         'capacity_bbl': 5_000_000,
         'products': 'gasoline,diesel,jet_fuel',
         'port_access': True,
         'notes': 'Largest terminal cluster in NY Harbor'},

        {'name': 'Tremley Point Terminal (Linden, NJ)',
         'lat': 40.6280, 'lon': -74.2200,
         'capacity_bbl': 3_000_000,
         'products': 'gasoline,diesel',
         'port_access': True,
         'notes': 'Phillips 66 / Buckeye Partners terminal'},

        {'name': 'Perth Amboy Terminal',
         'lat': 40.5080, 'lon': -74.2640,
         'capacity_bbl': 2_000_000,
         'products': 'gasoline,diesel,heating_oil',
         'port_access': True,
         'notes': 'NuStar / Buckeye terminal'},

        # NY side terminals
        {'name': 'Bronx Terminal (Hunts Point)',
         'lat': 40.8100, 'lon': -73.8800,
         'capacity_bbl': 500_000,
         'products': 'gasoline,diesel,heating_oil',
         'port_access': True,
         'notes': 'Serves Bronx and Upper Manhattan'},

        {'name': 'Brooklyn Terminal (Gowanus)',
         'lat': 40.6730, 'lon': -73.9900,
         'capacity_bbl': 1_000_000,
         'products': 'heating_oil,diesel',
         'port_access': True,
         'notes': 'Near Gowanus Canal; flooded during Sandy'},

        {'name': 'Staten Island Terminal (Howland Hook)',
         'lat': 40.6420, 'lon': -74.1700,
         'capacity_bbl': 800_000,
         'products': 'gasoline,diesel',
         'port_access': True,
         'notes': 'Global Partners terminal'},
    ])

    terminals['type'] = 'PETROLEUM_TERMINAL'
    terminals['source'] = 'hardcoded_nyserda'
    terminals['external'] = True  # all terminals are outside LM

    print(f"  {len(terminals)} major terminals defined")
    print(terminals[['name', 'lat', 'lon', 'capacity_bbl']].to_string())

    return terminals


# ============================================================
# Main execution
# ============================================================

if __name__ == "__main__":

    # ── 1. GAS STATIONS ──────────────────────────────────────

    print("\n" + "#" * 60)
    print("# PART 1: GAS STATIONS (Tier 2 — retail fuel)")
    print("#" * 60)

    # Try NYS MapServer first
    nys_stations = download_nys_gas_stations()

    # Always get OSM as backup/supplement
    osm_stations = download_osm_gas_stations()

    # Combine and deduplicate
    print("\n" + "=" * 60)
    print("COMBINING GAS STATION SOURCES")
    print("=" * 60)

    station_dfs = []

    if nys_stations is not None and len(nys_stations) > 0:
        # Normalize NYS data to common schema
        nys_norm = pd.DataFrame({
            'name':     nys_stations.get('Name', nys_stations.get('FACNAME',
                        nys_stations.get('STATION', pd.Series(['Unknown'] * len(nys_stations))))),
            'lat':      nys_stations.geometry.y,
            'lon':      nys_stations.geometry.x,
            'brand':    nys_stations.get('Brand', nys_stations.get('BRAND', '')),
            'operator': nys_stations.get('Owner', nys_stations.get('OWNER', '')),
            'source':   'nys_mapserver',
        })
        station_dfs.append(nys_norm)
        print(f"  NYS MapServer: {len(nys_norm)} stations")

    if osm_stations is not None and len(osm_stations) > 0:
        osm_norm = pd.DataFrame({
            'name':     osm_stations['name'],
            'lat':      osm_stations['lat'],
            'lon':      osm_stations['lon'],
            'brand':    osm_stations.get('brand', ''),
            'operator': osm_stations.get('operator', ''),
            'source':   'osm',
        })
        station_dfs.append(osm_norm)
        print(f"  OSM: {len(osm_norm)} stations")

    if not station_dfs:
        print("  WARNING: No gas station data from any source!")
        print("  Continuing with petroleum terminals only...")
        all_stations = pd.DataFrame()
    else:
        all_stations = pd.concat(station_dfs, ignore_index=True)

        # Deduplicate: round coords to ~50m to catch cross-source duplicates
        all_stations['_lat_r'] = all_stations['lat'].round(4)
        all_stations['_lon_r'] = all_stations['lon'].round(4)
        all_stations = (
            all_stations
            .sort_values('source', key=lambda s: s.map({'nys_mapserver': 0, 'osm': 1}))
            .drop_duplicates(subset=['_lat_r', '_lon_r'], keep='first')
            .drop(columns=['_lat_r', '_lon_r'])
        )
        print(f"\n  Total unique gas stations: {len(all_stations)}")

    # Filter to Lower Manhattan
    if len(all_stations) > 0:
        lm_mask = (
            (all_stations['lat'] >= LM_BOUNDS['min_lat']) &
            (all_stations['lat'] <= LM_BOUNDS['max_lat']) &
            (all_stations['lon'] >= LM_BOUNDS['min_lon']) &
            (all_stations['lon'] <= LM_BOUNDS['max_lon'])
        )
        stations_lm = all_stations[lm_mask].copy()
        print(f"  Gas stations in Lower Manhattan: {len(stations_lm)}")

        if len(stations_lm) > 0:
            print(stations_lm[['name', 'lat', 'lon', 'brand', 'source']].to_string())

        # Also save stations in expanded area (LM + immediate surroundings)
        # that could serve LM but are just outside bbox
        expanded_mask = (
            (all_stations['lat'] >= LM_BOUNDS['min_lat'] - 0.01) &
            (all_stations['lat'] <= LM_BOUNDS['max_lat'] + 0.01) &
            (all_stations['lon'] >= LM_BOUNDS['min_lon'] - 0.01) &
            (all_stations['lon'] <= LM_BOUNDS['max_lon'] + 0.01)
        )
        stations_expanded = all_stations[expanded_mask].copy()
        print(f"  Gas stations in expanded LM area: {len(stations_expanded)}")
    else:
        stations_lm = pd.DataFrame()
        stations_expanded = pd.DataFrame()

    # ── 2. PETROLEUM TERMINALS ────────────────────────────────

    print("\n" + "#" * 60)
    print("# PART 2: PETROLEUM TERMINALS (Tier 1 — bulk storage)")
    print("#" * 60)

    # Try EIA first
    eia_terminals = download_eia_terminals()

    # Always get hardcoded terminals
    hardcoded_terminals = get_hardcoded_terminals()

    # Combine and deduplicate
    print("\n" + "=" * 60)
    print("COMBINING TERMINAL SOURCES")
    print("=" * 60)

    terminal_dfs = []

    if eia_terminals is not None and len(eia_terminals) > 0:
        eia_norm = pd.DataFrame({
            'name':         eia_terminals.get('Terminal_N',
                            eia_terminals.get('Name',
                            eia_terminals.get('TERMINAL', 'Unknown Terminal'))),
            'lat':          eia_terminals.geometry.y,
            'lon':          eia_terminals.geometry.x,
            'capacity_bbl': eia_terminals.get('Total_She',
                            eia_terminals.get('CAPACITY', 0)),
            'products':     eia_terminals.get('Products',
                            eia_terminals.get('PRODUCTS', '')),
            'source':       'eia',
        })
        terminal_dfs.append(eia_norm)
        print(f"  EIA: {len(eia_norm)} terminals")

    # Always include hardcoded
    hc_norm = pd.DataFrame({
        'name':         hardcoded_terminals['name'],
        'lat':          hardcoded_terminals['lat'],
        'lon':          hardcoded_terminals['lon'],
        'capacity_bbl': hardcoded_terminals['capacity_bbl'],
        'products':     hardcoded_terminals['products'],
        'source':       'hardcoded_nyserda',
    })
    terminal_dfs.append(hc_norm)
    print(f"  Hardcoded: {len(hc_norm)} terminals")

    all_terminals = pd.concat(terminal_dfs, ignore_index=True)

    # Deduplicate
    all_terminals['_lat_r'] = all_terminals['lat'].round(2)
    all_terminals['_lon_r'] = all_terminals['lon'].round(2)
    all_terminals = (
        all_terminals
        .sort_values('source', key=lambda s: s.map({'eia': 0, 'hardcoded_nyserda': 1}))
        .drop_duplicates(subset=['_lat_r', '_lon_r'], keep='first')
        .drop(columns=['_lat_r', '_lon_r'])
    )
    all_terminals['external'] = True   # all terminals are external to LM
    all_terminals['type'] = 'PETROLEUM_TERMINAL'

    print(f"\n  Total unique terminals: {len(all_terminals)}")
    print(all_terminals[['name', 'lat', 'lon', 'capacity_bbl', 'source']].to_string())

    # ── 3. SAVE ALL OUTPUTS ───────────────────────────────────

    print("\n" + "=" * 60)
    print("SAVING OUTPUTS")
    print("=" * 60)

    # Save gas stations (LM only)
    if len(stations_lm) > 0:
        stations_gdf = gpd.GeoDataFrame(
            stations_lm,
            geometry=[Point(r.lon, r.lat) for r in stations_lm.itertuples()],
            crs='EPSG:4326'
        )
        stations_gdf['type'] = 'GAS_STATION'
        stations_gdf['external'] = False
        outpath = os.path.join(OUTPUT_DIR, 'gas_stations_lm.geojson')
        stations_gdf.to_file(outpath, driver='GeoJSON')
        print(f"  Saved → {outpath}  ({len(stations_gdf)} stations)")
    else:
        print("  WARNING: No gas stations in LM to save")

    # Save expanded area stations (for reference)
    if len(stations_expanded) > 0:
        exp_gdf = gpd.GeoDataFrame(
            stations_expanded,
            geometry=[Point(r.lon, r.lat) for r in stations_expanded.itertuples()],
            crs='EPSG:4326'
        )
        outpath = os.path.join(OUTPUT_DIR, 'gas_stations_expanded.geojson')
        exp_gdf.to_file(outpath, driver='GeoJSON')
        print(f"  Saved → {outpath}  ({len(exp_gdf)} stations)")

    # Save petroleum terminals
    terminals_gdf = gpd.GeoDataFrame(
        all_terminals,
        geometry=[Point(r.lon, r.lat) for r in all_terminals.itertuples()],
        crs='EPSG:4326'
    )
    outpath = os.path.join(OUTPUT_DIR, 'petroleum_terminals_nyc.geojson')
    terminals_gdf.to_file(outpath, driver='GeoJSON')
    print(f"  Saved → {outpath}  ({len(terminals_gdf)} terminals)")

    # Save combined fuel infrastructure (for build_graph.py)
    combined_rows = []

    if len(stations_lm) > 0:
        for _, r in stations_lm.iterrows():
            combined_rows.append({
                'name':     r['name'],
                'lat':      r['lat'],
                'lon':      r['lon'],
                'type':     'GAS_STATION',
                'subtype':  r.get('brand', 'RETAIL'),
                'source':   r['source'],
                'external': False,
            })

    for _, r in all_terminals.iterrows():
        combined_rows.append({
            'name':         r['name'],
            'lat':          r['lat'],
            'lon':          r['lon'],
            'type':         'PETROLEUM_TERMINAL',
            'subtype':      'BULK_STORAGE',
            'source':       r['source'],
            'external':     True,
            'capacity_bbl': r.get('capacity_bbl', 0),
        })

    combined_df = pd.DataFrame(combined_rows)
    combined_gdf = gpd.GeoDataFrame(
        combined_df,
        geometry=[Point(r['lon'], r['lat']) for _, r in combined_df.iterrows()],
        crs='EPSG:4326'
    )
    outpath = os.path.join(OUTPUT_DIR, 'fuel_infra_all.geojson')
    combined_gdf.to_file(outpath, driver='GeoJSON')
    print(f"  Saved → {outpath}  ({len(combined_gdf)} total fuel nodes)")

    # ── 4. SUMMARY ────────────────────────────────────────────

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    n_stations = len(stations_lm) if len(stations_lm) > 0 else 0
    n_terminals = len(all_terminals)
    print(f"  Gas stations in LM:        {n_stations}")
    print(f"  Petroleum terminals (ext):  {n_terminals}")
    print(f"  Total fuel nodes:           {n_stations + n_terminals}")
    print(f"\n  Files saved to: {os.path.abspath(OUTPUT_DIR)}/")

    print("\n" + "=" * 60)
    print("INTEGRATION NOTES FOR build_graph.py")
    print("=" * 60)
    print("""
New node type: 'fuel' with two subtypes:
  - GAS_STATION:        local nodes in LM, need power to pump fuel
  - PETROLEUM_TERMINAL: external nodes in NY Harbor, supply fuel to stations

New edge types to add:
  fuel_distribution:  terminal → gas_station (supply chain)
  power → fuel:       power_dependency with buffer=4h (backup generators)
  fuel → hospital:    fuel_supplies with buffer=96h (generator fuel)
  fuel → telecom:     fuel_supplies with buffer=24-72h (generator fuel)
  fuel → water:       fuel_supplies with buffer=48h (diesel pump backup)

Buffer durations:
  power → gas_station:  0h  (no backup — station can't pump without grid power)
  power → terminal:     4h  (some terminals have backup generators)
  terminal → station:   0h  (supply chain — if terminal down, no truck loading)

Key cascade pathway:
  Flood → port closed → terminals can't receive fuel → trucks can't load
  → gas stations run dry → hospital generators can't refuel → hospital fails

This creates a TIME-DELAYED FEEDBACK LOOP:
  Without fuel: hospital buffer appears infinite (96h then done)
  With fuel:    hospital buffer = min(96h, time_until_fuel_exhausted)
  If fuel supply disrupted at t=0, hospital WILL fail at t=96h

Feature vector for fuel nodes (dim=8):
  lat, lon, elevation, flood_depth, is_terminal, capacity_proxy,
  has_backup_power, is_external
""")