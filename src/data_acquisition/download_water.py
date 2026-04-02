"""
Get NYC Water Infrastructure Data
Sources:
1. NYC Facilities Database CSV (already downloaded)
2. OpenStreetMap Overpass API
3. NYC DEP wastewater treatment plants (hardcoded)
"""

import pandas as pd
import geopandas as gpd
import requests
from shapely.geometry import Point
import os

os.makedirs('data/water', exist_ok=True)

# ============================================================
# METHOD 1: NYC Facilities Database CSV
# ============================================================

print("=" * 60)
print("EXTRACTING FROM NYC FACILITIES DATABASE")
print("=" * 60)

facilities = pd.read_csv('data/raw/facilities_20260325.csv')
print(f"Total facilities: {len(facilities)}")

# Water infrastructure FACTYPE values confirmed in this dataset:
WATER_TYPES = [
    'PUMPING STATION',
    'WASTEWATER PUMPING STATION',
    'STORMWATER PUMPING STATION',
    'WATER POLLUTION CONTROL PLANT',
    'WASTEWATER TREATMENT PLANT',
    'SLUDGE DE-WATERING FACILITY',
    'POLLUTION CONTROL FACILITY',
]

water_fac = (
    facilities[facilities['FACTYPE'].isin(WATER_TYPES)]
    .dropna(subset=['LATITUDE', 'LONGITUDE'])
    .copy()
)

print(f"\nWater infrastructure from FacDB: {len(water_fac)}")
print(water_fac['FACTYPE'].value_counts())

facdb_rows = pd.DataFrame({
    'name':     water_fac['FACNAME'].values,
    'lat':      water_fac['LATITUDE'].astype(float).values,
    'lon':      water_fac['LONGITUDE'].astype(float).values,
    'type':     water_fac['FACTYPE'].values,
    'address':  water_fac['ADDRESS'].values,
    'boro':     water_fac['BORO'].values,
    'operator': water_fac['OPNAME'].values,
    'source':   'facdb',
})

# ============================================================
# METHOD 2: OpenStreetMap Overpass API
# ============================================================

print("\n" + "=" * 60)
print("QUERYING OPENSTREETMAP")
print("=" * 60)

overpass_url = "http://overpass-api.de/api/interpreter"

query = """
[out:json][timeout:120];
(
  node["man_made"="pumping_station"](40.49,-74.27,40.92,-73.70);
  way["man_made"="pumping_station"](40.49,-74.27,40.92,-73.70);
  node["man_made"="water_tower"](40.49,-74.27,40.92,-73.70);
  way["man_made"="water_tower"](40.49,-74.27,40.92,-73.70);
  node["man_made"="wastewater_plant"](40.49,-74.27,40.92,-73.70);
  way["man_made"="wastewater_plant"](40.49,-74.27,40.92,-73.70);
  node["man_made"="water_works"](40.49,-74.27,40.92,-73.70);
  way["man_made"="water_works"](40.49,-74.27,40.92,-73.70);
  node["man_made"="reservoir_covered"](40.49,-74.27,40.92,-73.70);
  way["man_made"="reservoir_covered"](40.49,-74.27,40.92,-73.70);
  relation["man_made"="wastewater_plant"](40.49,-74.27,40.92,-73.70);
);
out center;
"""

print("Querying OSM...")
try:
    response = requests.post(overpass_url, data={'data': query}, timeout=120)
    elements = response.json().get('elements', []) if response.status_code == 200 else []
    print(f"Found {len(elements)} features from OSM")

    osm_rows = []
    for e in elements:
        lat = e.get('lat') or e.get('center', {}).get('lat')
        lon = e.get('lon') or e.get('center', {}).get('lon')
        tags = e.get('tags', {})
        if lat and lon:
            osm_rows.append({
                'name':     tags.get('name', ''),
                'lat':      lat,
                'lon':      lon,
                'type':     tags.get('man_made', tags.get('amenity', 'unknown')),
                'address':  tags.get('addr:full', tags.get('addr:street', '')),
                'boro':     '',
                'operator': tags.get('operator', ''),
                'source':   'osm',
            })

    osm_df = pd.DataFrame(osm_rows) if osm_rows else pd.DataFrame()
    if len(osm_df):
        print(osm_df['type'].value_counts().to_string())

except Exception as ex:
    print(f"OSM query failed: {ex}")
    osm_df = pd.DataFrame()

# ============================================================
# METHOD 3: NYC DEP — all 14 wastewater treatment plants
# ============================================================

print("\n" + "=" * 60)
print("NYC DEP WASTEWATER TREATMENT PLANTS (hardcoded)")
print("=" * 60)

dep_plants = pd.DataFrame([
    {'name': 'North River WWTP',        'lat': 40.7997, 'lon': -73.9985, 'capacity_mgd': 170},
    {'name': 'Newtown Creek WWTP',       'lat': 40.7367, 'lon': -73.9388, 'capacity_mgd': 310},
    {'name': 'Red Hook WWTP',            'lat': 40.6729, 'lon': -74.0097, 'capacity_mgd': 60},
    {'name': '26th Ward WWTP',           'lat': 40.6424, 'lon': -73.8569, 'capacity_mgd': 85},
    {'name': 'Coney Island WWTP',        'lat': 40.5726, 'lon': -73.9677, 'capacity_mgd': 110},
    {'name': 'Jamaica WWTP',             'lat': 40.6316, 'lon': -73.8013, 'capacity_mgd': 100},
    {'name': 'Tallman Island WWTP',      'lat': 40.7925, 'lon': -73.8006, 'capacity_mgd': 80},
    {'name': 'Bowery Bay WWTP',          'lat': 40.7804, 'lon': -73.8837, 'capacity_mgd': 150},
    {'name': 'Hunts Point WWTP',         'lat': 40.8052, 'lon': -73.8702, 'capacity_mgd': 200},
    {'name': 'Wards Island WWTP',        'lat': 40.7912, 'lon': -73.9275, 'capacity_mgd': 275},
    {'name': 'Oakwood Beach WWTP',       'lat': 40.5556, 'lon': -74.1219, 'capacity_mgd': 40},
    {'name': 'Port Richmond WWTP',       'lat': 40.6366, 'lon': -74.1348, 'capacity_mgd': 60},
    {'name': 'Rockaway WWTP',            'lat': 40.5612, 'lon': -73.8576, 'capacity_mgd': 45},
    {'name': 'Owls Head WWTP',           'lat': 40.6450, 'lon': -74.0285, 'capacity_mgd': 120},
])
dep_plants['type']     = 'WASTEWATER TREATMENT PLANT'
dep_plants['address']  = ''
dep_plants['boro']     = ''
dep_plants['operator'] = 'NYC DEP'
dep_plants['source']   = 'nyc_dep'

print(f"NYC DEP plants: {len(dep_plants)}")
print(dep_plants[['name', 'lat', 'lon', 'capacity_mgd']].to_string())

# ============================================================
# COMBINE + DEDUPLICATE
# ============================================================

print("\n" + "=" * 60)
print("COMBINING ALL SOURCES")
print("=" * 60)

all_dfs = [facdb_rows, dep_plants[['name','lat','lon','type','address','boro','operator','source']]]
if len(osm_df):
    all_dfs.append(osm_df)

combined = pd.concat(all_dfs, ignore_index=True)

# Drop near-duplicates: FacDB already has most treatment plants — keep facdb over dep
# Round coords to ~100m to detect dupes across sources
combined['_lat_r'] = combined['lat'].round(3)
combined['_lon_r'] = combined['lon'].round(3)
combined = combined.sort_values('source', key=lambda s: s.map({'facdb': 0, 'osm': 1, 'nyc_dep': 2}))
combined = combined.drop_duplicates(subset=['_lat_r', '_lon_r'], keep='first').drop(columns=['_lat_r', '_lon_r'])

print(f"Total unique water features: {len(combined)}")
print("\nBy type:")
print(combined['type'].value_counts().to_string())
print("\nBy source:")
print(combined['source'].value_counts().to_string())

# ── GeoDataFrame ──
water_gdf = gpd.GeoDataFrame(
    combined,
    geometry=[Point(r.lon, r.lat) for r in combined.itertuples()],
    crs='EPSG:4326'
)

water_gdf.to_file('data/water/water_infra_nyc.geojson', driver='GeoJSON')
print("\nSaved → data/water/water_infra_nyc.geojson")

# ============================================================
# FILTER TO LOWER MANHATTAN
# ============================================================

print("\n" + "=" * 60)
print("LOWER MANHATTAN WATER INFRASTRUCTURE")
print("=" * 60)

LM = dict(lat_min=40.700, lat_max=40.755, lon_min=-74.020, lon_max=-73.970)

water_lm = water_gdf[
    (water_gdf['lat'] >= LM['lat_min']) & (water_gdf['lat'] <= LM['lat_max']) &
    (water_gdf['lon'] >= LM['lon_min']) & (water_gdf['lon'] <= LM['lon_max'])
].copy()

print(f"In-bbox features: {len(water_lm)}")

if len(water_lm):
    print(water_lm[['name', 'type', 'lat', 'lon', 'address']].to_string())
    water_lm.to_file('data/water/water_infra_lm.geojson', driver='GeoJSON')
    print("\nSaved → data/water/water_infra_lm.geojson")

# Always save LM-serving plants (outside bbox but serving LM sewershed)
lm_center = Point(-73.995, 40.725)
water_gdf['dist_to_lm'] = water_gdf.geometry.distance(lm_center)
serving_lm = water_gdf[
    water_gdf['type'].str.contains('TREATMENT|CONTROL PLANT', case=False, na=False)
].nsmallest(5, 'dist_to_lm')

print(f"\nNearest treatment plants serving Lower Manhattan:")
print(serving_lm[['name', 'type', 'lat', 'lon', 'dist_to_lm']].to_string())
serving_lm.to_file('data/water/water_treatment_serving_lm.geojson', driver='GeoJSON')
print("\nSaved → data/water/water_treatment_serving_lm.geojson")
