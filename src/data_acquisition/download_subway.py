# ============================================================
# SUBWAY STATIONS — from MTA data
# ============================================================
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# Load MTA subway stations
subway = pd.read_csv('data/raw/MTA_Subway_Stations_20260325.csv')
print(subway.columns.tolist())
print(subway.shape)  # (496, 19)

# Lat/lon columns are already clean floats
for col in subway.columns:
    if 'lat' in col.lower() or 'lon' in col.lower() or 'geom' in col.lower():
        print(f"  {col}: {subway[col].head(3).tolist()}")

# ── Convert to GeoDataFrame ──
subway['geometry'] = subway.apply(
    lambda r: Point(r['GTFS Longitude'], r['GTFS Latitude']), axis=1
)
subway_gdf = gpd.GeoDataFrame(subway, geometry='geometry', crs='EPSG:4326')

# ── Filter to Lower Manhattan ──
subway_lm = subway_gdf[
    (subway_gdf['GTFS Latitude'] >= 40.700) & (subway_gdf['GTFS Latitude'] <= 40.755) &
    (subway_gdf['GTFS Longitude'] >= -74.020) & (subway_gdf['GTFS Longitude'] <= -73.970)
]

print(f"\nSubway stations in Lower Manhattan: {len(subway_lm)}")
print(subway_lm[['Stop Name', 'Daytime Routes', 'GTFS Latitude', 'GTFS Longitude', 'ADA']].to_string())

# ── Save ──
import os
os.makedirs('data/transit', exist_ok=True)
subway_lm.to_file('data/transit/subway_stations_lm.geojson', driver='GeoJSON')
print("\nSaved to data/transit/subway_stations_lm.geojson")
