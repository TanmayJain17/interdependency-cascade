# ============================================================
# HOSPITALS — from NYC Facilities Database (CSV version)
# ============================================================
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# Load the CSV
facilities = pd.read_csv('facilities_20260325.csv')

print(facilities.columns.tolist())
print(facilities.shape)

# The key column is FACSUBGRP = "HOSPITALS AND CLINICS"
# FACTYPE breakdown:
#   HOSPITAL                        63
#   ACUTE CARE HOSPITAL             11
#   HOSPITAL EXTENSION CLINIC      232   ← these are outpatient clinics, not hospitals
#   SCHOOL BASED HOSP EXT CLINIC   127
#   MOBILE HOSPITAL EXT CLINIC      13
#   HOSPICE                          6

# ── Extract hospitals (full hospitals only, no extension clinics / hospice) ──
hospitals = facilities[
    facilities['FACTYPE'].str.contains('HOSPITAL', case=False, na=False) &
    ~facilities['FACTYPE'].str.contains('EXTENSION CLINIC|HOSPICE', case=False, na=False)
].dropna(subset=['LATITUDE', 'LONGITUDE']).copy()

print(f"\nTotal hospitals NYC: {len(hospitals)}")
print(hospitals['FACTYPE'].value_counts())

# ── Convert to GeoDataFrame ──
hospitals['geometry'] = hospitals.apply(
    lambda r: Point(r['LONGITUDE'], r['LATITUDE']), axis=1
)
hospitals_gdf = gpd.GeoDataFrame(hospitals, geometry='geometry', crs='EPSG:4326')

# ── Filter to Lower Manhattan ──
hospitals_lm = hospitals_gdf[
    (hospitals_gdf['LATITUDE'] >= 40.700) & (hospitals_gdf['LATITUDE'] <= 40.755) &
    (hospitals_gdf['LONGITUDE'] >= -74.020) & (hospitals_gdf['LONGITUDE'] <= -73.970)
]

print(f"\nHospitals in Lower Manhattan: {len(hospitals_lm)}")
print(hospitals_lm[['FACNAME', 'FACTYPE', 'LATITUDE', 'LONGITUDE', 'ADDRESS']].to_string())

# ── Save ──
import os
os.makedirs('data/healthcare', exist_ok=True)
hospitals_lm.to_file('data/healthcare/hospitals_lm.geojson', driver='GeoJSON')
print("\nSaved to data/healthcare/hospitals_lm.geojson")
