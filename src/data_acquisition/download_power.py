"""
Download NYC Power Infrastructure Data (Substations + Transmission Lines)
Source: Rutgers University HIFLD Mirror (ArcGIS MapServer)
Data version: HIFLD 9/19/2019

This script:
1. Downloads all substations and transmission lines within NYC bounding box
2. Handles pagination if results exceed server's MaxRecordCount (1000)
3. Saves full NYC data as GeoJSON
4. Filters to Lower Manhattan for flood cascade prototype
5. Prints summary statistics
"""

import requests
import geopandas as gpd
import pandas as pd
import io
import os
import time

# ============================================================
# Configuration
# ============================================================

BASE_URL = "https://oceandata.rad.rutgers.edu/arcgis/rest/services/RenewableEnergy/HIFLD_Electric_SubstationsTransmissionLines/MapServer"

# Full NYC + surrounding area (wider to catch lines crossing borders)
NYC_BBOX = '-74.27,40.49,-73.70,40.92'

# Lower Manhattan filter
LM_BBOX = {
    'min_lat': 40.700, 'max_lat': 40.755,
    'min_lon': -74.020, 'max_lon': -73.970
}

# Output directory
OUTPUT_DIR = 'data/power'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Helper: Paginated ArcGIS query
# ============================================================

def query_arcgis_layer(layer_id, bbox, max_per_request=1000):
    """
    Query an ArcGIS MapServer layer with automatic pagination.
    Returns a GeoDataFrame with all features.
    """
    url = f"{BASE_URL}/{layer_id}/query"
    all_features = []
    offset = 0
    
    while True:
        params = {
            'where': '1=1',
            'geometry': bbox,
            'geometryType': 'esriGeometryEnvelope',
            'inSR': '4326',
            'spatialRel': 'esriSpatialRelIntersects',
            'outFields': '*',
            'outSR': '4326',
            'f': 'geojson',
            'resultOffset': offset,
            'resultRecordCount': max_per_request
        }
        
        print(f"  Querying layer {layer_id}, offset={offset}...")
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
        
        gdf = gpd.read_file(io.StringIO(response.text))
        
        if len(gdf) == 0:
            break
            
        all_features.append(gdf)
        print(f"  Got {len(gdf)} features")
        
        if len(gdf) < max_per_request:
            break
            
        offset += max_per_request
        time.sleep(0.5)
    
    if not all_features:
        print(f"  WARNING: No features returned for layer {layer_id}")
        return gpd.GeoDataFrame()
    
    combined = pd.concat(all_features, ignore_index=True)
    return gpd.GeoDataFrame(combined, geometry='geometry', crs='EPSG:4326')


# ============================================================
# Download Substations (Layer 0)
# ============================================================

print("=" * 60)
print("DOWNLOADING SUBSTATIONS (Layer 0)")
print("=" * 60)

substations_nyc = query_arcgis_layer(0, NYC_BBOX)
print(f"\nTotal NYC substations: {len(substations_nyc)}")

if len(substations_nyc) > 0:
    print(f"\nAvailable columns: {list(substations_nyc.columns)}")
    
    if 'MAX_VOLT' in substations_nyc.columns:
        print("\nSubstations by max voltage:")
        print(substations_nyc['MAX_VOLT'].value_counts().head(10))
    
    if 'TYPE' in substations_nyc.columns:
        print("\nSubstations by type:")
        print(substations_nyc['TYPE'].value_counts())
    
    if 'STATUS' in substations_nyc.columns:
        print("\nSubstations by status:")
        print(substations_nyc['STATUS'].value_counts())
    
    key_cols = [c for c in ['NAME', 'MAX_VOLT', 'MIN_VOLT', 'TYPE', 'STATUS', 'LATITUDE', 'LONGITUDE'] 
                if c in substations_nyc.columns]
    if key_cols:
        print(f"\nFirst 10 substations:")
        print(substations_nyc[key_cols].head(10).to_string())
    
    # Save full NYC
    outpath = os.path.join(OUTPUT_DIR, 'substations_nyc.geojson')
    substations_nyc.to_file(outpath, driver='GeoJSON')
    print(f"\nSaved to {outpath}")
    
    # Filter to Lower Manhattan
    if 'LATITUDE' in substations_nyc.columns and 'LONGITUDE' in substations_nyc.columns:
        lm_mask = (
            (substations_nyc['LATITUDE'] >= LM_BBOX['min_lat']) &
            (substations_nyc['LATITUDE'] <= LM_BBOX['max_lat']) &
            (substations_nyc['LONGITUDE'] >= LM_BBOX['min_lon']) &
            (substations_nyc['LONGITUDE'] <= LM_BBOX['max_lon'])
        )
    else:
        lm_mask = (
            (substations_nyc.geometry.y >= LM_BBOX['min_lat']) &
            (substations_nyc.geometry.y <= LM_BBOX['max_lat']) &
            (substations_nyc.geometry.x >= LM_BBOX['min_lon']) &
            (substations_nyc.geometry.x <= LM_BBOX['max_lon'])
        )
    
    substations_lm = substations_nyc[lm_mask].copy()
    print(f"\nLower Manhattan substations: {len(substations_lm)}")
    
    if len(substations_lm) > 0:
        if key_cols:
            print(substations_lm[key_cols].to_string())
        outpath_lm = os.path.join(OUTPUT_DIR, 'substations_lower_manhattan.geojson')
        substations_lm.to_file(outpath_lm, driver='GeoJSON')
        print(f"Saved to {outpath_lm}")
    else:
        print("NOTE: No substations in strict LM bbox.")
        print("Con Edison substations serving LM may be just outside.")
        print("Try expanding bbox northward to 40.76 (includes East 13th St area).")
        
        # Auto-expand and retry
        expanded_mask = (
            (substations_nyc['LATITUDE'] >= 40.695) &
            (substations_nyc['LATITUDE'] <= 40.760) &
            (substations_nyc['LONGITUDE'] >= -74.025) &
            (substations_nyc['LONGITUDE'] <= -73.965)
        ) if 'LATITUDE' in substations_nyc.columns else None
        
        if expanded_mask is not None:
            substations_lm_expanded = substations_nyc[expanded_mask].copy()
            print(f"\nExpanded LM area substations: {len(substations_lm_expanded)}")
            if len(substations_lm_expanded) > 0 and key_cols:
                print(substations_lm_expanded[key_cols].to_string())
                outpath_lm = os.path.join(OUTPUT_DIR, 'substations_lower_manhattan_expanded.geojson')
                substations_lm_expanded.to_file(outpath_lm, driver='GeoJSON')
                print(f"Saved to {outpath_lm}")


# ============================================================
# Download Transmission Lines (Layer 1)
# ============================================================

print("\n" + "=" * 60)
print("DOWNLOADING TRANSMISSION LINES (Layer 1)")
print("=" * 60)

lines_nyc = query_arcgis_layer(1, NYC_BBOX)
print(f"\nTotal NYC transmission lines: {len(lines_nyc)}")

if len(lines_nyc) > 0:
    print(f"\nAvailable columns: {list(lines_nyc.columns)}")
    
    if 'VOLT_CLASS' in lines_nyc.columns:
        print("\nLines by voltage class:")
        print(lines_nyc['VOLT_CLASS'].value_counts())
    
    if 'OWNER' in lines_nyc.columns:
        print("\nLines by owner (top 10):")
        print(lines_nyc['OWNER'].value_counts().head(10))
    
    key_cols_lines = [c for c in ['VOLTAGE', 'VOLT_CLASS', 'OWNER', 'TYPE', 'STATUS', 'SUB_1', 'SUB_2'] 
                      if c in lines_nyc.columns]
    if key_cols_lines:
        print(f"\nFirst 10 transmission lines:")
        print(lines_nyc[key_cols_lines].head(10).to_string())
    
    # Save full NYC
    outpath = os.path.join(OUTPUT_DIR, 'transmission_lines_nyc.geojson')
    lines_nyc.to_file(outpath, driver='GeoJSON')
    print(f"\nSaved to {outpath}")
    
    # Filter lines intersecting Lower Manhattan
    from shapely.geometry import box
    lm_box = box(LM_BBOX['min_lon'], LM_BBOX['min_lat'], 
                 LM_BBOX['max_lon'], LM_BBOX['max_lat'])
    
    lines_lm = lines_nyc[lines_nyc.geometry.intersects(lm_box)].copy()
    print(f"\nTransmission lines intersecting Lower Manhattan: {len(lines_lm)}")
    
    if len(lines_lm) > 0:
        if key_cols_lines:
            print(lines_lm[key_cols_lines].to_string())
        outpath_lm = os.path.join(OUTPUT_DIR, 'transmission_lines_lower_manhattan.geojson')
        lines_lm.to_file(outpath_lm, driver='GeoJSON')
        print(f"Saved to {outpath_lm}")


# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
n_sub_nyc = len(substations_nyc) if len(substations_nyc) > 0 else 0
n_lines_nyc = len(lines_nyc) if len(lines_nyc) > 0 else 0
n_sub_lm = len(substations_lm) if 'substations_lm' in dir() and len(substations_lm) > 0 else 0
n_lines_lm = len(lines_lm) if 'lines_lm' in dir() and len(lines_lm) > 0 else 0

print(f"NYC Substations:           {n_sub_nyc}")
print(f"NYC Transmission Lines:    {n_lines_nyc}")
print(f"LM Substations:            {n_sub_lm}")
print(f"LM Transmission Lines:     {n_lines_lm}")
print(f"\nFiles saved to: {os.path.abspath(OUTPUT_DIR)}/")

print("\n" + "=" * 60)
print("NOTES")
print("=" * 60)
print("""
1. HIFLD covers substations >= 69kV (transmission level only).
   Distribution substations are Con Edison proprietary — flag for Dr. Lin.

2. Use SUB_1 and SUB_2 fields on transmission lines to build 
   the power grid graph topology (which substations connect to which).

3. VOLTAGE and VOLT_CLASS on lines = edge features for the GNN.

4. If LM shows 0 substations, the expanded bbox version captures 
   substations just north of the strict boundary that serve LM.
""")