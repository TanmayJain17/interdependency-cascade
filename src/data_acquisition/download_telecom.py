"""
Download NYC Cell Tower Data from OpenCelliD
Source: OpenCelliD.org (CC BY-SA 4.0)

This script:
1. Downloads cell tower data for NYC area via OpenCelliD API
2. Filters to NYC and Lower Manhattan
3. Saves as GeoJSON and CSV
4. Prints summary statistics by radio type and carrier
"""

import requests
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import os
import io

# ============================================================
# Configuration
# ============================================================

# PUT YOUR API TOKEN HERE
API_TOKEN = os.environ.get("OPENCELLID_TOKEN", "YOUR_TOKEN_HERE")  # set OPENCELLID_TOKEN env var

# NYC bounding box
NYC_BOUNDS = {
    'min_lat': 40.49, 'max_lat': 40.92,
    'min_lon': -74.27, 'max_lon': -73.70
}

# Lower Manhattan filter
LM_BOUNDS = {
    'min_lat': 40.700, 'max_lat': 40.755,
    'min_lon': -74.020, 'max_lon': -73.970
}

# US Mobile Country Codes
US_MCC = [310, 311, 312, 313, 316]

OUTPUT_DIR = 'data/telecom'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Method 1: Download via country-level CSV (Recommended)
# ============================================================

def download_via_csv():
    """
    Download the full US cell tower dataset and filter to NYC.
    This is the most reliable method — gets ALL towers.
    The file is large (~200MB compressed) but comprehensive.
    """
    print("=" * 60)
    print("METHOD 1: Country-level CSV download")
    print("=" * 60)
    
    # Download US cell tower data
    url = f"https://opencellid.org/ocid/downloads?token={API_TOKEN}&type=mcc&file=310.csv.gz"
    
    print("Downloading MCC=310 data (this may take a few minutes)...")
    print("File is ~200MB compressed, contains millions of rows")
    
    response = requests.get(url, stream=True, timeout=300)
    
    if response.status_code == 200:
        # Save compressed file
        gz_path = os.path.join(OUTPUT_DIR, '310.csv.gz')
        with open(gz_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded to {gz_path}")
        
        # Read and filter to NYC
        print("Reading and filtering to NYC bbox...")
        df = pd.read_csv(gz_path, 
                         names=['radio', 'mcc', 'net', 'area', 'cell', 
                                'unit', 'lon', 'lat', 'range', 'samples',
                                'changeable', 'created', 'updated', 'averageSignal'],
                         skiprows=1)
        
        # Filter to NYC
        nyc_mask = (
            (df['lat'] >= NYC_BOUNDS['min_lat']) & 
            (df['lat'] <= NYC_BOUNDS['max_lat']) &
            (df['lon'] >= NYC_BOUNDS['min_lon']) & 
            (df['lon'] <= NYC_BOUNDS['max_lon'])
        )
        towers_nyc = df[nyc_mask].copy()
        print(f"NYC towers: {len(towers_nyc)}")
        
        return towers_nyc
    else:
        print(f"Download failed: {response.status_code}")
        print(f"Response: {response.text[:500]}")
        return None


# ============================================================
# Method 2: Download via API cell queries (Smaller, faster)
# ============================================================

def download_via_api():
    """
    Query OpenCelliD API for cells in a bounding box.
    Faster but may hit rate limits and return fewer results.
    Use this if Method 1 is too slow.
    """
    print("=" * 60)
    print("METHOD 2: API bounding box query")
    print("=" * 60)
    
    # OpenCelliD API endpoint for cells in bbox
    url = "https://opencellid.org/cell/getInArea"
    
    # Query NYC area (API may limit results)
    # Split into smaller boxes if needed
    sub_boxes = [
        # Lower Manhattan + Downtown Brooklyn
        {'min_lat': 40.69, 'max_lat': 40.76, 'min_lon': -74.02, 'max_lon': -73.96},
        # Midtown
        {'min_lat': 40.75, 'max_lat': 40.80, 'min_lon': -74.01, 'max_lon': -73.95},
        # Upper Manhattan
        {'min_lat': 40.79, 'max_lat': 40.88, 'min_lon': -74.01, 'max_lon': -73.92},
        # Brooklyn
        {'min_lat': 40.63, 'max_lat': 40.70, 'min_lon': -74.02, 'max_lon': -73.90},
        # Queens
        {'min_lat': 40.70, 'max_lat': 40.80, 'min_lon': -73.96, 'max_lon': -73.75},
    ]
    
    all_towers = []
    
    for i, bbox in enumerate(sub_boxes):
        params = {
            'key': API_TOKEN,
            'BBOX': f"{bbox['min_lat']},{bbox['min_lon']},{bbox['max_lat']},{bbox['max_lon']}",
            'format': 'csv',
            'limit': 10000
        }
        
        print(f"  Querying sub-box {i+1}/{len(sub_boxes)}...")
        response = requests.get(url, params=params, timeout=60)
        
        if response.status_code == 200 and len(response.text) > 50:
            df = pd.read_csv(io.StringIO(response.text))
            all_towers.append(df)
            print(f"  Got {len(df)} towers")
        else:
            print(f"  Sub-box {i+1} returned: status={response.status_code}, length={len(response.text)}")
    
    if all_towers:
        towers_nyc = pd.concat(all_towers, ignore_index=True)
        # Remove duplicates (overlapping sub-boxes)
        towers_nyc = towers_nyc.drop_duplicates(subset=['lat', 'lon', 'radio', 'mcc', 'net', 'cell'])
        print(f"\nTotal NYC towers (deduplicated): {len(towers_nyc)}")
        return towers_nyc
    else:
        print("No data returned from API")
        return None


# ============================================================
# Main execution
# ============================================================

if __name__ == "__main__":
    
    if API_TOKEN == "YOUR_TOKEN_HERE":
        print("ERROR: Please set your API_TOKEN at the top of the script!")
        exit(1)
        
    print("Downloading via CSV method...\n")
    towers_nyc = download_via_csv()
    
    """ # Try Method 2 first (faster, smaller download)
    print("Attempting API query method first (faster)...\n")
    towers_nyc = download_via_api()
    
    # If API method returns too few results, try CSV method
    if towers_nyc is None or len(towers_nyc) < 100:
        print("\nAPI method returned few results. Trying CSV download...")
        towers_nyc = download_via_csv() """
    
    if towers_nyc is None or len(towers_nyc) == 0:
        print("\nERROR: No tower data retrieved. Check your API token.")
        exit(1)
    
    # ============================================================
    # Process and analyze
    # ============================================================
    
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    
    print(f"\nAvailable columns: {list(towers_nyc.columns)}")
    print(f"Total NYC towers: {len(towers_nyc)}")
    
    # Radio type breakdown
    if 'radio' in towers_nyc.columns:
        print("\nTowers by radio type:")
        print(towers_nyc['radio'].value_counts())
    
    # Network/carrier breakdown (net codes)
    # Common US net codes: AT&T=410, T-Mobile=260, Verizon=480/12, Sprint=120
    if 'net' in towers_nyc.columns:
        print("\nTop 10 network codes:")
        print(towers_nyc['net'].value_counts().head(10))
    
    # Sample data
    display_cols = [c for c in ['radio', 'mcc', 'net', 'lat', 'lon', 'range', 'samples'] 
                    if c in towers_nyc.columns]
    if display_cols:
        print(f"\nFirst 10 towers:")
        print(towers_nyc[display_cols].head(10).to_string())
    
    # ============================================================
    # Convert to GeoDataFrame and save
    # ============================================================
    
    # Identify lat/lon columns (OpenCelliD uses 'lat'/'lon')
    lat_col = 'lat' if 'lat' in towers_nyc.columns else 'LATITUDE'
    lon_col = 'lon' if 'lon' in towers_nyc.columns else 'LONGITUDE'
    
    geometry = [Point(xy) for xy in zip(towers_nyc[lon_col], towers_nyc[lat_col])]
    towers_gdf = gpd.GeoDataFrame(towers_nyc, geometry=geometry, crs='EPSG:4326')
    
    # Save NYC
    outpath = os.path.join(OUTPUT_DIR, 'cell_towers_nyc.geojson')
    towers_gdf.to_file(outpath, driver='GeoJSON')
    print(f"\nSaved NYC data to {outpath}")
    
    # Also save as CSV (smaller, easier to inspect)
    csv_path = os.path.join(OUTPUT_DIR, 'cell_towers_nyc.csv')
    towers_nyc.to_csv(csv_path, index=False)
    print(f"Saved NYC CSV to {csv_path}")
    
    # ============================================================
    # Filter to Lower Manhattan
    # ============================================================
    
    lm_mask = (
        (towers_nyc[lat_col] >= LM_BOUNDS['min_lat']) & 
        (towers_nyc[lat_col] <= LM_BOUNDS['max_lat']) &
        (towers_nyc[lon_col] >= LM_BOUNDS['min_lon']) & 
        (towers_nyc[lon_col] <= LM_BOUNDS['max_lon'])
    )
    towers_lm = towers_gdf[lm_mask].copy()
    
    print(f"\nLower Manhattan towers: {len(towers_lm)}")
    
    if len(towers_lm) > 0:
        if 'radio' in towers_lm.columns:
            print("\nLM towers by radio type:")
            print(towers_lm['radio'].value_counts())
        
        if display_cols:
            print(f"\nFirst 20 LM towers:")
            print(towers_lm[display_cols].head(20).to_string())
        
        outpath_lm = os.path.join(OUTPUT_DIR, 'cell_towers_lower_manhattan.geojson')
        towers_lm.to_file(outpath_lm, driver='GeoJSON')
        print(f"\nSaved to {outpath_lm}")
        
        csv_lm = os.path.join(OUTPUT_DIR, 'cell_towers_lower_manhattan.csv')
        towers_lm.drop(columns='geometry').to_csv(csv_lm, index=False)
        print(f"Saved to {csv_lm}")
    else:
        print("No towers found in Lower Manhattan bbox")
    
    # ============================================================
    # Summary
    # ============================================================
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"NYC Cell Towers:           {len(towers_nyc)}")
    print(f"LM Cell Towers:            {len(towers_lm)}")
    if 'radio' in towers_nyc.columns:
        for radio in towers_nyc['radio'].unique():
            count = len(towers_nyc[towers_nyc['radio'] == radio])
            print(f"  {radio}: {count}")
    print(f"\nFiles saved to: {os.path.abspath(OUTPUT_DIR)}/")
    
    print("\n" + "=" * 60)
    print("NOTES")
    print("=" * 60)
    print("""
1. OpenCelliD data represents observed cell sites, not physical towers.
   Multiple entries at the same location = multiple carriers/radios on 
   one physical tower. For your GNN graph, deduplicate by location:
   
   towers_unique = towers_lm.dissolve(
       by=[towers_lm.geometry.x.round(4), towers_lm.geometry.y.round(4)],
       aggfunc={'radio': 'count', 'net': 'first'}
   )
   
2. The 'range' field estimates coverage radius in meters — useful as 
   a node feature (coverage_radius) in your GNN.

3. The 'samples' field indicates data quality — more samples = more 
   reliable location. Filter to samples >= 5 for cleaner data.

4. For inter-infrastructure edges (power→telecom), connect each tower 
   to its nearest power substation using spatial proximity.
""")