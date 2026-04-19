#!/usr/bin/env python3
"""
fetch_dep_flood_maps.py

Download the three NYC DEP Stormwater Flood Map scenarios from ArcGIS Online
FeatureServer endpoints. Saves each as a GeoJSON in data/raw/flood/.

The NYC Open Data Socrata mirrors of these layers have been restricted to
agency-only access. We instead query the original ArcGIS FeatureServers
(hosted by Esri's NYC demo org), where the data is still public.

Service URLs and schema discovered via:
    https://www.arcgis.com/sharing/rest/search?q=owner:esri_dashboardpub+NYC+Stormwater

Attribute schema (all three layers):
    OBJECTID            internal row ID
    Flooding_Category   coded integer:
        1 = Nuisance Flooding    (ponding 4 in to 1 ft)
        2 = Deep and Contiguous  (ponding >= 1 ft)
        3 = Future High Tides    (tidal inundation; 2050 and 2080 scenarios only)
    Shape__Area         auto-computed polygon area
    Shape__Length       auto-computed polygon perimeter
    geometry            Polygon/MultiPolygon (service native CRS EPSG:3857;
                        we request EPSG:4326 output for easier downstream use)

Run from project root (~/Desktop/RA/):
    python3 src/data_acquisition/fetch_dep_flood_maps.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

import requests

try:
    import geopandas as gpd
except ImportError:
    print("ERROR: geopandas not installed. Activate your RA venv or run:")
    print("  pip install geopandas")
    sys.exit(1)


# -----------------------------------------------------------------------------
# Layer registry
# -----------------------------------------------------------------------------

BASE = (
    "https://services.arcgis.com/P3ePLMYs2RVChkJx/arcgis/rest/services"
)

LAYERS: dict[str, dict[str, str]] = {
    "moderate_current": {
        "service_url": f"{BASE}/NYC_Stormwater_Flood_Map_Moderate_Flood_with_Current_Sea_Levels/FeatureServer",
        "description": "Moderate Flood (2.13 in/hr) with Current Sea Levels",
        "filename": "moderate_current.geojson",
    },
    "moderate_2050": {
        "service_url": f"{BASE}/NYC_Stormwater_Flood_Map___Moderate_Flood_with_2050_Sea_Level_Rise_gdb/FeatureServer",
        "description": "Moderate Flood (2.13 in/hr) with 2050 Sea Level Rise",
        "filename": "moderate_2050.geojson",
    },
    "extreme_2080": {
        "service_url": f"{BASE}/NYC_Stormwater_Flood_Map___Extreme_Flood_with_2080_Sea_Level_Rise_gdb/FeatureServer",
        "description": "Extreme Flood (3.66 in/hr) with 2080 Sea Level Rise",
        "filename": "extreme_2080.geojson",
    },
}

OUT_DIR = Path("data/raw/flood")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, application/geo+json, */*",
}

# Output file size threshold below which we assume download is incomplete
MIN_VALID_SIZE_BYTES = 100_000  # 100 KB — citywide polygon layers are much larger


# -----------------------------------------------------------------------------
# ArcGIS REST helpers
# -----------------------------------------------------------------------------

def get_layer_info(service_url: str) -> dict[str, Any]:
    """Fetch metadata for layer 0 of a FeatureServer."""
    resp = requests.get(
        f"{service_url}/0",
        params={"f": "json"},
        headers=HEADERS,
        timeout=30,
    )
    resp.raise_for_status()
    info = resp.json()
    if "error" in info:
        raise RuntimeError(f"ArcGIS error: {info['error']}")
    return info


def get_feature_count(service_url: str) -> Optional[int]:
    """
    Return the total feature count via returnCountOnly, or None on failure.
    Used for progress estimation only.
    """
    try:
        resp = requests.get(
            f"{service_url}/0/query",
            params={"where": "1=1", "returnCountOnly": "true", "f": "json"},
            headers=HEADERS,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return int(data.get("count", 0)) or None
    except Exception:
        return None


def fetch_all_features(
    service_url: str,
    page_size: int,
    total_hint: Optional[int] = None,
) -> list[dict[str, Any]]:
    """
    Paginate through every feature in layer 0 of the FeatureServer.
    Returns a list of GeoJSON Feature dicts (already in EPSG:4326).
    """
    query_url = f"{service_url}/0/query"
    base_params = {
        "where": "1=1",
        "outFields": "*",
        "outSR": "4326",
        "f": "geojson",
        "resultRecordCount": page_size,
    }

    all_features: list[dict[str, Any]] = []
    offset = 0
    page = 1

    while True:
        params = {**base_params, "resultOffset": offset}

        for attempt in range(3):
            try:
                resp = requests.get(
                    query_url,
                    params=params,
                    headers=HEADERS,
                    timeout=300,
                )
                resp.raise_for_status()
                data = resp.json()
                break
            except requests.RequestException as e:
                if attempt == 2:
                    raise
                print(f"\n        WARN: page {page} attempt {attempt + 1} failed ({e}); retrying in 3s...")
                time.sleep(3)

        if isinstance(data, dict) and data.get("error"):
            raise RuntimeError(f"ArcGIS error on page {page}: {data['error']}")

        features = data.get("features", [])
        exceeded = data.get("exceededTransferLimit", False)

        all_features.extend(features)

        # Progress line
        total_str = f"/{total_hint:,}" if total_hint else ""
        pct_str = f" ({len(all_features) / total_hint * 100:.0f}%)" if total_hint else ""
        print(
            f"        page {page:>3}: offset={offset:>7}  "
            f"got {len(features):>5} features  "
            f"cumulative {len(all_features):>7,}{total_str}{pct_str}",
            flush=True,
        )

        # Termination
        if not features:
            break
        if not exceeded and len(features) < page_size:
            break

        offset += len(features)
        page += 1

    return all_features


# -----------------------------------------------------------------------------
# Saving and inspecting
# -----------------------------------------------------------------------------

def save_feature_collection(features: list[dict[str, Any]], out_path: Path) -> None:
    """Write a list of GeoJSON features as a single FeatureCollection."""
    collection = {
        "type": "FeatureCollection",
        "features": features,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(collection, f)


def inspect_layer(path: Path, description: str) -> None:
    """Print schema summary for a downloaded GeoJSON."""
    print("\n" + "=" * 72)
    print(f"SCHEMA: {description}")
    print(f"File:   {path}")
    print("=" * 72)

    try:
        gdf = gpd.read_file(path)
    except Exception as e:
        print(f"  ERROR reading file: {e}")
        return

    if len(gdf) == 0:
        print("  WARNING: zero rows in file")
        return

    geom_types = gdf.geometry.type.value_counts().to_dict()
    n_valid = int(gdf.geometry.notna().sum())
    n_empty = len(gdf) - n_valid

    print(f"\n  Rows:              {len(gdf):,}")
    print(f"  Rows w/ geometry:  {n_valid:,}")
    if n_empty > 0:
        print(f"  Rows w/ NULL geom: {n_empty:,}  <-- WARNING")
    print(f"  CRS:               {gdf.crs}")
    print(f"  Geometry type(s):  {geom_types}")

    if n_valid > 0:
        bbox = gdf.geometry.dropna().total_bounds
        print(
            f"  Bounds (lon/lat):  [{bbox[0]:.4f}, {bbox[1]:.4f}, "
            f"{bbox[2]:.4f}, {bbox[3]:.4f}]"
        )

    print(f"\n  Columns ({len(gdf.columns)}):")
    print(f"    {'name':<32} {'dtype':<14} {'unique':<10} sample")
    print(f"    {'-'*32} {'-'*14} {'-'*10} {'-'*30}")
    for col in gdf.columns:
        dtype = str(gdf[col].dtype)
        n_unique = gdf[col].nunique(dropna=True)
        if col == "geometry":
            sample = "<geom>"
        else:
            non_null = gdf[col].dropna()
            sample = repr(non_null.iloc[0]) if len(non_null) > 0 else "<all null>"
            if len(sample) > 30:
                sample = sample[:27] + "..."
        print(f"    {col:<32} {dtype:<14} {n_unique:<10} {sample}")

    # Breakdown of the depth-equivalent column
    if "Flooding_Category" in gdf.columns:
        print("\n  Flooding_Category breakdown:")
        vc = gdf["Flooding_Category"].value_counts(dropna=False).sort_index()
        for val, count in vc.items():
            print(f"    category {val!r:<5} -> {count:,} polygons")


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------

def download_layer(layer_key: str, config: dict[str, str]) -> Optional[Path]:
    print(f"\n[FETCH] {config['description']}")
    print(f"        Service: {config['service_url']}")

    out_path = OUT_DIR / config["filename"]
    if out_path.exists() and out_path.stat().st_size > MIN_VALID_SIZE_BYTES:
        size_mb = out_path.stat().st_size / (1024 * 1024)
        print(f"        [SKIP] Already present ({size_mb:.1f} MB)")
        return out_path

    try:
        info = get_layer_info(config["service_url"])
    except Exception as e:
        print(f"        ERROR fetching layer metadata: {e}")
        return None

    layer_name = info.get("name", "<unknown>")
    max_rec = int(info.get("maxRecordCount", 2000))
    print(f"        Layer name:      {layer_name}")
    print(f"        maxRecordCount:  {max_rec:,}")

    total = get_feature_count(config["service_url"])
    if total is not None:
        print(f"        Total features:  {total:,}")

    t0 = time.time()
    try:
        features = fetch_all_features(
            config["service_url"],
            page_size=max_rec,
            total_hint=total,
        )
    except Exception as e:
        print(f"        ERROR during pagination: {e}")
        return None
    elapsed = time.time() - t0

    if not features:
        print("        No features returned")
        return None

    save_feature_collection(features, out_path)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(
        f"        Saved {len(features):,} features to {out_path} "
        f"({size_mb:.1f} MB, {elapsed:.0f}s)"
    )
    return out_path


def main() -> int:
    print("=" * 72)
    print("NYC DEP Stormwater Flood Map Downloader (ArcGIS FeatureServer)")
    print("=" * 72)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUT_DIR.absolute()}")

    # Clean up any corrupt prior downloads (< 100 KB)
    for existing in OUT_DIR.glob("*.geojson"):
        if existing.stat().st_size < MIN_VALID_SIZE_BYTES:
            print(f"[CLEAN] Removing corrupt prior download: {existing.name}")
            existing.unlink()

    paths: dict[str, Path] = {}
    for layer_key, config in LAYERS.items():
        path = download_layer(layer_key, config)
        if path:
            paths[layer_key] = path

    # Schema inspection
    print("\n\n" + "=" * 72)
    print("SCHEMA SUMMARIES")
    print("=" * 72)

    for layer_key, path in paths.items():
        description = LAYERS[layer_key]["description"]
        try:
            inspect_layer(path, description)
        except Exception as e:
            print(f"\n[ERROR inspecting {description}]: {e}")

    # Final status
    print("\n" + "=" * 72)
    failed = set(LAYERS.keys()) - set(paths.keys())
    if failed:
        print(f"FAILED LAYERS: {sorted(failed)}")
        return 1

    print(f"OK: {len(paths)} / {len(LAYERS)} layers downloaded")
    return 0


if __name__ == "__main__":
    sys.exit(main())