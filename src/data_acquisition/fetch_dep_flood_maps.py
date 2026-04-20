#!/usr/bin/env python3
"""
fetch_dep_flood_maps.py

Download the three NYC DEP Stormwater Flood Map scenarios from the public
ArcGIS Online FeatureServer endpoints. Saves each as a GeoJSON in
data/raw/flood/.

The NYC Open Data Socrata mirrors of these layers have been restricted to
agency-only access. We instead query the original ArcGIS FeatureServers
(hosted by Esri's NYC demo org), where the data is still public.

Attribute schema (all three layers):
    OBJECTID            internal row ID
    Flooding_Category   coded integer:
        1 = Nuisance Flooding    (ponding 4 in to 1 ft)
        2 = Deep and Contiguous  (ponding >= 1 ft)
        3 = Future High Tides    (tidal inundation; 2050 and 2080 scenarios only)
    Shape__Area         auto-computed polygon area
    Shape__Length       auto-computed polygon perimeter
    geometry            one huge MultiPolygon per category (unioned citywide)

IMPORTANT: an earlier version used geometryPrecision=5 (1 m rounding) to reduce
payload size on the large Extreme-2080 layer. That rounding corrupted the
geometries (ring self-intersections plus entire sub-polygons collapsed below
3 vertices). This version requests FULL coordinate precision -- responses are
larger (up to ~200 MB per feature for Extreme 2080) but geometrically correct.
We also fetch one feature at a time by OBJECTID (not paginated bulk), so no
single HTTP response is bigger than one Flooding_Category polygon.

Run from project root (~/Desktop/RA/):
    python3 src/data_acquisition/fetch_dep_flood_maps.py
"""

from __future__ import annotations

import os
# Must be set BEFORE geopandas import for GDAL to honor it on read.
os.environ.setdefault("OGR_GEOJSON_MAX_OBJ_SIZE", "0")

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

BASE = "https://services.arcgis.com/P3ePLMYs2RVChkJx/arcgis/rest/services"

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

# Full precision -- do NOT send geometryPrecision. Responses are larger but
# geometrically valid.
REQUEST_TIMEOUT = 900   # 15 min per request
MAX_RETRIES = 4         # per feature
MIN_VALID_SIZE_BYTES = 100_000


# -----------------------------------------------------------------------------
# ArcGIS REST helpers
# -----------------------------------------------------------------------------

def get_layer_info(service_url: str) -> dict[str, Any]:
    resp = requests.get(
        f"{service_url}/0",
        params={"f": "json"},
        headers=HEADERS,
        timeout=60,
    )
    resp.raise_for_status()
    info = resp.json()
    if "error" in info:
        raise RuntimeError(f"ArcGIS error: {info['error']}")
    return info


def get_object_ids(service_url: str) -> list[int]:
    resp = requests.get(
        f"{service_url}/0/query",
        params={"where": "1=1", "returnIdsOnly": "true", "f": "json"},
        headers=HEADERS,
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    return list(data.get("objectIds", []) or [])


def fetch_one_feature(service_url: str, oid: int) -> dict[str, Any]:
    """
    Fetch one feature by OBJECTID at full precision. Retries on transport errors.
    Returns the single GeoJSON Feature dict.
    """
    query_url = f"{service_url}/0/query"
    params = {
        "where": f"OBJECTID={oid}",
        "outFields": "*",
        "outSR": "4326",
        "f": "geojson",
        # NOTE: deliberately NOT sending geometryPrecision
    }

    last_err: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        t0 = time.time()
        try:
            resp = requests.get(
                query_url,
                params=params,
                headers=HEADERS,
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()

            size_mb = len(resp.content) / (1024 * 1024)
            elapsed = time.time() - t0
            data = resp.json()

            if isinstance(data, dict) and data.get("error"):
                raise RuntimeError(f"ArcGIS error: {data['error']}")

            features = data.get("features", [])
            if not features:
                raise RuntimeError(f"Empty features array for OBJECTID={oid}")

            print(
                f"          attempt {attempt}: OK "
                f"({size_mb:.1f} MB received in {elapsed:.0f}s)"
            )
            return features[0]

        except (requests.RequestException, RuntimeError, ValueError) as e:
            last_err = e
            elapsed = time.time() - t0
            print(
                f"          attempt {attempt}: FAIL after {elapsed:.0f}s "
                f"({type(e).__name__}: {str(e)[:140]})"
            )
            if attempt < MAX_RETRIES:
                backoff = 5 * attempt
                print(f"          retrying in {backoff}s...")
                time.sleep(backoff)

    raise RuntimeError(f"All {MAX_RETRIES} attempts failed for OBJECTID={oid}: {last_err}")


def fetch_all_features(service_url: str) -> list[dict[str, Any]]:
    """Fetch every feature in layer 0, one OBJECTID at a time."""
    object_ids = get_object_ids(service_url)
    print(f"        Fetching {len(object_ids)} features by OBJECTID: {object_ids}")

    features: list[dict[str, Any]] = []
    for i, oid in enumerate(object_ids, start=1):
        print(f"        [{i}/{len(object_ids)}] OBJECTID={oid}")
        feat = fetch_one_feature(service_url, oid)
        features.append(feat)
    return features


# -----------------------------------------------------------------------------
# Save + inspect
# -----------------------------------------------------------------------------

def save_feature_collection(features: list[dict[str, Any]], out_path: Path) -> None:
    collection = {"type": "FeatureCollection", "features": features}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(collection, f)


def inspect_layer(path: Path, description: str) -> None:
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
    n_valid = int(gdf.geometry.is_valid.sum())
    n_invalid = len(gdf) - n_valid

    print(f"\n  Rows:              {len(gdf):,}")
    print(f"  Valid geometries:  {n_valid}/{len(gdf)}")
    if n_invalid > 0:
        print(f"  INVALID:           {n_invalid}  <-- will need make_valid() in overlay")
    print(f"  CRS:               {gdf.crs}")
    print(f"  Geometry type(s):  {geom_types}")

    bbox = gdf.total_bounds
    print(
        f"  Bounds (lon/lat):  [{bbox[0]:.4f}, {bbox[1]:.4f}, "
        f"{bbox[2]:.4f}, {bbox[3]:.4f}]"
    )

    if "Flooding_Category" in gdf.columns:
        print("\n  Flooding_Category breakdown:")
        vc = gdf["Flooding_Category"].value_counts(dropna=False).sort_index()
        for val, count in vc.items():
            print(f"    category {val!r:<5} -> {count:,} row(s)")


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
    print(f"        Layer name: {layer_name}")
    print(f"        Full geometry precision (no rounding)")

    t0 = time.time()
    try:
        features = fetch_all_features(config["service_url"])
    except Exception as e:
        print(f"        ERROR: {e}")
        return None
    elapsed = time.time() - t0

    if not features:
        print("        No features returned")
        return None

    save_feature_collection(features, out_path)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(
        f"        Saved {len(features):,} features to {out_path} "
        f"({size_mb:.1f} MB, {elapsed:.0f}s total)"
    )
    return out_path


def main() -> int:
    print("=" * 72)
    print("NYC DEP Stormwater Flood Map Downloader (FeatureServer, full precision)")
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

    print("\n\n" + "=" * 72)
    print("SCHEMA SUMMARIES")
    print("=" * 72)

    for layer_key, path in paths.items():
        description = LAYERS[layer_key]["description"]
        try:
            inspect_layer(path, description)
        except Exception as e:
            print(f"\n[ERROR inspecting {description}]: {e}")

    print("\n" + "=" * 72)
    failed = set(LAYERS.keys()) - set(paths.keys())
    if failed:
        print(f"FAILED LAYERS: {sorted(failed)}")
        return 1

    print(f"OK: {len(paths)} / {len(LAYERS)} layers downloaded")
    return 0


if __name__ == "__main__":
    sys.exit(main())