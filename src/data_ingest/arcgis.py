"""
ArcGIS FeatureServer / MapServer query helpers.

Two entry points:

    query_layer(service_url, layer_id, bbox=...)
        Paginated bulk fetch (resultOffset / resultRecordCount). Use for
        most point or small-polygon layers (substations, terminals, towers,
        BWSC outfalls, MassGIS NFHL, etc.).

    query_by_object_id(service_url, layer_id, object_ids=...)
        One-feature-at-a-time fetch. Use for huge polygon services where
        a single feature can be tens of megabytes (e.g. Climate Ready
        Boston SLR/AEP layers, NYC DEP flood layers).

Both return a GeoDataFrame in EPSG:4326. Reprojection happens on the
server when possible (outSR=4326); a client-side fallback re-projects if
the server ignores the request.

Lifted from the patterns in src/data_acquisition/download_power.py
(paginated MapServer) and src/data_acquisition/fetch_dep_flood_maps.py
(OBJECTID-by-OBJECTID for huge polygons).
"""
from __future__ import annotations

import io
import json
import logging
import time
from typing import Optional

import geopandas as gpd
import pandas as pd
import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

log = logging.getLogger(__name__)

DEFAULT_USER_AGENT = "NYU-CERA-flood-cascade-research/1.0 (tj2587@nyu.edu)"
DEFAULT_HEADERS = {
    "User-Agent": DEFAULT_USER_AGENT,
    "Accept": "application/json, application/geo+json, */*",
}
DEFAULT_TIMEOUT = 60
PAGE_SIZE = 1000
PAGE_PAUSE_SEC = 0.5


# -----------------------------------------------------------------------------
# Internals
# -----------------------------------------------------------------------------

@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    retry=retry_if_exception_type((requests.RequestException, RuntimeError)),
    reraise=True,
)
def _get(url: str, params: dict, timeout: int) -> str:
    resp = requests.get(url, params=params, headers=DEFAULT_HEADERS, timeout=timeout)
    resp.raise_for_status()
    text = resp.text
    # ArcGIS sometimes returns HTTP 200 with a JSON error body.
    stripped = text.lstrip()
    if stripped.startswith('{"error"') or '"error":{' in stripped[:200]:
        raise RuntimeError(f"ArcGIS error response: {text[:300]}")
    return text


def _to_4326(gdf: gpd.GeoDataFrame, expected_epsg: int = 4326) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=expected_epsg)
    if gdf.crs.to_epsg() != 4326:
        log.info("Reprojecting from %s to EPSG:4326", gdf.crs)
        gdf = gdf.to_crs(epsg=4326)
    return gdf


def _bbox_params(bbox: tuple[float, float, float, float]) -> dict:
    return {
        "geometry": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
    }


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def query_layer(
    service_url: str,
    layer_id: int = 0,
    *,
    bbox: Optional[tuple[float, float, float, float]] = None,
    where: str = "1=1",
    out_fields: str = "*",
    page_size: int = PAGE_SIZE,
    timeout: int = DEFAULT_TIMEOUT,
) -> gpd.GeoDataFrame:
    """Query a FeatureServer/MapServer layer with pagination.

    bbox: (min_lon, min_lat, max_lon, max_lat) in EPSG:4326. None = fetch all.
    Returns: GeoDataFrame in EPSG:4326 (empty GDF if no features).
    """
    query_url = f"{service_url.rstrip('/')}/{layer_id}/query"

    base_params: dict[str, str] = {
        "where": where,
        "outFields": out_fields,
        "outSR": "4326",
        "f": "geojson",
        "resultRecordCount": str(page_size),
    }
    if bbox is not None:
        base_params.update(_bbox_params(bbox))

    all_gdfs: list[gpd.GeoDataFrame] = []
    offset = 0
    while True:
        params = dict(base_params, resultOffset=str(offset))
        log.info(
            "ArcGIS GET %s layer=%d offset=%d page_size=%d",
            service_url, layer_id, offset, page_size,
        )
        text = _get(query_url, params, timeout)
        gdf = gpd.read_file(io.StringIO(text))
        n = len(gdf)
        log.info("  page returned %d feature(s)", n)
        if n == 0:
            break
        all_gdfs.append(gdf)
        # Advance by the actual number of features the server returned, not
        # the page_size we requested. Some services cap the per-request
        # count below page_size (e.g. by transfer-size budget) — if we
        # advanced by page_size in that case we would skip records.
        offset += n
        time.sleep(PAGE_PAUSE_SEC)

    if not all_gdfs:
        log.warning("ArcGIS query returned zero features (%s layer=%d)", service_url, layer_id)
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    combined = pd.concat(all_gdfs, ignore_index=True)
    out = gpd.GeoDataFrame(combined, geometry="geometry", crs=all_gdfs[0].crs)
    out = _to_4326(out)
    log.info("ArcGIS total: %d feature(s) from %s layer=%d", len(out), service_url, layer_id)
    return out


def list_object_ids(
    service_url: str,
    layer_id: int = 0,
    *,
    where: str = "1=1",
    timeout: int = DEFAULT_TIMEOUT,
) -> tuple[str, list[int]]:
    """Return (oid_field_name, object_ids) for a layer.

    The OID field is usually "OBJECTID" but services backed by shapefile
    sources commonly use "FID" instead. We surface the actual field name
    so callers can build correct WHERE clauses.
    """
    query_url = f"{service_url.rstrip('/')}/{layer_id}/query"
    params = {"where": where, "returnIdsOnly": "true", "f": "json"}
    text = _get(query_url, params, timeout)
    data = json.loads(text)
    oid_field = data.get("objectIdFieldName") or "OBJECTID"
    ids = list(data.get("objectIds", []) or [])
    return oid_field, ids


def query_by_object_id(
    service_url: str,
    layer_id: int = 0,
    *,
    object_ids: Optional[list[int]] = None,
    oid_field: Optional[str] = None,
    where: str = "1=1",
    out_fields: str = "*",
    timeout: int = 900,
) -> gpd.GeoDataFrame:
    """Fetch one OID per request. Use for huge polygon services.

    If object_ids is None, fetches the full OID list first and discovers
    the OID field name from the service. If you pass object_ids
    explicitly, also pass oid_field if the layer uses something other
    than "OBJECTID".
    """
    query_url = f"{service_url.rstrip('/')}/{layer_id}/query"
    if object_ids is None:
        discovered_field, object_ids = list_object_ids(service_url, layer_id, where=where)
        if oid_field is None:
            oid_field = discovered_field
    if oid_field is None:
        oid_field = "OBJECTID"
    log.info(
        "ArcGIS OID fetch: %d feature(s) via %s from %s layer=%d",
        len(object_ids), oid_field, service_url, layer_id,
    )

    all_gdfs: list[gpd.GeoDataFrame] = []
    for i, oid in enumerate(object_ids, start=1):
        params = {
            "where": f"{oid_field}={oid}",
            "outFields": out_fields,
            "outSR": "4326",
            "f": "geojson",
        }
        log.info("  [%d/%d] %s=%s", i, len(object_ids), oid_field, oid)
        text = _get(query_url, params, timeout)
        gdf = gpd.read_file(io.StringIO(text))
        if len(gdf) == 0:
            log.warning("  OBJECTID=%s returned 0 features", oid)
            continue
        all_gdfs.append(gdf)
        time.sleep(PAGE_PAUSE_SEC)

    if not all_gdfs:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    combined = pd.concat(all_gdfs, ignore_index=True)
    out = gpd.GeoDataFrame(combined, geometry="geometry", crs=all_gdfs[0].crs)
    return _to_4326(out)
