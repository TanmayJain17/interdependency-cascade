"""
OSM Overpass API helper.

    query(ql_template, bbox)
        Run an Overpass QL query and return a point GeoDataFrame in
        EPSG:4326. The ql_template must contain one literal "{bbox}"
        placeholder, which is replaced with the bbox in Overpass order
        ("min_lat,min_lon,max_lat,max_lon"). Ways and relations are
        reduced to their `out center` point.

Falls back across multiple Overpass endpoints (overpass-api.de,
kumi.systems, openstreetmap.ru) on failure, with exponential backoff
on transport errors and 15 s sleep on 429 Too Many Requests.

Lifted from the multi-endpoint pattern in
src/data_acquisition/download_fuel.py.
"""
from __future__ import annotations

import logging
import time
from typing import Iterable, Optional

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Point
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

log = logging.getLogger(__name__)

DEFAULT_ENDPOINTS: tuple[str, ...] = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
)
DEFAULT_USER_AGENT = "NYU-CERA-flood-cascade-research/1.0 (tj2587@nyu.edu)"
DEFAULT_TIMEOUT = 180


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=5, min=5, max=60),
    retry=retry_if_exception_type((requests.RequestException, ValueError)),
    reraise=True,
)
def _post(endpoint: str, query_text: str, timeout: int, headers: dict) -> list[dict]:
    resp = requests.post(
        endpoint,
        data={"data": query_text},
        headers=headers,
        timeout=timeout,
    )
    if resp.status_code == 429:
        log.warning("Overpass 429 at %s — sleeping 15s before retry", endpoint)
        time.sleep(15)
        raise requests.RequestException("Overpass 429 Too Many Requests")
    resp.raise_for_status()
    return resp.json().get("elements", []) or []


def query(
    ql_template: str,
    bbox: tuple[float, float, float, float],
    *,
    endpoints: Iterable[str] = DEFAULT_ENDPOINTS,
    timeout: int = DEFAULT_TIMEOUT,
) -> gpd.GeoDataFrame:
    """Run an Overpass QL query and parse nodes/ways into a Point GeoDataFrame.

    ql_template: Overpass QL string containing one literal "{bbox}" placeholder.
    bbox: (min_lon, min_lat, max_lon, max_lat) in EPSG:4326.

    Returns: GeoDataFrame with columns
        osm_id, osm_type, lat, lon, <flattened tag columns>
    in EPSG:4326. Empty GeoDataFrame if no elements found.
    """
    bbox_str = f"{bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]}"
    query_text = ql_template.format(bbox=bbox_str)
    headers = {"User-Agent": DEFAULT_USER_AGENT, "Accept": "application/json"}

    last_err: Optional[Exception] = None
    for endpoint in endpoints:
        log.info("Overpass POST → %s", endpoint)
        try:
            elements = _post(endpoint, query_text, timeout, headers)
        except Exception as e:
            last_err = e
            log.warning("Overpass %s failed: %s", endpoint, e)
            continue

        rows: list[dict] = []
        for el in elements:
            lat = el.get("lat") or (el.get("center") or {}).get("lat")
            lon = el.get("lon") or (el.get("center") or {}).get("lon")
            if lat is None or lon is None:
                continue
            tags = el.get("tags", {}) or {}
            row: dict = {
                "osm_id": el.get("id"),
                "osm_type": el.get("type"),
                "lat": float(lat),
                "lon": float(lon),
            }
            row.update(tags)
            rows.append(row)

        if not rows:
            log.info("Overpass: 0 features at %s", endpoint)
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

        df = pd.DataFrame(rows)
        gdf = gpd.GeoDataFrame(
            df,
            geometry=[Point(r.lon, r.lat) for r in df.itertuples()],
            crs="EPSG:4326",
        )
        log.info("Overpass: %d feature(s) from %s", len(gdf), endpoint)
        return gdf

    raise RuntimeError(f"All Overpass endpoints failed; last error: {last_err}")
