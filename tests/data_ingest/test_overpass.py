"""
Minimal smoke test for src.data_ingest.overpass.query.

Hits Overpass for a tiny bbox (downtown Boston, ~1 km x 0.5 km) querying
for `amenity=fuel`. Verifies the result is a GeoDataFrame in EPSG:4326
with Point geometries (or empty if the bbox truly has none).

Exits 0 on success, 1 on type/shape failure, 77 on network unavailable.

Run:
    python -m tests.data_ingest.test_overpass
"""
from __future__ import annotations

import logging
import sys

import geopandas as gpd
import requests

from src.data_ingest.overpass import query

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

# Tiny Boston bbox — Back Bay / Fenway area, almost certainly has at least one fuel pump.
TINY_BBOX = (-71.10, 42.34, -71.07, 42.36)

QL = """
[out:json][timeout:60];
(
  node["amenity"="fuel"]({bbox});
  way["amenity"="fuel"]({bbox});
);
out center;
"""


def main() -> int:
    try:
        gdf = query(QL, TINY_BBOX)
    except (requests.ConnectionError, requests.Timeout, RuntimeError) as e:
        print(f"SKIP: Overpass unavailable ({e})")
        return 77

    assert isinstance(gdf, gpd.GeoDataFrame), f"expected GeoDataFrame, got {type(gdf)}"
    if len(gdf) > 0:
        assert gdf.crs is not None and gdf.crs.to_epsg() == 4326, f"expected EPSG:4326, got {gdf.crs}"
        assert (gdf.geometry.type == "Point").all(), "expected all Point geometries"
    print(f"OK: overpass.query returned {len(gdf)} feature(s) in EPSG:4326")
    return 0


if __name__ == "__main__":
    sys.exit(main())
