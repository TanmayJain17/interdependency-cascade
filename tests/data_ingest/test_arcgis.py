"""
Minimal smoke test for src.data_ingest.arcgis.query_layer.

Hits the public Rutgers HIFLD substations FeatureServer with a tiny bbox
(downtown Boston, ~2 km x 1.5 km) and verifies that a GeoDataFrame in
EPSG:4326 comes back.

Exits 0 on success, 1 on type/shape failure, 77 on network unavailable
(matches the autotools convention so CI can decide whether to ignore).

Run:
    python -m tests.data_ingest.test_arcgis
"""
from __future__ import annotations

import logging
import sys

import geopandas as gpd
import requests

from src.data_ingest.arcgis import query_layer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

RUTGERS_HIFLD = (
    "https://oceandata.rad.rutgers.edu/arcgis/rest/services/"
    "RenewableEnergy/HIFLD_Electric_SubstationsTransmissionLines/MapServer"
)
# Tiny Boston bbox — substations layer 0 should return a handful (or zero).
TINY_BBOX = (-71.10, 42.34, -71.04, 42.38)


def main() -> int:
    try:
        gdf = query_layer(RUTGERS_HIFLD, layer_id=0, bbox=TINY_BBOX)
    except (requests.ConnectionError, requests.Timeout) as e:
        print(f"SKIP: network unavailable ({e})")
        return 77

    assert isinstance(gdf, gpd.GeoDataFrame), f"expected GeoDataFrame, got {type(gdf)}"
    assert gdf.crs is not None, "result has no CRS"
    assert gdf.crs.to_epsg() == 4326, f"expected EPSG:4326, got {gdf.crs}"
    print(f"OK: arcgis.query_layer returned {len(gdf)} feature(s) in EPSG:4326")
    return 0


if __name__ == "__main__":
    sys.exit(main())
