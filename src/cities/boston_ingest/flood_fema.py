"""
C7 — FEMA National Flood Hazard Layer (NFHL) via MassGIS.

The Massachusetts copy of FEMA's NFHL is the canonical state-level
serving of the federal flood-zone product. Source CRS is EPSG:26986
(NAD83 / Massachusetts Mainland); we reproject to EPSG:4326 on save.

Output:
    data/boston/flood/fema_nfhl.geojson

This is a cross-reference layer relative to CRB. CRB is a forward-looking
SLR + storm scenario product; NFHL is FEMA's current regulatory flood
zones (1% and 0.2% annual-chance hazard areas). The two should partially
overlap along coastal Boston but disagree in inland and projected-SLR
areas.

Schema preserved (all 24 NFHL columns). Notable fields:
    FLD_ZONE       Zone code (A, AE, AO, AH, V, VE, X, AREA NOT INCLUDED, etc.)
    SFHA_TF        Special Flood Hazard Area boolean
    STATIC_BFE     Base Flood Elevation, ft NAVD88 (sentinel -9999 if N/A)
    DEPTH          Sheet/shallow flow depth, ft (sentinel -9999 if N/A;
                   AO and AH zones carry actual values)
    VELOCITY       For V zones, ft/s (sentinel -9999 if N/A)
"""
from __future__ import annotations

import logging
import os

# Lift GDAL GeoJSON size cap in case any NFHL polygon comes back large
os.environ.setdefault("OGR_GEOJSON_MAX_OBJ_SIZE", "0")

import geopandas as gpd  # noqa: E402

from src.cities import boston  # noqa: E402
from src.data_ingest import arcgis  # noqa: E402

log = logging.getLogger(__name__)

LAYER = "flood_fema"
SOURCE_URL = (
    "https://arcgisserver.digital.mass.gov/arcgisserver/rest/services/"
    "FEMA/FEMA_National_Flood_Hazard_Layer/FeatureServer"
)
LAYER_ID = 0
OUT_PATH = boston.FLOOD_DIR / "fema_nfhl.geojson"


def download(force: bool = False) -> list[dict]:
    boston.FLOOD_DIR.mkdir(parents=True, exist_ok=True)
    if OUT_PATH.exists() and not force:
        size_mb = OUT_PATH.stat().st_size / 1024**2
        log.info("fema_nfhl.geojson already present (%.1f MB) — skipping", size_mb)
        n = len(gpd.read_file(OUT_PATH))
        return [{
            "layer": LAYER,
            "path": str(OUT_PATH),
            "source_url": SOURCE_URL,
            "record_count": n,
            "sub_layer": "fema_nfhl",
        }]

    log.info("Fetching FEMA NFHL for Boston bbox %s", boston.BBOX)
    gdf = arcgis.query_layer(
        SOURCE_URL,
        layer_id=LAYER_ID,
        bbox=boston.BBOX,
        page_size=1000,
    )
    n = len(gdf)
    log.info("FEMA NFHL returned: %d polygon(s)", n)
    log.info("Columns (%d): %s", len(gdf.columns), list(gdf.columns))

    if n == 0:
        raise RuntimeError("FEMA NFHL returned 0 polygons — endpoint or bbox likely broken")

    # Quick breakdown
    if "FLD_ZONE" in gdf.columns:
        log.info("FLD_ZONE breakdown: %s", gdf["FLD_ZONE"].value_counts().to_dict())
    if "DEPTH" in gdf.columns:
        real_depth = gdf[gdf["DEPTH"] > 0]
        log.info(
            "Polygons with real DEPTH (>0, not sentinel -9999): %d (zones: %s)",
            len(real_depth),
            real_depth["FLD_ZONE"].value_counts().to_dict() if len(real_depth) else "[]",
        )

    if OUT_PATH.exists():
        OUT_PATH.unlink()
    gdf.to_file(OUT_PATH, driver="GeoJSON")
    log.info("Saved %d NFHL polygon(s) → %s", n, OUT_PATH)

    return [{
        "layer": LAYER,
        "path": str(OUT_PATH),
        "source_url": SOURCE_URL,
        "record_count": n,
        "sub_layer": "fema_nfhl",
    }]


if __name__ == "__main__":
    from src.cities.boston_ingest._logging import configure as configure_logging

    configure_logging()
    entries = download()
    for e in entries:
        rc = e.get("record_count")
        rc_str = f"({rc} records)" if rc is not None else "(notice)"
        print(f"  {e['layer']:10s} [{e.get('sub_layer','')}]  {e.get('path')}  {rc_str}")
