"""
C1 — MBTA subway stops via the GTFS static feed.

Downloads the MBTA GTFS zip from https://cdn.mbta.com/MBTA_GTFS.zip,
extracts stops.txt, keeps rows with location_type in {0, 1}
(stops + stations), clips to the Boston bbox, and writes a GeoJSON
preserving every column from the original stops.txt.

Output files:
    data/boston/raw/subway/mbta_gtfs.zip   (raw GTFS archive)
    data/boston/raw/subway/stops.geojson   (filtered, all columns preserved)
"""
from __future__ import annotations

import logging
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Point

from src.cities import boston

log = logging.getLogger(__name__)

LAYER = "subway"
SOURCE_URL = "https://cdn.mbta.com/MBTA_GTFS.zip"
OUT_DIR = boston.RAW_DIR / "subway"
ZIP_PATH = OUT_DIR / "mbta_gtfs.zip"
GEOJSON_PATH = OUT_DIR / "stops.geojson"

KEEP_LOCATION_TYPES = {0, 1}
# GTFS route_type / vehicle_type: 0=tram, 1=subway, 2=rail, 3=bus, 4=ferry.
# Drop bus stops at ingest time; keep heavy rail, light rail, commuter rail,
# ferry, and parent stations (whose vehicle_type is null). Bus stops are
# functionally different infrastructure (flexible routing, no tunnels) and
# would otherwise dominate the file with ~3700 rows of noise.
EXCLUDE_VEHICLE_TYPES = {3}
USER_AGENT = "NYU-CERA-flood-cascade-research/1.0 (tj2587@nyu.edu)"
DOWNLOAD_TIMEOUT = 180


def _download_zip(force: bool) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if ZIP_PATH.exists() and not force:
        size_mb = ZIP_PATH.stat().st_size / (1024 * 1024)
        log.info("MBTA GTFS zip already present (%.1f MB) — skipping download", size_mb)
        return ZIP_PATH

    log.info("Downloading MBTA GTFS from %s", SOURCE_URL)
    tmp = ZIP_PATH.with_suffix(".zip.tmp")
    headers = {"User-Agent": USER_AGENT}
    with requests.get(SOURCE_URL, headers=headers, stream=True, timeout=DOWNLOAD_TIMEOUT) as resp:
        resp.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(chunk_size=64 * 1024):
                if chunk:
                    f.write(chunk)
    tmp.replace(ZIP_PATH)
    size_mb = ZIP_PATH.stat().st_size / (1024 * 1024)
    log.info("MBTA GTFS zip saved (%.1f MB) → %s", size_mb, ZIP_PATH)
    return ZIP_PATH


def _read_stops(zip_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open("stops.txt") as f:
            return pd.read_csv(f, dtype=str)


def _bbox_mask(df: pd.DataFrame, bbox: tuple[float, float, float, float]) -> pd.Series:
    min_lon, min_lat, max_lon, max_lat = bbox
    lat = df["stop_lat"].astype(float)
    lon = df["stop_lon"].astype(float)
    return (lon >= min_lon) & (lon <= max_lon) & (lat >= min_lat) & (lat <= max_lat)


def download(force: bool = False) -> list[dict]:
    """Run the C1 download. Returns manifest entries (path/source/record_count)."""
    zip_path = _download_zip(force)

    stops = _read_stops(zip_path)
    log.info(
        "GTFS stops.txt: %d row(s), %d column(s): %s",
        len(stops), len(stops.columns), list(stops.columns),
    )

    # GTFS spec: missing location_type defaults to 0 (stop)
    if "location_type" not in stops.columns:
        stops["location_type"] = "0"
    loc_int = stops["location_type"].fillna("0").replace("", "0").astype(int)
    loc_keep = loc_int.isin(KEEP_LOCATION_TYPES)
    bbox_keep = _bbox_mask(stops, boston.BBOX)

    # vehicle_type is set on platform/stop rows (location_type=0). Parent
    # stations (location_type=1) have it null; we keep them unconditionally.
    if "vehicle_type" in stops.columns:
        vt_str = stops["vehicle_type"].fillna("").astype(str)
        is_bus = vt_str.isin({str(v) for v in EXCLUDE_VEHICLE_TYPES})
        vt_keep = ~is_bus
    else:
        vt_keep = pd.Series(True, index=stops.index)

    keep = loc_keep & bbox_keep & vt_keep
    filtered = stops[keep].copy()
    log.info(
        "Filter pass: %d rows (location_type=%d, bbox=%d, non-bus=%d)",
        len(filtered), int(loc_keep.sum()), int(bbox_keep.sum()), int(vt_keep.sum()),
    )

    if len(filtered) == 0:
        log.warning("Zero MBTA stops survived filtering — check bbox %s", boston.BBOX)

    filtered["stop_lat"] = filtered["stop_lat"].astype(float)
    filtered["stop_lon"] = filtered["stop_lon"].astype(float)
    gdf = gpd.GeoDataFrame(
        filtered,
        geometry=[Point(lon, lat) for lon, lat in zip(filtered["stop_lon"], filtered["stop_lat"])],
        crs="EPSG:4326",
    )

    if GEOJSON_PATH.exists():
        GEOJSON_PATH.unlink()
    gdf.to_file(GEOJSON_PATH, driver="GeoJSON")
    log.info("Saved %d stop(s) → %s", len(gdf), GEOJSON_PATH)

    # Quick breakdown of what landed in the file
    if "location_type" in gdf.columns:
        log.info("Output by location_type: %s", gdf["location_type"].value_counts().to_dict())

    return [
        {
            "layer": LAYER,
            "path": str(ZIP_PATH),
            "source_url": SOURCE_URL,
            "record_count": None,
        },
        {
            "layer": LAYER,
            "path": str(GEOJSON_PATH),
            "source_url": SOURCE_URL,
            "record_count": len(gdf),
        },
    ]


if __name__ == "__main__":
    from src.cities.boston_ingest._logging import configure as configure_logging

    configure_logging()
    entries = download()
    for e in entries:
        rc = e["record_count"]
        rc_str = f"({rc} records)" if rc is not None else "(archive)"
        print(f"  {e['layer']:8s} {e['path']}  {rc_str}")
