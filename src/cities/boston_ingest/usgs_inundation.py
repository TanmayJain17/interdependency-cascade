"""
USGS published flood-inundation polygons for the January 2018 nor'easter (Boston).

This is the Phase-3 node-level "real wet zone" — the authoritative ground truth for
WHERE Boston actually flooded on January 4, 2018. It is the published companion to
SIR 2021-5109 (Lombard et al., 2021): USGS interpolated the surveyed high-water-mark
water-surface elevations onto a DEM and mapped the inundated extent. Using it spares
us from interpolating the 35 sparse HWM points ourselves.

IMPORTANT framing (carried into the writeup): this polygon is USGS's DEM interpolation
of the SAME high-water marks ingested in usgs_hwm.py, so "HWMs fall inside the polygon"
is near-tautological. The genuinely independent hazard anchors are the NOAA tide gauge
still-water peak and the documented Aquarium station closure. See
data/boston/raw/usgs_hwm/_PUBLISHED_INUNDATION_MAP.md.

Two products are pulled from ScienceBase item 60100f99d34e162231fecfa8:
    jan2018     New England January 2018 High-Water Mark Coastal Inundation Map
                -> the real wet zone for the event (REQUIRED)
    stillwater_100yr  New England USGS 100-year Stillwater Coastal Inundation Map
                -> secondary 1%-AEP comparator (the Jan 2018 event is rated 1-2% AEP)

Each is a polygon shapefile (native ESRI:102039 Albers, NAD83) carrying a WSE_FT
water-surface-elevation attribute; there is no depth raster, so node classification is
wet/dry by polygon containment (depth-RMSE is intentionally not attempted — the model
assigns a flat 0.30 m to every flooded node).

Outputs (data/boston/validation/):
    boston_2018_usgs_inundation.gpkg         clipped Jan-2018 wet zone (EPSG:4326, WSE_FT)
    boston_100yr_stillwater_inundation.gpkg  clipped 100-yr comparator (if it clips cleanly)

The raw 35 MB zips + full-New-England extracted shapefiles land under
data/boston/raw/usgs_hwm/inundation/ (gitignored; regeneratable from this module).
"""
from __future__ import annotations

import logging
import zipfile

import geopandas as gpd
import requests
from shapely.geometry import box

from src.cities import boston

log = logging.getLogger(__name__)

LAYER = "usgs_inundation"
RAW_DIR = boston.RAW_DIR / "usgs_hwm" / "inundation"
OUT_DIR = boston.DATA_DIR / "validation"

SB_FILE = "https://www.sciencebase.gov/catalog/file/get/60100f99d34e162231fecfa8"

PRODUCTS = {
    "jan2018": {
        "url": f"{SB_FILE}?f=__disk__b2%2F36%2F5d%2Fb2365db80be7c4d71966db856ca670316d0bee74",
        "zip_name": "jan2018_inundation.zip",
        "shp_stem": "New_England_January_2018_High_Water_Mark_Coastal_Inundation_Map",
        "out_name": "boston_2018_usgs_inundation.gpkg",
        "required": True,
    },
    "stillwater_100yr": {
        "url": f"{SB_FILE}?f=__disk__f0%2F13%2F4b%2Ff0134b08e030a517d0dd83ce6d623944874d0024",
        "zip_name": "stillwater_100yr_inundation.zip",
        "shp_stem": "New_England_USGS_100_year_Stillwater_Coastal_Inundation_Map",
        "out_name": "boston_100yr_stillwater_inundation.gpkg",
        "required": False,
    },
}

TIMEOUT = 600


def _download(url: str, dest, force: bool) -> None:
    if dest.exists() and not force:
        log.info("%s present (%.1f MB) — skipping download", dest.name, dest.stat().st_size / 1e6)
        return
    log.info("Downloading %s ...", dest.name)
    with requests.get(url, stream=True, timeout=TIMEOUT) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 16):
                f.write(chunk)
    log.info("Downloaded %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)


def _extract_shp(zip_path, shp_stem: str):
    out = RAW_DIR / zip_path.stem
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out)
    shp = out / f"{shp_stem}.shp"
    if not shp.exists():  # fall back to any .shp in the archive
        cands = list(out.glob("*.shp"))
        if not cands:
            raise FileNotFoundError(f"No .shp found in {zip_path.name}")
        shp = cands[0]
    return shp


def _clip_to_bbox(shp) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(shp)
    log.info("%s: native CRS=%s, %d features, geom=%s, cols=%s",
             shp.name, gdf.crs.to_string() if gdf.crs else "None", len(gdf),
             gdf.geom_type.value_counts().to_dict(), list(gdf.columns))
    gdf = gdf.to_crs("EPSG:4326")
    clip = gpd.clip(gdf, box(*boston.BBOX))
    clip = clip[~clip.geometry.is_empty & clip.geometry.notna()].reset_index(drop=True)
    return clip


def _process(key: str, spec: dict, force: bool) -> dict | None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / spec["zip_name"]
    out_path = OUT_DIR / spec["out_name"]

    try:
        _download(spec["url"], zip_path, force)
        shp = _extract_shp(zip_path, spec["shp_stem"])
        clip = _clip_to_bbox(shp)
    except Exception as exc:  # noqa: BLE001
        if spec["required"]:
            raise
        log.warning("Secondary product %s failed (%s) — skipping", key, exc)
        return None

    if len(clip) == 0:
        msg = f"{key}: inundation polygon does not intersect the Boston bbox."
        if spec["required"]:
            raise RuntimeError(msg)
        log.warning("%s — skipping secondary comparator", msg)
        return None

    # Area of the wet zone within the bbox (UTM 19N for metric area).
    area_km2 = float(clip.to_crs("EPSG:26919").geometry.area.sum() / 1e6)
    wse = clip["WSE_FT"] if "WSE_FT" in clip.columns else None
    wse_range = (float(wse.min()), float(wse.max())) if wse is not None else None

    if out_path.exists():
        out_path.unlink()
    clip.to_file(out_path, driver="GPKG")
    log.info("%s: %d clipped features, %.2f km^2 in bbox, WSE_FT range %s -> %s",
             key, len(clip), area_km2, wse_range, out_path.name)

    return {
        "layer": LAYER,
        "sub_layer": key,
        "path": str(out_path),
        "source_url": spec["url"],
        "record_count": int(len(clip)),
        "wet_area_km2": round(area_km2, 2),
        "wse_ft_range": wse_range,
    }


def download(force: bool = False) -> list[dict]:
    entries = []
    for key, spec in PRODUCTS.items():
        e = _process(key, spec, force)
        if e is not None:
            entries.append(e)
    return entries


if __name__ == "__main__":
    from src.cities.boston_ingest._logging import configure as configure_logging

    configure_logging()
    for e in download():
        print(f"  {e['sub_layer']:18s} {e['record_count']:3d} feats  "
              f"{e['wet_area_km2']:6.2f} km^2  WSE_FT {e['wse_ft_range']}  -> {e['path']}")
