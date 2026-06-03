"""
C5 — Telecom infrastructure for the Boston bbox.

The Week 12 handoff anticipated HIFLD Cellular Towers as the primary
source. HIFLD's surviving public layer
(Federal_User_Community / Cellular_Towers_in_the_United_States) returns
only **3** records for the Boston bbox — an FCC ULS Cellular Service
extract that captures one band class of Verizon licensee records only,
not a comprehensive macro-cell inventory. Pivoting to FCC's broader
Antenna Structure Registration (ASR), which is the federal authoritative
registry for all antenna structures above the FAA notification threshold.

Outputs:

    data/boston/raw/telecom/fcc_asr_towers.geojson
        FCC ASR registered antenna structures in the Boston bbox. Includes
        cellular, broadcast, microwave, dispatch — schema lets downstream
        filter to cellular-only by ENTITY name or LICID join.

    data/boston/raw/telecom/opencellid_towers.geojson
        OpenCelliD bbox extract (per-cell crowdsourced observations).
        Requires OPENCELLID_API_KEY env var. If not set, this layer is
        skipped and an instructions file is written instead.

    data/boston/raw/telecom/opencellid_TODO.txt   (written when key absent)
        Registration instructions for retroactively pulling OpenCelliD.

Tripwire: FCC ASR count must be in [20, 500] (expected 50-250 for greater
Boston). OpenCelliD is ungated — counts vary by crowdsourcing density.
"""
from __future__ import annotations

import io
import logging
import os
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Point

from src.cities import boston
from src.data_ingest import arcgis

log = logging.getLogger(__name__)

LAYER = "telecom"
OUT_DIR = boston.RAW_DIR / "telecom"

# ----- FCC ASR -----
FCC_ASR_URL = (
    "https://services.arcgis.com/B7X7NCOKKXditlwZ/arcgis/rest/services/"
    "FCC_Antenna_Structures/FeatureServer"
)
FCC_ASR_LAYER_ID = 0
FCC_ASR_PATH = OUT_DIR / "fcc_asr_towers.geojson"
FCC_ASR_TRIPWIRE_PATH = OUT_DIR / "fcc_asr_towers.tripwire.geojson"
FCC_ASR_TRIPWIRE_RANGE = (20, 500)

# ----- OpenCelliD -----
OPENCELLID_ENV = "OPENCELLID_API_KEY"
OPENCELLID_PATH = OUT_DIR / "opencellid_towers.geojson"
OPENCELLID_TODO_PATH = OUT_DIR / "opencellid_TODO.txt"
# OpenCelliD bbox API. Lat/lon order: min_lat, min_lon, max_lat, max_lon.
OPENCELLID_URL = "https://opencellid.org/cell/getInArea"
OPENCELLID_USER_AGENT = "NYU-CERA-flood-cascade-research/1.0 (tj2587@nyu.edu)"
OPENCELLID_TIMEOUT = 120
OPENCELLID_TODO_BODY = (
    "OpenCelliD coverage layer is not present.\n"
    "\n"
    "Reason: the OPENCELLID_API_KEY environment variable was not set when\n"
    "the Week 12 telecom ingestion ran. OpenCelliD requires a free API key\n"
    "for bulk and bbox queries.\n"
    "\n"
    "To backfill:\n"
    "  1. Register at https://opencellid.org/register (free account).\n"
    "  2. Generate an API key from your account settings.\n"
    "  3. Export it:  export OPENCELLID_API_KEY='your-key-here'\n"
    "  4. Re-run:     python -m src.cities.boston_ingest.telecom --force\n"
    "     (Or run the orchestrator with --layers telecom --force.)\n"
    "\n"
    "Until then, FCC ASR (fcc_asr_towers.geojson) is the only telecom\n"
    "source in this folder. ASR covers registered macro structures\n"
    "(cellular, broadcast, microwave) and is sufficient for cascade graph\n"
    "construction at the structure-level. OpenCelliD adds per-cell\n"
    "crowdsourced observations useful for carrier-level coverage analysis.\n"
)


class TripwireBreach(RuntimeError):
    pass


def _fetch_fcc_asr(force: bool) -> dict:
    if FCC_ASR_PATH.exists() and not force:
        size_kb = FCC_ASR_PATH.stat().st_size / 1024
        log.info("fcc_asr_towers.geojson already present (%.1f KB) — skipping", size_kb)
        n = len(gpd.read_file(FCC_ASR_PATH))
        return {
            "layer": LAYER,
            "path": str(FCC_ASR_PATH),
            "source_url": FCC_ASR_URL,
            "record_count": n,
            "sub_layer": "fcc_asr",
        }

    log.info("Fetching FCC ASR antenna structures for Boston bbox %s", boston.BBOX)
    gdf = arcgis.query_layer(
        FCC_ASR_URL,
        layer_id=FCC_ASR_LAYER_ID,
        bbox=boston.BBOX,
    )
    n = len(gdf)
    log.info("FCC ASR returned: %d feature(s)", n)
    log.info("FCC ASR columns (%d): %s", len(gdf.columns), list(gdf.columns))

    if not (FCC_ASR_TRIPWIRE_RANGE[0] <= n <= FCC_ASR_TRIPWIRE_RANGE[1]):
        if FCC_ASR_TRIPWIRE_PATH.exists():
            FCC_ASR_TRIPWIRE_PATH.unlink()
        if n > 0:
            gdf.to_file(FCC_ASR_TRIPWIRE_PATH, driver="GeoJSON")
        raise TripwireBreach(
            f"FCC ASR returned {n} features, outside tripwire range "
            f"{FCC_ASR_TRIPWIRE_RANGE}. Sidecar at {FCC_ASR_TRIPWIRE_PATH}."
        )

    if FCC_ASR_PATH.exists():
        FCC_ASR_PATH.unlink()
    gdf.to_file(FCC_ASR_PATH, driver="GeoJSON")
    log.info("Saved %d FCC ASR structure(s) → %s", n, FCC_ASR_PATH)

    return {
        "layer": LAYER,
        "path": str(FCC_ASR_PATH),
        "source_url": FCC_ASR_URL,
        "record_count": n,
        "sub_layer": "fcc_asr",
    }


def _write_opencellid_todo() -> dict:
    OPENCELLID_TODO_PATH.write_text(OPENCELLID_TODO_BODY)
    log.warning(
        "%s not set — wrote registration instructions to %s",
        OPENCELLID_ENV, OPENCELLID_TODO_PATH,
    )
    return {
        "layer": LAYER,
        "path": str(OPENCELLID_TODO_PATH),
        "source_url": OPENCELLID_URL,
        "record_count": None,
        "sub_layer": "opencellid_skipped",
    }


def _fetch_opencellid(api_key: str, force: bool) -> dict:
    if OPENCELLID_PATH.exists() and not force:
        size_kb = OPENCELLID_PATH.stat().st_size / 1024
        log.info("opencellid_towers.geojson already present (%.1f KB) — skipping", size_kb)
        n = len(gpd.read_file(OPENCELLID_PATH))
        return {
            "layer": LAYER,
            "path": str(OPENCELLID_PATH),
            "source_url": OPENCELLID_URL,
            "record_count": n,
            "sub_layer": "opencellid",
        }

    min_lon, min_lat, max_lon, max_lat = boston.BBOX
    params = {
        "key": api_key,
        "BBOX": f"{min_lat},{min_lon},{max_lat},{max_lon}",
        "format": "csv",
        "limit": 10000,
    }
    log.info("OpenCelliD bbox query: %s", params["BBOX"])
    resp = requests.get(
        OPENCELLID_URL,
        params=params,
        headers={"User-Agent": OPENCELLID_USER_AGENT},
        timeout=OPENCELLID_TIMEOUT,
    )
    resp.raise_for_status()
    if len(resp.text) < 50:
        raise RuntimeError(f"OpenCelliD returned suspiciously short response: {resp.text[:200]}")
    df = pd.read_csv(io.StringIO(resp.text))
    log.info("OpenCelliD returned %d row(s), columns: %s", len(df), list(df.columns))

    # OpenCelliD CSV columns: radio, mcc, net, area, cell, unit, lon, lat, range, samples, ...
    if "lat" not in df.columns or "lon" not in df.columns:
        raise RuntimeError(
            f"OpenCelliD response missing lat/lon columns: {list(df.columns)}"
        )
    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(r.lon, r.lat) for r in df.itertuples()],
        crs="EPSG:4326",
    )
    if OPENCELLID_PATH.exists():
        OPENCELLID_PATH.unlink()
    gdf.to_file(OPENCELLID_PATH, driver="GeoJSON")
    log.info("Saved %d OpenCelliD cell(s) → %s", len(gdf), OPENCELLID_PATH)

    return {
        "layer": LAYER,
        "path": str(OPENCELLID_PATH),
        "source_url": OPENCELLID_URL,
        "record_count": len(gdf),
        "sub_layer": "opencellid",
    }


def download(force: bool = False) -> list[dict]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    entries = [_fetch_fcc_asr(force)]

    api_key = os.environ.get(OPENCELLID_ENV, "").strip()
    if not api_key:
        entries.append(_write_opencellid_todo())
    else:
        log.info("%s is set; pulling OpenCelliD for Boston bbox", OPENCELLID_ENV)
        try:
            entries.append(_fetch_opencellid(api_key, force))
        except Exception as e:
            log.error("OpenCelliD pull failed: %s", e)
            entries.append({
                "layer": LAYER,
                "path": None,
                "source_url": OPENCELLID_URL,
                "record_count": None,
                "sub_layer": "opencellid_failed",
                "error": str(e),
            })

    return entries


if __name__ == "__main__":
    from src.cities.boston_ingest._logging import configure as configure_logging

    configure_logging()
    entries = download()
    for e in entries:
        rc = e.get("record_count")
        rc_str = f"({rc} records)" if rc is not None else "(notice)"
        print(f"  {e['layer']:8s} [{e.get('sub_layer','')}]  {e.get('path')}  {rc_str}")
