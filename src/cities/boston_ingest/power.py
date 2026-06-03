"""
C2 — HIFLD electric substations + transmission lines via the Rutgers mirror.

Same MapServer endpoint NYC's download_power.py uses
(src/data_acquisition/download_power.py:25). Substations live on layer 0;
transmission lines on layer 1.

Outputs:
    data/boston/raw/power/hifld_substations.geojson        (layer 0)
    data/boston/raw/power/hifld_transmission_lines.geojson (layer 1)

The transmission lines carry SUB_1 / SUB_2 endpoint names + VOLT_CLASS,
which the Boston graph builder uses to form the (power, power_line, power)
relation via the same SUB_1/SUB_2 name-matching rule NYC's build_graph.py
uses. (Added in Week 12-cont. Phase 2 — the substations-only ingest left
power_line with no input.)

Tripwire: greater Boston should yield ~30-80 substations. We hard-fail
on counts outside [10, 150] because the Rutgers MapServer is known to
truncate or drop the bbox silently when geometry params are malformed.
On tripwire breach the data is saved to a `.tripwire.geojson` sidecar so
it can be inspected, but the canonical path stays untouched.
"""
from __future__ import annotations

import logging

import geopandas as gpd

from src.cities import boston
from src.data_ingest import arcgis

log = logging.getLogger(__name__)

LAYER = "power"
SOURCE_URL = (
    "https://oceandata.rad.rutgers.edu/arcgis/rest/services/"
    "RenewableEnergy/HIFLD_Electric_SubstationsTransmissionLines/MapServer"
)
SUBSTATIONS_LAYER_ID = 0
LINES_LAYER_ID = 1
OUT_DIR = boston.RAW_DIR / "power"
SUBSTATIONS_PATH = OUT_DIR / "hifld_substations.geojson"
SUBSTATIONS_TRIPWIRE_PATH = OUT_DIR / "hifld_substations.tripwire.geojson"
LINES_PATH = OUT_DIR / "hifld_transmission_lines.geojson"

# Expected range: 30-80. Tripwire range: 10-150 (3x margin each side).
EXPECTED_RANGE = (30, 80)
TRIPWIRE_RANGE = (10, 150)


class TripwireBreach(RuntimeError):
    pass


def _record_existing(path) -> list[dict]:
    n = len(gpd.read_file(path))
    return [{
        "layer": LAYER,
        "path": str(path),
        "source_url": SOURCE_URL,
        "record_count": n,
        "arcgis_layer_id": SUBSTATIONS_LAYER_ID,
    }]


def _download_lines(force: bool) -> dict:
    """Fetch transmission lines (layer 1) — endpoints for the power_line relation."""
    if LINES_PATH.exists() and not force:
        size_kb = LINES_PATH.stat().st_size / 1024
        log.info("hifld_transmission_lines.geojson already present (%.1f KB) — skipping", size_kb)
        return {
            "layer": LAYER,
            "path": str(LINES_PATH),
            "source_url": SOURCE_URL,
            "record_count": len(gpd.read_file(LINES_PATH)),
            "sub_layer": "transmission_lines",
            "arcgis_layer_id": LINES_LAYER_ID,
        }

    log.info("Fetching HIFLD transmission lines for Boston bbox %s", boston.BBOX)
    gdf = arcgis.query_layer(SOURCE_URL, layer_id=LINES_LAYER_ID, bbox=boston.BBOX)
    n = len(gdf)
    log.info("HIFLD transmission lines returned: %d feature(s); cols: %s",
             n, list(gdf.columns))
    if LINES_PATH.exists():
        LINES_PATH.unlink()
    gdf.to_file(LINES_PATH, driver="GeoJSON")
    log.info("Saved %d transmission line(s) → %s", n, LINES_PATH)
    return {
        "layer": LAYER,
        "path": str(LINES_PATH),
        "source_url": SOURCE_URL,
        "record_count": n,
        "sub_layer": "transmission_lines",
        "arcgis_layer_id": LINES_LAYER_ID,
    }


def download(force: bool = False) -> list[dict]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if SUBSTATIONS_PATH.exists() and not force:
        size_kb = SUBSTATIONS_PATH.stat().st_size / 1024
        log.info("hifld_substations.geojson already present (%.1f KB) — skipping", size_kb)
        return _record_existing(SUBSTATIONS_PATH) + [_download_lines(force)]

    log.info("Fetching HIFLD substations for Boston bbox %s", boston.BBOX)
    gdf = arcgis.query_layer(
        SOURCE_URL,
        layer_id=SUBSTATIONS_LAYER_ID,
        bbox=boston.BBOX,
    )
    n = len(gdf)
    log.info("HIFLD substations returned: %d feature(s)", n)
    log.info(
        "Columns preserved (%d): %s",
        len(gdf.columns), list(gdf.columns),
    )

    if not (TRIPWIRE_RANGE[0] <= n <= TRIPWIRE_RANGE[1]):
        # Save sidecar so the user can inspect what came back
        if SUBSTATIONS_TRIPWIRE_PATH.exists():
            SUBSTATIONS_TRIPWIRE_PATH.unlink()
        if n > 0:
            gdf.to_file(SUBSTATIONS_TRIPWIRE_PATH, driver="GeoJSON")
            log.error(
                "TRIPWIRE: saved %d-feature sample to %s for inspection",
                n, SUBSTATIONS_TRIPWIRE_PATH,
            )
        raise TripwireBreach(
            f"HIFLD substations returned {n} features for Boston bbox "
            f"{boston.BBOX}; tripwire range is {TRIPWIRE_RANGE} "
            f"(expected {EXPECTED_RANGE}). Likely bbox geometry rejected "
            f"by Rutgers MapServer. Sidecar: {SUBSTATIONS_TRIPWIRE_PATH}"
        )

    if not (EXPECTED_RANGE[0] <= n <= EXPECTED_RANGE[1]):
        log.warning(
            "HIFLD substations count %d is outside the expected range %s "
            "(but within tripwire %s) — saving anyway",
            n, EXPECTED_RANGE, TRIPWIRE_RANGE,
        )

    if SUBSTATIONS_PATH.exists():
        SUBSTATIONS_PATH.unlink()
    gdf.to_file(SUBSTATIONS_PATH, driver="GeoJSON")
    log.info("Saved %d substation(s) → %s", n, SUBSTATIONS_PATH)

    return [
        {
            "layer": LAYER,
            "path": str(SUBSTATIONS_PATH),
            "source_url": SOURCE_URL,
            "record_count": n,
            "arcgis_layer_id": SUBSTATIONS_LAYER_ID,
        },
        _download_lines(force),
    ]


if __name__ == "__main__":
    from src.cities.boston_ingest._logging import configure as configure_logging

    configure_logging()
    entries = download()
    for e in entries:
        rc = e["record_count"]
        rc_str = f"({rc} records)" if rc is not None else "(archive)"
        print(f"  {e['layer']:8s} {e['path']}  {rc_str}")
