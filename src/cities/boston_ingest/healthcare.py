"""
Phase 1 — Boston hospitals via HIFLD Hospitals (FEMA national maps ArcGIS).

Hospitals are the 6th node type the NYC-trained cascade model requires.
Boston did not have them yet; this fills that gap.

Output:
    data/boston/raw/healthcare/hospitals.geojson

Full HIFLD schema (35 columns) is preserved, including BEDS (bed count)
for provenance. NOTE: the cascade model does NOT consume BEDS — NYC's
convert_to_pyg.py hardcodes bed_count=200 for every hospital, so to keep
Boston's feature distribution identical to NYC, the Phase 2 graph builder
will feed the same constant. BEDS is stored here only for the manifest /
future use.

Tripwire: greater Boston should yield ~20-60 hospitals. Outside [10, 90]
raises (with a sidecar for inspection).

Sanity check: the six flagship Boston hospitals (Mass General, Brigham &
Women's, Boston Medical Center, Tufts Medical Center, Beth Israel
Deaconess, Boston Children's) should appear; matched names are logged.
"""
from __future__ import annotations

import logging

import geopandas as gpd

from src.cities import boston
from src.data_ingest import arcgis

log = logging.getLogger(__name__)

LAYER = "healthcare"
SOURCE_URL = (
    "https://services2.arcgis.com/FiaPA4ga0iQKduv3/arcgis/rest/services/"
    "Hospitals/FeatureServer"
)
LAYER_ID = 0
OUT_DIR = boston.RAW_DIR / "healthcare"
OUT_PATH = OUT_DIR / "hospitals.geojson"
TRIPWIRE_PATH = OUT_DIR / "hospitals.tripwire.geojson"

EXPECTED_RANGE = (20, 60)
TRIPWIRE_RANGE = (10, 90)

# Flagship hospitals that must show up in greater Boston. Each entry is a
# list of keyword tuples; a hospital matches if any tuple's keywords ALL
# appear (case-insensitive) in the NAME field.
FLAGSHIP_EXPECTED: dict[str, list[tuple[str, ...]]] = {
    "Massachusetts General": [("massachusetts", "general")],
    "Brigham and Women's":   [("brigham",)],
    "Boston Medical Center": [("boston", "medical", "center")],
    "Tufts Medical Center":  [("tufts",)],
    "Beth Israel Deaconess": [("beth", "israel")],
    "Boston Children's":     [("children",)],
}


class TripwireBreach(RuntimeError):
    pass


def _record(path, n) -> list[dict]:
    return [{
        "layer": LAYER,
        "path": str(path),
        "source_url": SOURCE_URL,
        "record_count": n,
        "sub_layer": "hifld_hospitals",
        "arcgis_layer_id": LAYER_ID,
    }]


def _check_flagships(gdf: gpd.GeoDataFrame) -> list[str]:
    name_col = "NAME" if "NAME" in gdf.columns else None
    if name_col is None:
        log.warning("No NAME column on hospitals; skipping flagship check")
        return []
    names_lower = gdf[name_col].astype(str).str.lower()
    missing: list[str] = []
    for label, tuples in FLAGSHIP_EXPECTED.items():
        found = False
        for keywords in tuples:
            mask = names_lower.apply(lambda s: all(kw in s for kw in keywords))
            if mask.any():
                matched = gdf.loc[mask, name_col].tolist()
                log.info("  flagship match: %-24s → %s", label, matched[:2])
                found = True
                break
        if not found:
            log.warning("  flagship MISS: %s", label)
            missing.append(label)
    return missing


def download(force: bool = False) -> list[dict]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if OUT_PATH.exists() and not force:
        size_kb = OUT_PATH.stat().st_size / 1024
        log.info("hospitals.geojson already present (%.1f KB) — skipping", size_kb)
        return _record(OUT_PATH, len(gpd.read_file(OUT_PATH)))

    log.info("Fetching HIFLD hospitals for Boston bbox %s", boston.BBOX)
    gdf = arcgis.query_layer(SOURCE_URL, layer_id=LAYER_ID, bbox=boston.BBOX)
    n = len(gdf)
    log.info("HIFLD hospitals returned: %d feature(s)", n)
    log.info("Columns (%d): %s", len(gdf.columns), list(gdf.columns))

    if not (TRIPWIRE_RANGE[0] <= n <= TRIPWIRE_RANGE[1]):
        if TRIPWIRE_PATH.exists():
            TRIPWIRE_PATH.unlink()
        if n > 0:
            gdf.to_file(TRIPWIRE_PATH, driver="GeoJSON")
        raise TripwireBreach(
            f"HIFLD hospitals returned {n}, outside tripwire {TRIPWIRE_RANGE} "
            f"(expected {EXPECTED_RANGE}). Sidecar: {TRIPWIRE_PATH}"
        )
    if not (EXPECTED_RANGE[0] <= n <= EXPECTED_RANGE[1]):
        log.warning("Hospital count %d outside expected %s (within tripwire) — saving anyway",
                    n, EXPECTED_RANGE)

    # Flagship sanity check (log only; do not block — HIFLD naming varies)
    missing = _check_flagships(gdf)
    if missing:
        log.warning("Flagship hospitals not matched by name: %s "
                    "(may be present under alternate names — verify manually)", missing)

    # Bed-count availability note (provenance only)
    if "BEDS" in gdf.columns:
        with_beds = int((gdf["BEDS"].fillna(0) > 0).sum())
        log.info("BEDS populated on %d/%d hospitals (stored for provenance, not fed to model)",
                 with_beds, n)

    if OUT_PATH.exists():
        OUT_PATH.unlink()
    gdf.to_file(OUT_PATH, driver="GeoJSON")
    log.info("Saved %d hospital(s) → %s", n, OUT_PATH)
    return _record(OUT_PATH, n)


if __name__ == "__main__":
    from src.cities.boston_ingest._logging import configure as configure_logging

    configure_logging()
    entries = download()
    for e in entries:
        rc = e.get("record_count")
        print(f"  {e['layer']:10s} [{e.get('sub_layer','')}]  {e.get('path')}  ({rc} records)")
