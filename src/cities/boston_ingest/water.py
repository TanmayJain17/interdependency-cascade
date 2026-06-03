"""
C4 — Water infrastructure for the Boston bbox (best public proxy).

The handoff anticipated a BWSC GIS extract (9 pump stations / 267 outfalls /
201 tide gates). After exhausting the public-source priority order
(data.boston.gov BWSC org, BWSC's own ArcGIS, BostonGIS, MassGIS, broader
ArcGIS Online search), no canonical layer is publicly served. See the
README sidecar for the full audit trail.

This week's ingestion ships **the best public proxy** for the outfall
sub-layer, while pump stations and tide gates are explicitly deferred
pending a data-sharing agreement with BWSC:

    data/boston/raw/water/massdep_cso_outfalls.geojson
        MassDEP's CSO layer — BWSC + MWRA + Cambridge CSOs with per-outfall
        EPA NPDES permit URLs.

    data/boston/raw/water/epa_r1_cso_outfalls.geojson
        EPA Region 1's 2022 CSO Locations layer (layer 1) — overlapping but
        independent inventory of New England CSO outfalls.

    data/boston/raw/water/npdes_facilities_outfalls.geojson
        EPA's national NPDES Outfalls/Regulated Facilities layer — captures
        major facility-level permitted dischargers (Logan, Gillette, MWRA
        Deer Island Treatment Plant, Sunoco East Boston Terminal, etc.).

Tripwire: per the C4 directive, the CSO outfall sources are gated on
count in [30, 200]. The NPDES facilities layer is informational and
ungated. We do NOT backfill pump stations or tide gates with OSM or
hardcoded entries — the gap is documented instead.
"""
from __future__ import annotations

import logging

import geopandas as gpd

from src.cities import boston
from src.data_ingest import arcgis

log = logging.getLogger(__name__)

LAYER = "water"
OUT_DIR = boston.RAW_DIR / "water"

# ----- Sub-layer definitions -----

MASSDEP_CSO = {
    "key": "massdep_cso",
    "url": "https://services1.arcgis.com/7iJyYTjCtKsZS1LR/arcgis/rest/services/MassDEP_CSOs_2/FeatureServer",
    "layer_id": 0,
    "out_name": "massdep_cso_outfalls.geojson",
    "tripwire": (30, 200),
}

EPA_R1_CSO = {
    "key": "epa_r1_cso",
    "url": "https://services.arcgis.com/cJ9YHowT8TU7DUyn/arcgis/rest/services/R1_Combined_Sewer_Outfall__CSO__Locations__2022/FeatureServer",
    "layer_id": 1,
    "out_name": "epa_r1_cso_outfalls.geojson",
    "tripwire": (30, 200),
}

NPDES_FACILITIES = {
    "key": "npdes_facilities",
    "url": "https://services.arcgis.com/cJ9YHowT8TU7DUyn/arcgis/rest/services/oeca__echo__npdes_facilities_outfalls/FeatureServer",
    "layer_id": 0,
    "out_name": "npdes_facilities_outfalls.geojson",
    "tripwire": None,  # facility-level, count is expected to be small
}

ALL_SUBLAYERS = [MASSDEP_CSO, EPA_R1_CSO, NPDES_FACILITIES]


class TripwireBreach(RuntimeError):
    pass


def _fetch_one(sub: dict, force: bool) -> dict:
    out_path = OUT_DIR / sub["out_name"]
    tripwire_path = OUT_DIR / sub["out_name"].replace(".geojson", ".tripwire.geojson")

    if out_path.exists() and not force:
        size_kb = out_path.stat().st_size / 1024
        log.info("%s already present (%.1f KB) — skipping", sub["out_name"], size_kb)
        n = len(gpd.read_file(out_path))
        return {
            "layer": LAYER,
            "path": str(out_path),
            "source_url": sub["url"],
            "record_count": n,
            "sub_layer": sub["key"],
        }

    log.info("Fetching %s for Boston bbox %s", sub["key"], boston.BBOX)
    gdf = arcgis.query_layer(
        sub["url"],
        layer_id=sub["layer_id"],
        bbox=boston.BBOX,
    )
    n = len(gdf)
    log.info("%s returned: %d feature(s)", sub["key"], n)
    log.info("%s columns (%d): %s", sub["key"], len(gdf.columns), list(gdf.columns))

    if n == 0:
        raise TripwireBreach(f"{sub['key']}: returned 0 features.")

    tw = sub["tripwire"]
    if tw is not None and not (tw[0] <= n <= tw[1]):
        if tripwire_path.exists():
            tripwire_path.unlink()
        gdf.to_file(tripwire_path, driver="GeoJSON")
        raise TripwireBreach(
            f"{sub['key']}: {n} outside tripwire range {tw}; "
            f"sidecar at {tripwire_path}."
        )

    if out_path.exists():
        out_path.unlink()
    gdf.to_file(out_path, driver="GeoJSON")
    log.info("Saved %d feature(s) → %s", n, out_path)

    return {
        "layer": LAYER,
        "path": str(out_path),
        "source_url": sub["url"],
        "record_count": n,
        "sub_layer": sub["key"],
    }


def _write_gap_note() -> dict:
    """Document the BWSC operational-layer gap as a sidecar README."""
    gap_path = OUT_DIR / "_BWSC_GAP.md"
    gap_path.write_text(
        "# BWSC operational layers — gap notice\n"
        "\n"
        "The Week 12 handoff anticipated three BWSC operational sub-layers:\n"
        "\n"
        "| Asset | Handoff count |\n"
        "| --- | --- |\n"
        "| Pumping stations | 9 |\n"
        "| Outfalls | 267 |\n"
        "| Tide gates | 201 |\n"
        "\n"
        "After exhausting the public source priority order (data.boston.gov\n"
        "BWSC org page, BWSC's own ArcGIS org services5.arcgis.com/ji3WHeqN0AysEK5g,\n"
        "BostonGIS services.arcgis.com/sFnw0xNflSi8J0uh, MassGIS catalog,\n"
        "ArcGIS Online public search), no canonical layer for any of these\n"
        "three asset types is publicly served. The 2023 BWSC Coastal\n"
        "Stormwater Discharge Analysis report contains ~88 coastal-vulnerable\n"
        "outfall IDs with no lat/lon coordinates — insufficient for spatial\n"
        "use.\n"
        "\n"
        "**What this folder ships instead** (the best public proxy):\n"
        "\n"
        "- `massdep_cso_outfalls.geojson` — MassDEP's CSO layer (BWSC + MWRA + Cambridge)\n"
        "- `epa_r1_cso_outfalls.geojson` — EPA Region 1's 2022 CSO Locations (overlapping but independent inventory)\n"
        "- `npdes_facilities_outfalls.geojson` — EPA national NPDES facility-level dischargers in Boston (includes MWRA Deer Island)\n"
        "\n"
        "**Pump stations and tide gates are NOT in this folder.** OSM coverage\n"
        "of municipal water infrastructure is too sparse to use as a proxy,\n"
        "and hardcoding requires an authoritative inventory we do not have.\n"
        "\n"
        "**Path forward:** request the BWSC GIS extract through a data-sharing\n"
        "agreement. The handoff numbers (9/267/201) are consistent with\n"
        "BWSC's internal GIS database — that is the canonical source.\n"
    )
    log.info("Wrote BWSC gap notice → %s", gap_path)
    return {
        "layer": LAYER,
        "path": str(gap_path),
        "source_url": None,
        "record_count": None,
        "sub_layer": "_gap_note",
    }


def download(force: bool = False) -> list[dict]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []
    for sub in ALL_SUBLAYERS:
        entries.append(_fetch_one(sub, force))
    entries.append(_write_gap_note())
    return entries


if __name__ == "__main__":
    from src.cities.boston_ingest._logging import configure as configure_logging

    configure_logging()
    entries = download()
    for e in entries:
        rc = e["record_count"]
        rc_str = f"({rc} records)" if rc is not None else "(notice)"
        print(f"  {e['layer']:8s} [{e.get('sub_layer','')}]  {e['path']}  {rc_str}")
