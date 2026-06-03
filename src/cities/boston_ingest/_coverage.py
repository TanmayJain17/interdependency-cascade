"""
Coverage plot — single matplotlib figure showing every ingested layer
over the Boston bbox. Eyeball check for the Phase E report.

Output: data/boston/raw/_coverage.png
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

# Lift GDAL GeoJSON size cap so the worst-case CRB polygon reads back
os.environ.setdefault("OGR_GEOJSON_MAX_OBJ_SIZE", "0")

import geopandas as gpd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from src.cities import boston  # noqa: E402

log = logging.getLogger(__name__)

COVERAGE_PATH = boston.RAW_DIR / "_coverage.png"


def _safe_read(p: Path) -> Optional[gpd.GeoDataFrame]:
    if not p.exists():
        return None
    try:
        return gpd.read_file(p)
    except Exception as e:
        log.warning("Could not read %s for coverage plot: %s", p, e)
        return None


def write_coverage_plot() -> Path:
    fig, ax = plt.subplots(figsize=(11, 11), dpi=120)

    min_lon, min_lat, max_lon, max_lat = boston.BBOX

    # CRB worst case — fill (under everything)
    crb_worst = _safe_read(boston.FLOOD_DIR / "crb_slr36_aep01.geojson")
    if crb_worst is not None:
        crb_worst.plot(ax=ax, color="#0066cc", alpha=0.18, edgecolor="none",
                       label="CRB 36\" SLR + 1% AEP (worst case)")

    # CRB lightest case — outline only (for ordering visual)
    crb_light = _safe_read(boston.FLOOD_DIR / "crb_slr09_aep10.geojson")
    if crb_light is not None:
        crb_light.boundary.plot(ax=ax, color="#003366", linewidth=0.6, alpha=0.5,
                                label="CRB 9\" SLR + 10% AEP (light)")

    # FEMA NFHL — boundary only, light
    fema = _safe_read(boston.FLOOD_DIR / "fema_nfhl.geojson")
    if fema is not None:
        fema.boundary.plot(ax=ax, color="#999900", linewidth=0.3, alpha=0.45,
                           label=f"FEMA NFHL ({len(fema)})")

    # Subway stops
    subway = _safe_read(boston.RAW_DIR / "subway" / "stops.geojson")
    if subway is not None:
        subway.plot(ax=ax, color="#cc0000", markersize=4, alpha=0.6,
                    label=f"MBTA stops ({len(subway)})")

    # Power substations
    power = _safe_read(boston.RAW_DIR / "power" / "hifld_substations.geojson")
    if power is not None:
        power.plot(ax=ax, color="#000000", marker="s", markersize=22,
                   label=f"HIFLD substations ({len(power)})")

    # Fuel: OSM stations + man_made tanks
    osm_fuel = _safe_read(boston.RAW_DIR / "fuel" / "osm_stations.geojson")
    if osm_fuel is not None:
        only_fuel = osm_fuel[osm_fuel.get("osm_type") == "fuel"] if "osm_type" in osm_fuel.columns else osm_fuel
        only_tank = osm_fuel[osm_fuel.get("osm_type") == "storage_tank"] if "osm_type" in osm_fuel.columns else osm_fuel.iloc[0:0]
        only_fuel.plot(ax=ax, color="#ff8800", marker="^", markersize=8, alpha=0.7,
                       label=f"OSM gas stations ({len(only_fuel)})")
        if len(only_tank) > 0:
            only_tank.plot(ax=ax, color="#cc6600", marker=".", markersize=2, alpha=0.4,
                           label=f"OSM storage tanks ({len(only_tank)})")

    # Fuel terminals (highlight)
    terminals = _safe_read(boston.RAW_DIR / "fuel" / "major_terminals.geojson")
    if terminals is not None:
        terminals.plot(ax=ax, color="#ff0000", marker="*", markersize=200,
                       edgecolor="black", linewidth=0.8,
                       label=f"Chelsea Creek terminals ({len(terminals)})")

    # Water — three sub-layers
    for fname, color, label in [
        ("massdep_cso_outfalls.geojson", "#006699", "MassDEP CSOs"),
        ("epa_r1_cso_outfalls.geojson", "#0099cc", "EPA R1 CSOs"),
        ("npdes_facilities_outfalls.geojson", "#00cccc", "NPDES facilities"),
    ]:
        g = _safe_read(boston.RAW_DIR / "water" / fname)
        if g is not None:
            g.plot(ax=ax, color=color, marker="o", markersize=14, alpha=0.65,
                   label=f"{label} ({len(g)})")

    # Telecom
    telecom = _safe_read(boston.RAW_DIR / "telecom" / "fcc_asr_towers.geojson")
    if telecom is not None:
        telecom.plot(ax=ax, color="#660066", marker="x", markersize=24,
                     label=f"FCC ASR towers ({len(telecom)})")

    # Bbox frame
    ax.plot(
        [min_lon, max_lon, max_lon, min_lon, min_lon],
        [min_lat, min_lat, max_lat, max_lat, min_lat],
        color="black", linewidth=1.0, linestyle="--", label="Boston bbox",
    )

    ax.set_xlim(min_lon - 0.01, max_lon + 0.01)
    ax.set_ylim(min_lat - 0.01, max_lat + 0.01)
    ax.set_xlabel("Longitude (EPSG:4326)")
    ax.set_ylabel("Latitude (EPSG:4326)")
    ax.set_title(
        "Boston pilot — Week 12 ingestion coverage\n"
        f"bbox={boston.BBOX}",
        fontsize=11,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="lower left", fontsize=7, framealpha=0.85)
    ax.grid(True, alpha=0.25, linewidth=0.4)

    fig.tight_layout()
    fig.savefig(COVERAGE_PATH, dpi=150)
    plt.close(fig)
    log.info("Wrote coverage plot → %s", COVERAGE_PATH)
    return COVERAGE_PATH
