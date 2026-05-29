"""
Boston pilot — data sources, bounding box, folder paths.

Mirrors the planned structure for src/cities/nyc.py (future refactor).
Single source of truth for everything Boston-specific.
"""
from pathlib import Path

# Resolve project root relative to this file: src/cities/boston.py -> RA/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Geographic bounding box: (min_lon, min_lat, max_lon, max_lat)
BBOX = (-71.20, 42.22, -70.92, 42.40)

# Data directories (mirrors data/nyc/{raw,flood,graph,simulation})
DATA_DIR = PROJECT_ROOT / "data" / "boston"
RAW_DIR = DATA_DIR / "raw"
FLOOD_DIR = DATA_DIR / "flood"
GRAPH_DIR = DATA_DIR / "graph"
SIMULATION_DIR = DATA_DIR / "simulation"

ALL_DIRS = [RAW_DIR, FLOOD_DIR, GRAPH_DIR, SIMULATION_DIR]

# -----------------------------------------------------------------------------
# Data sources — URLs / endpoints per layer
# Confirmed URLs are filled in; TBDs will be set when we wire up downloads.
# -----------------------------------------------------------------------------
SOURCES = {
    # Flood (primary): Climate Ready Boston SLR scenarios from data.boston.gov
    "flood_climate_ready": None,         # TBD — data.boston.gov landing page

    # Flood (cross-check): FEMA NFHL via MassGIS
    "flood_fema_nfhl": None,             # TBD — MassGIS ArcGIS endpoint

    # Subway: MBTA GTFS feed (~153 stations)
    "subway_mbta_gtfs": "https://cdn.mbta.com/MBTA_GTFS.zip",

    # Power: HIFLD substations via Rutgers ArcGIS (same source as NYC)
    "power_hifld_substations": None,     # TBD — Rutgers ArcGIS FeatureServer

    # Water: BWSC (pumping stations, outfalls, tide gates) + MWRA + MassGIS
    "water_bwsc": None,                  # TBD
    "water_mwra": None,                  # TBD

    # Fuel: EIA Petroleum Terminals + OSM (Chelsea Creek terminal cluster)
    "fuel_eia_terminals": None,          # TBD — EIA ArcGIS FeatureServer
    "fuel_osm_overpass": "https://overpass-api.de/api/interpreter",

    # Telecom: OpenCelliD + HIFLD cellular towers
    "telecom_opencellid": None,          # TBD — OpenCelliD download
    "telecom_hifld_towers": None,        # TBD — HIFLD cellular towers
}

# -----------------------------------------------------------------------------
# Climate Ready Boston scenarios: 3 SLR levels x 2 AEP levels = 6 scenarios
# Matches the structure of NYC's 3 DEP scenarios (just a different grid)
# -----------------------------------------------------------------------------
CRB_SCENARIOS = [
    {"slr_in": 9,  "aep_pct": 10, "name": "slr09_aep10"},
    {"slr_in": 9,  "aep_pct": 1,  "name": "slr09_aep01"},
    {"slr_in": 21, "aep_pct": 10, "name": "slr21_aep10"},
    {"slr_in": 21, "aep_pct": 1,  "name": "slr21_aep01"},
    {"slr_in": 36, "aep_pct": 10, "name": "slr36_aep10"},
    {"slr_in": 36, "aep_pct": 1,  "name": "slr36_aep01"},
]


def ensure_dirs() -> None:
    """Create all Boston data directories if they don't exist."""
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    # Quick smoke test: print resolved paths and create directories
    print(f"Project root : {PROJECT_ROOT}")
    print(f"BBOX         : {BBOX}")
    print(f"Data dir     : {DATA_DIR}")
    print(f"Scenarios    : {len(CRB_SCENARIOS)}")
    ensure_dirs()
    print("Directories created.")