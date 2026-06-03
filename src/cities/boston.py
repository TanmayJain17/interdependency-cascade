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
#
# DEPRECATED markers carry a one-line reason and (where applicable) a
# pointer to the replacement source we actually use. Full audit trail
# is in data/boston/raw/_manifest.json and data/boston/raw/README.md.
#
SOURCES = {
    # -------- Subway --------
    "subway_mbta_gtfs": "https://cdn.mbta.com/MBTA_GTFS.zip",

    # -------- Power --------
    "power_hifld_substations": (
        "https://oceandata.rad.rutgers.edu/arcgis/rest/services/"
        "RenewableEnergy/HIFLD_Electric_SubstationsTransmissionLines/MapServer"
    ),

    # -------- Fuel --------
    # DEPRECATED — EIA's canonical FeatureServer was auth-gated some time
    # after NYC's last pull (HTTP 499 Token Required on the org-level
    # service URL). Replaced by operator-verified hardcoded entries.
    "fuel_eia_terminals": None,
    # Replacement: EPA Region 1 Chelsea Creek NPDES permit index, plus
    # operator-page geocodes for the 5 named terminals. The 5 records
    # carry per-row source_url pointing at the EPA permit PDF.
    "fuel_chelsea_creek_terminals": (
        "https://www.epa.gov/npdes-permits/chelsea-river-bulk-petroleum-storage-facilities-npdes-permits"
    ),
    "fuel_osm_overpass": "https://overpass-api.de/api/interpreter",

    # -------- Water --------
    # DEPRECATED — BWSC's operational pumping/tide-gate/MS4-outfall
    # layers are not publicly served on data.boston.gov, BWSC's own
    # ArcGIS org, BostonGIS, or MassGIS. The handoff numbers
    # (9 pumping / 267 outfalls / 201 tide gates) are consistent with
    # BWSC's internal GIS — pending a data-sharing agreement to release.
    "water_bwsc": None,
    # DEPRECATED — MWRA's own ArcGIS org (community ID COM_0018) does
    # not publicly serve regional pumping/headworks layers. MWRA Deer
    # Island Treatment Plant and 11 MWRA-permitted outfalls show up via
    # the NPDES / MassDEP layers below.
    "water_mwra": None,
    "water_massdep_cso": (
        "https://services1.arcgis.com/7iJyYTjCtKsZS1LR/arcgis/rest/services/"
        "MassDEP_CSOs_2/FeatureServer"
    ),
    "water_epa_r1_cso": (
        "https://services.arcgis.com/cJ9YHowT8TU7DUyn/arcgis/rest/services/"
        "R1_Combined_Sewer_Outfall__CSO__Locations__2022/FeatureServer"
    ),
    "water_npdes_facilities": (
        "https://services.arcgis.com/cJ9YHowT8TU7DUyn/arcgis/rest/services/"
        "oeca__echo__npdes_facilities_outfalls/FeatureServer"
    ),

    # -------- Healthcare (hospitals — 6th node type for the cascade model) --------
    "healthcare_hifld_hospitals": (
        "https://services2.arcgis.com/FiaPA4ga0iQKduv3/arcgis/rest/services/"
        "Hospitals/FeatureServer"
    ),

    # -------- Telecom --------
    # DEPRECATED — HIFLD's surviving public Cellular Towers layer
    # (Federal_User_Community / Cellular_Towers_in_the_United_States) is
    # now a single FCC ULS Cellular Service band-class extract that
    # returns only 3 records for the Boston bbox (all Cellco Partnership
    # / Verizon). Replaced by FCC ASR which is the broader federal
    # antenna structure registry (~85 records for Boston).
    "telecom_hifld_towers": None,
    "telecom_fcc_asr": (
        "https://services.arcgis.com/B7X7NCOKKXditlwZ/arcgis/rest/services/"
        "FCC_Antenna_Structures/FeatureServer"
    ),
    "telecom_opencellid": "https://opencellid.org/cell/getInArea",  # key-gated; skipped if OPENCELLID_API_KEY unset

    # -------- Flood --------
    "flood_climate_ready": (
        "https://services.arcgis.com/sFnw0xNflSi8J0uh/arcgis/rest/services/"
        "Climate_Ready_Boston_Sea_Level_Rise_Inundation/FeatureServer"
    ),
    "flood_fema_nfhl": (
        "https://arcgisserver.digital.mass.gov/arcgisserver/rest/services/"
        "FEMA/FEMA_National_Flood_Hazard_Layer/FeatureServer"
    ),
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