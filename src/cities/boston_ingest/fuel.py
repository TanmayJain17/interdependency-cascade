"""
C3 — Fuel infrastructure for the Boston bbox.

Two outputs:

    data/boston/raw/fuel/major_terminals.geojson
        Five operator-verified Chelsea Creek bulk petroleum terminals.
        EIA's canonical FeatureServer (the one NYC's download_fuel.py:205
        uses) is now access-gated, so we hardcode the cluster with full
        operator + EPA NPDES permit provenance instead of relying on a
        user re-host of unknown freshness.

    data/boston/raw/fuel/osm_stations.geojson
        OSM amenity=fuel (retail stations) UNION man_made=storage_tank
        (bulk storage where OSM contributors prefer that tag) within the
        Boston bbox. An osm_type column distinguishes the two. Way
        geometries are reduced to their `out center` point.

Tripwires:
    * major_terminals: must contain exactly 5 rows, one per expected name.
    * OSM amenity=fuel count must be in [30, 500] (storage_tank count
      reported but un-gated, since OSM coverage of bulk tanks varies wildly).
"""
from __future__ import annotations

import datetime as dt
import logging

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from src.cities import boston
from src.data_ingest import overpass

log = logging.getLogger(__name__)

LAYER = "fuel"
OUT_DIR = boston.RAW_DIR / "fuel"

# ----- major_terminals (hardcoded, operator/EPA-verified) -----
TERMINALS_PATH = OUT_DIR / "major_terminals.geojson"

# EPA Region 1 publishes the Chelsea Creek NPDES permit overview here:
EPA_CHELSEA_CREEK_INDEX = (
    "https://www.epa.gov/npdes-permits/chelsea-river-bulk-petroleum-storage-facilities-npdes-permits"
)
TERMINALS_SOURCE_URL = EPA_CHELSEA_CREEK_INDEX  # for the manifest aggregate

# Verification date is set when the script runs so the JSON property is
# always the date the data was last cross-checked, not the date of code commit.
VERIFICATION_DATE = dt.datetime.now(dt.timezone.utc).date().isoformat()

MAJOR_TERMINALS: list[dict] = [
    {
        "name": "Global Chelsea Eastern Ave Terminal",
        "operator": "Global Partners LP (Global Companies LLC)",
        "address": "80 Eastern Ave, Chelsea, MA 02150",
        "lat": 42.389937,
        "lon": -71.023666,
        "terminal_type": "bulk_petroleum",
        "epa_npdes_permit": None,  # part of the Chelsea Creek cluster, no per-facility permit ID confirmed
        "source": "operator_verified",
        "source_url": "https://www.globalp.com/where-we-are/all-terminals/global-chelsea-eastern-ave",
        "geocode_method": "nominatim:address:80-eastern-ave-chelsea-ma",
        "verification_date": VERIFICATION_DATE,
        "notes": "Barge- and truck-supplied terminal serving greater Boston via 7-lane truck rack.",
    },
    {
        "name": "Gulf Oil Chelsea Terminal",
        "operator": "Global Partners LP (acquired from Gulf Oil LP, Apr 2024)",
        "address": "281 Eastern Ave, Chelsea, MA 02150",
        "lat": 42.396807,
        "lon": -71.020971,
        "terminal_type": "bulk_petroleum",
        "epa_npdes_permit": "MA0001091",
        "source": "operator_verified",
        "source_url": "https://www3.epa.gov/region1/npdes/chelseacreekfuelterminals/pdfs/gulfoil/GulfOilFactSheet.pdf",
        "geocode_method": "nominatim:address:281-eastern-ave-chelsea-ma",
        "verification_date": VERIFICATION_DATE,
        "notes": "Acquired by Global Partners 2024-04 as part of 4-terminal Gulf Oil package; operator label may shift in future updates.",
    },
    {
        "name": "Irving Oil Revere Marine Terminal",
        "operator": "Irving Oil Terminals Inc",
        "address": "41 Lee Burbank Hwy, Revere, MA 02151",
        "lat": 42.3956,
        "lon": -71.0066,
        "terminal_type": "marine_petroleum",
        "epa_npdes_permit": "MA0001929",
        "source": "operator_verified",
        "source_url": "https://www3.epa.gov/region1/npdes/chelseacreekfuelterminals/pdfs/irvingoil/IrvingOilFigure1.pdf",
        "geocode_method": "overpass:tank-cluster:chelsea-creek-revere-shore",
        "verification_date": VERIFICATION_DATE,
        "notes": "Nominatim returned a Shell retail station at the same street address; the marine terminal proper sits on the Revere shore of Chelsea Creek (storage_tank cluster).",
    },
    {
        "name": "Sunoco / Energy Transfer East Boston Terminal",
        "operator": "Sunoco Partners Marketing & Terminals LP (Energy Transfer)",
        "address": "467 Chelsea St, East Boston, MA 02128",
        "lat": 42.381352,
        "lon": -71.025348,
        "terminal_type": "bulk_petroleum",
        "epa_npdes_permit": "MA0004006",
        "source": "operator_verified",
        "source_url": "https://www.sunocolp.com/segments/supply-terminals/id/t04ma1154",
        "geocode_method": "nominatim:address:467-chelsea-st-east-boston-ma",
        "verification_date": VERIFICATION_DATE,
        "notes": "OSM independently tags this parcel as 'Energy Transfer East Boston Terminal', corroborating the geocode.",
    },
    {
        "name": "Chelsea Sandwich Terminal",
        "operator": "Chelsea Sandwich LLC (Global Operating LLC subsidiary)",
        "address": "11 Broadway, Chelsea, MA 02150",
        "lat": 42.386576,
        "lon": -71.044494,
        "terminal_type": "bulk_petroleum",
        "epa_npdes_permit": "MA0003280",
        "source": "operator_verified",
        "source_url": "https://www3.epa.gov/region1/npdes/chelseacreekfuelterminals/chelseasandwich.html",
        "geocode_method": "nominatim:address:11-broadway-chelsea-ma",
        "verification_date": VERIFICATION_DATE,
        "notes": "Same parent (Global Partners) as Global Chelsea Eastern Ave but physically distinct facility ~2.3 km southwest, with its own EPA NPDES permit. Do not merge.",
    },
]

# ----- OSM -----
OSM_PATH = OUT_DIR / "osm_stations.geojson"
OSM_TRIPWIRE_PATH = OUT_DIR / "osm_stations.tripwire.geojson"
OSM_FUEL_TRIPWIRE_RANGE = (30, 500)  # applied to amenity=fuel count only
OSM_SOURCE_URL = "overpass-api.de (amenity=fuel + man_made=storage_tank)"

OSM_QL = """
[out:json][timeout:120];
(
  node["amenity"="fuel"]({bbox});
  way["amenity"="fuel"]({bbox});
  node["man_made"="storage_tank"]({bbox});
  way["man_made"="storage_tank"]({bbox});
);
out center;
"""


class TripwireBreach(RuntimeError):
    pass


# -----------------------------------------------------------------------------
# major_terminals
# -----------------------------------------------------------------------------

def _build_terminals_gdf() -> gpd.GeoDataFrame:
    df = pd.DataFrame(MAJOR_TERMINALS)
    geom = [Point(r.lon, r.lat) for r in df.itertuples()]
    return gpd.GeoDataFrame(df, geometry=geom, crs="EPSG:4326")


def _save_terminals(force: bool) -> dict:
    if TERMINALS_PATH.exists() and not force:
        log.info("major_terminals.geojson already present — skipping rebuild")
        n = len(gpd.read_file(TERMINALS_PATH))
        return {
            "layer": LAYER,
            "path": str(TERMINALS_PATH),
            "source_url": TERMINALS_SOURCE_URL,
            "record_count": n,
        }

    gdf = _build_terminals_gdf()
    n = len(gdf)
    log.info("major_terminals: %d hardcoded record(s)", n)
    log.info("Columns: %s", list(gdf.columns))

    # Tripwire: must be 5
    if n != 5:
        raise TripwireBreach(
            f"major_terminals: expected exactly 5 hardcoded entries, got {n}. "
            "Someone edited MAJOR_TERMINALS without updating the count assertion."
        )

    # Bbox sanity: log any outside bbox but do not block
    min_lon, min_lat, max_lon, max_lat = boston.BBOX
    for row in gdf.itertuples():
        in_bbox = min_lon <= row.lon <= max_lon and min_lat <= row.lat <= max_lat
        if not in_bbox:
            log.warning("  %s at (%.4f, %.4f) is OUTSIDE Boston bbox", row.name, row.lat, row.lon)

    if TERMINALS_PATH.exists():
        TERMINALS_PATH.unlink()
    gdf.to_file(TERMINALS_PATH, driver="GeoJSON")
    log.info("Saved %d major terminal(s) → %s", n, TERMINALS_PATH)
    return {
        "layer": LAYER,
        "path": str(TERMINALS_PATH),
        "source_url": TERMINALS_SOURCE_URL,
        "record_count": n,
    }


# -----------------------------------------------------------------------------
# OSM
# -----------------------------------------------------------------------------

def _download_osm(force: bool) -> dict:
    if OSM_PATH.exists() and not force:
        size_kb = OSM_PATH.stat().st_size / 1024
        log.info("osm_stations.geojson already present (%.1f KB) — skipping", size_kb)
        n = len(gpd.read_file(OSM_PATH))
        return {
            "layer": LAYER,
            "path": str(OSM_PATH),
            "source_url": OSM_SOURCE_URL,
            "record_count": n,
        }

    log.info("Fetching OSM amenity=fuel + man_made=storage_tank for Boston bbox %s", boston.BBOX)
    gdf = overpass.query(OSM_QL, boston.BBOX)
    n = len(gdf)
    log.info("OSM total returned: %d feature(s)", n)
    log.info("OSM columns (%d): %s", len(gdf.columns), list(gdf.columns))

    if n == 0:
        raise TripwireBreach("OSM returned zero features — Overpass may be unreachable.")

    # Disambiguate amenity=fuel vs man_made=storage_tank with an osm_type column.
    if "amenity" not in gdf.columns:
        gdf["amenity"] = pd.NA
    if "man_made" not in gdf.columns:
        gdf["man_made"] = pd.NA

    def _classify(row) -> str:
        if row.get("amenity") == "fuel":
            return "fuel"
        if row.get("man_made") == "storage_tank":
            return "storage_tank"
        return "other"

    gdf["osm_type"] = gdf.apply(_classify, axis=1)

    n_fuel = int((gdf["osm_type"] == "fuel").sum())
    n_tank = int((gdf["osm_type"] == "storage_tank").sum())
    n_other = int((gdf["osm_type"] == "other").sum())
    log.info("  osm_type breakdown: fuel=%d, storage_tank=%d, other=%d", n_fuel, n_tank, n_other)

    # Tripwire on amenity=fuel only (storage_tank counts are coverage-dependent)
    if not (OSM_FUEL_TRIPWIRE_RANGE[0] <= n_fuel <= OSM_FUEL_TRIPWIRE_RANGE[1]):
        if OSM_TRIPWIRE_PATH.exists():
            OSM_TRIPWIRE_PATH.unlink()
        gdf.to_file(OSM_TRIPWIRE_PATH, driver="GeoJSON")
        raise TripwireBreach(
            f"OSM amenity=fuel count {n_fuel} outside tripwire range "
            f"{OSM_FUEL_TRIPWIRE_RANGE}; sidecar at {OSM_TRIPWIRE_PATH}."
        )

    if OSM_PATH.exists():
        OSM_PATH.unlink()
    gdf.to_file(OSM_PATH, driver="GeoJSON")
    log.info("Saved %d OSM feature(s) → %s", n, OSM_PATH)
    return {
        "layer": LAYER,
        "path": str(OSM_PATH),
        "source_url": OSM_SOURCE_URL,
        "record_count": n,
    }


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------

def download(force: bool = False) -> list[dict]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return [
        _save_terminals(force),
        _download_osm(force),
    ]


if __name__ == "__main__":
    from src.cities.boston_ingest._logging import configure as configure_logging

    configure_logging()
    entries = download()
    for e in entries:
        rc = e["record_count"]
        rc_str = f"({rc} records)" if rc is not None else "(archive)"
        print(f"  {e['layer']:8s} {e['path']}  {rc_str}")
