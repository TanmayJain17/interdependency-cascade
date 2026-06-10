"""
USGS high-water marks for the January 2018 nor'easter (real surveyed flood elevations).

These are the field-surveyed peak water elevations USGS recorded after the
January 4, 2018 nor'easter — the authoritative public ground truth behind
SIR 2020-5048 ("Total water level data from the January and March 2018
nor'easters for coastal areas of New England"). Unlike the NOAA gauge (a single
still-water point), the HWMs are spatially distributed total-water-level marks
(they include wave runup/setup), so they tell us WHERE the water actually reached.

Acquisition is discovery-first against the USGS Short-Term Network (STN) flood-event
web service, with a documented local-file fallback:

    PRIMARY  STN Services REST API. The January 2018 event is resolved by name
             ("2018 January Extratropical Cyclone", event_id 208) rather than a
             blind hardcode — the id is verified against the live event name, and
             vertical-datum codes are resolved from the live lookup table.
    FALLBACK if STN is unreachable / unrecognizable, drop the SIR 2020-5048
             high-water-mark CSV at data/boston/raw/usgs_hwm/ and re-run; the CSV
             schema is inspected and printed before parsing.

Filtering for the Boston analysis keeps marks that are inside the bbox and Coastal.
Elevation (elev_ft, NAVD 88) is converted to meters; marks without a surveyed
elevation are retained for the point-containment test but flagged (no elevation).

Outputs (data/boston/raw/usgs_hwm/):
    event208_hwms_raw.json            full event pull (provenance)
    boston_2018_hwm_points.gpkg       filtered in-bbox coastal points (EPSG:4326)
    boston_2018_hwm_points.csv        same, human-readable
"""
from __future__ import annotations

import json
import logging

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Point

from src.cities import boston

log = logging.getLogger(__name__)

LAYER = "usgs_hwm"
OUT_DIR = boston.RAW_DIR / "usgs_hwm"

STN_BASE = "https://stn.wim.usgs.gov/STNServices"
EVENT_ID = 208
EVENT_NAME_EXPECT = "2018 January Extratropical Cyclone"
M_PER_FT = 0.3048
TIMEOUT = 120
HDR = {"Accept": "application/json"}

# Fallback datum map if the live lookup is unreachable (verified against the
# live VerticalDatums.json on acquisition).
_FALLBACK_VDATUM = {1: "local control point", 2: "NAVD88", 3: "Above Ground Level",
                    4: "NGVD29", 6: "PRVD02", 7: "VIVD09", 8: "IGLD85"}


def _get(path: str):
    return requests.get(f"{STN_BASE}/{path}", headers=HDR, timeout=TIMEOUT)


def _resolve_event() -> int:
    """Confirm EVENT_ID still maps to the January 2018 nor'easter by name."""
    events = _get("Events.json").json()
    by_id = {e["event_id"]: e for e in events}
    ev = by_id.get(EVENT_ID)
    if ev and EVENT_NAME_EXPECT.lower() in (ev.get("event_name") or "").lower():
        log.info("STN event %d confirmed: %s (%s)", EVENT_ID, ev["event_name"],
                 ev.get("event_start_date"))
        return EVENT_ID
    # Name-first search if the id drifted.
    for e in events:
        nm = (e.get("event_name") or "").lower()
        if "2018 january" in nm and "cyclone" in nm:
            log.warning("EVENT_ID drifted; matched by name -> id %d (%s)",
                        e["event_id"], e["event_name"])
            return e["event_id"]
    raise RuntimeError("Could not identify the January 2018 nor'easter in STN Events.")


def _vdatum_map() -> dict:
    try:
        vd = _get("VerticalDatums.json").json()
        return {d["datum_id"]: d["datum_name"] for d in vd}
    except Exception as exc:  # noqa: BLE001
        log.warning("VerticalDatums lookup failed (%s); using fallback map", exc)
        return dict(_FALLBACK_VDATUM)


def _fetch_stn() -> tuple[list[dict], dict]:
    event_id = _resolve_event()
    vdmap = _vdatum_map()
    hwms = _get(f"Events/{event_id}/HWMs.json").json()
    log.info("STN returned %d HWMs for event %d", len(hwms), event_id)
    log.info("HWM fields: %s", list(hwms[0].keys()))
    (OUT_DIR / "event208_hwms_raw.json").write_text(json.dumps(hwms, indent=2))
    return hwms, vdmap


def _fetch_local() -> tuple[list[dict], dict]:
    """Fallback: parse a user-provided SIR 2020-5048 CSV dropped in OUT_DIR."""
    csvs = sorted(OUT_DIR.glob("*.csv"))
    csvs = [c for c in csvs if c.name != "boston_2018_hwm_points.csv"]
    if not csvs:
        raise FileNotFoundError(
            "STN unreachable and no fallback CSV found. Download the SIR 2020-5048 "
            f"high-water-mark data release and place the CSV in {OUT_DIR}/, then re-run."
        )
    src = csvs[0]
    df = pd.read_csv(src)
    log.info("Fallback CSV %s columns: %s", src.name, list(df.columns))
    # Normalise to the STN field names this module expects downstream.
    ren = {"latitude": "latitude_dd", "longitude": "longitude_dd",
           "elevation_ft": "elev_ft", "elev_ft": "elev_ft",
           "environment": "hwm_environment", "verticalDatumName": "vdatum_name"}
    df = df.rename(columns={k: v for k, v in ren.items() if k in df.columns})
    return df.to_dict("records"), {}


def _to_gdf(hwms: list[dict], vdmap: dict) -> gpd.GeoDataFrame:
    minlon, minlat, maxlon, maxlat = boston.BBOX
    rows = []
    for h in hwms:
        lon, lat = h.get("longitude_dd"), h.get("latitude_dd")
        if lon is None or lat is None:
            continue
        if not (minlon <= lon <= maxlon and minlat <= lat <= maxlat):
            continue
        if h.get("hwm_environment") != "Coastal":
            continue
        vname = h.get("vdatum_name") or vdmap.get(h.get("vdatum_id"))
        elev_ft = h.get("elev_ft")
        elev_m = float(elev_ft) * M_PER_FT if elev_ft not in (None, "") else None
        rows.append({
            "hwm_id": h.get("hwm_id"),
            "lat": lat, "lon": lon,
            "elev_ft": elev_ft,
            "elev_m_navd88": round(elev_m, 4) if elev_m is not None else None,
            "vdatum": vname,
            "environment": h.get("hwm_environment"),
            "quality_id": h.get("hwm_quality_id"),
            "stillwater": h.get("stillwater"),
            "height_above_gnd_ft": h.get("height_above_gnd"),
            "desc": (h.get("hwm_locationdescription") or "")[:120],
            "geometry": Point(lon, lat),
        })
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    return gdf


def download(force: bool = False) -> list[dict]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gpkg = OUT_DIR / "boston_2018_hwm_points.gpkg"
    csv = OUT_DIR / "boston_2018_hwm_points.csv"

    if gpkg.exists() and not force:
        log.info("%s present — skipping STN pull (use force=True)", gpkg.name)
        gdf = gpd.read_file(gpkg)
    else:
        try:
            hwms, vdmap = _fetch_stn()
            path = "STN web service (live)"
        except Exception as exc:  # noqa: BLE001
            log.warning("STN acquisition failed (%s) — trying local fallback", exc)
            hwms, vdmap = _fetch_local()
            path = "local fallback CSV"
        gdf = _to_gdf(hwms, vdmap)
        log.info("Acquisition path: %s", path)
        if gpkg.exists():
            gpkg.unlink()
        gdf.to_file(gpkg, driver="GPKG")
        gdf.drop(columns="geometry").to_csv(csv, index=False)
        log.info("Saved %d in-bbox coastal HWMs -> %s", len(gdf), gpkg)

    navd = gdf[gdf["vdatum"] == "NAVD88"]
    with_elev = navd[navd["elev_m_navd88"].notna()]
    log.info("In-bbox coastal: %d | NAVD88: %d | NAVD88 with elevation: %d",
             len(gdf), len(navd), len(with_elev))
    if len(with_elev):
        log.info("Elevation range: %.2f - %.2f m NAVD88 (%.1f - %.1f ft)",
                 with_elev["elev_m_navd88"].min(), with_elev["elev_m_navd88"].max(),
                 with_elev["elev_ft"].astype(float).min(), with_elev["elev_ft"].astype(float).max())

    return [{
        "layer": LAYER,
        "path": str(gpkg),
        "source_url": f"{STN_BASE}/Events/{EVENT_ID}/HWMs.json",
        "record_count": int(len(gdf)),
        "sub_layer": "high_water_marks",
        "n_navd88": int(len(navd)),
        "n_navd88_with_elev": int(len(with_elev)),
    }]


if __name__ == "__main__":
    from src.cities.boston_ingest._logging import configure as configure_logging

    configure_logging()
    entries = download()
    for e in entries:
        print(f"  {e['layer']}  {e['path']}")
        print(f"    in-bbox coastal HWMs: {e['record_count']}  | NAVD88: {e['n_navd88']}  "
              f"| NAVD88 w/ elevation: {e['n_navd88_with_elev']}")
