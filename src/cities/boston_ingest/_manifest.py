"""
Provenance manifest + README writer for the Boston ingestion.

For every file in `data/boston/raw/**` and `data/boston/flood/**` produced
by the layer modules, the manifest records:

    layer, sub_layer, path, source_url, downloaded_utc, sha256,
    size_bytes, record_count, bbox

For ArcGIS-sourced entries (those carrying `arcgis_layer_id`), the
manifest ALSO records:

    server_count    — what `returnCountOnly=true` reports at manifest time
    count_match     — server_count == record_count
    server_extent   — the layer's reported extent at the time of check

This turns "I'm sure pagination got everything" into "the manifest
proves it." Especially important for FEMA NFHL after we hit the
underfetching bug that capped the first run at 987 of 1597 records.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Optional

# Lift GDAL's GeoJSON size cap so we can read the worst-case CRB scenario
# back without errors.
os.environ.setdefault("OGR_GEOJSON_MAX_OBJ_SIZE", "0")

import geopandas as gpd  # noqa: E402
import requests  # noqa: E402

from src.cities import boston  # noqa: E402

log = logging.getLogger(__name__)

MANIFEST_PATH = boston.RAW_DIR / "_manifest.json"
README_PATH = boston.RAW_DIR / "README.md"

DEFAULT_USER_AGENT = "NYU-CERA-flood-cascade-research/1.0 (tj2587@nyu.edu)"
SERVER_PROBE_TIMEOUT = 60


def _utc_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _bbox_of(path: Path) -> Optional[list[float]]:
    if path.suffix.lower() not in {".geojson", ".gpkg", ".shp"}:
        return None
    try:
        gdf = gpd.read_file(path)
        b = gdf.total_bounds
        if any((v != v) for v in b):  # NaN check
            return None
        return [float(b[0]), float(b[1]), float(b[2]), float(b[3])]
    except Exception as e:
        log.warning("Could not read bbox from %s: %s", path, e)
        return None


def _server_count(service_url: str, layer_id: int) -> Optional[int]:
    query_url = f"{service_url.rstrip('/')}/{layer_id}/query"
    params = {
        "where": "1=1",
        "geometry": f"{boston.BBOX[0]},{boston.BBOX[1]},{boston.BBOX[2]},{boston.BBOX[3]}",
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "returnCountOnly": "true",
        "f": "json",
    }
    try:
        resp = requests.get(
            query_url,
            params=params,
            headers={"User-Agent": DEFAULT_USER_AGENT},
            timeout=SERVER_PROBE_TIMEOUT,
        )
        resp.raise_for_status()
        return int(resp.json().get("count"))
    except Exception as e:
        log.warning("Server count probe failed for %s layer=%d: %s",
                    service_url, layer_id, e)
        return None


def _server_extent(service_url: str, layer_id: int) -> Optional[dict]:
    layer_url = f"{service_url.rstrip('/')}/{layer_id}"
    try:
        resp = requests.get(
            layer_url,
            params={"f": "json"},
            headers={"User-Agent": DEFAULT_USER_AGENT},
            timeout=SERVER_PROBE_TIMEOUT,
        )
        resp.raise_for_status()
        ext = resp.json().get("extent") or {}
        return {
            "xmin": ext.get("xmin"),
            "ymin": ext.get("ymin"),
            "xmax": ext.get("xmax"),
            "ymax": ext.get("ymax"),
            "wkid": (ext.get("spatialReference") or {}).get("latestWkid"),
        }
    except Exception as e:
        log.warning("Server extent probe failed for %s: %s", layer_url, e)
        return None


def _project_relative(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(boston.PROJECT_ROOT))
    except ValueError:
        return str(p)


def build_manifest(entries: list[dict]) -> list[dict]:
    """Compute SHA256 + server verification + bbox for each entry."""
    records: list[dict] = []
    for e in entries:
        path_str = e.get("path")
        if not path_str:
            continue
        path = Path(path_str)
        if not path.exists():
            log.warning("Manifest skip: %s does not exist", path)
            continue
        size = path.stat().st_size
        record: dict = {
            "layer": e.get("layer"),
            "sub_layer": e.get("sub_layer"),
            "path": _project_relative(path),
            "source_url": e.get("source_url"),
            "downloaded_utc": dt.datetime.fromtimestamp(
                path.stat().st_mtime, tz=dt.timezone.utc
            ).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "sha256": _sha256(path),
            "size_bytes": size,
            "record_count": e.get("record_count"),
            "bbox": _bbox_of(path),
        }
        layer_id = e.get("arcgis_layer_id")
        if layer_id is not None and e.get("source_url"):
            sc = _server_count(e["source_url"], int(layer_id))
            ext = _server_extent(e["source_url"], int(layer_id))
            record["arcgis_layer_id"] = int(layer_id)
            record["server_count"] = sc
            record["server_extent"] = ext
            record["count_match"] = (
                sc is not None
                and record["record_count"] is not None
                and sc == record["record_count"]
            )
        records.append(record)
    return records


def write_manifest(entries: list[dict]) -> Path:
    records = build_manifest(entries)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_utc": _utc_iso(),
        "project_root": str(boston.PROJECT_ROOT),
        "bbox": list(boston.BBOX),
        "n_records": len(records),
        "records": records,
    }
    with open(MANIFEST_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    log.info("Wrote manifest with %d record(s) → %s", len(records), MANIFEST_PATH)
    return MANIFEST_PATH


# -----------------------------------------------------------------------------
# README generator
# -----------------------------------------------------------------------------

README_BODY = """# Boston pilot — raw data ingestion

This folder is generated by `src/cities/boston_ingest/`.

Bbox: `(-71.20, 42.22, -70.92, 42.40)` (Boston metro + close suburbs)
CRS: every vector output is reprojected to **EPSG:4326** at save time.
Native source CRSes vary (EPSG:3857 for CRB, EPSG:26986 for MassGIS NFHL).

Per-file provenance (source URL, SHA256, server-vs-disk count match, bbox)
lives in [`_manifest.json`](_manifest.json). Run-by-run logs are under
[`_logs/`](_logs/).

## Regenerating

From the project root (`~/Desktop/RA/`):

```bash
python -m src.cities.boston_ingest                          # all layers
python -m src.cities.boston_ingest --layers subway,power    # subset
python -m src.cities.boston_ingest --force                  # force re-download
```

Per-layer modules are also callable standalone:

```bash
python -m src.cities.boston_ingest.subway
python -m src.cities.boston_ingest.power
# ... etc
```

## Layers

### subway — MBTA GTFS
Output: `subway/mbta_gtfs.zip`, `subway/stops.geojson`.
Filter: `location_type ∈ {0, 1}` ∩ Boston bbox ∩ `vehicle_type ≠ 3` (drop bus).
All 19 GTFS columns preserved including `parent_station`.

### power — HIFLD substations (Rutgers public mirror)
Output: `power/hifld_substations.geojson`. All 23 HIFLD columns preserved.
Schema note: HIFLD revision uses `MAX_INFER`/`MIN_INFER` as Y/N flags
(not kV); no `OPERATOR` column. NYC's existing file uses the same schema.

### fuel — Chelsea Creek bulk petroleum terminals + OSM
Outputs: `fuel/major_terminals.geojson`, `fuel/osm_stations.geojson`.
The terminals layer is hardcoded with operator-verified entries (EPA
NPDES permit URLs + Nominatim geocodes); 5 records. OSM ships
`amenity=fuel` ∪ `man_made=storage_tank` with `osm_type` distinguishing.

### water — CSO outfalls (best public proxy)
Outputs: `water/massdep_cso_outfalls.geojson`,
`water/epa_r1_cso_outfalls.geojson`,
`water/npdes_facilities_outfalls.geojson`, `water/_BWSC_GAP.md`.
MassDEP and EPA R1 cross-validate within ±1 per operating agency.
**BWSC pumping stations and tide gates are NOT covered** — see
[`water/_BWSC_GAP.md`](water/_BWSC_GAP.md) for the gap rationale.

### telecom — FCC Antenna Structure Registration
Output: `telecom/fcc_asr_towers.geojson`. All 21 ASR columns preserved.
OpenCelliD is key-gated; if `OPENCELLID_API_KEY` is not set at run time,
`telecom/opencellid_TODO.txt` is written with backfill instructions.

### flood (CRB primary) — Climate Ready Boston SLR scenarios
Outputs: 6 files under `../flood/` named `crb_slr<NN>_aep<NN>.geojson`
(filenames pinned to `CRB_SCENARIOS` in `src/cities/boston.py`).
Each scenario carries `scenario_name`, `slr_in`, `aep_pct` columns.

### flood (FEMA cross-reference) — FEMA NFHL via MassGIS
Output: `../flood/fema_nfhl.geojson`. 1,597 polygons, 25 columns.
`DEPTH` populated only on AO/AH zones.

## Known limitations index

| Sidecar | Layer | Issue | Status |
| --- | --- | --- | --- |
| [`water/_BWSC_GAP.md`](water/_BWSC_GAP.md) | water | BWSC pumping/tide-gate/full-outfall layers not publicly served | partial coverage shipped; pending data-sharing agreement |
| [`telecom/opencellid_TODO.txt`](telecom/opencellid_TODO.txt) | telecom | `OPENCELLID_API_KEY` not set on the run | backfill on demand |
| [`../flood/_CRB_KNOWN_LIMITATIONS.md`](../flood/_CRB_KNOWN_LIMITATIONS.md) | flood (CRB) | (a) no depth attribute, (b) eastern-bbox clip on 21"/36" 1pct | (a) deferred to fragility module with trigger condition; (b) shipped as-is — diagnostic confirmed 0 cascade label inversions |

## Pivots from the original handoff

| Original source | Why pivoted | Replacement |
| --- | --- | --- |
| EIA Petroleum Product Terminals FeatureServer | EIA canonical endpoint auth-gated since NYC's last pull (HTTP 499 Token Required) | 5 hardcoded Chelsea Creek terminals with EPA NPDES permit URLs + OSM `amenity=fuel`/`man_made=storage_tank` union |
| BWSC pumping / outfalls / tide gates | Not publicly served on data.boston.gov, BWSC's ArcGIS org, BostonGIS, or MassGIS | MassDEP CSOs + EPA R1 CSO + EPA NPDES facilities (best public outfall proxy; pump stations + tide gates remain a gap) |
| HIFLD Cellular Towers | Surviving public layer is a single FCC ULS Cellular Service band-class extract returning ~3 records for Boston | FCC Antenna Structure Registration (~85 macro structures, 21 columns) |

## Provenance notes

- **CRB source polygons have invalid topology (self-intersections).**
  Point-in-polygon labeling is unaffected, but area-based math on these
  layers is unreliable — clean with `make_valid()` and union in a
  projected CRS (EPSG:26986 for Massachusetts) if accurate area is
  needed.

- **`MAX_VOLT` / `MIN_VOLT` do not exist** in the current HIFLD
  substations data, despite being referenced in NYC's
  `src/data_acquisition/download_power.py` docstring (which describes
  the pre-revision HIFLD schema). The actual columns are `MAX_INFER` /
  `MIN_INFER` as Y/N flags.

- **Same physical facility cross-listed in multiple layers** —
  Sunoco / Energy Transfer East Boston Terminal appears in
  `fuel/major_terminals.geojson` (operator-verified) AND
  `water/npdes_facilities_outfalls.geojson` (NPDES permit MA0004006).
  MBTA and MWRA each appear as `ENTITY` in
  `telecom/fcc_asr_towers.geojson` AND as their own facility nodes
  elsewhere (subway, water). Joinable by NPDES permit ID, geographic
  proximity (~10 m), or operator-name match.
"""


def write_readme() -> Path:
    README_PATH.parent.mkdir(parents=True, exist_ok=True)
    README_PATH.write_text(README_BODY)
    log.info("Wrote README → %s", README_PATH)
    return README_PATH
