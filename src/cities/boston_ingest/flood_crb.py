"""
C6 — Climate Ready Boston (CRB) Sea Level Rise inundation polygons.

Six scenarios = 3 SLR levels (9", 21", 36") × 2 AEP levels (10%, 1%).
Filenames are pinned to CRB_SCENARIOS in src/cities/boston.py so the
multi-scenario runner can index by scenario name without ambiguity.

Output files:
    data/boston/flood/crb_slr09_aep10.geojson
    data/boston/flood/crb_slr09_aep01.geojson
    data/boston/flood/crb_slr21_aep10.geojson
    data/boston/flood/crb_slr21_aep01.geojson
    data/boston/flood/crb_slr36_aep10.geojson
    data/boston/flood/crb_slr36_aep01.geojson
    data/boston/flood/_CRB_DEPTH_LIMITATION.md   (decision-framework sidecar)

Each scenario layer is a single unioned MultiPolygon on the service side —
same packaging as NYC's DEP scenarios. We fetch one OBJECTID per layer.

DEPTH LIMITATION: CRB releases EXTENT polygons only (FID + Shape only,
no depth attribute on any layer). See the sidecar README for the
full decision framework and the trigger condition for revisit.

Post-download checks:
    * Monotone area ordering across the 6 scenarios (computed in EPSG:26986
      meters for area-accurate projection over Massachusetts).
    * Coverage of 7 named flood-vulnerable landmarks in the worst case
      (slr36_aep01): Logan, Back Bay, Seaport / Fort Point, East Boston
      waterfront, Charlestown waterfront, Chelsea Creek, Dorchester Bay.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

# Lift GDAL's GeoJSON max-feature-size cap so we can read back the worst-case
# scenario (slr36_aep01 is a single ~50 MB MultiPolygon). Mirrors the same
# guard NYC's fetch_dep_flood_maps.py sets for the Extreme 2080 layer.
os.environ.setdefault("OGR_GEOJSON_MAX_OBJ_SIZE", "0")

import geopandas as gpd  # noqa: E402  (must come AFTER the env var is set)
from shapely.geometry import Point  # noqa: E402

from src.cities import boston
from src.data_ingest import arcgis

log = logging.getLogger(__name__)

LAYER = "flood_crb"
OUT_DIR = boston.FLOOD_DIR

SERVICE_URL = (
    "https://services.arcgis.com/sFnw0xNflSi8J0uh/arcgis/rest/services/"
    "Climate_Ready_Boston_Sea_Level_Rise_Inundation/FeatureServer"
)

# Layer IDs in the service map to CRB_SCENARIOS by (slr_in, aep_pct).
# Service-side layer ordering is: 21,21,21HT, 36,36,36HT, 9,9,9HT.
LAYER_ID_FOR_SCENARIO: dict[tuple[int, int], int] = {
    (21, 10): 0,
    (21, 1):  1,
    (36, 10): 3,
    (36, 1):  4,
    (9, 10):  6,
    (9, 1):   7,
}

# Landmark coverage check: representative low-lying points the worst-case
# scenario (36" SLR + 1% AEP) MUST cover. Each is either a documented
# waterfront/tidal-flat location or a verified physical asset coordinate
# from our own layers (the 4 fuel terminals around Chelsea Creek).
LANDMARK_POINTS: list[tuple[str, float, float]] = [
    ("Logan Airport (apron near Chelsea Creek)",   42.3680, -71.0140),
    ("Back Bay (Charles River esplanade)",         42.3540, -71.0820),
    ("Seaport / Fort Point Channel",               42.3470, -71.0470),
    ("East Boston waterfront",                     42.3790, -71.0380),
    ("Charlestown waterfront (Navy Yard)",         42.3730, -71.0540),
    ("Gulf Oil Chelsea Terminal",                  42.3968, -71.0210),
    ("Sunoco / Energy Transfer East Boston",       42.3814, -71.0253),
    ("Chelsea Sandwich Terminal",                  42.3866, -71.0445),
    ("Dorchester Bay (Carson Beach)",              42.3275, -71.0395),
]

# CRB does NOT publish depth. This sidecar documents the four paths we
# evaluated and the trigger condition for revisiting the choice.
DEPTH_LIMITATION_DOC = """# CRB known limitations — decision framework

Two limitations of the canonical Climate Ready Boston source are
documented here. Each section follows the same shape: what the limit
is, why we accepted it for this snapshot, the alternative paths we
evaluated, and the trigger condition that would cause us to revisit.

---

## LIMITATION 1 — Extent-only, no depth attribute

### What CRB provides
The 6 SLR × AEP scenarios at
`services.arcgis.com/sFnw0xNflSi8J0uh/.../Climate_Ready_Boston_Sea_Level_Rise_Inundation/FeatureServer`
are **flood extent polygons only**: attributes are `FID`, `Shape_Length`,
`Shape_Area`. No depth, no water-surface elevation, no scenario tag in
the feature payload. Confirmed by direct schema and feature inspection
on all 9 service layers (the 6 we use + 3 high-tide bonus).

### What CRB does NOT provide
- Per-node water depth at infrastructure locations.
- Continuous water-surface elevation surfaces.
- Anything that distinguishes "ankle-deep nuisance" from "fully submerged."

### NYC parity context
NYC's DEP flood layers in `data/raw/flood/` are similarly extent +
categorical (`Flooding_Category` ∈ {Nuisance, Deep and Contiguous,
Future High Tides}). The existing pipeline already handles
"category → implicit depth" mapping. CRB extent slots into that same
pattern. Diverging here (Boston gets depth, NYC doesn't) would muddy
cross-city model comparison.

### Four depth-addition paths evaluated

| # | Path | Effort | Trade-offs |
| - | --- | --- | --- |
| 1 | **Ship CRB extent-only (this folder).** | 0 hr | NYC parity preserved. No depth signal for fragility. |
| 2 | Pivot to MassDOT FHWA depth-threshold rasters (D1 / Dpt5 / Dpt1 × 2030 / 2050 / 2070). | 60–90 min | Real depth signal. Loses CRB scenario structure; rescaling against handoff. |
| 3 | Ship CRB extent + MassDOT depth raster as sidecar. | ~90 min | Keeps CRB scenario names AND adds depth. Scenario mismatch is fatal: CRB "36" SLR + 1% AEP" does not cleanly map to MassDOT "2070 D1" — wrong depth is worse than no depth. |
| 4 | **Derive depth from CRB extent + MassGIS LiDAR DEM + scenario-specific WSE.** | 2–3 hr | Depth at full LiDAR resolution matching CRB scenarios exactly. Requires DEM ingestion, raster math, NAVD88 WSE lookup per scenario. Strongest research outcome. |

### Trigger to revisit
If the fragility module's continuous-depth response materially affects
cascade predictions vs. categorical wet/dry treatment, return to
**Option 4 (LiDAR DEM derivation)**. Concretely:
- During fragility-module wiring, run an ablation comparing
  (a) categorical wet/dry from CRB extent vs.
  (b) a depth proxy (e.g., constant 0.5 m within extent + DEM-based
  decay near edges).
- If predictions differ materially between (a) and (b), cascade is
  depth-sensitive → go to Option 4.
- If predictions agree within tolerance, categorical wet/dry suffices →
  keep this folder as-is.

---

## LIMITATION 2 — Cross-AEP inversion / eastern-bbox clipping

### What we observed
Computed area in EPSG:26986 (NAD83 / Massachusetts Mainland, area-accurate
projection):

| Scenario | Area (km²) |
| --- | --- |
| slr09_aep10 | 80.42 |
| slr09_aep01 | 107.69 |
| slr21_aep10 | 119.93 |
| **slr21_aep01** | **110.62**  ← smaller than 21" 10pct |
| slr36_aep10 | 122.47 |
| **slr36_aep01** | **117.96**  ← smaller than 36" 10pct |

Cross-AEP at fixed SLR should obey **area(1pct) ≥ area(10pct)** because a
1% storm event is a strict superset of 10% events. At 9" SLR this
holds. At 21" and 36" SLR it inverts.

Smoking gun is in the eastern bounds:

| Layer | eastern max_lon |
| --- | --- |
| All 10pct layers (and both 9" layers) | -70.872 |
| **slr21_aep01 and slr36_aep01** | **-70.923** (~4.5 km clipped) |

The 21" and 36" 1pct layers are missing ~4.5 km of eastern coverage
(Winthrop / Deer Island / outer harbor) that the corresponding 10pct
layers include. This is a publication artifact of the canonical
BostonGIS service, **not** a download or parsing error — layer IDs are
verified by inspecting layer JSON metadata, and our 10pct/1pct
assignment matches the source layer names exactly.

### Within-AEP monotonicity IS correct
Holding AEP fixed and increasing SLR, area grows monotonically:
- 10pct sequence (9" → 21" → 36"):  80.4 → 119.9 → 122.5
- 1pct sequence  (9" → 21" → 36"): 107.7 → 110.6 → 118.0

So the SLR axis is reliable in the source. Only cross-AEP at 21" and 36"
SLR is broken.

### Three correction paths evaluated

| # | Path | Effort | Trade-offs |
| - | --- | --- | --- |
| 1 | **Ship as-is, document the artifact (this folder).** | 0 hr | Preserves source provenance — reviewers can verify against the BostonGIS service. NYC parity (we don't post-process NYC DEP layers either). Downstream model has to be aware of the inversion. |
| 2 | Ship corrected `1pct := union(1pct_raw, 10pct_raw)` and keep raw as sidecar. | ~30 min | Mathematically enforces area(1pct) ≥ area(10pct). Justified by definition (1% event is a superset of 10%). Cost: we are modifying the source; a reviewer will ask why and we must defend it. |
| 3 | Wait for upstream fix — file an issue with BostonGIS / Boston Environment Dept. | unknown | Right thing to do for the next release of CRB. Does not unblock this week. |

### Pre-Phase-D diagnostic (run 2026-06-03)

Two questions answered before deciding whether to ship as-is or correct:

**(a) How many infrastructure nodes fall in the truncated strip
(lon ∈ (-70.923, -70.872))?**

| Layer | Total | In strip |
| --- | ---:| ---:|
| subway | 505 | 0 |
| power | 34 | **1** (EAST WEYMOUTH substation, Weymouth) |
| fuel terminals | 5 | 0 |
| fuel OSM | 377 | 0 |
| water massdep | 45 | 0 |
| water epa_r1 | 47 | 0 |
| water npdes_fac | 9 | 0 |
| telecom | 85 | **1** (HINGHAM SHIPYARD LLC tower, Hingham) |
| **TOTAL** | | **2** |

**(b) Of those 2 nodes, how many would actually be label-inverted (wet in
10pct, dry in 1pct at same SLR)?**

Both nodes sit on dry land at the southern edge of the bbox (south-shore
towns). Neither is inside the 10pct OR the 1pct flood polygon at any SLR
level. So neither is label-inverted: both are dry under both scenarios.

**Conclusion: zero cascade labels are inverted. Source-canonical 1pct
layers are shipped as-is.** The truncation artifact remains a known
limitation of the upstream publication but is non-load-bearing for our
infrastructure inventory.

The diagnostic also confirmed (via server.count, server.objectIds, and
the layer's reported extent at xmax=-7895022 in EPSG:3857 ≈ -70.923 in
EPSG:4326) that the eastern clip is published by BostonGIS at source,
not introduced by our download.

### Helper-level fix that came out of this investigation

While diagnosing CRB pagination as a hypothesis (it wasn't the cause
here — CRB has 1 OID per layer), we discovered that
`src/data_ingest/arcgis.py`'s `query_layer()` had been advancing
`offset` by `page_size` instead of by the actual count returned per
page. That under-fetched FEMA NFHL (got 987 records, source has 1597).
Now fixed by `offset += n`. The CRB downloads were never affected
because each layer holds a single OID.

### Trigger to revisit (still applicable for future Boston pilots)

If a future addition to any infrastructure layer (e.g. broader bbox,
new sensor network) places a node inside the truncated strip AND inside
the 10pct flood polygon at 21" or 36" SLR, the inversion would
mislabel that node. At that point: call
`_apply_aep_inversion_correction()` (already implemented in
`flood_crb.py`) to swap each affected 1pct layer for
`union(1pct_raw, 10pct_raw)` and preserve `*_raw.geojson` sidecars.

In parallel, file an issue with BostonGIS pointing at the eastern-clip
artifact so the upstream layer eventually gets corrected at source.

---

## Provenance

Source: `services.arcgis.com/sFnw0xNflSi8J0uh/.../Climate_Ready_Boston_Sea_Level_Rise_Inundation/FeatureServer`
Owner: BostonGIS (City of Boston official GIS)
Methodology: HDR / Kleinfelder modeling, scenarios published by Boston
Environment Department through the Climate Ready Boston initiative
(https://www.boston.gov/departments/environment/climate-ready-boston).
"""


def _save_one_scenario(slr_in: int, aep_pct: int, name: str, force: bool) -> dict:
    out_path = OUT_DIR / f"crb_{name}.geojson"
    if out_path.exists() and not force:
        size_kb = out_path.stat().st_size / 1024
        log.info("%s already present (%.1f KB) — skipping", out_path.name, size_kb)
        n = len(gpd.read_file(out_path))
        return {
            "layer": LAYER,
            "path": str(out_path),
            "source_url": SERVICE_URL,
            "record_count": n,
            "sub_layer": name,
            "slr_in": slr_in,
            "aep_pct": aep_pct,
        }

    layer_id = LAYER_ID_FOR_SCENARIO[(slr_in, aep_pct)]
    log.info("CRB %s: SLR=%d\" AEP=%d%%, ArcGIS layer %d", name, slr_in, aep_pct, layer_id)
    # One huge MultiPolygon per layer — fetch by OBJECTID
    gdf = arcgis.query_by_object_id(
        SERVICE_URL,
        layer_id=layer_id,
        timeout=900,
    )
    n = len(gdf)
    log.info("%s returned %d feature(s); cols: %s", name, n, list(gdf.columns))

    # Tag every row with the scenario name so downstream concatenation is unambiguous
    gdf["scenario_name"] = name
    gdf["slr_in"] = slr_in
    gdf["aep_pct"] = aep_pct

    if out_path.exists():
        out_path.unlink()
    gdf.to_file(out_path, driver="GeoJSON")
    log.info("Saved %s → %s", name, out_path)
    return {
        "layer": LAYER,
        "path": str(out_path),
        "source_url": SERVICE_URL,
        "record_count": n,
        "sub_layer": name,
        "slr_in": slr_in,
        "aep_pct": aep_pct,
    }


def _apply_aep_inversion_correction(entries: list[dict]) -> None:
    """Fix the source-side eastern-bbox clip on slr21_aep01 and slr36_aep01.

    The canonical CRB layers for the 1% AEP scenarios at 21" and 36" SLR
    are published with ~4.5 km of eastern coverage missing relative to
    the corresponding 10% AEP layers. This violates the physical
    invariant area(1pct) >= area(10pct) and causes 2 infrastructure nodes
    (1 power substation in East Weymouth, 1 telecom tower in Hingham
    Shipyard) to be incorrectly labelled "not flooded under worst case"
    when they ARE flooded under the milder 10pct scenario.

    Correction: replace each affected 1pct layer with the union of itself
    and the corresponding 10pct layer at the same SLR. This is the
    mathematically required behavior (a 1% storm event is a superset of
    10% events). Originals are preserved as `*_raw.geojson` sidecars.

    Mutates entries in place to keep the post-correction paths visible
    to downstream callers.
    """
    by_name: dict[str, dict] = {e["sub_layer"]: e for e in entries if e.get("sub_layer", "").startswith("slr")}
    pairs = [("slr21_aep01", "slr21_aep10"), ("slr36_aep01", "slr36_aep10")]
    for one_pct_name, ten_pct_name in pairs:
        one_pct = by_name.get(one_pct_name)
        ten_pct = by_name.get(ten_pct_name)
        if one_pct is None or ten_pct is None:
            continue
        one_pct_path = Path(one_pct["path"])
        ten_pct_path = Path(ten_pct["path"])
        raw_path = one_pct_path.with_name(one_pct_path.stem + "_raw.geojson")

        # If a previous run already produced the corrected file, the raw
        # sidecar will exist. In that case the canonical path already
        # holds the union, so don't re-apply.
        if raw_path.exists():
            log.info("%s already corrected (raw sidecar present) — skipping", one_pct_name)
            continue

        log.info("Applying AEP-inversion correction to %s", one_pct_name)
        g_one = gpd.read_file(one_pct_path)
        g_ten = gpd.read_file(ten_pct_path)

        # Preserve raw, then write union to the canonical path
        one_pct_path.rename(raw_path)
        log.info("  preserved raw → %s", raw_path)

        union_geom = g_one.geometry.union_all().union(g_ten.geometry.union_all())
        corrected = gpd.GeoDataFrame(
            {
                "FID": [1],
                "scenario_name": [one_pct_name],
                "slr_in": [g_one.iloc[0].get("slr_in")],
                "aep_pct": [g_one.iloc[0].get("aep_pct")],
                "correction_applied": ["union_with_10pct_at_same_slr"],
            },
            geometry=[union_geom],
            crs="EPSG:4326",
        )
        corrected.to_file(one_pct_path, driver="GeoJSON")
        log.info("  wrote corrected → %s (raw preserved at %s)", one_pct_path, raw_path)


def _verify_monotone_and_coverage(entries: list[dict]) -> None:
    """Compute area-accurate km² per scenario and check the ordering + landmark coverage."""
    # Order entries by (slr_in, descending aep_pct) so the sequence is:
    #   9/10, 9/1, 21/10, 21/1, 36/10, 36/1
    ordered = sorted(
        [e for e in entries if e.get("sub_layer", "").startswith("slr")],
        key=lambda e: (e["slr_in"], -e["aep_pct"]),
    )
    areas_km2: dict[str, float] = {}
    worst_gdf: gpd.GeoDataFrame | None = None

    for e in ordered:
        gdf = gpd.read_file(e["path"])
        # Reproject to NAD83 / Massachusetts Mainland for accurate area
        gdf_mass = gdf.to_crs(epsg=26986)
        area_km2 = float(gdf_mass.geometry.area.sum() / 1e6)
        areas_km2[e["sub_layer"]] = area_km2
        log.info("  %s: %.2f km²", e["sub_layer"], area_km2)
        if e["sub_layer"] == "slr36_aep01":
            worst_gdf = gdf

    seq = list(areas_km2.values())
    is_monotone = all(seq[i] <= seq[i + 1] for i in range(len(seq) - 1))
    if is_monotone:
        log.info("Monotone area ordering CONFIRMED across 6 scenarios.")
    else:
        log.error("Monotone area ordering FAILED. Sequence: %s", areas_km2)

    if worst_gdf is None:
        log.warning("Worst-case scenario (slr36_aep01) not in output; skipping landmark check.")
        return

    # Landmark coverage in worst case
    worst_union = worst_gdf.geometry.union_all()
    log.info("Landmark coverage in worst-case (slr36_aep01):")
    misses: list[str] = []
    for nm, lat, lon in LANDMARK_POINTS:
        pt = Point(lon, lat)
        covered = worst_union.contains(pt) or worst_union.intersects(pt)
        mark = "✓" if covered else "✗"
        log.info("  %s  %s  (%.4f, %.4f)", mark, nm, lat, lon)
        if not covered:
            misses.append(nm)
    if misses:
        log.warning("Landmarks NOT covered in worst case: %s", misses)
    else:
        log.info("All %d landmarks covered in worst-case scenario.", len(LANDMARK_POINTS))


def _write_limitation_doc() -> dict:
    out_path = OUT_DIR / "_CRB_KNOWN_LIMITATIONS.md"
    out_path.write_text(DEPTH_LIMITATION_DOC)
    log.info("Wrote CRB known-limitations decision framework → %s", out_path)
    return {
        "layer": LAYER,
        "path": str(out_path),
        "source_url": None,
        "record_count": None,
        "sub_layer": "_limitation_doc",
    }


def download(force: bool = False) -> list[dict]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []
    for sc in boston.CRB_SCENARIOS:
        entries.append(_save_one_scenario(sc["slr_in"], sc["aep_pct"], sc["name"], force))
    entries.append(_write_limitation_doc())

    # Note: an _apply_aep_inversion_correction helper exists in this module
    # but is not called by default. The pre-Phase-D diagnostic showed that
    # while 2 infrastructure nodes (East Weymouth substation, Hingham
    # Shipyard tower) fall geographically inside the truncated eastern
    # strip, neither node is inside any CRB flood polygon — they sit on
    # dry land in the south-shore towns at the edge of the bbox. So no
    # cascade label would actually flip between the 10pct and 1pct
    # scenarios at the same SLR. The source-canonical 1pct layers are
    # therefore shipped as-is. See _CRB_KNOWN_LIMITATIONS.md for the
    # diagnostic trail and the trigger that would activate the
    # correction (a node being labelled wet@10pct but dry@1pct at the
    # same SLR).

    # Post-download verification (best-effort; never raises)
    try:
        _verify_monotone_and_coverage(entries)
    except Exception as e:
        log.error("Post-download verification failed: %s", e)

    return entries


if __name__ == "__main__":
    from src.cities.boston_ingest._logging import configure as configure_logging

    configure_logging()
    entries = download()
    for e in entries:
        rc = e.get("record_count")
        rc_str = f"({rc} records)" if rc is not None else "(notice)"
        print(f"  {e['layer']}  [{e.get('sub_layer','')}]  {e.get('path')}  {rc_str}")
