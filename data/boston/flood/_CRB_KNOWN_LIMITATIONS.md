# CRB known limitations — decision framework

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
