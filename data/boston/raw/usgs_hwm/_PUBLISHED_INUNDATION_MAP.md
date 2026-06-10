# Published USGS Jan-2018 inundation map — discovery note

A published, peer-reviewed inundation extent for the January 4, 2018 nor'easter
exists and is used as the Phase-3 node-level "real wet zone" (better ground truth
than interpolating the 35 sparse high-water marks ourselves).

- **Publication:** Lombard et al. (2021), USGS SIR 2021-5109 — "Documentation and
  mapping of flooding from the January and March 2018 nor'easters in coastal New England."
  https://pubs.usgs.gov/publication/sir20215109
- **Data release (ver. 2.0, Nov 2021):**
  https://www.usgs.gov/data/data-and-shapefiles-used-document-floods-associated-january-and-march-2018-noreasters-coastal
- **ScienceBase parent item:** 6010090fd34e162231fecf90
- **ScienceBase child (shapefiles):** 60100f99d34e162231fecfa8
  - `New England January 2018 High-Water Mark Coastal Inundation Map.zip` (~35 MB) — the Jan-2018 wet zone we use.
  - `New England USGS 100-year Stillwater Coastal Inundation Map.zip` (~36 MB) — secondary 1%-AEP comparator.
  - `New England January and March 2018 Coastal Storm Event Data Points.zip` — the point survey (= our STN HWMs).

## Honest framing (must carry into the writeup)

The Jan-2018 inundation polygon is USGS's professional **interpolation of the same
high-water marks** onto a DEM. So "HWMs fall inside the polygon" is near-tautological,
and the points-in-CRB-extent (primary) and nodes-in-polygon (secondary) metrics both
rest on one survey. The genuinely INDEPENDENT anchors are:
  1. the NOAA tide gauge still-water peak (2.944 m NAVD88, record-matching), and
  2. the documented Aquarium Blue Line station closure on Jan 4, 2018.

## A-priori expectation (not just a ranking)

USGS frequency analysis rates Boston's Jan-2018 stillwater (9.66 ft NAVD88) at a
**1-2% annual exceedance probability (50-100-yr recurrence)**. The model's near-term
1% scenario is `slr09_aep01`, so the real event SHOULD best-match `slr09_aep01`, with
the USGS polygon plausibly sitting at-or-just-inside that extent. If a higher scenario
fits better, that is a finding (CRB near-term depths under-calibrated), not a failure.

## Decision

Use the published polygon as the node-level wet zone. Do NOT pull a 3DEP DEM / do NOT
compute depth-RMSE: the model assigns a flat 0.30 m to every flooded node, so there is
no depth variation to score against.
