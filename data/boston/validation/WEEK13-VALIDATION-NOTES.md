# Week 13 — Real-data hazard validation (Boston, January 4 2018 nor'easter)

First validation of the Boston pipeline against a **real historic event** instead of synthetic
Climate Ready Boston (CRB) extents. The January 4, 2018 nor'easter set Boston's all-time record
water level and flooded the same coastal zones the model flags. We ingested real NOAA observed
water levels and real USGS surveyed flood elevations + the published USGS inundation polygon, then
tested whether the model's flood footprint lands where the water actually went.

**Lead with the model-INDEPENDENT geometry result.** Nothing here retrains the model or scores
node-level cascade accuracy. The on-branch model is the known-leaky v1; any model-dependent number
is DIRECTIONAL. The strong results are hazard geometry, which does not depend on the cascade GNN.

---

## 1. Data sources (exact)

| Source | Product / endpoint | Use |
| --- | --- | --- |
| NOAA CO-OPS | `datagetter` API, station **8443970**, products water_level / predictions / air_pressure, 2018-01-01..07, `datum=NAVD`, `application=cera-boston-pilot` | Independent still-water anchor |
| NOAA CO-OPS | `mdapi .../stations/8443970/datums.json` | Datum offsets |
| USGS STN | Flood-event web service, **event_id 208** ("2018 January Extratropical Cyclone"), `Events/208/HWMs.json` (behind SIR 2020-5048) | High-water marks |
| USGS (SIR 2021-5109) | ScienceBase `60100f99d34e162231fecfa8` → "New England January 2018 High-Water Mark Coastal Inundation Map" (+ "100-year Stillwater" comparator) | Published real wet zone |
| ORNL EAGLE-I (STAGED) | Figshare article 24237376, `eaglei_outages_2018.csv` (~1 GB), FIPS **25025** | County outage timing (experimental) |

Modules: `src/cities/boston_ingest/{noaa_water_levels,usgs_hwm,usgs_inundation,eagle_i_outages}.py`;
validation `src/cities/boston_hazard_validation.py`.

## 2. Datum handling (the most likely source of silent error — handled explicitly)

- Observed + predicted were pulled **directly in NAVD 88** (verified available for this
  station-product), matching the repo's DEM / CRB / USGS layers — so no hand-applied conversion
  is in the saved series.
- Published offsets (m above station datum): MLLW 1.074, MSL 2.66, MHHW 4.205, **NAVD88 2.752** →
  **NAVD88 − MLLW = 1.678 m** (recorded in `data/boston/raw/noaa/datum_reference.json`, not hardcoded).
- **Record cross-check:** published record 15.16 ft MLLW → 4.621 m MLLW → 2.943 m NAVD88. Our pulled
  observed peak = **2.944 m NAVD88 (9.66 ft)** — agreement to **1 mm**, and identical to USGS's
  9.66 ft NAVD88 gauge peak.

## 3. Results

### Phase 1 — NOAA gauge (independent anchor)
- Peak observed **2.944 m NAVD88 (9.66 ft) @ 2018-01-04 17:42 UTC (12:42 EST)** — record-matching.
- Storm surge **1.02 m**, peaking **0.3 h after** astronomical high tide → surge rode on top of the
  high tide, which is why the record was set.

### Phase 2 — real ground truth
- **35** USGS high-water marks inside the Boston bbox, all Coastal; **31** NAVD88; **22** with a
  surveyed elevation (range **2.04–3.78 m NAVD88**; the gauge still-water peak sits mid-range —
  total-water-level marks with wave runup go higher).
- Published USGS Jan-2018 inundation polygon clipped to bbox: **16 features, 61 km²** wet zone;
  100-yr stillwater comparator: 15 features, 66 km², WSE 9.8–10.0 ft.

### Phase 3 — does real water land where the model floods? (hazard geometry)

| Scenario | HWM in extent (in-domain /16) | Node P / R / F1 | In-domain recall | IoU vs USGS | USGS zone covered |
| --- | --- | --- | --- | --- | --- |
| slr09_aep10 | 93.8% | 0.76 / 0.76 / 0.76 | 0.84 | 0.48 | 93% |
| **slr09_aep01** (near-term 1%) | 93.8% | 0.58 / 0.90 / 0.70 | **1.00** | 0.42 | **98%** |
| slr21_aep10 | 93.8% | 0.49 / 0.90 / 0.63 | 1.00 | 0.40 | 99% |
| slr21_aep01 | 68.8% | 0.45 / 0.90 / 0.60 | 1.00 | 0.37 | 94% |
| slr36_aep10 | 93.8% | 0.41 / 0.90 / 0.56 | 1.00 | 0.38 | 100% |
| slr36_aep01 | 75.0% | 0.30 / 0.90 / 0.45 | 1.00 | 0.34 | 94% |

**Headline — AEP-band triangulation (two independent lines of evidence agree):**
- The real event **exceeds** the near-term 10% extent — `slr09_aep10` in-domain recall is only **0.84**
  (it misses real-wet nodes).
- The real event is **bounded by** the near-term 1% extent — `slr09_aep01` contains **98%** of the
  USGS wet zone, in-domain recall **1.00**.
- So the event sits **between the model's 10% and 1% near-term scenarios → a ~1–2% AEP event**, which
  is exactly what USGS's **independent** frequency analysis rates the Jan-2018 stillwater (1–2% AEP,
  50–100-yr). The model's near-term hazard input is correctly calibrated.
- Do **not** lead with "slr09_aep10 has the best IoU/F1": scoring growing model extents against a
  fixed real zone mechanically rewards the smallest adequate extent (precision falls monotonically as
  the extent grows). The raw best-IoU scenario is an artifact, not independent evidence. The slight
  over-prediction of `slr09_aep01` (precision 0.58) is the expected signature of a 1–2% event being a
  touch smaller than a full 1% scenario.

**Aquarium station** (`subway_aquarium_place_aqucl`): model-flooded under the near-term scenario
(p_fail t96 = 1.0) **and** inside the USGS Jan-2018 wet zone → a **documented true positive** (the Blue
Line Aquarium station was closed by flooding on Jan 4, 2018).

### Check A — CRB scenario non-nesting (documented data property, not a bug)
The HWM-in-extent column is non-monotonic (slr21_aep01 68.8%, slr36_aep01 75% vs 93.8%). Verified: all
CRB polygons are geometrically **valid** (raw and after `buffer(0)`); `buffer(0)` does not change
containment. At the **same SLR**, the 1%-AEP extent should contain the 10%-AEP extent, but at slr21 and
slr36 **~11% of the 10% extent falls outside the 1% extent** (slr09 is properly nested, 0.0%). CRB's six
scenarios are **independently-modeled probabilistic surfaces, not nested depth thresholds**, so they are
not guaranteed to be spatially nested; the outer-harbor (Winthrop) area drops out of the high-SLR 1%
extents. Footnote, not a fix. The headline (slr09 band) is unaffected.

### Check B — Chelsea Creek fuel terminals (relevant to the fuel-cascade contribution)
Of 3 Chelsea Creek fuel-terminal nodes: **2** (`gulf_oil_chelsea`, `chelsea_sandwich`) fall inside the
CRB domain **and** the real USGS wet zone — floodable in the model and really flooded, supporting the
fuel feedback loop. The 3rd (`global_chelsea_eastern_ave`, up-creek) is **not mapped as inundated by
USGS** — consistent with dry, but *not* a verified true negative: Chelsea Creek's industrial stretch is
lightly surveyed (few HWMs there), so absence from the polygon means "not mapped as flooded," not
"confirmed dry." It also falls **outside the CRB modeled domain entirely** — CRB never wets it even in
the worst slr36_aep01 case, a minor coverage caveat worth flagging since the fuel feedback loop is the
novel contribution and an advisor may probe exactly there. Conclusion unchanged: the fuel-cascade
premise holds for Boston (2 of 3 terminals floodable and really flooded).

### Phase 4 — EAGLE-I county outage timing (STAGED / EXPERIMENTAL — not a headline)
Suffolk County (FIPS 25025) customers-out vs the model's aggregate power-failure curve, Jan 2018
(495 fifteen-min rows stream-filtered from the ~1 GB national file; no full-file storage). Real
Suffolk peak: **1,393 customers out @ 2018-01-07 08:30 UTC** — note this lands **~3 days after** the
Jan-4 flood/storm peak, which is itself the clearest evidence the January 2018 Boston outages were
largely **not** flood-driven (wind/snow/cold). The model's flood-driven power ramp is small (~1.5–1.7
expected failed power nodes) and should not be expected to reproduce the Jan-7 peak. Both curves are on
one clock (UTC; model t=0 anchored to the storm peak 2018-01-04 17:42 UTC = NOAA observed peak).
**Directional aggregate timing only — cannot validate node-level predictions.** See
`validation/staged/README.md` for the full confounds.

## 4. Honest limitations

1. **Leaky-v1 model** → any model-dependent cascade number is directional. Phase 3 is deliberately
   model-INDEPENDENT (flood-extent geometry vs real water), so it survives this.
2. **Cross-validation, not circularity.** CRB (the model's flood input) and the USGS inundation polygon
   are **independent** products, so model-extent-vs-USGS agreement is a genuine cross-validation. The
   only circular comparison — validating the USGS polygon against the high-water marks it was
   interpolated from — is deliberately **not** made. Independent anchors: NOAA gauge + Aquarium.
3. **CRB domain = City of Boston + immediate harbor** (starts ~lat 42.27). Quincy / south shore / outer
   harbor islands are outside it, so ~half the HWMs and **all** node recall-misses fall outside CRB's
   domain (North Quincy, **Chelsea Creek**, Field St). Within CRB's domain the model misses **zero**
   real-wet nodes — the gap is CRB coverage, not flood-extent error.
4. **Still water vs total water level.** NOAA gauge = still water; USGS HWMs = total water level (incl.
   wave runup/setup). Not identical quantities — a known offset, not noise.
5. **No DEM / no depth.** CRB and USGS are extent-only here; the model assigns a flat 0.30 m to every
   in-extent node, so depth-RMSE is not meaningful and was intentionally not computed. The published
   USGS polygon (DEM-derived) supplies the wet/dry zone without us needing a 3DEP DEM.
6. **EAGLE-I** is county-level and partly wind/snow-driven → directional aggregate timing only.

## 5. Mapping to the two open advisor decisions

- **Dr. Miura — real (non-HAZUS) label provenance.** This week demonstrates a real, non-HAZUS hazard
  ground truth exists and is usable: USGS field HWMs + the published USGS inundation polygon (SIR
  2021-5109) + the NOAA record gauge. These are the seed of genuine Boston labels (the next step adds
  documented outage records for cascade labels). HAZUS-derived labels remain circular and are avoided.
- **Dr. Lin — the t=0 oracle leak.** Validating against a **real forward event** (real flood in → model
  predicts → compare to real inundation) is exactly the regime that does not feed the t=0 initial-failure
  mask as an oracle. Real-event forward simulation is the path to a leak-free evaluation; lock the next
  model so it cannot lean on the early-flood hint before training.
