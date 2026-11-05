# Time-variant flooding in the cascade model — how failures are timed

*For Jesse. Repository `interdependency-cascade`, branch `feat/direct-model` (the behaviour below has been unchanged since the Week 23 implementation; code in `src/simulation/dynamic_forcing.py`, `src/simulation/cascade_sim.py`, `src/graph/build_graph.py`). 3 Nov 2026.*

## Short answer to "at what time is the electricity failure implemented?"

There are two label campaigns, and they differ exactly on this point.

| | **static** campaign | **arrival** campaign (time-variant) |
|---|---|---|
| *Whether* a power node fails from flooding | fragility draw on its **peak** depth | fragility draw on its **peak** depth (same) |
| *When* that failure is placed | at the **peak** (hour 0 of the record) | at the node's own **arrival hour**: the first hour the hydrograph puts any water at that node |
| Typical lead before the peak (power nodes) | 0 | 1–30 h, depending on the storm (see table below) |

So: **in the time-variant runs, a substation does not fail at peak inundation. It fails the first hour water reaches it**, which is before the peak for every wet site in every storm we have. The *decision* to fail is still made from the peak depth; only the *clock* moves. In the static runs everything is at the peak.

## How it works, step by step

1. **Peak depth map.** Each storm has one depth per site (metres), the maximum over the storm. These are the tables `jesse22_node_depths_v1.csv` (the 23 maps you have) and `gissr48_node_depths_v1.csv` (the 48 new ones).

2. **Which sites the water knocks out.** For each of the 1,000 Monte Carlo runs, each wet site fails directly with probability from the Hazus lognormal curve on its peak depth: P = Φ((ln d − ln δ)/β). For power: δ = 0.6 m, β = 0.5 (so 0.3 m → 8 %, 0.6 m → 50 %, 1.2 m → 92 %). External sites (99 in all, 91 of them power nodes outside the city: the NJ and upstate substations) never fail from flooding.

3. **When each knocked-out site fails.**
   - Static: at hour 0 = the peak.
   - Arrival: at `arrival_h`, read from the timing table — the first hour, on the storm's own clock (hour 0 = first hydrograph sample), at which the time-resolved map gives that site a depth > 0. A site that is wet at the peak but has no timing row is placed at the peak (the latest defensible time); in our storms this does not occur.

4. **The cascade after that.** Dependency links carry a buffer: the downstream site fails when its last supplier has been down longer than the buffer. Buffers are Weibull-distributed per run around engineering medians:
   - power → telecom 6 h (battery), → hospital 96 h (generator), → water 3 h, → subway 0 h, → gas station 0 h, → fuel terminal 4 h;
   - telecom → power 2 h (SCADA monitoring link; this is the only way a power node fails *other than* flooding);
   - water → hospital 24 h; fuel → hospital 96 h, → telecom 48 h, → water 48 h.
   Shapes: hospital 2.5, telecom 3.0, water 2.5, subway 4.0, fuel 2.0, power 3.0. Each site has exactly one supplier of each kind (its nearest within 5 km), and the gate is OR: a site fails at the first evaluation look at which *any one* of its suppliers has been down longer than that link's buffer (e.g. a telecom cluster fails when its substation has been out more than its ~6 h battery, or its fuel source more than 48 h, whichever comes first).

5. **What is *not* in these runs.** No power-flow physics: `power_line` edges are structural only and do not propagate failure, and the pandapower coupling arm (`allow_power_coupling: false`) is switched off in every label campaign. No recovery: a failed site stays failed. Within-system rules exist for subway (Motter–Lai load shedding), telecom (capacity + demand surge after the peak) and hospitals (patient diversion), none for power.

6. **Evaluation grid and record frame.** The engine looks at the system every 6 h from onset to the peak, then at peak + 0, 6, 24, 48, 96 (and 144, 240, 360 in the arrival campaign). Failures are stamped at those looks. Records are written **peak-relative**: `fail_time_per_node` is hours after the peak, negative = before the peak. The learning targets are "failed by peak + 6 / 24 / 48 / 96 h".

## The timing table

`node_timing_jesse22_plus_gissr48_v1.csv` — one row per (site, storm), for every wet site of all 71 storms (the 23 you have plus the 48 new). Columns:

| column | meaning (hours on the storm's own clock, hour 0 = first hydrograph sample) |
|---|---|
| `peak_depth_m` | the site's peak depth (same number as the depth table) |
| `arrival_h` | first hour with depth > 0 — **this is when the site fails in the arrival campaign, if its fragility draw says it fails** |
| `cross_h` | first hour depth exceeds a threshold; with the threshold at 0 m it equals `arrival_h` (threshold mode exists but was never used) |
| `recede_h` | last hour with depth > 0 |
| `duration_h` | `recede_h − arrival_h` |
| `duration_thr_h` | same above the threshold (equal to `duration_h` here) |
| `time_to_peak_h` | `t_peak_h − arrival_h`: hours before the storm peak that the water reached the site (≥ 0 for every wet site in every storm) |

`..._summary.csv` — one row per storm: `t_peak_h` (hour of the peak on that storm's clock), `span_h` (length of the hydrograph), `peak_wl_m`, number of wet sites, median and 90th-percentile arrival hours.

Hydrographs: GISSR storms use their own water-level time series (hourly steps, the as-delivered convention); the three GeoClaw storms use the NOAA Battery gauge (8518750) hourly record from Sandy 2012, which is why their `t_peak_h` is 25 h and their span 47 h.

## Numbers to anchor on (power nodes)

| storm | peak level | t_peak_h | power nodes wet | example: node, arrival_h, hours before peak |
|---|---|---|---|---|
| syn_ts_914_6 | 2.17 m | 8.7 | 5 | power_east_13th_street: 7.7 h, 1.0 h before |
| geoclaw_2050 | 3.48 m | 25.0 | 24 | power_east_13th_street: 22.9 h, 2.1 h before |
| syn2_t198w16 | 4.72 m | 39.3 | 31 | power_bruckner: 12.6 h, 26.7 h before; power_east_64th_street: 33.7 h, 5.6 h before |

In the biggest storm, then, some substations are placed 27 hours before the peak and others 5 hours before; in the static campaign all of them are placed at the peak. That difference is what the arrival labels encode, and it is why arrival-label cascades are larger (more time for buffers to run out before the peak-relative horizons are counted).

## If you want to align your power model with ours

- Use `arrival_h` per power node (or `time_to_peak_h` if you work on a peak-relative clock) from the timing table as the outage time, and the peak depth with the Hazus curve above as the outage probability — that reproduces our seeding exactly.
- If you need the full time series of depth per site rather than arrival/recede/duration, say so: it is not in the repository tables, but it is what the timing table was computed from.
- The 203 power nodes with coordinates, voltage class and the `external` flag are in `data/graph/nyc_infra_nodes.geojson` (property `infra_type = power`).

## Files

| file | what |
|---|---|
| `data/flood/jesse22_node_depths_v1.csv` | peak depth per site, the 23 maps you have (20 GISSR + 3 GeoClaw) |
| `data/flood/gissr48_node_depths_v1.csv` | peak depth per site, the 48 new GISSR storms |
| `data/flood/gissr48_manifest_v1.csv` | the 48 storms: GISSR catalogue coordinates (timeline, season, row), target and realised peak level, fast/slow lead class, train/test role |
| `data/flood/timing/node_timing_jesse22_plus_gissr48_v1.csv` | arrival / recede / duration / time-to-peak per wet site, all 71 storms |
| `data/flood/timing/node_timing_jesse22_plus_gissr48_v1_summary.csv` | t_peak, span, wet-site count per storm |
| `config/dynamic_forcing_gissr48_v1.yaml` | the arrival-mode settings the labels were produced with |

Note on one site: `telecom_cluster_03042` is 0.105 m deeper in the repository depth table than in the depths the frozen campaign actually sampled (a known discrepancy, on record since Week 29). It does not affect power nodes.
