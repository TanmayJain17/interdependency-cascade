# Cascade Simulation Data — For Economic Impact Analysis

Exported from the NYC Infrastructure Cascade pipeline for use in downstream economic impact modeling. 

---

## Files in this package

### 1. nodes_metadata.csv
Master reference — one row per infrastructure node in the NYC graph.

| Column | Description |
|---|---|
| node_id | Unique identifier (stable across scenarios) |
| name | Human-readable name |
| infra_type | One of: power, telecom, hospital, subway, water, fuel |
| borough | Manhattan / Brooklyn / Queens / Bronx / Staten Island / NJ (external) |
| lat, lon | WGS84 coordinates (6 decimal places) |
| external | True if node is outside NYC (NJ substations, petroleum terminals) |
| flood_*_depth_m | Flood depth in meters under each DEP scenario |

Scope: 6,231 nodes across all five boroughs plus external NJ substations and
petroleum terminals.

---

### 2. extreme_2080_per_timestep.csv
Per-run, per-timestep failure records for the DEP Extreme 2080 scenario
(500-year rainfall + 2080 sea level rise).

| Column | Description |
|---|---|
| run_id | Monte Carlo run identifier (0-999) |
| timestep_h | Hours after flood onset: 0, 6, 24, 48, or 96 |
| node_id | Which node failed at this timestep in this run |
| fail_cause | 'flood' (t=0, direct) or 'cascade' (t>0, propagated) |

Use this to track how failures propagate over time. For example, to find all
nodes that fail within 24 hours across all runs:
    df[df.timestep_h <= 24].node_id.unique()

Row count: varies by run, roughly 300-800k rows total.

---

### 3. extreme_2080_aggregated.csv
Aggregated across all 1000 Monte Carlo runs. One row per node that failed
in at least one run.

| Column | Description |
|---|---|
| node_id | Node identifier |
| fail_freq | Fraction of runs where this node failed (0.0 to 1.0) |
| mean_fail_time_h | Average time to failure across runs that failed it |
| p50_fail_time_h | Median failure time |
| p95_fail_time_h | 95th percentile failure time |
| (plus all columns from nodes_metadata for joined context) |

Use this for expected-value analysis. For example, to find the 50 nodes most
likely to fail and the typical timing:
    df.nlargest(50, 'fail_freq')

---

### 4. sandy_lm_per_timestep.csv + sandy_lm_aggregated.csv + sandy_lm_nodes_metadata.csv
Same format as files 2 and 3 but for Hurricane Sandy scenario at Lower
Manhattan scope (347 nodes).

Use this for validation against 2012 Sandy historical economic impact data.

IMPORTANT: Sandy data is LM-only. A citywide Sandy scenario would require
running the GISSR surge model for all boroughs, which is future work.

---

## Known limitations (please factor into your analysis)

1. **DEP Extreme 2080 excludes storm surge.** The DEP flood maps model pluvial
   (rainfall) and tidal flooding only. Storm surge from hurricanes is not
   included. As a result, surge-exposed infrastructure like hospitals along
   the FDR corridor (Bellevue, NYU Langone, Mount Sinai Beth Israel, Mount
   Sinai NYEE) show up as "dry" in DEP's footprint — but they still fail in
   our cascade because their upstream power and fuel suppliers are flooded.
   The Sandy LM data captures surge exposure for validation.

2. **Cascade model is rule-based.** Each dependency edge carries a buffer
   time representing how long the downstream node survives after the
   upstream source fails (e.g., hospital generator = 96h on NFPA 110 fuel
   reserve). We do NOT yet model intra-infrastructure cascade (power grid
   load redistribution via pandapower). Future versions of this data will
   have more conservative total failure counts once intra-grid cascade is
   added.

3. **Redundancy = OR-gate.** If a hospital has both a power supply and a
   fuel supply, either failing causes the hospital to fail in our model.
   In reality, redundancy should be AND-gate (both must fail). This tends
   to over-estimate failure rates for well-redundant systems.

4. **NJ substations and petroleum terminals can receive cascade but not
   originate it.** Their failure probability is always 0.0 in our fragility
   model since we don't have their flood exposure data. They may cascade-fail
   if their dependencies fail.

5. **Monte Carlo runs are noisy.** Individual run-level data varies; for
   expected-value impact estimates, use the aggregated files.

---

## Recommended usage

For direct economic impact (which blocks lose which services):
- Join `nodes_metadata.csv` with `extreme_2080_aggregated.csv` on `node_id`
- Filter to nodes with `fail_freq > 0.5` for "likely to fail" analysis
- Use `lat/lon` to spatially join with NYC census blocks

For time-dependent impact (how long is the outage):
- Use `extreme_2080_per_timestep.csv` to get failure timing distribution
- For each node, look at `mean_fail_time_h` and `p95_fail_time_h` to
  understand how quickly and how variably it fails

For Sandy validation:
- Compare `sandy_lm_aggregated.csv` failure frequencies against documented
  2012 Sandy infrastructure outages
- Four hospitals (Bellevue, NYU Langone, Mount Sinai Beth Israel, Mount
  Sinai NYEE) evacuated in 2012 due to cascade-failure — check whether
  these appear with high `fail_freq` in your Sandy LM data

---


