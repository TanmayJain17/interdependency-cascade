# Power Cascade Handoff — Flood-Driven Monte Carlo Failure Rates

This folder is the deliverable for Tanmay's Stage 4 (inter-infrastructure
cascade), covering the intra-power-network cascade work described in
`power-cascade.pdf`. It packages, per GeoClaw (`gc_*`) flood scenario, the
percentage of Monte Carlo runs in which each node in the network ended up
electrically dead — both in this project's own network representation (Ward
equivalent buses) and translated onto OSI substation nodes.

**Read the caveats below before using this data** — several are load-bearing
for correct interpretation, not just disclaimers.

---

## 1. What's in each scenario folder

`gc_2026/`, `gc_2050/`, `gc_2080/` each contain:

| File | Contents |
|---|---|
| `osi_node_failure_pct_<scenario>.csv` | **Primary deliverable.** One row per OSI node: matched Ward bus, match type/distance, and `pct_failed` — the % of converged Monte Carlo runs in which that node's matched Ward bus ended up out of service. |
| `mc_element_failure_pct_<scenario>.csv` | Same failure-rate concept, but at the native Ward-network level (50 buses, 94 lines, 16 transformers) — the source data `osi_node_failure_pct` is derived from. |
| `mc_run_summary_<scenario>.csv` | Full per-run detail: 1,000 runs, each row showing which buses/lines flooded, cascade iterations, solver used (AC/DC), and the final out-of-service set. |
| `mc_failure_map_<scenario>.html` | Interactive map of the Ward network colored by failure rate. |

Each scenario ran **1,000 Monte Carlo trials**. All 1,000 converged to a
resolved final state in every scenario (via AC power flow, or a DC fallback
when AC didn't converge — see §5).

Top-level files (shared across all scenarios):

- `osi_node_ward_assignment.csv` — the full OSI node → Ward bus crosswalk (independent of any flood scenario).
- `osi_ward_assignment_map.html` — map of that crosswalk.
- `ward_elements.csv` — see §4.

---

## 2. Three scenarios produced no cascade — not a bug

Only the three **GeoClaw (`gc_2026`, `gc_2050`, `gc_2080`)** scenarios are
included here. The other three flood scenarios in this project's dataset —
`dep_extreme_2080`, `dep_moderate_2050`, `dep_moderate_current` — were
checked directly against their HAZUS bus- and line-flood-depth CSVs and have
**zero nonzero failure probabilities on every bus and every line**. With
nothing to draw a failure from, the Monte Carlo cascade would trivially
produce zero failures every run, so no simulation or output was generated
for those three. If DEP-scenario cascade data is actually needed, that's a
question for whoever produced the underlying DEP flood-depth rasters —
the HAZUS translation step found nothing to work with, not this cascade
model.

---

## 3. The node set here is NOT the one you supplied

The nodes in `osi_node_failure_pct_*.csv` are keyed to **this project's own
`OSI_nodes.csv`** (`Injection_Model/Grid_modeling/OSI_nodes.csv`) — a
Rutgers/HIFLD-mirror + OpenStreetMap merge built independently in this
codebase — filtered to NY-state rows with `Ignore` != TRUE (175 of 402 total
rows; NJ rows and 26 explicitly-ignored rows excluded). **This is not
guaranteed to be the same node/ID space your Stage 2/4 pipeline uses.**
Your plan's master node list was meant to be Miura's shapefiles, with
HIFLD/OSM merged in only to fill gaps — `OSI_nodes.csv` has no Miura data at
all, and its `Unique ID` scheme (`S1`, `S8`, `P3`, ...) was invented in this
project, not derived from or verified against your live node table. If you
can join on name + lat/lon instead of ID, that should still work; a direct
ID join is not guaranteed to.

**If you'd rather work from nodes closer to your own original data:**
`osi_node_ward_assignment.csv` (and every `osi_node_failure_pct_*.csv`) has
a `Rutgers_data` column — `TRUE` means that OSI node's `source` field in
`OSI_nodes.csv` was `rutgers` or `both` (i.e. it really was in the original
Rutgers/HIFLD export, 116 of 175 nodes); `FALSE` means it was added from
OpenStreetMap only (59 of 175). Filtering to `Rutgers_data == TRUE` gets you
the subset closest to your original dataset.

**Also incorporate `ward_elements.csv`** — 12 rows representing this
project's Ward-equivalent boundary/external-tie nodes (e.g. MARCY, DUNWOODIE,
SPRAIN BROOK, VALLEY STREAM). These are real electrical nodes in the power
model — the points where the truncated external grid folds into this
network's equivalent injections — but they are not substations in the OSI
sense and won't appear in the OSI crosswalk. They should be added to your
graph as their own nodes alongside the OSI-matched set.

---

## 4. The OSI crosswalk is many-to-one — read failure rates accordingly

The Ward network is a 50-bus equivalent; OSI's substation set is much finer
(175 NY nodes after filtering). The nearest-neighbor assignment
(`assign_osi_nodes_to_ward.py`) means **multiple OSI nodes share one Ward
bus and therefore share that bus's exact failure percentage.** The extreme
case: 27 Staten Island OSI substations (the network has no real Staten
Island representation) all inherit the identical failure rate of one bus,
`UNKNOWN129920` (13.6% in `gc_2050`, 17.3% in `gc_2080`) — see any
`osi_node_failure_pct_*.csv`, sorted descending, for the repeated blocks.
**"OSI node X failed in Y% of runs" should be read as "the equivalent Ward
bus representing X's neighborhood failed in Y% of runs,"** not as an
independent per-substation result. Match type and distance are included in
every row so you can judge confidence per node — `coincident` (≤5 m, 23
nodes) is the highest-confidence tier, `staten_island_override` (27 nodes)
is explicitly the coarsest.

---

## 5. Cascade mechanics — what generates these failure rates

Per run: buses/lines are drawn as flooded independently (fresh random draw
per element per run) using HAZUS failure probabilities; a flooded bus also
de-energizes everything physically attached to it. The network is then
iteratively stabilized — solve power flow, trip any line loaded over 100%,
remove newly-islanded buses, and re-solve — repeating until the grid
stabilizes or a solve fails outright (capped at 20 iterations; the cap was
never hit in any of the 3,000 runs across all scenarios). **AC power flow is
tried first at every iteration; DC power flow is used as a fallback only
when AC fails to converge at that iteration** (116–282 of 1,000 runs per
scenario needed the DC fallback at least once, see below). This differs from your
plan's `cascade_sim.py` pseudocode, which runs in either pure-DC
(training) or pure-AC (validation) mode — this implementation always
prefers the more accurate AC solve and only degrades to DC when necessary,
rather than committing to one mode for the whole run. (DC fallback usage
across the three scenarios: 116/1,000 runs for `gc_2026`, 248/1,000 for
`gc_2050`, 282/1,000 for `gc_2080` — rising with storm severity, as expected.)

**Queensbridge–Vernon manual override:** the three parallel 138 kV
QUEENSBRIDGE↔VERNON circuits (Ward line indices 1285/1286/1287) are
excluded from the overload-trip step. In the undamaged network they run at
~89% loading already (322 MW each) — the closest thing to a real load
constraint in this equivalent model — so the >100%-loading overload
heuristic would trip them from ordinary redispatch noise, not from a
flood-driven event. They can still fail, but only via their own HAZUS flood
draw (currently 0% in every `gc_*` scenario, so in practice they never
appear as failed in this data). Flagging this because it's a modeling
judgment call, not a default.

---

## 6. Transmission line modeling — two separate things

**Electrical parameters** (impedance, thermal rating) used for the actual
power-flow solve come directly from a real NYISO on-peak MATPOWER case
(`nyiso_on_peak_v23_shunts_as_gen.m`), not from voltage-class lookup tables.
Per-unit R/X/B values are converted to ohms using each line's own base kV —
i.e. these are real utility-grade system parameters, not the
standard-table estimates your plan's Open Question 1 anticipated needing.

**Geographic routing** (for mapping only — it has no effect on the power
flow solve) is separate: **39 of the Ward network's 94 lines (41.5%) have
real right-of-way geometry**, matched against cached OpenStreetMap
power-line ways by a two-tier matcher (name match first, then
proximity + voltage-class match). The remaining 58.5% of lines are drawn as
straight bus-to-bus segments with no routing data. This routing match rate
is unrelated to and should not be confused with the OSI node crosswalk
(§3–4), which matches point substations, not line paths.

---

## 7. Outstanding — not done yet

**ConEd secondary-network-area assignment.** A separate heuristic already
exists in this codebase (`Grid_modeling/assign_networks_to_nodes.py`,
visualized in `Grid_modeling/network_node_map.html`) that assigns OSI nodes
to ConEd's ~89 secondary network polygons (nodes inside a polygon are
assigned directly; if fewer than K=5 nodes fall inside, the nearest
additional nodes within 6,500 m of the polygon centroid are added). It has
**not** been reconciled with or folded into this handoff — the Ward-bus
crosswalk and the ConEd-network assignment are two independent, currently
disconnected pieces of node-grouping logic. If your Stage 4 work needs
nodes grouped by ConEd network area rather than by nearest Ward bus, that
integration is still open.
