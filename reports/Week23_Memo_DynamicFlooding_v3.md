# Week 23 Memo — Does *when* the water arrives change the cascade?

Tanmay Jain · CERA Lab · v3, September 10, 2026 (v1 August 28) · Branch `feat/dynamic-flood-maps` (commit `[fill after the Week 25 commit]`)

*v3 replaces the v2 arrival-error paragraph with §4.7–4.8: the translation test that answers the Week 24 question, the surge-phase decomposition, and a new grid-resolution sensitivity. All v1 numbers are unchanged.*

---

## 1. Summary (one paragraph)

Until this week every simulation started with the whole flood already at its peak: every node that would get wet was wet at hour zero. This week the simulator learned to flood nodes when the storm surge actually reaches them, using the hydrograph behind each of Gwen's synthetic maps (and the Sandy tide gauge for the GeoClaw maps). Across all 23 storm scenarios, at 1,000 Monte Carlo runs each and with identical random seeds so every comparison is paired, the answer has three parts. **Timing never changes whether the city crashes** (22 of 23 crash probabilities identical; the 23rd differs by 0.013). **Below roughly 3.1 m of water at The Battery, timing changes when things fail but not where the city ends up:** failures at peak+96 h are 2–15 % higher and the median node fails 0–6 hours earlier, but by 15 days the difference is 1–5 %. **Above roughly 3.2 m the city ends up somewhere worse and stays there:** +21 % to +58 % more failures at peak+96 h, unchanged at 15 days, because by the time the surge peaks 1,000–2,200 nodes are already down — three to four times what the static model assumes — and the telecom demand surge lands on a network that has already lost that capacity. That threshold sits exactly where last week's grid-physics result found variance amplification switching on. The static-at-peak convention is right about *if*, slightly early about *when* in ordinary storms, and wrong about *how bad* in the saturated ones.

---

## 2. What was done (plain language)

1. **Every synthetic map now has a storm clock.** The 20 maps were matched to their storms in Gwen's catalogue (peak water level = surge + tide, 1.91–4.46 m at The Battery), which gives each a 20-point water-level curve from storm onset to recession. The GeoClaw trio uses the NOAA Battery record for Sandy (48 h).
2. **A node gets wet when the coastal water level comes within its map depth of the peak.** This works because Gwen's model fills each coastal division within an hour of being overtopped (verified by running her code), so the map depth is the truth at the peak and the hydrograph supplies the shape. Only differences of water level enter, so datum and sea-level rise cancel. Setting the curve flat reproduces the static map exactly.
3. **Flood failures are scheduled at their arrival hour** through the same hook that already delivered Jesse's power states at t = 6. Nothing else in the cascade engine changed; two additive parameters let a static run be replayed at any hour.
4. **The dynamic code reduces to the frozen campaign byte-for-byte** when every seed is placed at the peak — checked record-for-record against the Week 22 legacy campaign on all 23 scenarios at N = 1,000 (23 of 23 passed). Every dynamic number below is therefore a controlled difference from the frozen results, same seeds, same buffers.
5. **Three baselines bracket the effect:** static-at-peak (the frozen campaign), arrival-ordered (the new mode), and static-at-onset (everything wet at hour 0, the other extreme).
6. **Horizon extended to peak + 360 h** so that "different end state" can be separated from "same cascade observed earlier".

---

## 3. Findings for the December story

### 3.1 The crash decision does not move

| storms | crash probability, static | crash probability, arrival-ordered |
|---|---|---|
| 13 storms ≤ 2.63 m | 0.0 → 0.8 | identical, storm by storm |
| 880_9 (3.09 m) | 0.982 | 0.995 |
| 9 storms ≥ 3.24 m (incl. GeoClaw trio) | 1.0 | 1.0 |

Threshold 1,200 failures by peak + 96 h, as in Week 22.

### 3.2 Two regimes along the S-curve

| peak level (m, MSL) | wet nodes | example | extra failures at peak+96 h | at peak+360 h | median node fails earlier by |
|---|---|---|---|---|---|
| 1.9–2.2 | 108–155 | 914_6 | +2 % to +5 % | +1 % to +4 % | 2–6 h |
| 2.3–2.8 | 186–311 | 156_15, 192_15 | +3 % to +15 % | +0.4 % to +11 % | 0–24 h |
| 3.1 | 393 | 880_9 | +13 % | +5 % | 24 h |
| **3.2–4.5** | **430–703** | 514_13 … 605_5 | **+21 % to +53 %** | **+21 % to +53 %** | 14–40 h |
| GeoClaw 2026 / 2050 / 2080 | 264 / 368 / 482 | Sandy hydrograph | +12 % / +58 % / +47 % | +6 % / +59 % / +46 % | 18–24 h |

Below the transition the peak+96 excess decays by 15 days (same destination, different timing). Above it the excess does not decay at all. The transition is sharp — 880_9 at 3.09 m converges, 514_13 at 3.24 m does not — and it coincides with the onset of variance amplification in the Week 22 physics arm (514_13 ×1.88, 258_9 ×2.21, ≥3.4 m ×3.4–5.8; gc_2026 ×1.10 converges here too). The GeoClaw trio straddles the same transition, so it is not a synthetic-map artifact; that gc_2050 persists at 368 wet nodes while 880_9 converges at 393 says the spatial pattern of the flood matters, not only its size.

### 3.3 What the static model is missing in the big storms

At the moment the static model starts its clock (the peak), the arrival-ordered world already has 1.4× (small storms) to 3.6× (605_5: 2,231 vs 612) the static seed count down. In 605_5, a 59-hour storm, 1,804 node-failures per run happen before the peak; the median node fails 40 h earlier than the static model says; every one of 1,000 paired runs ends worse. In the fast-rising storms (peak within 3–10 h of onset) the two worlds are nearly the same, which is why the effect is confined to the saturated regime.

### 3.4 Why the end state differs — and the one assumption it rests on

*(v3: the assumption's leverage is now measured directly — §4.7: in the crash-edge storm, moving the demand surge six hours relative to the flood moves the end state by about ±240 nodes and the super-cascade fraction between 7 % and 39 %.)*

Decomposing 605_5 by cause and by infrastructure type (N = 100, t360): telecom +714 nodes, fuel +405, subway +84, water +40, power +24, hospital +9. The subway part is the classic order-dependence of load-redistribution cascades (Motter-Lai overload: sequential removal sheds load differently from simultaneous removal). The telecom part — the bulk of it — is the demand-surge model: the frozen engine applies a 1.5× call-volume surge at the event, decaying over 12 h; in the arrival-ordered world that surge hits a network that has already lost four times as many sites. Fuel and most of the inter-layer growth are downstream of those two.

This is a real interaction (surge on a pre-weakened network), but it is conditional on **when the demand surge is assumed to happen**. Anchoring it at storm onset instead of the peak (so it has decayed by the time a slow storm peaks) changes the answer: the end-state *mean* comes back to within 1–4 % of static, but 75–84 % of runs end smaller than static and 16–25 % much larger — a bimodal outcome with a spread 2–5× the static world's. Either way, timing does to the tail what the grid physics did in Week 22. Sandy's call-volume record (spike at and after landfall) argues for the peak-anchored version; this is the first decision to put to the advisors (§7).

### 3.5 What it means for World 2

The intermediate labels the surrogate is trained on (failed by t = 6, 24, 48) change by hundreds of nodes in the big storms under either clock assumption. A surrogate trained on static labels is therefore wrong about the middle of the event even where it is right about the end. The timing quantities (arrival, duration, time-to-peak) are now available as input features behind a switch, and dynamic labels exist for all 23 scenarios, so the planned experiment is a controlled 2 × 2: labels {static, dynamic} × inputs {mask only, mask + timing}. The "static labels, mask only" cell is the twin already trained (§5).

---

## 4. Appendix for Dr. Lin — definitions, gates, tables, caveats

### 4.1 Definitions

- Time origin `t = 0` is the first hydrograph sample (storm onset). Evaluation grid: every 6 h up to the peak, then peak + {0, 6, 24, 48, 96} (the frozen offsets; unchanged for comparability) with +144, +240, +360 added for horizon studies.
- Node timing from the static map depth `d` and coastal water level `W(t)`: `depth(t) = max(d − (W_peak − W(t)), 0)`; arrival = first `t` with `depth > 0`; duration = hours with `depth > 0`; time-to-peak = `t_peak − arrival`. Crossings solved on the piecewise-linear hydrograph. Datum and SLR cancel (differences only). `W(t) ≡ W_peak` reduces exactly to the static map.
- Seeds are sampled from the static peak-depth map by the unchanged fragility step; only their landing hour changes (scheduled-failure hook, cause `flood`, earliest cause wins). The seed *set* is identical across all three baselines.
- Records are written peak-relative (`fail_time_per_node` = hours after the peak, negative before it) so `data.py`, the compare scripts and the cascade-only mask read them unchanged; onset-clock times are stored alongside.
- The intra-layer closure receives a clock relative to the peak (frozen assumption) and a fully decayed clock before the origin; buffers on inter-layer edges start from each source's exact failure hour, so "buffer clocks start at arrival" holds by construction.

### 4.2 Exact-reduction gate

`static_peak` = every seed at `t_peak`, grid translated, intra clock translated — a pure time shift of the frozen run. Compared record-for-record (direct failures, totals, by-horizon counts, failed node lists, per-node failure hours, cause counts) against `legacy_v1_n1000`: 23/23 scenarios at N = 1,000 identical (Torch job 16456615), plus the local N = 20/100/1,000 checks. The sampler is prefix-consistent (the first N draws of an N-run job equal the first N of the 1,000-run campaign), which is what makes every comparison paired.

### 4.3 Three-storm deep dive, N = 1,000, paired (arrival − static_peak)

| | 914_6 (2.17 m) | 880_9 (3.09 m) | 605_5 (4.46 m) |
|---|---|---|---|
| static total at t360 | 335.6 | 2,297.4 | 2,749.4 |
| arrival total at t360 | 342.6 | 2,417.5 | 4,055.2 |
| paired Δ mean ± sd | +7.0 ± 15.5 | +120.2 ± 320.1 | +1,305.8 ± 176.0 |
| runs larger / smaller | 974 / 3 | 980 / 19 | 1,000 / 0 |
| KS on totals | 0.136 | 0.381 | 0.989 |
| down at the peak: static → arrival | 91 → 176 | 276 → 1,006 | 612 → 2,231 |
| Δ at t6 / t24 / t48 / t96 | +19 / +19 / +10 / +9 | +614 / +369 / +621 / +260 | +1,081 / +1,269 / +1,138 / +1,204 |
| Δ at t144 / t240 / t360 | +8 / +7 / +7 | +136 / +135 / +120 | +1,224 / +1,290 / +1,306 |
| nodes failing earlier / same / later | 60 % / 40 % / 0 % | 78 % / 22 % / 0.6 % | 98 % / 1.5 % / 0.4 % |
| median shift (h) | −2 | −24 | −40 |
| pre-peak failures per run | 100 | 814 | 1,804 |
| crash prob. static → arrival | 0.001 → 0.001 | 0.982 → 0.995 | 1.0 → 1.0 |

Bracket (N = 100): static_onset totals 331 / 2,739 / 3,784 at t96 vs arrival 317 / 2,217 / 3,735 vs static_peak 310 / 1,982 / 2,554 — arrival lies between the two static baselines in every storm.

### 4.4 Intra-clock sensitivity (demand surge anchored at onset), N = 1,000, paired vs static_peak

| | 880_9 | 605_5 |
|---|---|---|
| paired Δ at t360, mean ± sd | +18.1 ± 204.7 | +118.5 ± 528.5 |
| runs larger / smaller | 212 / 753 | 162 / 838 |
| Δ at t6 / t96 / t360 | +431 / +158 / +18 | +748 / +215 / +118 |
| down at the peak (arrival) | 879 | 1,980 |

The elapsed-time effect (t0–t48) is clock-independent; the persistent end-state gap is the surge interaction. Onset-anchored, the mean converges but the distribution becomes bimodal with 2–5× the static spread.

### 4.5 Mechanism decomposition, 605_5, N = 100, t360 (peak clock)

By cause: flood −181 (seeds reached by the cascade before their scheduled hour, relabelled), inter +557, intra_telecom +392, intra_fuel +453, intra_subway +44, intra_water +10. By type: telecom +714, fuel +405, subway +84, water +40, power +24, hospital +9. With the onset clock: telecom +24, fuel +12, subway +7 (the `intra_subway` cause still +44 — relabelling, not new nodes).

### 4.6 Caveats, stated

1. **Hydrograph assumptions.** (a) Bathtub timing: a division fills within the hour it is overtopped — verified in Gwen's model, appropriate for her maps; on GeoClaw surfaces the surge arrives at different times around the city, so the trio's timing carries a ±3–6 h uncertainty until fgmax arrival times replace it. (b) The Battery hydrograph is applied citywide (no propagation delay). (c) The NOAA Sandy record begins with the water already at 1.47 m, so the trio's earliest arrivals are a window edge. (d) Catalogue storms begin at the surge threshold (0.68 m + tide), so in small storms many nodes are "wet at onset".
2. **The demand-surge clock** (§3.4) decides whether the saturated-regime gap persists; it is a modeling choice with a Sandy argument for the peak-anchored version.
3. **The frozen 96-hour horizon truncates an unfinished cascade** even in the static world (+14–16 % from t96 to t360 in the big storms). Week 22's `total_failures` is a 96-hour snapshot, not an end state.
4. **Model-conditional**, as in Week 22: one model perturbed in one dimension (seed timing), legacy arm only (Jesse's states are peak-anchored and were kept out of the first ablation).
5. **Arrival perturbation: run (§4.7).** Not yet run: Sandy hindcast timeline check (ConEd 21:00, NYU Langone 19:30–23:00, tunnels, FCC).
6. **Evaluation-grid resolution is a model parameter in the tipping zone (§4.8).** The engine realizes cascade kills only at grid instants (pre-peak every 6 h, then peak + 6/24/48/96 h); refining to 3 h raises the fraction of runs that tip into the super-cascade mode for both forcing modes (880_9: static_peak 0 → 16 %, arrival 3 → 23 %) while the contained-mode end state is unchanged. The §4.3 magnitudes are 6-h-grid values until the 3-h N = 1,000 rerun (§7, item 6). The 880_9 static-vs-arrival difference grows from +84 to +141 on the finer grid, so the direction of §3 is not at risk.
7. **Hour-0 origin rule.** No propagation is evaluated at the origin instant, so seeds already wet at hour 0 have their zero-buffer dependents realized one grid step late; the end state is unaffected, the t6 labels of scenarios with many hour-0 arrivals carry a one-bin bias.

### 4.7 Arrival-error robustness, the translation test, and the surge-phase decomposition (v3)

**Shift study (Week 24; ±3/±6 h uniform shifts, N = 100, 3 storms, 6-h grid).** No shift changes any crash decision. End-state deltas ≤ 2.2 % (914_6) and ≤ 0.4 % (605_5); the crash-edge storm 880_9 responded asymmetrically (earlier arrival +13 % / +315 ± 619 nodes; later −2 % / −47 ± 241).

**The question (Dr. Lin, Week 24 meeting).** A uniform shift of an autonomous model is a relabeling and should change nothing.

**The translation test.** Translating the *whole* model — arrivals, evaluation grid, surge clock, record frame — by +6 h and again by +12 h reproduces every record key (`direct_failures, total_failures, by_timestep, failed_nodes_t96, fail_time_per_node, cause_counts`) in 100/100 paired runs of 880_9. The engine is translation-invariant. The shift-study deltas therefore come from the one term that does not translate with the flood: the telecom demand surge `m(t) = 1 + 0.5·exp(−(t − t_peak)/12)`, anchored at the storm peak (§3.4). Shifting the flood against a fixed peak changes the flood–surge phase. Two implementation details add distortion: clipping of arrivals at hour 0 (absent in 880_9 — 75 nodes are already at hour 0 and the next arrival is 10.65 h) and the evaluation grid (below). All randomness is drawn before the cascade loop, so paired differences are deterministic; the wide spreads are the two outcome modes of a bistable storm, not noise.

**Decomposition (880_9, N = 100, paired, reference = the translated world).** Moving only the surge clock, everything else fixed, and — separately — moving only the flood, on the standard 6-h grid and on a 3-h grid:

| arm (880_9) | 6-h grid: Δ t360, runs up/down | 3-h grid: Δ t360, runs up/down, super-cascade |
|---|---|---|
| surge 6 h later than flood (P+) | +75 ± 321, 66/18 | **+229 ± 539, 69/21, 0.23 → 0.38** |
| surge 6 h earlier than flood (P−) | −82 ± 243, 4/96 | **−242 ± 587, 14/82, 0.23 → 0.07** |
| flood 6 h earlier, peak fixed (Week 24 −6) | +315 ± 619, 70/24 | **+244 ± 554, 63/33, 0.23 → 0.39** |
| flood 6 h later, peak fixed (Week 24 +6) | −47 ± 241, 30/63 | **−242 ± 587, 20/78, 0.23 → 0.07** |
| 914_6 control (P+ / P−) | +3 ± 21 / −6 ± 12 | — |

On the 3-h grid the two ways of changing the phase coincide (+244 vs +229; −242 vs −242) and the effect is symmetric: about ±240 nodes (±9 % of the contained-mode mean), moving the super-cascade fraction between 7 % and 39 %. On the 6-h grid they disagree (+315 vs +75) because the post-peak offsets (6, 24, 48 h) sample the shifted surge's decay at m = 1.5 then 1.11 instead of 1.5, 1.30, 1.07. **The Week 24 early/late asymmetry was the 6-h grid, not the tipping zone; the physics is a symmetric flood–surge phase effect through the demand surge.** The shift study is a robustness check on the §3.4 interaction, not an independent instrument for the tipping zone.

### 4.8 Grid-resolution sensitivity (v3; 880_9, N = 100, paired)

| grid | static_peak t360 | arrival t360 | arrival − static_peak | super-cascade (≥ 3,000): static / arrival |
|---|---|---|---|---|
| 6 h (campaign) | 2,304 ± 118 | 2,389 ± 289 | +84 ± 242 (99/1) | 0.00 / 0.03 |
| 3 h | 2,550 ± 626 | 2,691 ± 710 | +141 ± 435 (95/5) | 0.16 / 0.23 |

The contained-mode end state is grid-stable (arrival contained runs 2,342 vs 2,305); what the grid changes is how many runs tip, for both forcing modes. Direction of §3 preserved and strengthened; magnitudes in the tipping zone are grid-conditional. Files: `analysis/static_vs_dynamic_v1/week25_arm_table_v1.csv` and the `surgephase_*`, `arrival_vs_staticpeak_grid3_*`, `arrival_shift*_grid3_*`, `translation_*` files; method note `reports/Week25_ConceptNote_DynamicForcing_v3.md`.

---

## 5. The twin CascadeGNN, held-out (job 16438671; completes Week 22's open item)

Both arms: frozen v1 graph, seed 42, 6 epochs (best val legacy 0.10824, jesse 0.11969, both at epoch 5), 5-way scenario holdout. Task: given the direct-failure mask as input, predict failure by t ∈ {6, 24, 48, 96}; cascade-only metrics score the nodes that were not direct failures.

| held-out | loss legacy | loss jesse | ratio | cascade PR-AUC legacy (t6/24/48/96) | jesse | Δ pp |
|---|---|---|---|---|---|---|
| geoclaw_2050 | 0.1406 | 0.1807 | 1.29 | 0.955/0.983/0.959/0.981 | 0.956/0.968/0.941/0.969 | +0.1/−1.5/−1.9/−1.2 |
| 914_6 | 0.0855 | 0.0900 | 1.05 | 0.967/0.863/0.858/0.856 | 0.969/0.851/0.855/0.845 | +0.2/−1.2/−0.3/−1.1 |
| 258_9 | 0.1248 | 0.1491 | 1.19 | 0.966/0.981/0.955/0.978 | 0.971/0.969/0.945/0.971 | +0.5/−1.2/−1.0/−0.7 |
| 605_5 | 0.2052 | 0.3441 | 1.68 | 0.933/0.975/0.969/0.988 | 0.884/0.921/0.909/0.947 | −5.0/−5.4/−5.9/−4.1 |
| 808_27 | 0.1400 | 0.2072 | 1.48 | 0.955/0.978/0.963/0.984 | 0.957/0.952/0.926/0.957 | +0.2/−2.6/−3.6/−2.7 |

The loss ratio rises with Week 22's variance amplification (1.05 → 1.68), on held-out storms: the physics labels carry more irreducible entropy and the BCE floor cannot go below it. Skill is indistinguishable at t6 everywhere and separates only at later horizons in the saturated storms — the same storms where timing changes the destination. These 0.95–0.98 cascade-only values are on 17 training scenarios and the frozen graph; the earlier 0.73 was the same metric on far fewer scenarios, which is the learning-curve hypothesis, not yet a result.

---

## 6. GISSR (Gwen's flood model) — review summary

Read and run end to end; the notebook's Sandy run reproduced to machine precision and the delivered raster bit-for-bit. Facts: (1) with two coastline segments' missing elevations and an overflow-to-zero bug fixed, the model gives a clean bathtub at the peak water level in every one of 206 divisions — the Sandy extent (120 km²) matches; (2) all 20 delivered maps carry the nine-division northern-Manhattan hole (≈1 % of footprint, no rerun needed), and the 7 largest show the overflow bug as small non-nested patches (132–5,259 cells); (3) the synthetic maps are nested bathtubs indexed by one number, peak water level at The Battery — which explains Week 22's Spearman 0.998 on wet-node count vs 0.926 on the surge label (the label omits tide), gives the S-curve in metres (156 wet nodes ≈ 2.17 m, 218 ≈ 2.45 m, 390 ≈ 3.08 m, present-day sea level), and means more GISSR maps densify one axis without adding spatial diversity; (4) catalogue water levels contain no SLR (to be added by storm year), the NOAA file is MSL vs the DEM's NAVD88 (−0.063 m), and depth rasters include sub-zero DEM cells (depths to 17.7 m). All sent to Gwen as a question list; fixes belong in her repository.

---

## 7. Decisions requested

**Dr. Miura**
1. *When should the telecom demand surge be anchored — storm peak (current, Sandy-consistent) or storm onset?* This one modeling choice decides whether the saturated-regime damage is +21–58 % and persistent, or mean-neutral with a fat bimodal tail; v3 §4.7 shows a six-hour change in that anchor moves the crash-edge storm's end state by about ±240 nodes and its super-cascade fraction between 7 % and 39 %. Either is a finding; the memo needs to know which one to lead with.
2. Approval to start the World 2 2×2 (dynamic labels are computed; GPU time inside the policer window).
3. GISSR: confirm the fixes go into Gwen's repo, whether SLR should be added by storm year for the production batch, and whether a per-gauge spatial scaling of GISSR (Kings Point / Sandy Hook / Bergen Point ratios from Sandy) is something you would endorse for spatial diversity — the alternative is more GeoClaw runs.

**Dr. Lin**
4. Whether the ±3–6 h arrival perturbation and the Sandy hindcast timeline are the two reviewer checks to run next, or whether the fgmax arrival-time request to the GeoClaw side should come first.
5. N_MC for surrogate-label campaigns (200–300 vs 1,000) — the label-stability check can be run on the dynamic campaign now that it exists.
6. *(v3)* Approval for the grid-resolution rerun: static_peak vs arrival on the 3-h grid for 914_6 / 880_9 / 605_5 at N = 1,000 on Torch (one env change to the existing campaign script; ~2× the Week 23 wall time). This decides whether the §4.3 saturated-storm magnitudes are quoted as 6-h-grid values or replaced.

---

## 8. Files (all on `feat/dynamic-flood-maps`)

`analysis/static_vs_dynamic_v1/campaign_table_v1.{md,csv}` (23 rows), `fig_timing_effect_vs_level_v1.png`, `fig_prepeak_failures_v1.png`; three-storm N = 1,000 tables `arrival_vs_staticpeak_h360_n1000_v1.csv`, `arrival_clockonset_vs_staticpeak_h360_n1000_v1.csv`; `analysis/gnn_twin_v1/twin_heldout_v1.{md,csv}`, `fig_twin_cascade_pr_v1.png`, `fig_twin_val_loss_v1.png`; `analysis/TwinCampaign_20maps_resolved_v1.csv`; `analysis/gissr_delivered_maps_audit_v1.txt`; timing tables `data/flood/timing/node_timing_jesse22_v1{,_summary}.csv`. Dynamic campaign outputs: Torch `$SCRATCH/results/dynamic_v1/` and Mac `data/hpc_results_aug2026/dynamic_v1/` (46 files). v3: `analysis/static_vs_dynamic_v1/week25_arm_table_v1.csv`, `translation_test_T12_vs_R6_grid6_880_9_n100_v1.txt`, `surgephase_{plus6,minus6}_vs_translatedref_grid{6,3}_h360_n100_v1.csv`, `arrival_vs_staticpeak_grid3_h360_n100_v1.csv`, `arrival_shift{-6,6}_vs_arrival_grid3_h360_n100_v1.csv`; `scripts/week25_arm_table_v1.py`; `reports/Week25_ConceptNote_DynamicForcing_v3.md`; env knobs `DYNAMIC_TPEAK_SHIFT_H`, `DYNAMIC_INTRA_ORIGIN_SHIFT_H`, `DYNAMIC_PRE_PEAK_STEP_H` in `src/simulation/dynamic_forcing.py`.
