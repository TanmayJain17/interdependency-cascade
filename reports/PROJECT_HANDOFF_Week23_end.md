# PROJECT HANDOFF — Week 23 end (Aug 28, 2026)

Continuation of the NYC flood-cascade research (World 1 Monte Carlo simulator + World 2 CascadeGNN
surrogate). Read before acting. Theme of the week: **prove every step reduces exactly to the frozen
result before trusting the step that goes beyond it** — that discipline is why the timing result can be
stated as a controlled difference rather than a new model's opinion.

---

## 1. HEADLINE: seed timing is in, validated, and characterized across the whole S-curve

Branch `feat/dynamic-flood-maps` (cut from `feat/intra-power-coupling` at 4ae1f79, merged its final
sbatch commit a97fe91; HEAD c910d25 + this handoff). Old branch closed at 40c7e5a (twin analysis).
Tag `week22-twin-campaign` on 4ae1f79 (the commit the frozen campaign ran from).

### What was built
- **Timing features** (`external/gissr_tools/hydrograph_timing.py`, kept outside the repo until Gwen's
  code is on GitHub): per node, arrival / threshold-crossing / recession / duration / time-to-peak from
  the coastal hydrograph and the static map depth: `depth(t) = max(d − (W_peak − W(t)), 0)`. Only
  differences enter → datum and SLR cancel; flat hydrograph reduces exactly to static. Tables:
  `data/flood/timing/node_timing_jesse22_v1{,_summary}.csv` (23 scenarios, 6,941 wet node-scenario rows;
  synthetic-20 table kept as the simulator source of record).
- **Arrival-ordered seeding** (`src/simulation/dynamic_forcing.py`, engine + runner patched by
  `scripts/apply_dynamic_forcing_patch.py`): flood seeds go through the Week 19 `scheduled_failures`
  hook at their hour, cause `flood`; engine gains `t_origin` and `intra_clock_origin` (default-preserving);
  time axis = hours after onset, grid every 6 h to the peak then peak + {0,6,24,48,96}(+144,240,360 via
  `DYNAMIC_OFFSETS`); records peak-relative (data.py reads them unchanged). Config gate
  `config/dynamic_forcing.yaml` ships `enabled: false`; env `DYNAMIC_FORCING`, `DYNAMIC_MODE`
  (static_peak | static_onset | arrival | threshold), `DYNAMIC_OFFSETS`, `DYNAMIC_INTRA_CLOCK` (peak|onset),
  `DYNAMIC_ARRIVAL_SHIFT_H`.
- **Exact-reduction gate** (`scripts/dynamic_forcing_reduction_run.py` + `_check.py`): `static_peak`
  reproduces the frozen legacy campaign record-for-record (six fields) — proven on toy graphs with a
  clock-dependent closure (18/18), on the real system at N = 20/100/1,000 locally, and **23/23 scenarios
  at N = 1,000 on Torch (job 16456615, 0 failures)**. Sampler is prefix-consistent → all comparisons paired.
- **Dynamic campaign v1** (`hpc/campaign_dynamic_v1.sbatch`, cpu_short, 46 tasks ≈ 25 min):
  `$SCRATCH/results/dynamic_v1/{static_peak,arrival}_h360/` and Mac
  `data/hpc_results_aug2026/dynamic_v1/` (46 files, 7.6 GB).
- **GNN timing inputs** (`scripts/apply_gnn_timing_patch.py` → `src/gnn/{data,train}.py`): 3 features
  (arrival, duration, time-to-peak; hours/96) after the mask when `GNN_TIMING_FEATURES=1`,
  `GNN_TIMING_CSV=...`; width 9 → 12; default build byte-identical. Smoke-tested on 605_5 and gc_2050.

### Validated findings (all paired, same seeds and buffers, legacy arm)
- **Crash decision unchanged in 23/23** (22 identical; 880_9 0.982 → 0.995). Footprint decides IF.
- **Two regimes on the S-curve axis** (peak water level at The Battery, MSL, no SLR):
  ≤ 3.1 m → +2–15 % failures at peak+96, decaying to +1–5 % at peak+360 (same destination, earlier);
  ≥ 3.2 m → +21 % to +58 % at peak+96, **unchanged at peak+360** (different end state). Transition between
  880_9 (3.09 m: +13 → +5 %) and 514_13 (3.24 m: +22 → +21 %); coincides with the Week 22
  variance-amplification onset. GeoClaw trio straddles it (2026 converges +12 → +6 %; 2050 +58 %, 2080 +46 %
  persist) → not a synthetic-map artifact; gc_2050 persisting at 368 wet vs 880_9 converging at 393 → spatial
  pattern matters, not only size.
- **Pre-peak failures**: 58–1,804 per run; at the peak the dynamic world has 1.4–3.6× the static seed count
  down (605_5: 2,231 vs 612). Median node fails 0–6 h earlier in fast storms, 14–40 h earlier above 3.2 m.
- **Three-storm deep dive, N = 1,000, t360**: 914_6 +7.0 ± 15.5 (974/1000 larger); 880_9 +120 ± 320
  (980/1000); 605_5 +1,306 ± 176 (1000/1000, KS 0.989). Bracket: arrival lies between static_peak and
  static_onset in every storm.
- **Mechanism** (605_5, N = 100, t360): telecom +714, fuel +405, subway +84, water +40, power +24,
  hospital +9. Subway = Motter-Lai order dependence; telecom = the peak-anchored 1.5× demand surge landing
  on a network with 4× the dead sites; fuel/inter downstream.
- **Clock sensitivity** (surge anchored at onset, N = 1,000): end-state MEAN within 1–4 % of static
  (880_9 +18 ± 205, 605_5 +118 ± 529) but **753/1000 and 838/1000 runs end smaller**, spread 2–5× static →
  bimodal. Elapsed-time effect (t0–t48) is clock-independent; persistence is the surge interaction.
  ⇒ Decision 1 for Dr. Miura: peak-anchored (Sandy-consistent) vs onset-anchored surge.
- **Frozen 96-h horizon truncates the static cascade too** (+14–16 % from t96 to t360 in big storms):
  Week 22 `total_failures` is a 96-h snapshot.

### Twin CascadeGNN (job 16438671, legacy vs jesse labels; completes Week 22 item 2)
Both arms 6 epochs inside the policer window (1h10/1h11); best val 0.10824 / 0.11969 @ epoch 5.
Held-out loss ratio jesse/legacy 1.05 (914_6) → 1.19 (258_9) → 1.29 (gc_2050) → 1.48 (808_27) → 1.68
(605_5), rising with Week 22 σ-amplification. Cascade-only PR-AUC (input = t0 mask, scored on non-mask
nodes) within 1 pp at t6 everywhere; −3 to −6 pp at t24–96 only in 605_5/808_27. Files
`analysis/gnn_twin_v1/` on both branches. **Comms rule**: 0.95–0.98 here vs old 0.73 = same metric,
17 vs few training scenarios — the learning-curve HYPOTHESIS, not a result.

### GISSR (Gwen's model) — reviewed and run
`reports/`-level facts in the memo §6. Code: bathtub at peak per division once two data bugs are fixed
(NaN elevations → 9 northern-Manhattan divisions dry in ALL 20 delivered maps; overflow → 0 visible as
non-nesting in the 7 largest maps); cell-area in ft²; synthetic branch of the notebook cannot run as saved;
NOAA MSL vs DEM NAVD88 (−0.063 m); no SLR in catalogue water levels; depths up to 17.7 m on sub-zero DEM
cells. Maps = nested bathtubs indexed by peak water level → S-curve in metres: 156 wet ≈ 2.17 m,
218 ≈ 2.45 m, 390 ≈ 3.08 m. Tag convention: label = surge peak (999_12 = water level; 463_29 = warm row 1;
156_15 label truncated). Message to Gwen drafted/sent (memo §6); fixes belong in her repo.

---

## 2. PROCESS RULES LEARNED THIS WEEK (each one cost time)

1. **Never switch branches while a loop that spawns fresh `python` processes is running** (each scenario
   re-imports the code from disk). Check `jobs` first.
2. **`PYTHONUNBUFFERED=1` on every background loop**, not just sbatch — an empty log is not a stalled job.
3. **v1/v2 node files**: `data/flood/nyc_infra_{nodes,graph}_all_flood.*` are gitignored and LOCAL; the
   Mac copies are v2 (7,452). The frozen v1 inputs are `data/graph_v1_frozen/` (md5s: nodes 0934b000…,
   graph 0442c464…), on both machines. Every campaign sbatch sets NODES_IN/GRAPH_IN to them and gates on
   6,231 nodes. The driver's `unmatched ids` check caught this once.
4. **Torch has no push credentials** — commit on Torch only to move a diff (`git show HEAD -- path >
   file.diff`), apply and commit on the Mac, push from the Mac, `reset --hard origin/...` on Torch.
   Identity on Torch now set (Tanmay Jain, 127398328+TanmayJ17@users.noreply.github.com).
5. **`run_synthetic20.py` overwrites `data/flood/synthetic20_node_depths.csv` and
   `data/analysis/synthetic20_summary.json` with the filtered map set** (Torch's copy was a 1-column clobber;
   stashed there). Archive before, `git checkout --` after. The campaign's real depth record is its
   `temp_nodes_nyc_*.geojson`; the committed CSV differs on one node by 0.105 m (telecom_cluster_03042).
6. **Never let a dynamic run fall back to static silently**: the driver now exits if the scenario has no
   timing entry. Same class as the resolver-gate bug of Week 22.
7. **Extended horizons change cumulative fields** (`cause_counts`, fail_time entries beyond t96); the
   reduction checker compares on the reference horizons only and says so.
8. **GPU policer**: 6 epochs ≈ 70 min per arm on l40s_public is the working recipe until batching lands.
9. **CPU work goes to `cpu_short`/`cs`** (no GRES); 46 one-core tasks at 16-wide finished in ~25 min.
10. **Paste blocks once; every copy verified by `wc -l` and a grep for the new symbol** — held all week.

---

## 3. OPEN ITEMS (ordered)

1. **Send the memo** (`reports/Week23_Memo_DynamicFlooding_v1.md`): §1–3 + figures to Dr. Miura, full to
   Dr. Lin; the two decisions in §7 gate the December framing (surge clock) and World 2 (2×2 approval).
2. **±3–6 h arrival perturbation** (loop launched Fri; `analysis/static_vs_dynamic_v1/arrival_shift*`) →
   one sentence in memo §4.6 (→ v2).
3. **World 2 2×2**: labels {static: `legacy_v1_n1000`, dynamic: `dynamic_v1/arrival_h360`} × inputs
   {mask, mask+timing}. Sbatch = `campaign_gnn_twin_v1.sbatch` pattern with `CASCADE_RESULTS_DIR`/
   `SYN_RESULTS_DIR` → the arrival dir and `GNN_TIMING_FEATURES=1 GNN_TIMING_CSV=...`; 6 epochs; the
   heterodata gate stays. Phase-0: confirm `data.py` labels from peak-relative negative fail times (verified
   in the smoke: 605_5 t96 label count 1,608) and that `initial_failures` lists all seeds (it does).
4. **Sandy hindcast timeline** (ConEd E 13th ~21:00, NYU Langone 19:30–23:00, tunnels, FCC >25 %) against
   the geoclaw arrival-ordered run — the qualitative validation the lit review asked for.
5. **Jesse arm under dynamic forcing**: his t=6 floor is peak-anchored; re-derive as `t_peak + 6` (or per
   his time-resolved states) before any physics-arm dynamic run. Driver currently refuses it.
6. **Intra-infrastructure verification pass** (side goal): start with the `regime=pluvial` fragility flag
   printed on coastal maps; then one type per pass (power → subway → water → fuel → telecom → hospital),
   each against a Sandy anchor. The telecom demand-surge timing question belongs here too.
7. **GISSR production batch**: after Gwen's fixes (F1–F4) land; SLR by storm year; stratified catalogue
   sample with ~60 % in 2.2–3.1 m; spatial diversity decision (GeoClaw ensemble vs gauge-scaled GISSR).
   Reproduce two delivered maps first. `check_delivered_map.py` is the audit tool.
8. **Batched training step + learning-curve experiment** (Dr. Lin's 200–300-map question; the 0.73 → 0.95
   jump is the hypothesis).
9. **Option B remainder**: depth-duration fragility (Nofal & van de Lindt 2020), recovery gated on
   recession (duration_h and recede_h are already in the timing table).
10. Housekeeping still open: loader cosmetic (print resolved per-scenario coverage); move `gissr_tools`
    into `scripts/gissr/` once Gwen's repo is public; December deck figures in Gamma theme.

---

## 4. FILE / PATH QUICK REFERENCE

- Repo: TanmayJain17/interdependency-cascade. Mac remote `personal`, Torch remote `origin` (same repo).
  Branches: `feat/intra-power-coupling` (closed, 40c7e5a), `feat/dynamic-flood-maps` (active). Torch is on
  the dynamic branch at 29cbe85+ (`git pull` there before any new sbatch).
- Key commits: 777b61f (dynamic forcing wiring + timing table), 6306b3f (reduction driver), 2860ee0
  (N=100 smoke + horizon), ff0af65 (twin GNN comparison + clock sensitivity), 0c75f77 (GNN timing inputs +
  jesse22 table), 29cbe85 (campaign sbatch, strict driver), f239256 (N=1000 3-storm), 91f44a7 (23-scenario
  campaign), c910d25 (memo).
- Analysis: `analysis/static_vs_dynamic_v1/` (campaign_table_v1.{md,csv}, fig_timing_effect_vs_level_v1.png,
  fig_prepeak_failures_v1.png, 3-storm N=1000 tables, clock-onset tables), `analysis/gnn_twin_v1/`,
  `analysis/TwinCampaign_20maps_resolved_v1.csv`, `analysis/gissr_delivered_maps_audit_v1.txt`.
- Scripts: `scripts/{apply_dynamic_forcing_patch,dynamic_forcing_reduction_run,dynamic_forcing_reduction_check,
  compare_static_dynamic_v1,plot_static_dynamic_campaign_v1,compare_twin_gnn_v1,apply_gnn_timing_patch}.py`;
  `hpc/campaign_dynamic_v1.sbatch`. External: `external/GISSR_generalized/` (Gwen's zip, unzipped as
  delivered), `external/gissr_tools/{gissr_core,gissr_raster,check_delivered_map,hydrograph_timing}.py`.
- Data on the Mac: `data/graph_v1_frozen/` (v1 inputs), `data/hpc_results_aug2026/{legacy,jesse}_v1_n1000/`,
  `.../gnn_twin_v1/`, `.../dynamic_v1/`; Gwen's rasters `~/Downloads/nyc_synthetic_flood/` (20 GeoTIFFs);
  Jesse's library `Tanmay_Handoff/`.
- Job-ID history: 16438671 (twin retrain, COMPLETED), 16456569 (dyn-v1 smoke), 16456615 (dyn-v1 campaign,
  46/46 COMPLETED, 23 reduction OK).

**First message of next session should ask: advisor answers to memo §7 (surge clock; 2×2 approval),
Gwen's reply, and the perturbation table — then proceed down §3 in order.**
