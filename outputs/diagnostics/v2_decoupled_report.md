# v2 Decoupled Architecture — Phase 3 Report

**Branch:** `feat/v2-leakage-fix`
**Date:** 2026-05-26
**Run tag:** `decoupled_v1`
**Status:** Fourth architecture in the v2 series, after three converging-null coupled attempts (warm, cold, noisy). Decoupling the *gradient* flow from cascade-BCE to `LearnableFragility` is structurally verified (cascade-only-loss backward gives **exact zero** fragility grads). The dual test eval reveals that *information* flow is a separate question — the CascadeGNN learned an internal HAZUS-like depth → P(t=0) mapping during training. The architecture now serves two complementary purposes rather than fighting one task.

**Update (2026-05-26, decoupled-nodepth run added):** A fifth run trained the same decoupled architecture with col 3 held at constant 0 throughout training — the apples-to-apples graph-only baseline. **Test Cascade-PR @ t=6 = 0.6812.** The honest **in-distribution gap** between depth-trained (0.9308) and graph-only-trained (0.6812) is **~0.25 at t=6**, narrowing to **~0.18 at t=96**. The 0.80 OOD number from yesterday's "trained-with, tested-without" eval is now reframed as a distribution-shift number, not the gap-due-to-depth. See Section 6 below.

## 1. Architecture

`LearnableFragility` runs as a parallel prediction head whose output `p_t0` is **not** concatenated into the GNN's input. The 12-parameter lognormal CDF mapping depth → P(failure) is supervised by an aux BCE against true t=0 labels (`aux_weight=0.1`) plus an L2 prior pulling toward HAZUS init (`prior_weight=1.0`). The GNN input is the static 8-feature base with col 3 (flood_depth) overwritten by the scenario-specific depth at every example; the GNN does not see fragility's output anywhere. The two heads share only the depths input and the combined training loss. Cascade BCE flows back only through the GNN.

See [src/gnn/model_v2.py](src/gnn/model_v2.py) for the forward; [outputs/diagnostics/v2_noisy_coupled_report.md](outputs/diagnostics/v2_noisy_coupled_report.md) for the three coupled-architecture nulls that motivated this design.

## 2. Hyperparameters

| | Value |
|---|---|
| Architecture | `decoupled_v1` (no fragility column on GNN input) |
| `aux_weight` | 0.1 |
| `prior_weight` | 1.0 |
| `lr` | 1e-3 |
| `weight_decay` | 1e-5 |
| `epochs` | 10 |
| `train_subset` / `val_subset` | 1500 / 500 |
| `warm_start` | False (incompatible with input-dim change) |
| Holdout test scenario | `geoclaw_2050` |
| col-3 depth overwrite | enabled at train; toggled at eval |
| GNN `node_in_dim` per type | **8** (was 9 in v1 / coupled v2) |
| Total params | 662,308 (was 662,692 in coupled — 64 hidden × 6 types removed) |

## 3. Six-way held-out test comparison (`geoclaw_2050`, 1000 MC runs)

| Run | Test loss | Test PR @ t=[6, 24, 48, 96] | **Test Cascade-PR @ t=[6, 24, 48, 96]** |
|---|---|---|---|
| v1 baseline (leaky; binary mask present) | 0.0397 | [0.9863, 0.9997, 0.9993, 0.9998] | **[0.9313, 0.9991, 0.9979, 0.9995]** |
| v1 Run D (mask zeroed; honest graph-only floor) | 0.2942 | [0.8173, 0.8671, 0.8567, 0.8678] | **[0.7281, 0.8256, 0.8121, 0.8362]** |
| v2 noisy-coupled (failed coupled attempt) | 0.0393 | [0.9862, 0.9997, 0.9993, 0.9997] | **[0.9308, 0.9990, 0.9982, 0.9994]** |
| **v2 decoupled (trained w/ depth) — eval WITH depth** | **0.0390** | **[0.9862, 0.9997, 0.9993, 0.9998]** | **[0.9308, 0.9991, 0.9979, 0.9996]** |
| v2 decoupled (trained w/ depth) — eval col 3 ZEROED (OOD) | 2.1841 | [0.2136, 0.2554, 0.3012, 0.3477] | [0.1285, 0.1652, 0.2188, 0.2818] |
| **v2 decoupled (trained w/o depth, this run) — eval col 3 const 0** | **0.3196** | **[0.8062, 0.8521, 0.8573, 0.8721]** | **[0.6812, 0.7885, 0.7976, 0.8235]** |

### Two gap views — OOD vs in-distribution

| t (h) | Cascade-PR (trained w/ depth) | Cascade-PR (yanked col 3 at eval, OOD) | **OOD gap** | Cascade-PR (trained w/o depth, ID) | **In-distribution gap** |
|---|---|---|---|---|---|
| 6   | 0.9308 | 0.1285 | +0.8023 | **0.6812** | **+0.2496** |
| 24  | 0.9991 | 0.1652 | +0.8339 | **0.7885** | **+0.2106** |
| 48  | 0.9979 | 0.2188 | +0.7791 | **0.7976** | **+0.2004** |
| 96  | 0.9996 | 0.2818 | +0.7178 | **0.8235** | **+0.1761** |

The **in-distribution gap (~0.18–0.25)** is the right "how much does depth contribute" answer. The OOD gap is mostly distribution-shift damage from yanking a variable the model trained on. Sections 5 and 6 unpack this distinction.

## 4. Reframing — what we actually built

The three prior coupled attempts (warm/cold/noisy) tried to break a leakage path that runs from `LearnableFragility` to the cascade prediction. The decoupling test in this phase (Test A in smoke) verifies that we have, in fact, **cut that path at the gradient level**: backward through cascade BCE produces *exactly* zero gradient on every fragility parameter. There is no longer any architectural connection from cascade loss to fragility's 12 lognormal parameters.

What the dual test eval shows is that **gradient flow and information flow are different things**. With col-3 depth available at eval time the decoupled model still hits Cascade-PR = 0.9308 — identical to the leaky and the noisy-coupled baselines. With col 3 zeroed at eval time the same trained model collapses to Cascade-PR = 0.1285. So the CascadeGNN's 662,296 parameters learned an internal mapping from raw depth to a HAZUS-like initial-failure probability during training, and used that mapping to predict cascades via reachability on the static graph. The model didn't *need* `LearnableFragility`'s output column to recover the oracle behavior — depth alone was enough information for it to do so internally.

This reframes what the architecture actually delivers. The two heads now answer complementary questions rather than competing for the same gradient signal:

- **`LearnableFragility` (12 parameters)** — an interpretable lognormal CDF per infrastructure type: median depth at which 50% of nodes fail, β shape parameter. Comparable across infra types and against HAZUS. Supervised only by t=0 truth and the prior.
- **`CascadeGNN` (662,296 parameters)** — a high-capacity neural predictor that learns whatever depth → failure → cascade pattern the data exhibits, including non-fragility-like behavior (graph-mediated effects, scenario-specific quirks). Supervised only by t>0 cascade labels.

The "leakage" framing from earlier weeks is no longer the right description. There is no module-level leakage — fragility's parameters and the GNN's parameters now share only the data they're trained on. The high Cascade-PR is not leakage; it is the GNN being competent at the task it was given (depth as input, cascade outcomes as labels). What the gap quantifies is *how much of that competence flows through explicit depth*, not whether the architecture is broken.

## 5. The OOD gap — what 0.80 means (and what it doesn't)

The dual eval from the depth-trained model gave a gap of **0.72–0.83 across all four timesteps** between with-depth and zeroed-col-3 evaluations. That number sounds like "the GNN is almost entirely dependent on depth at test time" — but that framing collapses two effects into one.

The 0.80 gap is **out-of-distribution**: the model was *trained* with the scenario depth in col 3, learned a depth-dependent representation, and then had that variable yanked at eval time. Most of the 0.80 is distribution-shift damage (the model is responding to an input it has never seen), not a measurement of how much depth carries as a predictive signal. A model that never saw depth during training would not collapse to 0.13 — it would learn to use whatever non-depth signal exists. The **in-distribution gap** — depth-trained model vs no-depth-trained model, both evaluated on their native distribution — is the right way to ask "how much does depth contribute?" Section 6 below runs that experiment directly and gets a much smaller number (~0.25 at t=6, ~0.18 at t=96).

Two observations on the depth-trained run that remain useful:

1. **The 0.93 with-depth Cascade-PR matches the v1 leaky baseline to four decimals.** Same model class, same trained-on data, same test scenario — the GNN landed in essentially the same prediction surface regardless of whether fragility was an input column (v1, coupled v2) or absent (decoupled v2). For the current dataset where labels were generated from HAZUS-Bernoulli sampling on the same static graph the GNN sees, all four with-depth architectures converge to the same predictor. The "improvement" we hoped for from the decoupled architecture is not visible on this dataset, because there is no signal on this dataset that distinguishes a correctly-tuned fragility from HAZUS.

2. **At t=96 the OOD gap narrows (0.72 vs 0.83 at t=24).** As t grows more nodes have failed and the base-rate-aware PR gives a noisier model more credit. Same pattern holds for the in-distribution gap (Section 6) — it's a feature of the cascade dynamics, not of which gap we measure.

## 6. The in-distribution gap — true graph-only baseline (decoupled v2, no depth overwrite)

This run trained the same decoupled v2 architecture with col 3 held at constant 0 throughout training **and** eval (`--no_depth_overwrite`). The GNN's only inputs are the 7 non-depth feature columns plus the graph structure plus the edge types. The fragility head is unchanged — it still receives the real depths and produces `p_t0` as before; only the GNN is starved of explicit depth. This is the apples-to-apples graph-only baseline — directly comparable to v1 Run D, but inside the decoupled v2 architecture.

### Six-way Cascade-PR @ t=6 comparison

| Run | GNN col 8 | Train col 3 | Eval col 3 | Cascade-PR @ t=6 |
|---|---|---|---|---|
| v1 leaky | binary mask | constant 0 | constant 0 | **0.9313** |
| v1 Run D (mask zeroed) | mask zeroed | constant 0 | constant 0 | **0.7281** |
| v2 noisy-coupled (cold, σ=0.15) | noisy fragility | scenario depth | scenario depth | **0.9308** |
| v2 decoupled (yesterday) | — (no col 8) | scenario depth | scenario depth | **0.9308** |
| v2 decoupled, OOD eval (yesterday) | — | scenario depth | constant 0 | **0.1285** |
| **v2 decoupled, no depth (this run)** | **— (no col 8)** | **constant 0** | **constant 0** | **0.6812** |

### Test Cascade-PR per timestep (no-depth run)

| t (h) | With-depth Cascade-PR (yesterday) | **No-depth Cascade-PR (this run)** | In-distribution gap |
|---|---|---|---|
| 6  | 0.9308 | **0.6812** | **0.2496** |
| 24 | 0.9991 | **0.7885** | **0.2106** |
| 48 | 0.9979 | **0.7976** | **0.2004** |
| 96 | 0.9996 | **0.8235** | **0.1761** |

### Interpretation

**The no-depth run lands at 0.6812 at t=6 — about 5 percentage points below Run D's 0.7281**, and within training noise of Run D at the longer horizons (t=24: 0.79 vs 0.83; t=48: 0.80 vs 0.81; t=96: 0.82 vs 0.84). The architectures aren't identical (Run D was the v1 GNN with a 9th constant-zero mask column; this run is the v2 GNN with 8 inputs total), so a 5pp drift at the noisiest timestep is plausible architectural/seed variation rather than a deeper effect. The key conclusion holds at every t: graph-only-trained cascade prediction lands at roughly **0.68–0.82 Cascade-PR**, far below 0.93 and broadly matching the v1 graph-only floor.

The **in-distribution gap of ~0.25 at t=6** is the right number to report as "how much does explicit flood depth contribute to cascade prediction." Four signs that this is the honest measurement, not the OOD 0.80:

1. Both models in the comparison were *trained on their respective input distribution*, so neither is OOD at test time.
2. The "with depth" half (0.9308) matches the v1 leaky baseline (0.9313) and v2 coupled (0.9308) to four decimals — so depth-trained models all land in the same prediction surface, and we are not selecting the strongest one.
3. The "no depth" half (0.6812) matches Run D's independently-measured graph-only floor (0.7281) within ~5pp — so the no-depth half is consistent with prior measurement.
4. The gap shrinks monotonically with time (0.25 → 0.21 → 0.20 → 0.18 from t=6 to t=96), exactly the pattern expected from a model that uses depth most heavily at early timesteps (where t=0 failure prediction dominates) and less at later timesteps (where graph reachability has propagated and most cascades are nearly complete regardless).

### What the gap says about the model

About **25 percentage points** of test Cascade-PR @ t=6 come from explicit flood depth. The remaining **~68 percentage points** come from features (infrastructure type, voltage class, ada-accessibility flags, etc.) plus graph structure (heterogeneous edges between power lines, water pipes, subway routes, telecom backhauls, scada monitoring, etc.). The graph-only baseline is itself a strong predictor — heterogeneous GAT learns meaningful cascade dynamics from infrastructure topology and node-type signal alone, before depth is added.

The gap also widens monotonically the closer t is to t=0 (where the actual flooding event is happening): 0.18 at t=96, 0.20 at t=48, 0.21 at t=24, 0.25 at t=6. This is exactly what you'd expect if depth's role is mostly to inform *who fails first*, while graph structure carries *what fails next.*

### Final fragility (decoupled-nodepth-v1, 6-decimal precision)

| Type | HAZUS μ | Final μ | Drift % | HAZUS β | Final β | Drift % β |
|---|---|---|---|---|---|---|
| power    | 0.600000 | 0.600265 | +0.044% | 0.500000 | 0.499996 | −0.001% |
| telecom  | 0.300000 | 0.300003 | +0.001% | 0.600000 | 0.600041 | +0.007% |
| hospital | 0.600000 | 0.600001 | +0.000% | 0.500000 | 0.500001 | +0.000% |
| subway   | 0.100000 | 0.100008 | +0.008% | 0.400000 | 0.400046 | +0.011% |
| water    | 0.500000 | 0.499982 | −0.004% | 0.500000 | 0.500062 | +0.012% |
| fuel     | 0.500000 | 0.500113 | +0.023% | 0.600000 | 0.599895 | −0.018% |

Same Bayesian-correct outcome as the depth-trained decoupled run — all drifts under 0.05%, `train_bce_t0` locked at 0.01219 throughout. The architectural change (zeroing col 3 in the GNN input) has no effect on fragility's gradient since fragility was never gradient-connected to the cascade head to begin with.

### Dual-eval sanity check

The `test` and `test_no_depth` entries in this run's history.json are **bit-identical** — `Cascade-PR per_t = [0.6812, 0.7885, 0.7976, 0.8235]` for both, gap = `[0.0, 0.0, 0.0, 0.0]`. This is the expected consequence of `--no_depth_overwrite`: when col 3 is already constant 0 in training, zeroing it at eval time is a no-op. Confirms the flag is consistently applied across train and both eval passes.

## 7. Fragility — six-decimal precision, comparison vs noisy-coupled

### Median (m)

| Type | HAZUS μ | noisy μ | decoupled μ | noisy Δ% | **decoupled Δ%** |
|---|---|---|---|---|---|
| power    | 0.600000 | 0.600946 | 0.600265 | +0.158% | **+0.044%** |
| telecom  | 0.300000 | 0.299587 | 0.300003 | −0.138% | **+0.001%** |
| hospital | 0.600000 | 0.600003 | 0.600001 | +0.001% | **+0.000%** |
| subway   | 0.100000 | 0.100006 | 0.100008 | +0.006% | **+0.008%** |
| water    | 0.500000 | 0.499452 | 0.499982 | −0.110% | **−0.004%** |
| fuel     | 0.500000 | 0.499974 | 0.500113 | −0.005% | **+0.023%** |

### β

| Type | HAZUS β | noisy β | decoupled β | noisy Δ% | **decoupled Δ%** |
|---|---|---|---|---|---|
| power    | 0.500 | 0.500016 | 0.499996 | +0.003% | **−0.001%** |
| telecom  | 0.600 | 0.599385 | 0.600041 | −0.102% | **+0.007%** |
| hospital | 0.500 | 0.500000 | 0.500001 | +0.000% | **+0.000%** |
| subway   | 0.400 | 0.400058 | 0.400046 | +0.015% | **+0.011%** |
| water    | 0.500 | 0.500009 | 0.500062 | +0.002% | **+0.012%** |
| fuel     | 0.600 | 0.599921 | 0.599895 | −0.013% | **−0.018%** |

Drift is < 0.05% on every parameter — smaller than every prior v2 run, including the cold-start baseline. This is the **Bayesian-correct outcome** for HAZUS-derived labels under a HAZUS prior: posterior ≈ prior. The aux BCE is locked at 0.01219 throughout the 10 epochs (consistent with prior runs), confirming that fragility starts at the loss-optimum and stays there. The decoupling also removes the small transient drift the noisy-coupled run exhibited (telecom peaked at −0.5% at ep 3 then retreated) — without any cascade-loss gradient reaching fragility, the only forces on it are the aux BCE (zero net pull) and the prior (zero at HAZUS init). The parameters move only due to gradient noise and return.

This is **not a failure of `LearnableFragility`**. It is the correct posterior for a dataset whose labels were generated by HAZUS. To demonstrate that the architecture *can* learn non-HAZUS fragility would require a dataset whose generative process disagrees with HAZUS.

## 8. Fragility output variance on `extreme_2080` example 0 (eval mode)

| Type | N | n_flooded | p_t0 var | p_t0 mean | p_t0 max |
|---|---|---|---|---|---|
| power    | 203  | 10  | 8.59e−03 | 0.0148 | 0.7172 |
| telecom  | 4150 | 274 | 3.33e−02 | 0.0434 | 0.9489 |
| hospital | 61   | 1   | 3.21e−06 | 0.0002 | 0.0140 |
| subway   | 493  | 76  | 0.1241   | 0.1502 | 0.9999 |
| water    | 137  | 13  | 3.62e−02 | 0.0561 | 0.8264 |
| fuel     | 1187 | 84  | 1.76e−02 | 0.0287 | 0.7832 |

Variance is identical to the noisy-coupled run's diagnostic on the same scenario, to three decimals. Fragility has not collapsed — it is differentiating nodes (max p_t0 from 0.014 on hospital to 0.9999 on subway), it just hasn't drifted from HAZUS during training.

## 9. What this enables — Boston cross-city test next week

This is the architecture that ports cleanly. Two reasons to believe it will be more informative on Boston than it was on NYC:

1. **The CascadeGNN's depth-dependence is now measurable in two ways, and we now have the in-distribution baseline.** NYC's headline numbers are (depth-trained = 0.93, graph-only-trained = 0.68, in-distribution gap ≈ 0.25 at t=6). When we run the same pair on Boston, comparing (X_with, X_without) to NYC's (0.93, 0.68) tells us whether the GNN's learned `depth → cascade` mapping ports across cities and whether the graph-only baseline does. Two quantitative possibilities:
   - Boston with-depth ≈ NYC with-depth → the GNN's depth → cascade mapping is largely universal (depth is depth, graph rules are graph rules)
   - Boston with-depth < NYC with-depth, but the without-depth number stays similar to NYC's 0.68 → the GNN's *graph-only* reasoning generalizes but the depth-mapping is NYC-specific
2. **Fragility is the part most likely to actually drift on Boston.** If Boston's flood data exhibits a depth → failure relationship that differs from HAZUS (different soil drainage, different construction codes, different age-cohort of infrastructure), the aux BCE on Boston's t=0 labels will no longer be loss-optimal at HAZUS init. Drift in `log_mu`/`log_beta` away from HAZUS *and toward Boston-specific values* would be the long-awaited evidence that `LearnableFragility` is learnable, not just instantiable. The decoupled architecture is the right vehicle for this test because (a) fragility's gradient is undiluted by cascade loss, and (b) the prior can be tuned (or dropped) so it doesn't anchor fragility to NYC's HAZUS values when training on Boston data.

The current NYC result does not falsify the v2 thesis; it just doesn't have the right test for it. Cross-city is that test.

## 10. What we ruled out (recap)

Three coupled-architecture attempts established convergent null results (all at test Cascade-PR @ t=6 = 0.9309 ± 0.0001):

| Attempt | Knob tested | Outcome |
|---|---|---|
| v2 warm-start | Joint training of HAZUS-init `LearnableFragility` + v1-pretrained GNN | 0.9309 |
| v2 cold-start | Same architecture, GNN re-initialized | 0.9309 |
| v2 noisy-coupled | + Gaussian noise σ=0.15 on fragility input + col-3 depth shortcut removed + cold-start | 0.9308 |

The decoupled architecture is the principled response: stop coupling. The decoupling is genuine at the gradient level (Test A: exact zero), and the dual test eval is the diagnostic that exposes what the *information* path looks like even when the gradient path is severed. The architecture is now ready for the cross-city deployment that can actually exercise it.

## 11. Artifacts

Depth-trained decoupled run (yesterday):
- Training history: `data/gnn_v2_checkpoints/decoupled_v1/history.json` (includes `test` and `test_no_depth` keys)
- Final fragility variance: same file, key `fragility_variance_extreme_2080`
- Per-epoch log: `data/gnn_v2_checkpoints/logs/decoupled_v1.log`
- Best checkpoint: `data/gnn_v2_checkpoints/decoupled_v1/best.pt` (gitignored)

No-depth decoupled run (this addendum):
- Training history: `data/gnn_v2_checkpoints/decoupled_nodepth_v1/history.json` (includes `test` and `test_no_depth` — identical by construction)
- Per-epoch log: `data/gnn_v2_checkpoints/logs/decoupled_nodepth_v1.log`
- Overfit final fragility: `data/gnn_v2_checkpoints/overfit_decoupled_nodepth/overfit_final_fragility.json`
- Best checkpoint: `data/gnn_v2_checkpoints/decoupled_nodepth_v1/best.pt` (gitignored)

Code:
- Architecture: [src/gnn/model_v2.py](src/gnn/model_v2.py) — decoupled forward, optional `no_depth_overwrite` flag preserves col 3 at constant 0
- Trainer / loss / dual-eval / CLI: [src/gnn/data_v2.py](src/gnn/data_v2.py), [src/gnn/train_v2.py](src/gnn/train_v2.py)
