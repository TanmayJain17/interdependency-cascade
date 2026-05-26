# v2 Decoupled Architecture — Phase 3 Report

**Branch:** `feat/v2-leakage-fix`
**Date:** 2026-05-26
**Run tag:** `decoupled_v1`
**Status:** Fourth architecture in the v2 series, after three converging-null coupled attempts (warm, cold, noisy). Decoupling the *gradient* flow from cascade-BCE to `LearnableFragility` is structurally verified (cascade-only-loss backward gives **exact zero** fragility grads). The dual test eval reveals that *information* flow is a separate question — the CascadeGNN learned an internal HAZUS-like depth → P(t=0) mapping during training. The architecture now serves two complementary purposes rather than fighting one task.

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

## 3. Five-way held-out test comparison (`geoclaw_2050`, 1000 MC runs)

| Run | Test loss | Test PR @ t=[6, 24, 48, 96] | **Test Cascade-PR @ t=[6, 24, 48, 96]** |
|---|---|---|---|
| v1 baseline (leaky; binary mask present) | 0.0397 | [0.9863, 0.9997, 0.9993, 0.9998] | **[0.9313, 0.9991, 0.9979, 0.9995]** |
| v1 Run D (mask zeroed; honest graph-only floor) | 0.2942 | [0.8173, 0.8671, 0.8567, 0.8678] | **[0.7281, 0.8256, 0.8121, 0.8362]** |
| v2 noisy-coupled (failed coupled attempt) | 0.0393 | [0.9862, 0.9997, 0.9993, 0.9997] | **[0.9308, 0.9990, 0.9982, 0.9994]** |
| **v2 decoupled — eval WITH depth in col 3** | **0.0390** | **[0.9862, 0.9997, 0.9993, 0.9998]** | **[0.9308, 0.9991, 0.9979, 0.9996]** |
| **v2 decoupled — eval with col 3 ZEROED** | **2.1841** | **[0.2136, 0.2554, 0.3012, 0.3477]** | **[0.1285, 0.1652, 0.2188, 0.2818]** |

### The headline gap (decoupled, with depth − without depth)

| t (h) | Cascade-PR with depth | Cascade-PR no depth | **Gap** |
|---|---|---|---|
| 6   | 0.9308 | 0.1285 | **+0.8023** |
| 24  | 0.9991 | 0.1652 | **+0.8339** |
| 48  | 0.9979 | 0.2188 | **+0.7791** |
| 96  | 0.9996 | 0.2818 | **+0.7178** |

## 4. Reframing — what we actually built

The three prior coupled attempts (warm/cold/noisy) tried to break a leakage path that runs from `LearnableFragility` to the cascade prediction. The decoupling test in this phase (Test A in smoke) verifies that we have, in fact, **cut that path at the gradient level**: backward through cascade BCE produces *exactly* zero gradient on every fragility parameter. There is no longer any architectural connection from cascade loss to fragility's 12 lognormal parameters.

What the dual test eval shows is that **gradient flow and information flow are different things**. With col-3 depth available at eval time the decoupled model still hits Cascade-PR = 0.9308 — identical to the leaky and the noisy-coupled baselines. With col 3 zeroed at eval time the same trained model collapses to Cascade-PR = 0.1285. So the CascadeGNN's 662,296 parameters learned an internal mapping from raw depth to a HAZUS-like initial-failure probability during training, and used that mapping to predict cascades via reachability on the static graph. The model didn't *need* `LearnableFragility`'s output column to recover the oracle behavior — depth alone was enough information for it to do so internally.

This reframes what the architecture actually delivers. The two heads now answer complementary questions rather than competing for the same gradient signal:

- **`LearnableFragility` (12 parameters)** — an interpretable lognormal CDF per infrastructure type: median depth at which 50% of nodes fail, β shape parameter. Comparable across infra types and against HAZUS. Supervised only by t=0 truth and the prior.
- **`CascadeGNN` (662,296 parameters)** — a high-capacity neural predictor that learns whatever depth → failure → cascade pattern the data exhibits, including non-fragility-like behavior (graph-mediated effects, scenario-specific quirks). Supervised only by t>0 cascade labels.

The "leakage" framing from earlier weeks is no longer the right description. There is no module-level leakage — fragility's parameters and the GNN's parameters now share only the data they're trained on. The high Cascade-PR is not leakage; it is the GNN being competent at the task it was given (depth as input, cascade outcomes as labels). What the gap quantifies is *how much of that competence flows through explicit depth*, not whether the architecture is broken.

## 5. The gap — what 0.80 means

Predicted gap was 0.10–0.20. Observed gap is **0.72–0.83 across all four timesteps.** The decoupled GNN is almost entirely dependent on the col-3 depth signal at test time. Three observations:

1. **The no-depth Cascade-PR (0.13 at t=6) is below v1 Run D's honest floor (0.73).** That looks paradoxical at first — Run D was the "blind cascade" baseline. But the experiments are not the same setup: Run D was *trained* with the binary failure mask zeroed, so its GNN learned to predict cascades from features + graph alone. Our decoupled v2 was *trained* with depth, learned a depth-dependent representation, and then had depth removed at test time. The "without depth" eval here is "what does this model think when you yank the variable it relied on," not "how well can a model trained without depth do." A direct apples-to-apples comparison would require retraining the decoupled v2 with col 3 also constant (left for future work).

2. **The 0.93 with-depth Cascade-PR matches the v1 leaky baseline to four decimals.** Same model class, same trained-on data, same test scenario — the GNN landed in essentially the same prediction surface regardless of whether fragility was an input column (v1, coupled v2) or absent (decoupled v2). For the current dataset where labels were generated from HAZUS-Bernoulli sampling on the same static graph the GNN sees, all four architectures converge to the same predictor. The "improvement" we hoped for from the decoupled architecture is not visible on this dataset, because there is no signal on this dataset that distinguishes a correctly-tuned fragility from HAZUS.

3. **At t=96 the gap narrows (0.72 vs 0.83 at t=24).** As t grows more nodes have failed and the base-rate-aware PR gives a noisier model more credit. The gap is largest at t=6 and t=24 — the timesteps closest to the t=0 oracle — and shrinks at t=96 where reachability has saturated.

## 6. Fragility — six-decimal precision, comparison vs noisy-coupled

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

## 7. Fragility output variance on `extreme_2080` example 0 (eval mode)

| Type | N | n_flooded | p_t0 var | p_t0 mean | p_t0 max |
|---|---|---|---|---|---|
| power    | 203  | 10  | 8.59e−03 | 0.0148 | 0.7172 |
| telecom  | 4150 | 274 | 3.33e−02 | 0.0434 | 0.9489 |
| hospital | 61   | 1   | 3.21e−06 | 0.0002 | 0.0140 |
| subway   | 493  | 76  | 0.1241   | 0.1502 | 0.9999 |
| water    | 137  | 13  | 3.62e−02 | 0.0561 | 0.8264 |
| fuel     | 1187 | 84  | 1.76e−02 | 0.0287 | 0.7832 |

Variance is identical to the noisy-coupled run's diagnostic on the same scenario, to three decimals. Fragility has not collapsed — it is differentiating nodes (max p_t0 from 0.014 on hospital to 0.9999 on subway), it just hasn't drifted from HAZUS during training.

## 8. What this enables — Boston cross-city test next week

This is the architecture that ports cleanly. Two reasons to believe it will be more informative on Boston than it was on NYC:

1. **The CascadeGNN's depth-dependence is now measurable.** On NYC the gap was 0.80, which is large but uninterpretable in isolation — we have no city where we'd expect a meaningfully different gap. If Boston gives Cascade-PR with depth = 0.X and without depth = 0.Y, comparing (X, Y) to NYC's (0.93, 0.13) tells us whether the GNN's depth-mapping transfers (graph topology + features + depth → cascade) or whether Boston demands re-learning. Two quantitative possibilities:
   - Boston with-depth ≈ NYC with-depth → the GNN's depth → cascade mapping is largely universal (depth is depth, graph rules are graph rules)
   - Boston with-depth < NYC with-depth, but the without-depth number stays similar → the GNN's *graph-only* reasoning generalizes but the depth-mapping is NYC-specific
2. **Fragility is the part most likely to actually drift on Boston.** If Boston's flood data exhibits a depth → failure relationship that differs from HAZUS (different soil drainage, different construction codes, different age-cohort of infrastructure), the aux BCE on Boston's t=0 labels will no longer be loss-optimal at HAZUS init. Drift in `log_mu`/`log_beta` away from HAZUS *and toward Boston-specific values* would be the long-awaited evidence that `LearnableFragility` is learnable, not just instantiable. The decoupled architecture is the right vehicle for this test because (a) fragility's gradient is undiluted by cascade loss, and (b) the prior can be tuned (or dropped) so it doesn't anchor fragility to NYC's HAZUS values when training on Boston data.

The current NYC result does not falsify the v2 thesis; it just doesn't have the right test for it. Cross-city is that test.

## 9. What we ruled out (recap)

Three coupled-architecture attempts established convergent null results (all at test Cascade-PR @ t=6 = 0.9309 ± 0.0001):

| Attempt | Knob tested | Outcome |
|---|---|---|
| v2 warm-start | Joint training of HAZUS-init `LearnableFragility` + v1-pretrained GNN | 0.9309 |
| v2 cold-start | Same architecture, GNN re-initialized | 0.9309 |
| v2 noisy-coupled | + Gaussian noise σ=0.15 on fragility input + col-3 depth shortcut removed + cold-start | 0.9308 |

The decoupled architecture is the principled response: stop coupling. The decoupling is genuine at the gradient level (Test A: exact zero), and the dual test eval is the diagnostic that exposes what the *information* path looks like even when the gradient path is severed. The architecture is now ready for the cross-city deployment that can actually exercise it.

## 10. Artifacts

- Training history: `data/gnn_v2_checkpoints/decoupled_v1/history.json` (now includes `test` and `test_no_depth` keys)
- Final fragility variance: same file, key `fragility_variance_extreme_2080`
- Per-epoch log: `data/gnn_v2_checkpoints/logs/decoupled_v1.log`
- Best checkpoint: `data/gnn_v2_checkpoints/decoupled_v1/best.pt` (gitignored)
- Architecture: [src/gnn/model_v2.py](src/gnn/model_v2.py) — full rewrite for decoupling
- t=0 labels + loss: [src/gnn/data_v2.py](src/gnn/data_v2.py), [src/gnn/train_v2.py](src/gnn/train_v2.py)
