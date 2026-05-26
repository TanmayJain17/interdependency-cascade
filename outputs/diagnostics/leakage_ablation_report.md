# Feature-Leakage Ablation Report

**Branch:** `diag/feature-leakage-ablation`
**Date:** 2026-05-25
**Model:** v1 `CascadeGNN` (warm-init-eligible, 662,680 params), CPU, 5 epochs each
**Subsets:** 1500 train / 500 val per epoch; held-out test = `geoclaw_2050` (1000 MC runs)
**Timesteps reported:** t = [6, 24, 48, 96] hours (t=0 excluded — label leakage from `initial_failure_mask`)
**PR = average precision (area under PR curve); CASCADE-only = same metric excluding nodes that were initial failures.**

## What each run zeroed

| Run | `--run_tag` | `--drop_features` argument | Columns zeroed (across all 6 node types) |
|---|---|---|---|
| A | `ablation_full` | *(none)* | 0 |
| B | `ablation_no_latlon` | `lat,lon` | 12 (lat+lon on each of 6 types) |
| C | `ablation_no_id` | `lat,lon,degree_centrality,tower_count,num_routes,capacity_proxy` | 17 (B's 12 + `power.degree_centrality`, `telecom.tower_count`, `subway.num_routes`, `water.capacity_proxy`, `fuel.capacity_proxy`) |

Run C drops every continuous, high-cardinality feature that could act as an implicit node identifier. Constant-default columns (elevation = 3.0, flood_depth = 0.0, hospital's bed_count/generator/water/critical, etc.) were not dropped because they already carry zero information.

## Final-epoch comparison

(All values are last-epoch numbers, except `test_*` which is the one-shot eval after epoch 5.)

| Metric | Run A (baseline) | Run B (no lat/lon) | Run C (no lat/lon + no unique cont.) |
|---|---|---|---|
| Train BCE (final epoch) | 0.00729 | 0.00735 | 0.00722 |
| Val BCE | 0.02181 | 0.02188 | 0.02271 |
| Val PR @ t = [6,24,48,96] | [0.9948, 0.9997, 0.9989, 0.9993] | [0.9948, 0.9997, 0.9988, 0.9990] | [0.9948, 0.9997, 0.9986, 0.9983] |
| Val Cascade-PR @ t = [6,24,48,96] | [0.9716, 0.9896, 0.9850, 0.9977] | [0.9717, 0.9935, 0.9843, 0.9972] | [0.9716, 0.9908, 0.9835, 0.9960] |
| Test BCE | 0.03967 | 0.03978 | 0.04142 |
| Test PR @ t = [6,24,48,96] | [0.9863, 0.9997, 0.9993, 0.9998] | [0.9863, 0.9997, 0.9993, 0.9998] | [0.9863, 0.9997, 0.9992, 0.9996] |
| Test Cascade-PR @ t = [6,24,48,96] | [0.9313, 0.9991, 0.9979, 0.9995] | [0.9312, 0.9991, 0.9981, 0.9994] | [0.9311, 0.9991, 0.9979, 0.9991] |

## Fragility parameters

Not applicable. This is the v1 GNN (no `LearnableFragility` module). Fragility is provided by the precomputed cascade simulator in `data/simulation/cascade_results_nyc_*.json`; nothing in the GNN's forward pass touches a HAZUS curve. Logged here for the record only — drift cannot be observed in v1.

## Interpretation

**Dropping lat/lon did not move test PR off 0.99+. Dropping lat/lon *plus* every high-cardinality continuous feature also did not move it.** Across all three runs, test Cascade-PR at t=6 is 0.9313 / 0.9312 / 0.9311 — variance is at the fourth decimal place, well inside training noise. Final train and val BCE are also essentially identical across runs.

The lat/lon hypothesis is **falsified**. The leakage is not in node features. Three places to look next, in order of likelihood:

1. **The `initial_failure_mask` input column.** `build_input_x_dict` ([src/gnn/data.py:106-116](src/gnn/data.py#L106-L116)) appends the per-node t=0 failure indicator as the 9th input feature. The cascade labels at later timesteps are extremely tightly coupled to the initial-failure set on a static graph: once you know who got flooded at t=0, downstream failures are largely a deterministic function of graph reachability under that initial set. The cascade simulator is itself a deterministic procedure over the same static graph the GNN sees, so the mapping `initial_mask → labels[t>0]` is almost a lookup table. The model isn't predicting cascades — it's learning the simulator's reachability function, conditioned on the mask we hand it as input. The "drop t=0 from CASCADE-only metrics" filter excludes nodes that failed at t=0 but does **not** remove the mask from the input, so the predictor still sees who the initial victims were.
2. **Graph structure as identity.** With features zeroed, every node is still uniquely identified by its position in the heterograph (degree, neighborhood multiset, edge types). A heterogeneous GAT with 2 layers and 4 heads has more than enough capacity to memorize position embeddings for 6,231 nodes across 5 training scenarios — especially since the test scenario shares the same graph, only the initial-failure mask changes.
3. **Scenario-depth coupling.** Train scenarios include `extreme_2080` and `geoclaw_2080`; held-out test is `geoclaw_2050`. If the cascade depth distributions overlap enough, "harder" training scenarios may bracket the test scenario rather than testing extrapolation. Worth checking distributions of `fail_time_per_node` across scenarios before drawing harder conclusions.

**Recommended next diagnostic:** retrain with the `initial_failure_mask` input column *also* zeroed (a true "predict the cascade from features + graph alone, blind to who failed at t=0" task). If test Cascade-PR collapses there but not here, hypothesis #1 is confirmed and the cascade-only metric needs to be redefined for v2 — the current setup measures "reachability decoding" not "cascade prediction."

## Artifacts

- Per-run history: `data/gnn_diag_checkpoints/{ablation_full,ablation_no_latlon,ablation_no_id,ablation_no_mask}/history.json`
- Per-run drop manifest: `…/dropped_features.json`
- Per-run training logs: `data/gnn_diag_checkpoints/logs/{ablation_full,ablation_no_latlon,ablation_no_id,ablation_no_mask}.log`
- Feature schema used: `outputs/diagnostics/feature_schema.json`

---

## Phase 3: Initial-Mask Ablation

### Run D setup

| Run | `--run_tag` | What changed vs. Run A | How |
|---|---|---|---|
| D | `ablation_no_mask` | The appended t=0 failure-mask column (input feature index `-1`) is zeroed on the model input | New `--drop_initial_mask` flag in `src/gnn/train.py`. `train_step` and `eval_step` zero `x_dict[nt][:, -1]` after `example_from_run`. `eval_step` clones the *true* mask **before** zeroing so the CASCADE-only metric still excludes the correct set of t=0-victim nodes from scoring. All other settings identical to Run A. |

### Final-epoch comparison (Run A vs Run D)

| Metric | Run A (baseline) | Run D (mask zeroed) | Δ |
|---|---|---|---|
| Train BCE (final epoch) | 0.00729 | 0.19337 | **+0.186** (~27× harder to fit) |
| Val BCE | 0.02181 | 0.25390 | **+0.232** |
| Val PR @ t = [6,24,48,96] | [0.9948, 0.9997, 0.9989, 0.9993] | [0.4794, 0.5311, 0.5303, 0.5326] | ~ −0.47 across all t |
| Val Cascade-PR @ t = [6,24,48,96] | [0.9716, 0.9896, 0.9850, 0.9977] | [0.4592, 0.5034, 0.4962, 0.5044] | ~ −0.49 across all t |
| Test BCE | 0.03967 | 0.29420 | **+0.255** |
| Test PR @ t = [6,24,48,96] | [0.9863, 0.9997, 0.9993, 0.9998] | [0.8173, 0.8671, 0.8567, 0.8678] | ~ −0.13 |
| Test Cascade-PR @ t = [6,24,48,96] | [0.9313, 0.9991, 0.9979, 0.9995] | **[0.7281, 0.8256, 0.8121, 0.8362]** | **−0.20, −0.17, −0.19, −0.16** |

### Four-run roll-up (test Cascade-PR @ t=6, the most sensitive metric)

| Run | Drop | Test Cascade-PR @ t=6 |
|---|---|---|
| A (baseline) | nothing | 0.9313 |
| B (no lat/lon) | lat, lon | 0.9312 |
| C (no unique features) | lat, lon, degree_centrality, tower_count, num_routes, capacity_proxy | 0.9311 |
| **D (no mask)** | initial_failure_mask | **0.7281** |

### Interpretation

**Hypothesis #1 is confirmed.** Zeroing the appended t=0 failure-mask column collapses every PR metric by 0.13 – 0.49 (depending on slice), while zeroing 17 node-feature columns across all 6 node types in Run C moved the needle by < 0.001. The model's near-perfect v1 scores were not coming from any geographic, structural, or capacity-style node feature — they were coming directly from the t=0 mask we hand it as input. With the mask removed, the cascade-prediction problem is genuinely hard (test Cascade-PR ≈ 0.73 – 0.84 instead of ≈ 0.93 – 1.00), and training BCE rises from 0.007 to 0.19, confirming the model can no longer trivially copy the answer from the input.

This also explains why v2's `LearnableFragility` parameters never drifted from HAZUS initialization. The v1 GNN — which is what v2 warm-starts from — solves the task without using fragility at all: it learns the simulator's reachability function conditioned on the mask. There is no gradient signal asking fragility to be anything in particular, so a HAZUS-init fragility module attached to a trained v1 backbone has nothing to learn.

### What this implies for v2

1. **Cascade-only is not actually cascade-only.** The metric drops nodes that failed at t=0 from scoring but leaves the t=0 mask fully observable on the input. Future cascade-prediction runs should either zero the mask on the input or replace it with something that doesn't trivially identify the initial-failure set (e.g. a learned hazard embedding from depth-at-node).
2. **Joint v2 training needs a different observation structure.** If fragility is supposed to be learnable, the GNN must be forced to *predict* who fails — not be told. Two options: (a) feed depth/exposure features instead of the binary mask and let fragility convert them, or (b) keep the mask but mask it out a fraction of the time during training so the network can't depend on it.
3. **The 0.73 – 0.84 numbers in Run D are likely now a more honest performance baseline** for any v2 architecture that doesn't get to see the initial-failure set for free. They should be the reference, not the ≈ 1.0 from v1.

### Notes on Run D numbers

- Test Cascade-PR is *higher* than val Cascade-PR in Run D (e.g. 0.7281 vs 0.4592 at t=6). The likely reason: the val set is mixed across 5 training scenarios (some with sparse, hard-to-predict cascades), while the test set is a single GeoClaw scenario with reasonably consistent flood geometry. Worth checking per-scenario val PR before reading anything stronger into this. Doesn't affect the headline finding.
- Train BCE plateaus around 0.19 within ~3 epochs — there is signal the model is exploiting (graph structure + features still help), it is just nowhere near the cheap signal it had access to before.

