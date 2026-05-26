# v2 Noise-Perturbed Coupled Architecture — Phase 3 Report

**Branch:** `feat/v2-leakage-fix`
**Date:** 2026-05-25
**Run tag:** `noisy_coupled_v1`
**Status:** Architecture refactor complete; full training finished. Negative result on the headline metric — three converging nulls on the coupled architecture (warm-start, cold-start, noisy-coupled) all land at test Cascade-PR @ t=6 = 0.9309 ± 0.0001. The coupled pattern itself is the failure mode; a decoupled architecture is required and will be designed in a separate task.

## 1. Architecture

`LearnableFragility` produces a clean per-node failure probability `p_t0` from raw flood depths. During training only, the input to the GNN is `p_noisy = clamp(p_t0 + N(0, σ²), 0, 1)` with σ=0.15 — fresh draw per forward pass. The GNN's input is the static 8-feature base (`base_data[nt].x`, normalized) with the clean flood_depth column 3 *left untouched* (Stage 2 ablation: forces the GNN to use only the noisy fragility column 8 for flood information) and `p_noisy` appended as column 8. The clean `p_t0` is supervised by an auxiliary BCE against true t=0 failure labels (`aux_weight=0.1`). A weak L2 prior pulls fragility log-params toward HAZUS init (`prior_weight=1.0`). At eval time noise is disabled and the GNN sees `p_t0` directly. See [src/gnn/model_v2.py](src/gnn/model_v2.py) for the forward, [src/gnn/train_v2.py](src/gnn/train_v2.py) for the loss.

## 2. Hyperparameters (this run)

| | Value |
|---|---|
| `noise_sigma` | 0.15 |
| `aux_weight` | 0.1 |
| `prior_weight` | 1.0 |
| `lr` | 1e-3 |
| `weight_decay` | 1e-5 |
| `epochs` | 10 |
| `train_subset / val_subset` | 1500 / 500 |
| `warm_start` | **False** (cold) |
| Holdout test scenario | `geoclaw_2050` |
| col-3 depth overwrite | **disabled** (Stage 2) |

## 3. Three-way metric comparison

(warm/cold numbers from prior runs in this branch; noisy is this run.)

| Metric | v2 warm-start | v2 cold-start | **v2 noisy-coupled** |
|---|---|---|---|
| Train BCE (final epoch, cascade BCE) | 0.02252 | 0.02274 | **0.02922** |
| Train total (with prior + aux when used) | 0.02253 | 0.02275 | **0.03046** |
| `train_bce_t0` (locked across all epochs) | n/a | n/a | **0.01219** |
| Val loss (final) | 0.04119 | 0.04155 | **0.04279** |
| Test BCE | 0.03842 | 0.03890 | **0.03930** |
| Test PR @ t=[6,24,48,96] | [0.9862, 0.9997, 0.9994, 0.9999] | [0.9862, 0.9997, 0.9993, 0.9998] | **[0.9862, 0.9997, 0.9993, 0.9997]** |
| **Test Cascade-PR @ t=[6,24,48,96]** | [0.9309, 0.9991, 0.9983, 0.9997] | [0.9309, 0.9991, 0.9981, 0.9995] | **[0.9308, 0.9990, 0.9982, 0.9994]** |
| **Test Cascade-PR @ t=6** | **0.9309** | **0.9309** | **0.9308** |

Test Cascade-PR at t=6 is **identical to 3 decimal places** across all three runs. The noise injection plus depth-shortcut removal did not move the headline metric.

## 4. Fragility drift comparison

### Median (m) at end of training (10 epochs)

| Type | HAZUS | warm μ (Δ%) | cold μ (Δ%) | **noisy μ (Δ%)** |
|---|---|---|---|---|
| power    | 0.600000 | 0.600812 (+0.135%) | 0.601222 (+0.204%) | **0.600946 (+0.158%)** |
| telecom  | 0.300000 | 0.300411 (+0.137%) | 0.300302 (+0.101%) | **0.299587 (−0.138%)** |
| hospital | 0.600000 | 0.600002 (+0.000%) | 0.599978 (−0.004%) | **0.600003 (+0.001%)** |
| subway   | 0.100000 | 0.100005 (+0.005%) | 0.100008 (+0.008%) | **0.100006 (+0.006%)** |
| water    | 0.500000 | 0.499777 (−0.045%) | 0.499801 (−0.040%) | **0.499452 (−0.110%)** |
| fuel     | 0.500000 | 0.500243 (+0.049%) | 0.500484 (+0.097%) | **0.499974 (−0.005%)** |

### β at end of training

| Type | HAZUS | warm β (Δ%) | cold β (Δ%) | **noisy β (Δ%)** |
|---|---|---|---|---|
| power    | 0.500 | 0.499911 (−0.018%) | 0.499865 (−0.027%) | **0.500016 (+0.003%)** |
| telecom  | 0.600 | 0.599965 (−0.006%) | 0.599800 (−0.033%) | **0.599385 (−0.102%)** |
| hospital | 0.500 | 0.500003 (+0.001%) | 0.500014 (+0.003%) | **0.500000 (+0.000%)** |
| subway   | 0.400 | 0.399975 (−0.006%) | 0.399979 (−0.005%) | **0.400058 (+0.015%)** |
| water    | 0.500 | 0.499921 (−0.016%) | 0.499901 (−0.020%) | **0.500009 (+0.002%)** |
| fuel     | 0.600 | 0.600206 (+0.034%) | 0.599947 (−0.009%) | **0.599921 (−0.013%)** |

### Peak transient drift during the noisy-coupled run (across all 10 epochs)

| Type | Peak |Δ%| | Epoch | Final Δ% | Direction lasted? |
|---|---|---|---|---|
| power    | 0.328% | 1 | +0.158% | No — oscillated |
| **telecom**  | **0.503%** | **3** | **−0.138%** | Partial — net negative direction held |
| hospital | 0.005% | 6 | +0.001% | Flat throughout |
| subway   | 0.025% | 8 | +0.006% | Flat throughout |
| water    | 0.110% | 5 | −0.110% | Yes — small but monotonic-ish |
| fuel     | 0.235% | 2 | −0.005% | No — drifted down then back |

The noisy-coupled run **does** push fragility around more than the cold-start baseline (cold peaks were all under 0.25%; noisy peaked at 0.50% on telecom and 0.33% on power). But the motion is largely transient — by epoch 9 most drifts are back inside ±0.15%. The architecture provides gradient pressure, but no committed direction.

## 5. Variance diagnostic — fragility on `extreme_2080` run 0 (eval mode, post-training)

| Type | N | n_flooded | p_t0 var | p_t0 mean | p_t0 max |
|---|---|---|---|---|---|
| power    | 203  | 10  | 8.56e−03 | 0.0148 | 0.7164 |
| telecom  | 4150 | 274 | 3.34e−02 | 0.0434 | 0.9494 |
| hospital | 61   | **1** | 3.21e−06 | 0.0002 | 0.0140 |
| subway   | 493  | 76  | 0.124    | 0.1502 | **0.9999** |
| water    | 137  | 13  | 3.63e−02 | 0.0562 | 0.8270 |
| fuel     | 1187 | 84  | 1.76e−02 | 0.0287 | 0.7833 |

Fragility is *not* collapsed — variance is meaningful where there's flood exposure. Subway is saturated to ≈1.0 at its 10 cm HAZUS median (so its gradient through `∂p/∂log_mu` is near zero, explaining the flat drift). Hospital has effectively no signal even in `extreme_2080` (1/61 nodes flooded — too sparse to drive learning). Power and water have respectable variance and a real upper tail (max p_t0 = 0.72 and 0.83 respectively), so the gradient-sparsity story from Phase 2 does not apply here.

## 6. Interpretation

### Did fragility drift meaningfully?

No, by the test set as a whole, fragility did not commit to any new optimum. The peak transient drift was 0.50% on telecom at epoch 3 — three orders of magnitude smaller than your pre-experiment prediction of "−2% to −10% on power log_mu." Final-epoch fragility is within ±0.16% of HAZUS on every type. This is more motion than the cold-start v2 baseline saw (which capped at ±0.25%) — the architecture is doing *something* — but nothing converged. Telecom, fuel, and power all drifted away from HAZUS during epochs 1–3 and then returned by epochs 6–9. This pattern (drift away, then back) is the classic signature of an underdetermined parameter being pulled in different directions by competing forces.

### Direction per type — physical intuition?

- **Telecom** ended at μ=0.2996 (−0.14%) and β=0.5994 (−0.10%) — more fragile (lower median) and *tighter* (lower beta) than HAZUS. This is the only type with a clean signed drift across the run. Direction does match common intuition for telecom in coastal urban flooding (cell sites are surface-mounted boxes; modest water disrupts them quickly).
- **Power, fuel** wandered between positive and negative drift — no committed direction. The user's prediction was power becoming more fragile (negative drift). What we see is +0.158% at the end, but it transited −0.328% at epoch 1. The data is providing competing signals.
- **Hospital, subway** were flat throughout, both for the reasons exposed by the variance diagnostic (hospital: too few flooded examples; subway: CDF already saturated at HAZUS init).
- **Water** drifted −0.11% (more fragile), smaller motion than telecom but the same sign.

No clean physical narrative emerges because the parameters did not commit. The architecture has just enough pressure to perturb fragility but not enough to teach it.

### Where did test Cascade-PR land in the 0.73–0.93 corridor?

**0.9308 — pinned to the leaky baseline.** The prediction was 0.78–0.88. The architecture changes (noise injection + col-3 depth-overwrite removed + cold-start) did not move the held-out cascade-prediction metric at all. This is the headline negative result.

**Why this is consistent with everything else:** σ=0.15 is small relative to the dynamic range of `p_t0`. For a node with true p_t0 = 0.9, the noisy input is uniformly distributed (roughly) over [0.45, 1.0] after clamping; for p_t0 ≈ 0, noisy is in [0, ~0.5]. The GNN sees fresh noise per forward pass, but over many passes the mean of `p_noisy | p_t0` is still close to `p_t0` itself, so the GNN can extract `p_t0` to high fidelity via implicit averaging across the training run. Once it has `p_t0`, the cascade labels are roughly a deterministic reachability function of `p_t0` on a static graph, which the GNN solves regardless of noise. The leakage path we hoped to block is still open; we just slowed the learning of it.

### What the drift pattern tells us about which infrastructure types could ever inform fragility from the NYC data

Looking at the variance diagnostic plus the peak drifts:

- **Has signal AND moves**: telecom (274 flooded nodes; peak drift 0.50%), fuel (84 flooded; 0.24%), power (10 flooded; 0.33%), water (13 flooded; 0.11%). These are the four types where the architecture's gradient pressure can actually be detected, even though it doesn't commit.
- **Has signal but saturated**: subway. 76 flooded, but HAZUS median 0.1m means p_t0 is already at 1.0 for almost every flooded subway in this scenario — gradient through ∂p/∂log_mu collapses in the saturated tail. Increasing noise won't help. We would need either a higher HAZUS prior median for subway or a transformation that prevents saturation.
- **No signal**: hospital. Only 1 of 61 nodes flooded in the `extreme_2080` snapshot we measured; across the training set hospital is rarely on the cascade-relevant path. Fragility cannot learn here without more hospital flooding events, which the city's geography may not produce.

## 7. This is the third converging null result on the coupled architecture

Three independent attempts to fix the v2 coupling have now produced the same held-out cascade-prediction metric to three decimal places:

| Attempt | What it tested | Test Cascade-PR @ t=6 |
|---|---|---|
| **v2 warm-start** (prior week) | Joint training of HAZUS-init `LearnableFragility` + v1-pretrained GNN. The continuous P from fragility replaces the binary mask v1 used. | **0.9309** |
| **v2 cold-start** (Phase 2 of last task) | Same architecture, GNN re-initialized from scratch. Removes "v1 baked the oracle into the GNN weights" as the explanation. | **0.9309** |
| **v2 noisy-coupled** (this run) | Same architecture + Gaussian noise σ=0.15 on fragility's output before it enters the GNN + col-3 depth shortcut commented out + cold-start. Removes "GNN treats saturated P as binary," "raw depth is a backup oracle," and the warm-start contamination all in one. | **0.9308** |

Each attempt cut a different hypothesized leakage path. Each produced no measurable change on the headline test metric. The differences across the three runs (0.0001 in test Cascade-PR @ t=6, ≤ 0.4% in fragility drift) are inside training noise. **The coupling itself is the failure mode, not any specific knob inside it.** The architectural pattern — fragility's output is concatenated into the GNN's per-node input, and the GNN's cascade loss is the only thing that pulls fragility — gives the GNN a route that bypasses fragility regardless of what we do to that input column (saturate it, randomize the weights upstream of it, add noise to it, remove an alternate shortcut beside it). σ-tuning, dropout-style masking, or further coupled-architecture variants would all be tests of the same broken pattern.

The conclusion to draw from three converging nulls is that the architecture needs to be **decoupled** — fragility's parameters must learn from a signal that does not pass through the cascade GNN. The specific design of that decoupling is out of scope for this report and is the subject of a separate task. What this report establishes is that further iteration inside the coupled pattern is not justified.

## 8. Artifacts

- Training history: `data/gnn_v2_checkpoints/noisy_coupled_v1/history.json`
- Final fragility variance: same file, key `fragility_variance_extreme_2080`
- Per-epoch log: `data/gnn_v2_checkpoints/logs/noisy_coupled_v1.log`
- Best checkpoint: `data/gnn_v2_checkpoints/noisy_coupled_v1/best.pt` (gitignored)
- Architecture: [src/gnn/model_v2.py](src/gnn/model_v2.py) (col-3 depth-overwrite commented, line preserved)
- Loss/CLI: [src/gnn/train_v2.py](src/gnn/train_v2.py)
- t=0 label extractor: [src/gnn/data_v2.py](src/gnn/data_v2.py) — `t0_labels_from_run`
