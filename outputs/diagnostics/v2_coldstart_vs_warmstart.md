# v2 Cold-Start vs Warm-Start Comparison

**Branch:** `feat/v2-leakage-fix`
**Date:** 2026-05-25
**Question:** Does warm-starting the v2 GNN from v1's (leaky) weights prevent `LearnableFragility` from drifting during joint training? If so, cold-start should unfreeze it.

## Setup

Both runs use the same `CascadeGNNv2` architecture (`LearnableFragility` → continuous initial-failure probability concatenated as input column 9 → `CascadeGNN` → per-timestep failure logits), the same hyperparameters (10 epochs, 1500 train / 500 val subset, `prior_weight=1.0`, lr `1e-3`, weight_decay `1e-5`, `geoclaw_2050` held out), and the same data (5 train scenarios × 1000 MC runs each, 80/20 train/val split by run).

| Run | Source | GNN initial weights |
|---|---|---|
| **Warm** | `data/gnn_v2_checkpoints/run01_warmstart_prior1.0_lr1e-3.json` (committed last week) | v1 checkpoint `data/gnn_checkpoints/best.pt` |
| **Cold** | `data/gnn_v2_checkpoints/coldstart_run/history.json` (this run) | Fresh PyTorch random init (`--no_warm_start`) |

## Final-epoch comparison

| Metric | v2 warm-start | v2 cold-start | Δ |
|---|---|---|---|
| Train BCE (final epoch) | 0.02252 | 0.02274 | +0.0002 |
| Train total (BCE + λ·prior) | 0.02253 | 0.02275 | +0.0002 |
| Val loss | 0.04119 | 0.04155 | +0.0004 |
| Val PR @ t=[6,24,48,96] | [0.9453, 0.9456, 0.9416, 0.9448] | [0.9451, 0.9443, 0.9407, 0.9424] | ≤ −0.002 |
| Val Cascade-PR @ t=[6,24,48,96] | [0.8080, 0.8631, 0.8566, 0.8733] | [0.8061, 0.8605, 0.8549, 0.8680] | ≤ −0.005 |
| **Test BCE** | 0.03842 | 0.03890 | +0.0005 |
| **Test PR @ t=[6,24,48,96]** | [0.9862, 0.9997, 0.9994, 0.9999] | [0.9862, 0.9997, 0.9993, 0.9998] | ≤ −0.0001 |
| **Test Cascade-PR @ t=[6,24,48,96]** | **[0.9309, 0.9991, 0.9983, 0.9997]** | **[0.9309, 0.9991, 0.9981, 0.9995]** | ≤ −0.0002 |

Cold-start is statistically indistinguishable from warm-start on every metric. The metric that mattered most — test Cascade-PR @ t=6 — is **0.9309 in both runs to four decimal places.**

## Fragility drift after 10 epochs

(Drift = (final − HAZUS) / HAZUS · 100%; positive = curve shifted right, more flood-tolerant.)

| Type | HAZUS median (m) | Warm final | Warm Δ | Cold final | Cold Δ | HAZUS β | Warm β Δ | Cold β Δ |
|---|---|---|---|---|---|---|---|---|
| power    | 0.600 | 0.60081 | **+0.14%** | 0.60122 | **+0.20%** | 0.500 | −0.02% | −0.03% |
| telecom  | 0.300 | 0.30041 | **+0.14%** | 0.30030 | **+0.10%** | 0.600 | −0.01% | −0.03% |
| hospital | 0.600 | 0.60000 |  +0.00% | 0.59998 |  −0.00% | 0.500 | +0.00% | +0.00% |
| subway   | 0.100 | 0.10000 |  +0.01% | 0.10001 |  +0.01% | 0.400 | −0.01% | −0.01% |
| water    | 0.500 | 0.49978 | **−0.04%** | 0.49980 | **−0.04%** | 0.500 | −0.02% | −0.02% |
| fuel     | 0.500 | 0.50024 | **+0.05%** | 0.50048 | **+0.10%** | 0.600 | +0.03% | −0.01% |

**The largest cold-start drift is 0.20% on `power.median` (from 0.600 m → 0.60122 m, a shift of 1.2 mm).** Every other parameter moved by less than 0.1%. The HAZUS prior is doing essentially nothing here — `train_prior` averaged ~7e−6, four orders of magnitude smaller than `train_bce` (~0.023) — so this is not the prior holding fragility back. The drift is real-but-tiny because the GNN provides almost no gradient pressure on fragility regardless of starting weights.

## Interpretation

**The warm-start hypothesis is falsified.** Removing v1's pretrained weights had no effect on either (a) test accuracy or (b) fragility drift. Cold-start v2 lands in essentially the same minimum.

We are firmly in **outcome 3** of the pre-experiment decision matrix: *"The leakage is structural — saturated continuous prob looks binary to the GNN regardless. Architecture change definitely needed."*

The mechanism is: at the HAZUS-init lognormal CDF, the mapping `depth → P(failure)` is already near-binary for the depth distributions present in our scenarios. Nodes with depth well below median get P ≈ 0; nodes with depth well above median get P ≈ 1; the saturated middle is small. The v2 GNN inherits the same "trust input column 9 as the answer" shortcut that the diag/feature-leakage-ablation experiment confirmed for v1 — and from the GNN's perspective, a near-binary P from fragility is functionally identical to the binary mask v1 was trained on. Since the GNN explains the labels almost perfectly via column 9 alone, the chain-rule gradient that reaches `LearnableFragility` is tiny, and the prior — though weak — is enough to hold the lognormal params at HAZUS init.

This also explains the Run D number from last week's diagnostic. When we zeroed the binary mask on v1, test Cascade-PR @ t=6 dropped to 0.7281. v2 cold-start gives the GNN a **near-binary** continuous probability instead of a binary mask, and the test Cascade-PR @ t=6 returns to 0.9309 — i.e. v2 has roughly recovered the v1 leakage path through fragility's saturated output, rather than learning fragility as a calibrated belief.

## Numerical sanity checks (so we don't over-read the result)

- Val Cascade-PR is much lower than test Cascade-PR in both runs (0.81 vs 0.93 at t=6). This isn't new — it appeared in last week's diag report. Likely because val is mixed across 5 training scenarios with varying cascade depth, while test is the single `geoclaw_2050` scenario with more uniform flood geometry. Not a confound for the warm-vs-cold comparison since both runs see the same val/test sets.
- Train_prior remained near zero throughout both runs (~7e−6), confirming the prior is too weak to dominate any signal. Bumping `--prior_weight` higher would tighten fragility further toward HAZUS — that's the *opposite* of what we want here.
- Cold-start train BCE converged to within 1% of warm-start train BCE within 2 epochs (0.0248 by ep02 vs 0.0225 warm final). The model has more than enough capacity to recover v1's behavior from scratch.

## What to do next (recommendations, not yet acted on)

Two architecture-level levers, in order of cheapness:

1. **Pass depth directly, not P(depth).** Replace the appended `compute_initial_fragility(depths)` column with the raw normalized depth tensor at each node. The GNN then has to learn `depth → failure` itself, and the gradient that reaches `LearnableFragility` is no longer routed through a saturated CDF. Risk: `LearnableFragility` could become irrelevant (no input at all to the GNN).
2. **Supervise fragility with an aux loss.** Keep the column-9 architecture but add an explicit BCE between `compute_initial_fragility(depths)` and the *true t=0 failure indicator*, with its own weight. This forces fragility to be a calibrated belief about t=0 failure independent of cascade outcomes. Risk: gives the GNN even more leakage because fragility now becomes a very accurate predictor of t=0 victims.

Worth discussing whether to try (1) first, (2) first, or both in parallel. We may also need to **redefine the cascade-prediction task** so the model isn't given any t=0 hint at all — i.e. the "blind cascade" setup the diag Run D measured (test Cascade-PR ≈ 0.73). That number is probably the honest performance baseline; ~0.93 is "cascade reconstruction from initial victims," which is a different task.

## Artifacts

- Cold-start history: `data/gnn_v2_checkpoints/coldstart_run/history.json`
- Warm-start history (last week): `data/gnn_v2_checkpoints/run01_warmstart_prior1.0_lr1e-3.json`
- Cold-start log: `data/gnn_v2_checkpoints/logs/coldstart_run.log`
- Modified trainer: `src/gnn/train_v2.py` (added `--run_tag` to route outputs to subdir; `--no_warm_start` was already present)
