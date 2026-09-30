# Week 28 note: recovery, or an enforced rising order? (v3, 30 Sep 2026, with the Torch audit, held-out storms and fall rates by label row)

*v3 answers the two points v2 left open (held-out storms, and whether falls concentrate on flat label rows), records one hypothesis that did not survive, and adds a finding about what `--seed` controls. v2 added the audit results (Torch job 18896971, code at `bc0e5f4`). v1 was written before the run; its predictions are kept unchanged in section 7. Everything traces to a file and line in the repo or to `analysis/gnn/week28_mono_enforce_v1.csv`.*

## 1. The question (24 Sep meeting)

The surrogate's P(failed by t) only grows with t, so it has no notion of recovery. Two options were raised:

- **A.** Add recovery to the pipeline, at the risk of making it too complicated.
- **B.** Skip recovery for now and enforce the rising order, either inside the model or on the output side.

## 2. The answer in four sentences

1. The simulator has no recovery, so the training labels contain none, and a surrogate cannot learn what its labels do not contain. Option B is forced by the data, not chosen for convenience.
2. Recovery, when added, does not make the failed-by curve fall. It is a second rising curve, "restored by t", and the number of sites out of service is the gap between the two.
3. The rising order is already enforced inside the model (hazard head, Week 27) with no loss of fit.
4. Measured on the saved checkpoints, the two sides fit equally: an output-side fix moves validation loss by at most 0.00002, because the falls are many but tiny (median 0.00001 to 0.0001 in probability). The hazard head is kept for what it enables next, not for accuracy.

## 3. Why there is nothing to learn about recovery (code evidence)

| Fact | Where |
|---|---|
| A site gets one fail time and keeps it. The relaxation loop skips any node already in `fail_time`; nothing is ever removed. | `src/simulation/cascade_joint.py:232` (loop at 226–248) |
| The engine is described as monotone: "dead stays dead". Telecom congestion would really ease as the surge decays, so telecom counts are an upper bound. | `src/simulation/intra_telecom.py:66–70` |
| Labels are `1 if fail_time <= t`, so every row is 0…01…1. | `src/gnn/data.py:153` |
| A restoration layer exists: days-to-restore per site from flood depth. It feeds the economic handoff, not the cascade loop. | `src/simulation/downtime.py`, `src/export/build_handshake.py` |
| The recover-gating hook (a site cannot start restoring until its upstream is back) is written and unit-tested but not applied. | `downtime.py:162–194`, `tests/test_downtime_gating.py` |
| The restoration times are placeholders marked "VERIFY against the HAZUS Flood Technical Manual". | `config/restoration_curves.yaml:3–5` |

One caution on the last row. The HAZUS flood manual I could check lists restoration time for buildings by occupancy; restoration functions for utilities were confirmed only in the HAZUS earthquake manual. The source for flood restoration times of lifelines needs to be settled before any recovery result is shown.

## 4. What adding recovery would take

Let F(t) = P(site has failed by t) and R(t) = P(site has failed and been restored by t). If a restored site does not fail again:

- F and R both only rise, and R(t) ≤ F(t).
- P(site is out of service at t) = F(t) − R(t), which rises and then falls.

So recovery is a second absorbing step (working → failed → restored), not a violation of monotonicity. This is the progressive multi-state model of survival analysis (Putter, Fiocco & Geskus 2007); the non-monotone quantity is the state occupancy, not either transition curve.

Work needed, in order:

1. **World 1.** Verified restoration times; the gating hook switched on; a `recover_time_per_node` written beside `fail_time_per_node`. The placeholder times are 1 to 90 days, mostly beyond the 96 h the surrogate covers, so the label horizons would need extending too.
2. **World 2.** A second hazard head for "restored by t", conditional on failure. The hazard head from Week 27 is the right base for this; an output-side patch is not.

Doing this now would mean building and validating a second simulator component before the first surrogate question (timing accuracy) is answered. That is the complexity cost of option A.

## 5. Four readings of the same four outputs

Worked example, outputs (−2, 0, 1, −1):

| Reading | 6 h | 24 h | 48 h | 96 h |
|---|---|---|---|---|
| Independent head, as trained | 0.119 | 0.500 | 0.731 | 0.269 |
| + running max | 0.119 | 0.500 | 0.731 | 0.731 |
| + sort | 0.119 | 0.269 | 0.500 | 0.731 |
| + isotonic regression | 0.119 | 0.500 | 0.500 | 0.500 |
| Hazard head | 0.119 | 0.560 | 0.882 | 0.913 |

The first four rows take a trained independent head and repair its output. The last row is a different model: the four numbers are hazards and the curve is 1 − ∏(1 − h).

## 6. Two results proved this week

Setting: one site, predictions p₁…p_K, labels 0…01…1 with the first r equal to 0. The loss is Σ_{k≤r} f₀(p_k) + Σ_{k>r} f₁(p_k). For BCE, f₀(p) = −log(1−p) and f₁(p) = −log p; for Brier, f₀ = p² and f₁ = (1−p)². In both, f₀ is increasing, f₁ is decreasing, and both are convex.

**Result 1: sorting never raises the loss.** Sorting is a sequence of swaps of adjacent out-of-order pairs (a at the earlier horizon, b at the later, a > b). If the two labels are equal, the sum is unchanged. If they are (0, 1), the loss goes from f₀(a) + f₁(b) to f₀(b) + f₁(a); f₀(b) ≤ f₀(a) because f₀ is increasing and f₁(a) ≤ f₁(b) because f₁ is decreasing. Labels (1, 0) cannot occur. ∎

**Result 2: isotonic regression never raises the loss.** Isotonic regression replaces each pooled block by its mean m. Within a pooled block every initial segment has mean ≥ m and every final segment has mean ≤ m (otherwise the block would split). The labels in the block are s zeros followed by ones. By Jensen and monotonicity, Σ_{zeros} f₀(p_i) ≥ s·f₀(mean of the first s) ≥ s·f₀(m), and likewise the ones give at least (n−s)·f₁(m). ∎

**Running max has no such guarantee.** Predictions (0.9, 0.1) with labels (0, 0) become (0.9, 0.9), which is worse.

Status of these results: derived here and checked numerically (`scripts/week28_mono_enforce_fingerprint_v1.py`: no increase in 1,000,000 cases for sort and isotonic; running max worse in 36%). The known published relative is Chernozhukov, Fernández-Val & Galichon (2010): sorting a non-monotone estimate of a monotone curve brings it weakly closer to the true curve in every L^p norm. I have not found the label-based statement above in a paper, so present it as our own short argument.

Both results hold per row, so they hold for the type-averaged validation loss and for every held-out storm.

## 7. The audit and its result

`hpc/audit_gnn_mono_enforce_v1.sbatch` re-scores 8 saved checkpoints (4 independent, 4 hazard) on the Week 27 validation split and held-out storms. No training; `src/gnn` untouched. Smoke job 18896824 (8 of 8 passed), production job 18896971 (8 of 8 passed, about 8 minutes each).

Gates, all passed: the raw row reproduces the saved validation loss to 5e-6; fixed readings show zero falling pairs; sort and isotonic do not raise BCE or Brier in any run; the fixes are no-ops on hazard checkpoints.

### Validation loss (3,400 runs, seed 42)

| Cell | Independent head | + running max | + sort | + isotonic | Hazard head |
|---|---|---|---|---|---|
| static, mask | 0.10824 | 0.10825 | 0.10824 | 0.10823 | 0.10796 |
| static, mask + timing | 0.09808 | 0.09808 | 0.09808 | 0.09807 | 0.09796 |
| dynamic, mask | 0.18774 | 0.18773 | 0.18774 | 0.18774 | 0.18798 |
| dynamic, mask + timing | 0.16529 | 0.16526 | 0.16528 | 0.16528 | 0.16508 |

Change against the independent head, as trained (negative is better):

| Cell | Running max | Sort | Isotonic | Hazard head |
|---|---|---|---|---|
| static, mask | +0.000008 | −0.000004 | −0.000006 | −0.000274 |
| static, mask + timing | +0.000000 | −0.000002 | −0.000004 | −0.000116 |
| dynamic, mask | −0.000014 | −0.000002 | −0.000003 | +0.000241 |
| dynamic, mask + timing | −0.000021 | −0.000002 | −0.000005 | −0.000204 |

Cascade-only PR-AUC per horizon moves by at most 0.0012 under any of the three fixes.

### The falls themselves (independent head, validation)

| Cell | Falls per run (of 18,693 pairs) | Share of pairs | Median size | 90th pct | 99th pct | Largest | Pairs falling > 0.01 | > 0.05 |
|---|---|---|---|---|---|---|---|---|
| static, mask | 1,265 | 6.8% | 0.000027 | 0.0027 | 0.036 | 0.31 | 0.29% | 0.03% |
| static, mask + timing | 1,458 | 7.8% | 0.000012 | 0.0010 | 0.027 | 0.25 | 0.22% | 0.03% |
| dynamic, mask | 740 | 4.0% | 0.000019 | 0.0066 | 0.032 | 0.16 | 0.22% | 0.01% |
| dynamic, mask + timing | 1,185 | 6.3% | 0.00013 | 0.0018 | 0.032 | 0.33 | 0.26% | 0.03% |

### Predictions written before the run, and what happened

| | Prediction | Result |
|---|---|---|
| P2 | Isotonic improves val loss by at most 0.002 in every cell | Pass. Actual improvement 0.000003 to 0.000006, several hundred times smaller than the bound |
| P3 | Hazard head and independent + isotonic agree within ±0.002 | Pass. Actual −0.00027 to +0.00025 |
| P4 | Median dip below 0.01; fewer than 1% of pairs dip by more than 0.05 | Pass. Median 0.00001 to 0.0001; 0.01% to 0.03% of pairs |
| P5 | More than half of all dips sit on flat label rows | Pass as written (91% to 97%), but uninformative: never-failed rows fall at about the overall rate (see below) |
| P6 | Cascade-only PR-AUC moves by at most 0.005 under sort and isotonic | Pass. At most 0.0012 |

The bounds were loose. They passed by one to two orders of magnitude, so they tested little; the measured sizes are the finding.

### How to read it

1. **The rising-order constraint is worth nothing measurable in fit.** Applied after the model, it changes the validation loss in the sixth decimal.
2. **The falls are a consistency defect, not an accuracy defect.** Week 27 reported the count (4% to 8% of pairs). The size is typically 0.00001 to 0.0001 in probability; 99% of falls are below 0.04. Rare large ones exist, up to 0.16 to 0.33.
3. **The ±0.0003 between the hazard head and the independent head is not the effect of the constraint.** The constraint itself is worth at most 0.00002. The rest is the difference between two separately trained models on one seed, and it has both signs across cells.
4. **Running max carried no guarantee and behaved that way:** on validation slightly worse in one cell and slightly better in two; on held-out storms worse on 4 of 20. The differences are too small to matter.
5. **So fit cannot choose between model side and output side.** The reason to keep the hazard head is structural: its outputs are per-window hazards, which the survival likelihood trains on and a recovery head would reuse, and the rare large falls are removed with no post-processing step.

### Held-out storms (1,000 runs each): change against the independent head, as trained

| Cell | Storm | Independent head | Running max | Sort | Isotonic | Hazard head |
|---|---|---|---|---|---|---|
| static, mask | 914_6 | 0.08548 | +0.00000 | −0.00001 | −0.00001 | +0.00012 |
| static, mask | 258_9 | 0.12480 | −0.00004 | +0.00000 | +0.00000 | +0.00042 |
| static, mask | 808_27 | 0.14004 | −0.00011 | −0.00001 | −0.00001 | +0.00015 |
| static, mask | gc_2050 | 0.14061 | −0.00002 | −0.00001 | −0.00001 | +0.00166 |
| static, mask | 605_5 | 0.20517 | −0.00047 | −0.00008 | −0.00011 | +0.00328 |
| static, mask + timing | 914_6 | 0.07298 | −0.00002 | +0.00000 | −0.00001 | +0.00063 |
| static, mask + timing | 808_27 | 0.11366 | −0.00002 | +0.00000 | +0.00000 | −0.00084 |
| static, mask + timing | 258_9 | 0.12119 | −0.00003 | +0.00000 | −0.00001 | −0.00005 |
| static, mask + timing | gc_2050 | 0.13506 | −0.00002 | +0.00000 | +0.00000 | +0.00090 |
| static, mask + timing | 605_5 | 0.20009 | −0.00055 | −0.00024 | −0.00028 | −0.00100 |
| dynamic, mask | 914_6 | 0.12620 | +0.00002 | +0.00000 | +0.00000 | +0.00172 |
| dynamic, mask | 258_9 | 0.27341 | −0.00013 | −0.00001 | −0.00001 | −0.00230 |
| dynamic, mask | gc_2050 | 0.33227 | +0.00001 | −0.00003 | −0.00001 | −0.00404 |
| dynamic, mask | 808_27 | 0.38056 | −0.00018 | −0.00001 | −0.00001 | +0.00132 |
| dynamic, mask | 605_5 | 0.47850 | −0.00060 | −0.00004 | −0.00010 | +0.00918 |
| dynamic, mask + timing | 914_6 | 0.10478 | +0.00003 | +0.00000 | +0.00000 | +0.00146 |
| dynamic, mask + timing | 258_9 | 0.26391 | −0.00014 | +0.00000 | −0.00001 | −0.00495 |
| dynamic, mask + timing | 808_27 | 0.32292 | −0.00004 | −0.00002 | −0.00002 | −0.01450 |
| dynamic, mask + timing | gc_2050 | 0.32883 | +0.00004 | +0.00000 | −0.00001 | −0.00884 |
| dynamic, mask + timing | 605_5 | 0.44143 | −0.00042 | −0.00008 | −0.00028 | −0.01115 |

- An output-side fix moves a held-out storm loss by at most 0.00060 (running max), 0.00024 (sort), 0.00028 (isotonic). The largest effects are all on 605_5, the hardest storm.
- Sort and isotonic are never worse, on any of the 20 results. Running max is worse on 4 of 20.
- The hazard head differs from the independent head by −0.0145 to +0.0092. It is better on 0, 3, 2 and 4 of the 5 storms in the four cells: 9 of 20 overall.

**Reading.** The Week 27 observation (dynamic + timing, hazard head better on 4 of 5 held-out storms, up to −0.0145) cannot be the rising-order constraint acting on the output, because that constraint is worth at most 0.0006 on those storms. It is either an effect of training with the hazard parameterisation or run-to-run variation. One seed cannot tell these apart, and other cells show differences of similar size with the opposite sign (dynamic, mask on 605_5: +0.0092). The seed check decides.

### Where the falls sit: share of pairs that fall, by label row (non-seed sites)

| Cell | All rows | [0000] never failed | [0001] | [0011] | [0111] | [1111] | Seeds |
|---|---|---|---|---|---|---|---|
| static, mask | 6.8% | 7.2% | 2.4% | 2.1% | 8.5% | 8.0% | 3.1% |
| static, mask + timing | 7.8% | 8.7% | 1.2% | 1.1% | 8.9% | 6.9% | 0.5% |
| dynamic, mask | 4.0% | 4.4% | 0.4% | 0.4% | 2.4% | 5.5% | 0.9% |
| dynamic, mask + timing | 6.3% | 6.7% | 0.6% | 2.6% | 10.4% | 7.7% | 1.4% |

| Cell | 6→24 h | 24→48 h | 48→96 h | Sites per run with a fall > 0.01 (of 6,231) | > 0.05 |
|---|---|---|---|---|---|
| static, mask | 5.8% | 8.9% | 5.6% | 51 | 5.5 |
| static, mask + timing | 7.7% | 6.9% | 8.8% | 36 | 5.2 |
| dynamic, mask | 7.7% | 2.3% | 1.8% | 37 | 2.1 |
| dynamic, mask + timing | 7.5% | 2.9% | 8.6% | 38 | 5.4 |

- Never-failed rows fall at about the overall rate (within one percentage point in every cell). So "91% to 97% of falls are on flat rows" reflects that most rows are flat, not that falls concentrate there. P5 passed as written but was not informative.
- Rows that first fail in the last two windows ([0001], [0011]) fall much less often. Rows that fail in (6, 24] h ([0111]) fall as often as flat rows or more, except in dynamic, mask.
- Seeds rarely fall (0.5% to 3.1% of their pairs).

**A hypothesis that did not survive.** The Week 27 handoff proposed that static cells dip more than dynamic cells because static labels have more flat rows. Within never-failed rows alone, the mask arms still differ (7.2% static against 4.4% dynamic), so the difference is not a matter of row mix. What the table does show is that dynamic, mask is the low cell, and that its falls are rare at the two later pairs (2.3% and 1.8%). The reason is open.

### What `--seed` controls (found while reading `train.py` for the seed check)

`src/gnn/train.py` never calls `torch.manual_seed`. `--seed` feeds only `random.Random(args.seed)` (lines 308 and 517), which fixes the train/validation split and the per-epoch shuffle order. Weight initialisation and dropout use PyTorch's own generator.

- Whether two jobs share an initialisation therefore depends on the PyTorch version's default seed. In the PyTorch build in this chat's sandbox the default seed differs from process to process, and two fresh processes built different initial weights. The version on Torch has not been checked.
- One-line check on Torch, run twice: `python -c "import torch; print(torch.__version__, torch.default_generator.initial_seed())"`. The same number twice means a fixed default seed; two different numbers mean every campaign so far started from a different random initialisation.
- Tonight's audit is unaffected: it scores saved checkpoints and involves no training randomness.
- For the seed check, set the torch seed explicitly (a flag that is off by default, so earlier campaigns stay as they were), and compare the two heads paired by seed.

## 8. What this does and does not claim

- It compares two ways of enforcing the rising order in **fit** on four fixed horizons, one seed.
- It does not claim any gain in timing accuracy. The dynamic-label gap (+73%) is untouched.
- It does not claim the simulator's lack of recovery is harmless. Telecom counts are an upper bound, and lightly damaged sites with one-day restoration could be back inside the 96 h window.

## 9. Questions to expect

**If the output fix is free and cannot hurt, why change the model?** Because the next step trains on the survival likelihood, which needs per-window hazards as trainable outputs. A sorted or pooled curve has none; pooled stretches imply a hazard of exactly zero. The output fix remains the right patch for old checkpoints.

**Does skipping recovery bias the 96 h numbers?** Upward, by an amount we have not measured. The placeholder restoration times are mostly days to weeks, so the effect inside 96 h should come mainly from the lightest damage state and from telecom congestion.

**Could a restored site fail again?** Not in the design above. If it could, the process alternates and neither curve is monotone; that is a different model and is not proposed.

**Is the 4–8% figure alarming?** No. It counts any drop above 1e-6. The median drop is 0.00001 to 0.0001; 0.2% to 0.3% of pairs drop by more than 0.01; the largest single drop is 0.16 to 0.33. Removing them changes the validation loss in the sixth decimal.

**Then was Week 27 wasted?** No. It produced a model whose outputs are hazards, at no cost in fit. What changes is the justification: Week 27 framed the hazard head as fixing a defect; the defect turns out to be small in size (though the rare large falls are real), and the head's value is that the survival likelihood and a later recovery head need it.

## 10. One correction on the record

The Week 28 handoff quotes "20 relation types, 13,709 edges" for the frozen graph. From the files:

| Graph | Relation types | Edges |
|---|---|---|
| `nyc_infra_heterodata_v1_frozen_base.pt` (simulator's dependency graph) | 19 | 13,709 |
| `nyc_infra_heterodata_v1_frozen_failover_edges.pt` (what every GNN campaign trains on) | 20 | 37,355 |

The difference is the 23,646 telecom failover edges. The carry card should read 6,231 / 20 / 37,355 for the surrogate.

## 11. References

Checked against the publisher or arXiv page unless marked.

- Gensheimer, M. F., & Narasimhan, B. (2019). A scalable discrete-time survival model for neural networks. *PeerJ* 7:e6257. https://doi.org/10.7717/peerj.6257
- Kvamme, H., & Borgan, Ø. (2021). Continuous and discrete-time survival prediction with neural networks. *Lifetime Data Analysis* 27:710–736. https://doi.org/10.1007/s10985-021-09532-6 (This is the Logistic-Hazard paper. The 2019 JMLR paper by Kvamme, Borgan & Scheel is about Cox models and is not the right citation.)
- Tutz, G., & Schmid, M. (2016). *Modeling Discrete Time-to-Event Data*. Springer. https://doi.org/10.1007/978-3-319-28158-2
- Cao, W., Mirjalili, V., & Raschka, S. (2020). Rank consistent ordinal regression for neural networks. *Pattern Recognition Letters* 140:325–331. https://doi.org/10.1016/j.patrec.2020.11.008
- Chernozhukov, V., Fernández-Val, I., & Galichon, A. (2010). Quantile and probability curves without crossing. *Econometrica* 78(3):1093–1125. https://doi.org/10.3982/ECTA7880 (Proposition number read from the arXiv version.)
- Barlow, R. E., Bartholomew, D. J., Bremner, J. M., & Brunk, H. D. (1972). *Statistical Inference under Order Restrictions*. Wiley. (Existence confirmed by citation only; not read.)
- Putter, H., Fiocco, M., & Geskus, R. B. (2007). Tutorial in biostatistics: competing risks and multi-state models. *Statistics in Medicine* 26:2389–2430. https://doi.org/10.1002/sim.2712 (Covers reversible transitions; the statement that state occupancy is non-monotone is our own inference.)
- Bruneau, M., et al. (2003). A framework to quantitatively assess and enhance the seismic resilience of communities. *Earthquake Spectra* 19(4):733–752. https://doi.org/10.1193/1.1623497
- Varbella, A., Gjorgiev, B., & Sansavini, G. (2023). Geometric deep learning for online prediction of cascading failures in power grids. *Reliability Engineering & System Safety* 237:109341. https://doi.org/10.1016/j.ress.2023.109341 (A GNN cascade surrogate with no recovery. The search found no GNN cascade surrogate that also models recovery.)
