# Week 25 Concept Note v3 — Dynamic flood forcing, Dr. Lin's question, and what the designed decomposition showed

*Supersedes v1 and v2 (v3 adds the Mac-reproduced N = 100 dense-grid arms W3, P−3 and the closing arm V3, which make the decomposition converge). Branch `feat/dynamic-flood-maps` @ `4a05bf3` + the three-knob patch to `src/simulation/dynamic_forcing.py` (default build byte-identical, gate in §7.1). All runs N = 100, paired seeds (42/43), legacy arm, 360-h horizon. Every number below comes from a file in `week25_results/`; the sandbox that produced them reproduced the committed Week 24 files bit-for-bit (§7.1), and the Mac re-run matched every shared file to the last digit.*

Sections 1–3 (the bathtub rule, spread-vs-clock, the null stated exactly) are unchanged from v1 and not repeated. This version replaces v1 §4–§9.

---

## 4. The corrected mechanism inventory (v2)

### 4.1 The peak-anchored demand surge — the physics, confirmed

`m(t) = 1 + 0.5·exp(−(t − t_peak)/12)`, `m = 1.0` before the peak, jump to 1.5 at the peak instant (`cascade_joint.py:243–244`, `intra_telecom.py:226–228`). The v1 simulator is not autonomous, by design (Week 23 §3.4). Moving the surge 6 h *later* relative to the flood, with everything else held fixed (§7.3, dense grid): **+229 ± 539 nodes at t360, super-cascade fraction 23 % → 38 %, 69 runs up / 21 down.** Moving it 6 h *earlier*: **−242 ± 587, 14 up / 82 down** on the dense grid (−82 ± 243, 4 / 96 on the standard grid). **The phase effect is symmetric** — an earlier surge removes about as much as a later one adds. Telecom-led in both directions (+131 / −139 telecom, +69 / −73 fuel on the dense grid). On the dense grid this one clock reproduces the flood-shift arm almost exactly (§4.2).

### 4.2 The evaluation grid — not an "artifact that explains the remainder"; a genuine sensitivity of the tipping zone

v1 claimed the grid was the likely source of most of the shift-study delta. The refinement test (§7.4) says otherwise, and something more interesting:

- The −6 h flood-shift delta shrinks only modestly under refinement (+315 → **+244 ± 554**, 63/33, N = 100) — and on the dense grid it **coincides with the surge-shift arm** (+244 vs +229, within 15 nodes). Shifting the flood and shifting the surge are two ways of changing the same phase; on a fine grid they agree, on the coarse grid they disagree (+315 vs +75). That convergence is the proof that the phase is the physics and the residual is grid.
- The grid changes **how many runs tip** into the super-cascade mode, for *both* forcing modes: static_peak 0 % → 16 %, arrival 3 % → 23 %. The contained-mode end state is grid-stable (arrival contained runs: 2,342 on 6 h vs 2,305 on 3 h). Coarser grid ⇒ more inter-kill latency (kills realized only at grid instants, `cascade_joint.py:235–236`) ⇒ fewer runs reach the tipping threshold while the surge is high.
- Consequently the **6-h grid under-samples the surge decay after a +6 h surge shift** (samples at m = 1.5 then 1.11 instead of 1.5, 1.30, 1.07). That is why the phase effect measured on the standard grid was only +75 (24 % of the delta) and on the dense grid +229 (73–76 %). The 24 % figure in the first pass was a measurement confound, now understood.

On the dense grid, grid placement contributes ~15 of the +244 (about 6 %). On the coarse grid the flood-shift arm overshoots the converged value (+315) and the surge-shift arm undershoots it (+75), in opposite directions, for the sampling reason above. **The Week 24 asymmetry (+13 % early / ≤ 2 % late) is therefore a coarse-grid artifact, not a property of the tipping zone.** Closed by V3 (flood +6 h, peak fixed, dense grid): **−242.0 ± 587.4, 20/78, super-cascade 0.23 → 0.07** — within 0.2 nodes of the earlier-surge arm P−3 (−241.8). On the dense grid the four arms pair up: flood earlier +244 ↔ surge later +229; flood later −242 ↔ surge earlier −242.

### 4.3 t = 0 clipping — absent in 880_9 (unchanged from v1)

880_9 has 75 arrivals at exactly 0 h and none before 10.65 h; ±3/±6 create no new slug. Present in 914_6 (12 nodes at −6) and 605_5 (12 / 75).

### 4.4 Fixed-camera evaluation — intermediate horizons only (unchanged)

End states are plateaued (t240 → t360: +5 unshifted, +26 at −6, +7 at +6), so t360 deltas are structural.

### 4.5 The hour-0 origin rule — a real, small, now-measured artifact of the *unshifted* world

No propagation is evaluated at the origin instant (`cascade_joint.py:216–217`), so the 75 hour-0 seeds' zero-buffer dependents are realized one bin late in the unshifted run only. Measured: translating the whole model by +6 h gives an **identical end state** (paired Δ = 0.00, identical fingerprint) but ~25 nodes/run realized 6 h earlier at intermediate times (peak-relative −2 → −8, −20 → −26). This is why the translation test's reference had to be a shifted world, and why the Week 23 intermediate-horizon labels (t6/t24) for scenarios with many hour-0 arrivals carry a one-bin bias worth noting.

### 4.6 Not a mechanism: random-stream misalignment (unchanged)

All draws precede the loop (`fragility.py:193–196`; `multi_scenario_runner.py:226, 233`); no random call inside the cascade or the closures. Every paired difference above is deterministic. Wide paired SDs are bimodality, not noise.

---

## 5. What this does to the record

**Week 24 numbers:** all stand, reproduced bit-for-bit (§7.1).

**Week 24 words — four corrections:**
1. "The surge peak has NO mechanical role" → retract; it anchors the demand surge, and that surge is three quarters of the shift effect.
2. "Clipping-induced ~150-node concentration difference" → wrong for 880_9; no clipping there. The +153 at t0 is failed-by-peak under a fixed camera.
3. "2.7× more damage carried into the surge peak" → mechanically meaningful *via the demand surge*: the dead set at the peak instant is the argument of the telecom closure at m = 1.5. Reword accordingly.
4. "Fourth independent instrument" → a robustness check on the Week 23 timing divergence through the same lever; not independent.
5. "Asymmetric — early arrival raises the end state, late arrival barely moves it" → withdrawn. On a 3-h grid the flood-shift arms are symmetric (+244 / −242) and coincide with the surge-shift arms (+229 / −242). The asymmetry was the 6-h grid's (§4.2).

**Week 23 — one new caveat, one strengthening:** the tipping-zone super-cascade fraction depends on the evaluation-grid resolution (§4.2), so the *magnitude* of end-state divergence in 880_9 is grid-conditional. The *direction* strengthens under refinement: arrival − static_peak = +84 ± 242 (99/1) on 6 h → **+141 ± 435 (95/5)** on 3 h, and arrival tips more runs than static_peak on both grids (3 vs 0; 23 vs 16). "Timing opens outcome space" is more true on a finer grid, not less.

**v1 of this note — one correction of my own:** v1 §9 attributed the non-surge remainder to grid quantization and told memo v3 to say so. The refinement test contradicts that (§4.2). v2 §9 replaces it.

---

## 6. Honest scope statement (unchanged from v1, one line added)

Time-dependent things v1 models: Weibull inter-layer buffers (autonomous); the peak-anchored telecom demand surge (non-autonomous, a modeling choice with a Sandy argument); order-dependent Motter–Lai subway overload with no recovery; directed fuel/water starvation. Not modeled: depth–duration fragility; any recovery in the loop; citywide surge propagation delay; the power-physics arm under dynamic forcing; a ramp-up of the demand surge (it steps 1.0 → 1.5). **Added:** the engine evaluates inter-kills only at grid instants, so the evaluation grid is a model parameter of the tipping zone, not merely an output resolution.

---

## 7. The experiment — design, gates, results, adjudication

### 7.1 Gates (all passed)

- **Default build unchanged:** patched module, no knobs, N = 20 vs the unshifted N = 100 file, first 20 runs: six keys identical, 0 mismatches.
- **Sandbox = pipeline:** unshifted arrival N = 100 on 880_9 reproduced the committed Week 24 reference exactly (t360 mean 2388.71 / sd 288.91 / t0 996.66); the Week 24 −6 h arm likewise (+315.39 ± 619.24, t0 +152.97); 914_6 unshifted 337.04 vs 337.00.

### 7.2 Arms (880_9 unless noted; env knobs `DYNAMIC_ARRIVAL_SHIFT_H` / `DYNAMIC_TPEAK_SHIFT_H` / `DYNAMIC_INTRA_ORIGIN_SHIFT_H`; dense grid = `DYNAMIC_PRE_PEAK_STEP_H=3` with 3-h post-peak offsets to 96 h then 120/144/192/240/300/360)

| arm | grid | arrival | t_peak | intra | meaning | t360 mean ± sd | super-cascade (≥3,000) |
|---|---|---|---|---|---|---|---|
| U | 6 h | 0 | 0 | 0 | unshifted arrival world | 2388.7 ± 289 | 0.03 |
| W | 6 h | −6 | 0 | 0 | Week 24 −6 h arm | 2704.1 ± 704 | 0.25 |
| R | 6 h | +6 | +6 | 0 | whole model translated (reference) | 2388.7 ± 289 | 0.03 |
| T | 6 h | +12 | +12 | 0 | R translated again — **translation test** | 2388.7 ± 289 | 0.03 |
| P− | 6 h | +6 | +6 | −6 | surge 6 h earlier than flood | 2307.0 ± 120 | 0.00 |
| P+ | 6 h | +6 | +6 | +6 | surge 6 h later than flood | 2463.6 ± 441 | 0.08 |
| S3 | 3 h | static_peak | 0 | 0 | Week 23 reference, dense grid | 2549.7 ± 626 | 0.16 |
| U3 | 3 h | 0 | 0 | 0 | arrival, dense grid | 2691.0 ± 710 | 0.23 |
| R3 | 3 h | +6 | +6 | 0 | translated reference, dense grid | 2691.0 ± 710 | 0.23 |
| P+3 | 3 h | +6 | +6 | +6 | surge 6 h later, dense grid | 2920.0 ± 835 | 0.38 |
| P−3 | 3 h | +6 | +6 | −6 | surge 6 h earlier, dense grid | 2449.2 ± 433 | 0.07 |
| W3 | 3 h | −6 | 0 | 0 | Week 24 −6 h arm, dense grid | 2934.7 ± 838 | 0.39 |
| V3 | 3 h | +6 | 0 | 0 | Week 24 +6 h arm, dense grid (closing arm) | 2448.9 ± 433 | 0.07 |
| R6 / P−6 / P+6 | 6 h | 914_6 control | | | | 337.0 / 330.7 / 340.4 | 0 |

### 7.3 Paired comparisons

| comparison | Δ t360 (mean ± sd) | runs up / down | reads as |
|---|---|---|---|
| T − R | **0.00 ± 0.00, six keys identical, 100/100 runs** | — | Lin's null holds on our simulator |
| R − U | 0.00 (fingerprint identical); intermediate-time diffs on ~25 nodes/run | — | origin-rule artifact of the unshifted world |
| R3 − U3 | 0.00; t0 +1.6 | — | same, dense grid |
| P+ − R (6 h) | +74.9 ± 320.7 | 66 / 18 | surge phase, under-sampled decay |
| P− − R (6 h) | −81.7 ± 242.7 | 4 / 96 | surge phase, earlier |
| **P+3 − R3 (3 h)** | **+229.0 ± 539.5** | **69 / 21** | **surge phase, properly sampled** |
| W − U (6 h) | +315.4 ± 619.2 | 70 / 24 | Week 24 −6 h arm (reproduced) |
| **P−3 − R3 (3 h)** | **−241.8 ± 587.5** | **14 / 82** | **surge phase, earlier — symmetric with P+3** |
| **W3 − U3 (3 h)** | **+243.8 ± 554.0** | **63 / 33** | **flood earlier on the dense grid — converges to P+3** |
| **V3 − U3 (3 h)** | **−242.0 ± 587.4** | **20 / 78** | **flood later on the dense grid — coincides with P−3** |
| U3 − S3 (3 h) | +141.3 ± 434.8 | 95 / 5 | Week 23 timing effect, dense grid |
| P+6 − R6 / P−6 − R6 (914_6) | +3.3 ± 21.0 / −6.3 ± 12.0 | 36/57 / 0/99 | control: absorbed |

Crash decisions (≥ 1,000 / 1,200 / 1,500): unchanged in every arm.

### 7.4 Adjudication against the pre-registration (v1 §7)

1. T ≡ R exactly — **pass.**
2. P+ Δ > 0 with wider spread; P− Δ ≤ 0 with narrower spread — **pass.** `|Δ(P+)| > |Δ(P−)|` on the standard grid — **fail** (+75 vs −82): the mean phase effect is roughly symmetric there; the asymmetry is in the *distribution* (later surge opens a fat upper tail — 8 % super-cascade; earlier surge pulls 96 % of runs down). On the dense grid the effect is symmetric (+229 / −242): **the asymmetry part of prediction 2 fails on both grids.** The Week 24 asymmetry is not a property of the phase mechanism.
3. Surge-phase share of the −6 h delta: standard grid **24 %** (below the pre-registered 25 % threshold, which would have meant "mostly grid"); dense grid: the flood-shift arm (+244) and the surge-shift arm (+229) agree within 15 nodes — the phase is ~94 % of the dense-grid delta. The pre-registered expectation (majority) holds once the confound in §4.2 is removed. All three numbers are reported; the dense-grid convergence is the result.
4. 914_6 control ≤ 10 — **pass** (3.3 / 6.3).
5. No crash flips — **pass.**

6. V3 (added after the refinement result): Δ in [−280, −200] — **pass** (−242.0); super-cascade ≤ 0.10 — **pass** (0.07); ≥ 80 runs smaller — **miss by two** (78).

One thing the original pre-registration did not anticipate: the grid-refinement result (§4.2), which became the week's most consequential finding for Week 23.

### 7.5 Open, in priority order

1. ~~V3~~ — run. Pre-registered Δ ≈ −240 in [−280, −200] and super-cascade ≤ 0.10: **both hold** (−242.0; 0.07). "≥ 80 runs smaller": **78 — missed by two**, recorded as such.
2. Grid sensitivity of the Week 23 campaign: static_peak vs arrival on the 3-h grid for the three deep-dive storms at N = 1,000 (Torch `cpu_short`, same sbatch pattern as `campaign_dynamic_v1` with the two env knobs). This is the reviewer-ready sensitivity Dr. Lin will ask for, and it decides whether the "+21–58 %" saturated-storm numbers are grid-conditional.
3. Grid convergence: 1-h step on 880_9 at N = 100 (does the super-cascade fraction keep rising, or plateau near the 3-h value?).

---

## 8. The one-paragraph answer to Dr. Lin (v2)

You were right: for a model with no clock of its own, shifting every arrival by the same amount is a relabeling, and we now have that as a measurement — translating the whole model by 6 and 12 hours gives records that are identical on every key in every run. Our shift study was not a relabeling for one physical reason and one implementation reason. The physical one: the telecom demand surge is anchored to the storm peak (memo §3.4), so moving the flood while the peak stays put changes the phase between the flood and the surge. Measured on its own, with everything else translated, that phase effect is about +230 nodes at the end state for a 6-hour later surge and −240 for a 6-hour earlier one — symmetric — and on a fine evaluation grid shifting the flood gives the same number as shifting the surge (+244 vs +229), so the phase is the whole physics. The asymmetry we showed you was the coarse grid, not the city: on a 3-hour grid a 6-hour later flood gives −242 and a 6-hour earlier flood +244. The implementation one: the engine realizes cascade kills only at evaluation-grid instants, and in the tipping-zone storm the grid resolution changes how many runs tip at all (0 → 16 % for static forcing, 3 → 23 % for arrival forcing, from 6-hour to 3-hour steps) while leaving the contained-mode outcome unchanged. Clipping at hour zero, which we first blamed, does not occur in this storm; there is no random-stream effect because every draw precedes the cascade loop. The Week 23 claim survives refinement and strengthens (+84 → +141, arrival always tipping more runs than static); its magnitude in the tipping zone is grid-conditional, and we will report it that way.

---

## 9. Replacement wording

**Memo v3 §4.6 (keep every number; replace the interpretation):**

> Arrival-error robustness (±3/±6 h uniform shifts, N = 100, 3 storms): no shift changes any crash decision. End-state deltas ≤ 2.2 % (914_6) and ≤ 0.4 % (605_5); the crash-edge storm 880_9 responds asymmetrically (earlier arrival up to +13 % / +315 nodes; later ≤ 2 %). A uniform shift of an autonomous model changes nothing (verified: translating the whole model — arrivals, grid, surge clock — reproduces every record exactly, §4.7). The deltas arise because the telecom demand surge is anchored to the storm peak: shifting the flood against a fixed peak changes the flood–surge phase. Isolated on a 3-h evaluation grid, a 6-h later surge adds +229 ± 539 nodes (super-cascade 23 % → 38 %) and a 6-h earlier surge removes −242 ± 587: the phase effect is symmetric. On that grid the flood-shift arm gives +244 ± 554, within 15 nodes of the surge-shift arm — and the flood +6 h arm gives −242 ± 587 (super-cascade 0.23 → 0.07), coinciding with the earlier-surge arm. Flood shifts and surge shifts coincide on the fine grid, so the phase is the mechanism, and the Week 24 asymmetry (early +13 % / late ≤ 2 %) was the 6-h grid, not the tipping zone. Clipping at hour zero does not occur in 880_9. All randomness precedes the cascade loop, so paired differences are deterministic; the 2.4× wider spread under −6 h is bimodality. The shift asymmetry is a robustness check on the §3.4 surge interaction, not an independent instrument.

**Memo v3 new §4.7 — Translation test, surge-phase decomposition, and grid sensitivity:** the tables of §7.2–7.3 above, plus: *the evaluation-grid resolution is a parameter of the tipping zone: refining 6 h → 3 h raises the super-cascade fraction for both forcing modes (static_peak 0 → 16 %, arrival 3 → 23 %) with the contained-mode end state unchanged; the static-vs-arrival difference grows from +84 to +141. Week 23's saturated-storm magnitudes are therefore reported as 6-h-grid values pending the grid sensitivity on Torch (§7.5).*

**Deck slide (Lin standard):**
> **Title:** Shifting the flood against a fixed demand-surge clock is a phase change, not a translation — on a fine grid, moving the flood and moving the surge give the same answer.
> **On slide:** mechanism diagram (arrival curve vs m(t) step-and-decay, two phases) before the numbers; T ≡ R "identical on every key" as the proof line; +244 (flood shift) vs +229 (surge shift), and −242 for the earlier surge, interpreted in one line each.
> **SAY:** Dr. Lin was right: a pure shift changes nothing, and we proved that on our own model. Ours wasn't pure — the call-volume surge stays pinned to the peak. We moved only the surge and measured the effect by itself.

**Deck slide 2 (new, the caveat we found ourselves):**
> **Title:** In the tipping-zone storm, how many runs go super-cascade depends on how often the engine looks — 6-h vs 3-h steps moves it by 15–20 points for both forcing modes.
> **SAY:** This is a property of our engine, not of the city. We're re-running the Week 23 deep dive on the finer grid before we quote the big numbers again.

---

## Appendix — files

`dynamic_forcing.py` (patched; 156 lines; grep count of the three knob names = 8), `week25_results/week25_arm_table_v1.csv`, `surgephase_{plus6,minus6}_vs_translatedref_grid6_h360_n100_v1.csv`, `surgephase_plus6_vs_translatedref_grid3_h360_n100_v1.csv`, `arrival_vs_staticpeak_grid3_h360_n100_v1.csv`, `translation_test_T12_vs_R6_grid6_880_9_n100_v1.txt`, `translation_R6_vs_U_grid3_880_9_n100_v1.txt`.
