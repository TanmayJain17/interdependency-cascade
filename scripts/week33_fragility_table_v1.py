#!/usr/bin/env python3
"""
scripts/week33_fragility_table_v1.py — end-to-end fragility: verdicts P0-P7 on the JSONs written by
hpc/audit_gnn_fragility_v1.sbatch (scripts/week33_fragility_audit_v1.py), one per checkpoint:

  <dir>/e2e_<arm>_<cell>_s<seed>.json     cell in static_plain | arrival_timing;  seed 1, 2

  arm                 fragility                 init (seed 1 / seed 2)        seed loss  regime      detach
  a0_formula          Hazus-style formula       -                             -          -           -      (Week 32 checkpoints, re-audited)
  a1a_lognormal_l1    learnable lognormal       medians x2, beta x0.5 / x0.5, x2   lambda 1   joint       yes    MAIN DESIGN
  a1b_lognormal_l0    learnable lognormal       same                          lambda 0   joint       yes    identifiability demonstration
  a1c_lognormal_l0p1  learnable lognormal       same                          lambda 0.1 joint       yes
  a1d_lognormal_l10   learnable lognormal       same                          lambda 10  joint       yes
  a1e_lognormal_nodetach learnable lognormal    same                          lambda 1   joint       no     leakage test
  a4_sequential       learnable lognormal       same                          stage 1 on seed frequency alone, then frozen
  a2_mixture          mixture of 3 lognormals   medians x{0.5,1,2}, equal weights   lambda 1  joint    yes    flexible monotone head
  a3_blackbox         none (depth is a feature, no composition)                -          -           -

    python scripts/week33_fragility_table_v1.py --dir <json dir> [--fingerprint analysis/gnn/week33_fragility_fingerprint_v1.json]
                                                [--out analysis/gnn/week33_fragility_v1.csv]

VOCABULARY
  "Assumed" parameters: the six (median, beta) pairs in src/simulation/fragility.py that generated every direct
  failure of the simulator (power 0.6/0.5, telecom 0.3/0.6, hospital 0.6/0.5, subway 0.1/0.4, water 0.5/0.5,
  fuel 0.5/0.6). Recovery error: |learned/assumed - 1| for the median, |learned - assumed|/assumed for beta.
  Soft log loss, floor, count error, GeoClaw: as in week31_direct_table_v1.py. Arm value on a storm = mean of
  the two seeds (which here also means the mean over the two opposite perturbed starts).
  Curve error: mean |phi_hat - phi_formula| over the wet non-external site-storms of the 13 test storms.
  Decomposition error: mean |g - g_hat| over wet test sites with seed frequency <= 0.99, where
  g_hat = (pi_hat - s)/(1 - s) is the simulator's own cascade-given-not-a-seed probability.
  Monotonicity violation rate: share of wet test site-storms where raising the depth by 10% lowers P(failed by 96 h)
  by more than 1e-6.
  "Big four" types: power, telecom, water, fuel (>= 300 informative site-storms each in the 50 training storms);
  "small two": subway (114 informative site-storms, median 0.1 m) and hospital (105).

PRE-REGISTRATION (written Wed 4 Nov 2026, before any code of the fragility head existed and before any training).
Seen before writing: the Week 32 direct-model results (commit f1a1e48 and the Torch jobs 21158520 / 21225432), the
study guide's count of informative site-storms per type, and the Week 11 warm-start run (parameters moved < 0.2 %
from a start AT the assumed values, with a penalty keeping them there: not a test of recovery). No output of any
learned-fragility arm has been seen. Verdicts are read in both cells unless stated; "in every cell" = both.
  P0  gates: fingerprint F1-F6 pass; 32 of 32 new checkpoints finished with a history; every audit JSON passes its
      gates with an empty failure list; the re-audited A0 checkpoints reproduce their Week 32 13-storm soft log
      losses within 1e-4. If P0 fails no other verdict is given.
  P1  recovery (main design a1a, both seeds, i.e. from both opposite starts, in every cell): medians of the big four
      within 3 % and betas within 5 % of the assumed values; subway and hospital within 10 % on both.
  P2  identifiability (a1b, no seed loss): its 13-storm mean soft log loss is within 2 % of A0's in every cell
      (the totals fit), yet in at least 3 of its 4 runs (2 cells x 2 seeds) power or telecom ends with a median
      error >= 10 %, in the "more fragile" direction (learned median below the assumed one).
  P3  parity: a1a's 13-storm mean soft log loss is within +-2 % of A0's in every cell.
  P4  flexible head (a2): (a) its curve error exceeds a1a's in every cell; (b) its 13-storm mean soft log loss is
      within 3 % of a1a's in every cell (flexibility costs little in distribution, buys nothing on this data).
  P5  black box (a3): (a) worse than A0 on geoclaw_2050 in every cell; (b) a monotonicity violation rate above
      0.1 % of wet test site-storms in every cell.
  P6  regimes: a1c, a1d and a4_sequential each have a 13-storm mean soft log loss within 1 % of a1a's in every
      cell, and each recovers the big-four medians within 3 % (both seeds, every cell): the choice of lambda in
      [0.1, 10] and of joint vs sequential does not matter once the seed loss is present.
  P7  leakage (a1e, no detach): recovers the big-four medians within 3 % and betas within 5 % (both seeds, every
      cell), and its 13-storm mean soft log loss is within 1 % of a1a's: the complete, correctly specified concept
      does not leak even without the detach.
Named failure modes, to be read if they occur: P1 missing with the big four landing within 3 % of the ORACLE's
estimate (R, probit fit on the seed frequencies) but not of the assumed values = the finite sample, not the
network, sets the error; P2's second clause failing = the decomposition is identifiable from the totals alone on
this graph (cascade leverage too weak to move phi), which would be a finding, not a defect.

JSON expected per checkpoint (written by week33_fragility_audit_v1.py; the keys read here):
  identity: arm, cell, label_mode, timing, torch_seed, fragility (formula|lognormal|mixture|none), lambda_seed,
            regime, detach_phi, frag_init, train_storms, val_storms, test_storms, epochs_run, early_stopped
  gates: {name: bool}, failures: list
  fragility_params: {type: {median, beta, assumed_median, assumed_beta}}  (lognormal arms; mixture gives effective
            curve only; absent for formula/none)
  curve_error: {type: {mean_abs, max_abs}, all: {mean_abs, max_abs}}   (learned arms)
  test_per_scenario[storm]: level_m, model{soft_bce, count_err_per_t, ...}, b0_fragility{...}, floor_split_half{...},
            decomposition_error (float or null), mono_violation_rate (float)
  week32_match: {per_storm_max_abs_diff: float} (a0 only)
"""
import argparse
import csv
import json
import sys
from pathlib import Path

CELLS = ("static_plain", "arrival_timing")
SEEDS = (1, 2)
ARMS = ("a0_formula", "a1a_lognormal_l1", "a1b_lognormal_l0", "a1c_lognormal_l0p1", "a1d_lognormal_l10",
        "a1e_lognormal_nodetach", "a4_sequential", "a2_mixture", "a3_blackbox")
NEW_ARMS = tuple(a for a in ARMS if a != "a0_formula")
LOGNORMAL_ARMS = ("a1a_lognormal_l1", "a1b_lognormal_l0", "a1c_lognormal_l0p1", "a1d_lognormal_l10",
                  "a1e_lognormal_nodetach", "a4_sequential")
ASSUMED = {"power": (0.6, 0.5), "telecom": (0.3, 0.6), "hospital": (0.6, 0.5), "subway": (0.1, 0.4),
           "water": (0.5, 0.5), "fuel": (0.5, 0.6)}
BIG4 = ("power", "telecom", "water", "fuel")
SMALL2 = ("subway", "hospital")
GEOCLAW = "geoclaw_2050"
FINGERPRINT_KEYS = ("F1", "F2", "F3", "F4", "F5", "F6")


def load(d):
    out, missing = {}, []
    for arm in ARMS:
        for cell in CELLS:
            for s in SEEDS:
                p = Path(d) / f"e2e_{arm}_{cell}_s{s}.json"
                if not p.exists():
                    missing.append(p.name); continue
                with open(p) as fh:
                    out[(arm, cell, s)] = json.load(fh)
    return out, missing


def mean13(j, who="model"):
    tps = j["test_per_scenario"]
    return sum(tps[s][who]["soft_bce"] for s in tps) / len(tps)


def arm_mean(js, arm, cell):
    return sum(mean13(js[(arm, cell, s)]) for s in SEEDS) / len(SEEDS)


def storm_val(js, arm, cell, storm, who="model"):
    return sum(js[(arm, cell, s)]["test_per_scenario"][storm][who]["soft_bce"] for s in SEEDS) / len(SEEDS)


def rec_err(j, t):
    fp = j["fragility_params"][t]
    return abs(fp["median"] / fp["assumed_median"] - 1.0), abs(fp["beta"] - fp["assumed_beta"]) / fp["assumed_beta"]


def recovered(j, types, med_tol, beta_tol):
    return all(rec_err(j, t)[0] <= med_tol and rec_err(j, t)[1] <= beta_tol for t in types)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--fingerprint", default="analysis/gnn/week33_fragility_fingerprint_v1.json")
    ap.add_argument("--out", default="analysis/gnn/week33_fragility_v1.csv")
    a = ap.parse_args()
    js, missing = load(a.dir)
    V = {}

    # ---- P0 --------------------------------------------------------------------------------------------
    fail = list(missing)
    if Path(a.fingerprint).exists():
        fp = json.load(open(a.fingerprint))
        for k in FINGERPRINT_KEYS:
            if fp.get(k, {}).get("pass") is not True:
                fail.append(f"fingerprint {k}: {fp.get(k, {}).get('pass')}")
    else:
        fail.append(f"fingerprint file not found: {a.fingerprint}")
    for key, j in js.items():
        arm, cell, s = key
        i = j.get("identity", {})
        if i.get("arm") != arm or i.get("cell") != cell or int(i.get("torch_seed", -1)) != s:
            fail.append(f"{key}: identity does not match the file name")
        g = j.get("gates", {})
        if not g or not all(bool(v) for v in g.values()) or j.get("failures"):
            fail.append(f"{key}: gates {g} failures {j.get('failures')}")
        if arm == "a0_formula":
            m = j.get("week32_match", {}).get("per_storm_max_abs_diff")
            if m is None or m > 1e-4:
                fail.append(f"{key}: A0 does not reproduce Week 32 (max abs diff {m})")
    V["P0"] = (not fail, "; ".join(fail) if fail else "fingerprint F1-F6, 36/36 JSONs gate-clean, A0 reproduces Week 32")
    print(f"P0 gates: {'PASS' if V['P0'][0] else 'FAIL'} — {V['P0'][1]}")
    if not V["P0"][0]:
        print("No other verdict is given (P0 failed)."); sys.exit(1)

    storms = sorted(js[("a0_formula", CELLS[0], 1)]["test_per_scenario"],
                    key=lambda st: js[("a0_formula", CELLS[0], 1)]["test_per_scenario"][st]["level_m"])

    # ---- tables ---------------------------------------------------------------------------------------
    rows = []
    for cell in CELLS:
        print(f"\n=== {cell} — 13-storm mean soft log loss, curve error, decomposition error, mono violations (mean of seeds) ===")
        print(f"{'arm':<24}{'loss':>8}{'vs A0':>8}{'curve':>8}{'decomp':>8}{'mono%':>7}  recovered (median | beta) per type")
        a0 = arm_mean(js, "a0_formula", cell)
        for arm in ARMS:
            m = arm_mean(js, arm, cell)
            ce = [js[(arm, cell, s)].get("curve_error", {}).get("all", {}).get("mean_abs") for s in SEEDS]
            ce = sum(ce) / 2 if all(v is not None for v in ce) else float("nan")
            de = []
            for s in SEEDS:
                vals = [r.get("decomposition_error") for r in js[(arm, cell, s)]["test_per_scenario"].values()]
                vals = [v for v in vals if v is not None]
                de.append(sum(vals) / len(vals) if vals else float("nan"))
            de = sum(de) / 2
            mv = sum(sum(r.get("mono_violation_rate", 0.0) for r in js[(arm, cell, s)]["test_per_scenario"].values())
                     / 13 for s in SEEDS) / 2 * 100
            rec = ""
            if arm in LOGNORMAL_ARMS:
                parts = []
                for t in ASSUMED:
                    e = [rec_err(js[(arm, cell, s)], t) for s in SEEDS]
                    parts.append(f"{t[:4]} {max(x[0] for x in e)*100:.1f}|{max(x[1] for x in e)*100:.1f}%")
                rec = "  ".join(parts)
            print(f"{arm:<24}{m:8.4f}{m / a0:8.3f}{ce:8.4f}{de:8.4f}{mv:7.2f}  {rec}")
            for s in SEEDS:
                j = js[(arm, cell, s)]
                for st in storms:
                    r = j["test_per_scenario"][st]
                    row = {"arm": arm, "cell": cell, "torch_seed": s, "storm": st, "level_m": r["level_m"],
                           "model_soft_bce": r["model"]["soft_bce"], "b0_soft_bce": r["b0_fragility"]["soft_bce"],
                           "floor_soft_bce": r["floor_split_half"]["soft_bce"],
                           "count_err_96": r["model"]["count_err_per_t"][-1], "decomp_err": r.get("decomposition_error"),
                           "mono_violation_rate": r.get("mono_violation_rate")}
                    if arm in LOGNORMAL_ARMS:
                        for t in ASSUMED:
                            row[f"median_{t}"] = j["fragility_params"][t]["median"]; row[f"beta_{t}"] = j["fragility_params"][t]["beta"]
                    rows.append(row)
        print(f"{'geoclaw_2050 by arm':<24}" + "  ".join(f"{arm.split('_')[0]} {storm_val(js, arm, cell, GEOCLAW):.4f}" for arm in ARMS))

    # ---- P1-P7 -----------------------------------------------------------------------------------------
    def every_cell(fn):
        res = {c: fn(c) for c in CELLS}
        return all(v[0] for v in res.values()), "; ".join(f"{c}: {v[1]}" for c, v in res.items())

    def p1(c):
        ok = all(recovered(js[("a1a_lognormal_l1", c, s)], BIG4, 0.03, 0.05) and
                 recovered(js[("a1a_lognormal_l1", c, s)], SMALL2, 0.10, 0.10) for s in SEEDS)
        worst = max((rec_err(js[("a1a_lognormal_l1", c, s)], t)[0], t, s) for s in SEEDS for t in BIG4)
        return ok, f"big-four worst median error {worst[0]*100:.1f}% ({worst[1]}, seed {worst[2]})"
    V["P1"] = every_cell(p1)

    def p2(c):
        ratio = arm_mean(js, "a1b_lognormal_l0", c) / arm_mean(js, "a0_formula", c)
        return abs(ratio - 1) <= 0.02, f"loss ratio to A0 {ratio:.3f}"
    ok2, txt2 = every_cell(p2)
    biased = 0
    for c in CELLS:
        for s in SEEDS:
            j = js[("a1b_lognormal_l0", c, s)]
            if any(j["fragility_params"][t]["median"] <= 0.9 * j["fragility_params"][t]["assumed_median"] for t in ("power", "telecom")):
                biased += 1
    V["P2"] = (ok2 and biased >= 3, f"{txt2}; runs with power/telecom median >= 10% too low: {biased}/4")

    def p3(c):
        r = arm_mean(js, "a1a_lognormal_l1", c) / arm_mean(js, "a0_formula", c)
        return abs(r - 1) <= 0.02, f"a1a/A0 = {r:.3f}"
    V["P3"] = every_cell(p3)

    def p4(c):
        ce = lambda arm: sum(js[(arm, c, s)]["curve_error"]["all"]["mean_abs"] for s in SEEDS) / 2
        r = arm_mean(js, "a2_mixture", c) / arm_mean(js, "a1a_lognormal_l1", c)
        return (ce("a2_mixture") > ce("a1a_lognormal_l1")) and abs(r - 1) <= 0.03, \
            f"curve error a2 {ce('a2_mixture'):.4f} vs a1a {ce('a1a_lognormal_l1'):.4f}; loss ratio a2/a1a {r:.3f}"
    V["P4"] = every_cell(p4)

    def p5(c):
        gc = storm_val(js, "a3_blackbox", c, GEOCLAW) > storm_val(js, "a0_formula", c, GEOCLAW)
        mv = sum(sum(r.get("mono_violation_rate", 0.0) for r in js[("a3_blackbox", c, s)]["test_per_scenario"].values()) / 13
                 for s in SEEDS) / 2
        return gc and mv > 0.001, f"geoclaw a3 {storm_val(js, 'a3_blackbox', c, GEOCLAW):.4f} vs A0 {storm_val(js, 'a0_formula', c, GEOCLAW):.4f}; mono violations {mv*100:.2f}%"
    V["P5"] = every_cell(p5)

    def p6(c):
        base = arm_mean(js, "a1a_lognormal_l1", c)
        parts, ok = [], True
        for arm in ("a1c_lognormal_l0p1", "a1d_lognormal_l10", "a4_sequential"):
            r = arm_mean(js, arm, c) / base
            rec = all(recovered(js[(arm, c, s)], BIG4, 0.03, 1.0) for s in SEEDS)
            ok = ok and abs(r - 1) <= 0.01 and rec
            parts.append(f"{arm.split('_')[0]} {r:.3f}{'' if rec else ' (medians NOT within 3%)'}")
        return ok, ", ".join(parts)
    V["P6"] = every_cell(p6)

    def p7(c):
        r = arm_mean(js, "a1e_lognormal_nodetach", c) / arm_mean(js, "a1a_lognormal_l1", c)
        rec = all(recovered(js[("a1e_lognormal_nodetach", c, s)], BIG4, 0.03, 0.05) for s in SEEDS)
        return abs(r - 1) <= 0.01 and rec, f"loss ratio {r:.3f}; big four recovered: {rec}"
    V["P7"] = every_cell(p7)

    print("\n=== VERDICTS (pre-registered 4 Nov 2026) ===")
    for k in ("P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7"):
        ok, txt = V[k]
        print(f"  {k}  {'PASS' if ok else 'MISS':<6} {txt}")
    print(f"  {sum(1 for v in V.values() if v[0])} of {len(V)} predictions passed")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("arm", "cell", "torch_seed", "storm"), k))
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {a.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
