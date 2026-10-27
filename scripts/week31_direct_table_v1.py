#!/usr/bin/env python3
"""
scripts/week31_direct_table_v1.py — the direct model (flood map -> per-site failure probability in one
pass, no seed sampling): verdicts D0-D6 on the JSONs written by hpc/audit_gnn_direct_v1.sbatch
(scripts/week31_direct_audit_v1.py), one per checkpoint:

  <dir>/direct_<cell>_n<17|34|50>_s<seed>.json   cell in static_plain | static_timing | arrival_plain | arrival_timing
                                                 ("plain" here = no per-site timing columns; the expected-seed
                                                 summary is always on), n = number of training storms, seed 1, 2

Prints, per cell, the 13 held-out storms by water level with the soft log loss of each training size, the
fragility-only, lookup and two-stage baselines and the split-half floor; then the verdicts below. Writes one
CSV row per cell x training size x seed x storm.

    python scripts/week31_direct_table_v1.py --dir <json dir> [--fingerprint analysis/gnn/week31_direct_fingerprint_v1.json]
                                             [--out analysis/gnn/week31_direct_v1.csv]

VOCABULARY
  Target pi_hat_ik: share of the storm's 1,000 simulator runs in which site i has failed by horizon k
  (seeds included; the same <= rule as build_labels). Soft log loss (E1): mean over sites and horizons of
  -[pi_hat ln p + (1 - pi_hat) ln(1 - p)]; its minimum, reached at p = pi_hat, is the entropy of pi_hat, so the
  split-half floor (pi_hat from runs 1-500 scored against pi_hat from runs 501-1,000) is the number a perfect
  predictor would score. Count error (E3): |sum_i p_ik - sum_i pi_hat_ik| / sum_i pi_hat_ik per horizon.
  Arm value on a storm = mean over PyTorch seeds 1 and 2. "Ahead" = lower soft log loss.
  Baselines: B0 fragility only, p = phi (the Hazus seed probability, no cascade); B1 lookup50, the training
  storm whose expected seed count sum_i phi_i is closest to this storm's (the direct-model analogue of the
  two-stage nearest-storm lookup, which matches on the run's seed count); B2 the two-stage ctx57 checkpoint
  of the same cell averaged over 100 seed sets sampled from phi (scored by the patched predict_cascade.py).
  Storm groups by peak level: ABOVE = the 5 held-out storms above every old training storm (old maximum
  3.64 m), INSIDE = the 7 held-out GISSR storms at or below 3.64 m, OTHER = geoclaw_2050.
  Split by storm: 50 training (17 old + 33 new), 7 validation (new storms, every 5th by level from the 4th,
  mixing fast and slow), 13 test (unchanged from the two-stage experiment). Training-size curve: nested sets
  17 (the old storms) / 34 (17 + every other of the 33 new by level) / 50; the same 7 validation storms and
  early stopping on their soft log loss in every arm.

PRE-REGISTRATION (written Tue 27 Oct 2026, before any code of the direct model existed and before any
training). Seen before writing: the two-stage results of commit f2b40ea (plain and storm-summary variants,
13 held-out storms, lookup57), the count-error floor of 21 Oct, and the 20-run simulator preview on the 48
new storms. No output of any direct model has been seen. Agreed with the advisors on 22 Oct (split by storm,
predictions as drafted).
  D0  gates: fingerprint F1-F5 pass (F1 default build bit-identical to HEAD; F2 pi_hat recount on two
      storms equals the audit's; F3 phi-consistency: the seed frequency over 1,000 runs matches phi_i within
      0.03 on at least 99% of wet non-external sites, telecom_cluster_03042 excepted, and the 99 external
      sites have phi = 0 and zero seed frequency; F4 p >= phi and p non-decreasing in the horizon on random
      weights; F5 the loss at p = pi_hat equals the entropy of pi_hat); 24 of 24 checkpoints finished with a
      history and every JSON passes its own gates with an empty failure list. If D0 fails no other verdict.
  D1  the model learns a cascade: in every cell, on 13 of 13 storms, the direct model is ahead of B0.
  D2  with 50 maps the lookup is still hard to beat where it can be built: in every cell B1 is ahead of the
      direct model on at least 7 of the 12 GISSR storms; the direct model is ahead of B1 on geoclaw_2050 in
      at least 3 of the 4 cells.
  D3  the two-stage model stays ahead: in every cell B2 is ahead of the direct model on at least 9 of 13
      storms, and the direct model's 13-storm mean soft log loss is within 1.5x of B2's.
  D4  more maps help, and are not yet saturating: in every cell the 13-storm mean soft log loss falls
      17 -> 34 -> 50 and the 34 -> 50 gain is at least half the 17 -> 34 gain. Named failure mode: a 34 -> 50
      gain below 20% of the 17 -> 34 gain reads as "map diversity, not map count, is the limit".
  D5  counts: in the two static cells the count error at 96 h is within 0.05 of the split-half floor on at
      least 9 of 13 storms.
  D6  two starts agree: in every cell the two seeds agree in direction on at least 11 of 13 storms in each of
      the D2 (model vs B1), D3 (model vs B2) and D4 (17 vs 50) comparisons.

JSON expected per checkpoint (written by week31_direct_audit_v1.py; the keys read here):
  identity: cell, timing (0/1), label_mode (static|arrival), train_storms (int), torch_seed, target ("marginal"),
            val_storms (list), test_storms (list), early_stopped (bool), best_epoch
  gates:    {name: bool, ...}, failures: list
  test_per_scenario[storm]: level_m, model/b0_fragility/b1_lookup/b2_two_stage (or null)/floor_split_half,
            each with soft_bce (float, mean over horizons) and count_err_per_t (list of 4)
"""
import argparse
import csv
import json
import sys
from pathlib import Path

CELLS = ("static_plain", "static_timing", "arrival_plain", "arrival_timing")
SIZES = (17, 34, 50)
SEEDS = (1, 2)
OLD_MAX_LEVEL = 3.64
GEOCLAW = "geoclaw_2050"
FULL = 50
FINGERPRINT_KEYS = ("F1", "F2", "F3", "F4", "F5")


def load(d):
    out, missing = {}, []
    for cell in CELLS:
        for n in SIZES:
            for s in SEEDS:
                p = Path(d) / f"direct_{cell}_n{n}_s{s}.json"
                if not p.exists():
                    missing.append(p.name); continue
                with open(p) as fh:
                    out[(cell, n, s)] = json.load(fh)
    return out, missing


def identity_ok(key, j):
    cell, n, s = key
    i = j.get("identity", {})
    want_timing = 1 if cell.endswith("timing") else 0
    want_mode = "static" if cell.startswith("static") else "arrival"
    return (i.get("cell") == cell and int(i.get("timing", -1)) == want_timing and i.get("label_mode") == want_mode
            and int(i.get("train_storms", -1)) == n and int(i.get("torch_seed", -1)) == s
            and i.get("target") == "marginal" and len(i.get("test_storms", [])) == 13 and len(i.get("val_storms", [])) == 7)


def storms_by_level(j):
    tps = j["test_per_scenario"]
    return sorted(tps, key=lambda st: (tps[st]["level_m"], st))


def group(st, level):
    if st == GEOCLAW:
        return "OTHER"
    return "ABOVE" if level > OLD_MAX_LEVEL else "INSIDE"


def arm(js, cell, n, storm, who, field="soft_bce"):
    """Mean over the two seeds of js[(cell,n,seed)].test_per_scenario[storm][who][field]; None if absent."""
    vals = []
    for s in SEEDS:
        rec = js[(cell, n, s)]["test_per_scenario"][storm].get(who)
        if rec is None:
            return None
        v = rec[field]
        vals.append(v if not isinstance(v, list) else v)
    if isinstance(vals[0], list):
        return [sum(v[k] for v in vals) / len(vals) for k in range(len(vals[0]))]
    return sum(vals) / len(vals)


def per_seed(js, cell, n, storm, who, field="soft_bce"):
    return [js[(cell, n, s)]["test_per_scenario"][storm][who][field] for s in SEEDS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--fingerprint", default="analysis/gnn/week31_direct_fingerprint_v1.json")
    ap.add_argument("--out", default="analysis/gnn/week31_direct_v1.csv")
    a = ap.parse_args()

    js, missing = load(a.dir)
    verdicts = {}

    # ---- D0 -------------------------------------------------------------------------------------------
    d0_fail = list(missing)
    fp = {}
    if Path(a.fingerprint).exists():
        with open(a.fingerprint) as fh:
            fp = json.load(fh)
        for k in FINGERPRINT_KEYS:
            if fp.get(k, {}).get("pass") is not True:
                d0_fail.append(f"fingerprint {k}: {fp.get(k)}")
        pc = fp.get("F3", {})
        if pc.get("frac_wet_within_0.03", 0.0) < 0.99:
            d0_fail.append(f"phi-consistency below 99%: {pc.get('frac_wet_within_0.03')}")
        if pc.get("external_phi_zero_and_never_seed") is not True:
            d0_fail.append("external sites: phi or seed frequency not zero")
    else:
        d0_fail.append(f"fingerprint file not found: {a.fingerprint}")
    for key, j in js.items():
        if not identity_ok(key, j):
            d0_fail.append(f"{key}: identity does not match file name")
        g = j.get("gates", {})
        if not g or not all(bool(v) for v in g.values()) or j.get("failures"):
            d0_fail.append(f"{key}: gates {g} failures {j.get('failures')}")
    verdicts["D0"] = (not d0_fail, "; ".join(d0_fail) if d0_fail else "24/24 checkpoints, all gates and F1-F5 pass")
    print(f"D0 gates: {'PASS' if verdicts['D0'][0] else 'FAIL'} — {verdicts['D0'][1]}")
    if not verdicts["D0"][0]:
        print("No other verdict is given (D0 failed).")
        sys.exit(1)

    storms = storms_by_level(js[(CELLS[0], FULL, 1)])
    level = {st: js[(CELLS[0], FULL, 1)]["test_per_scenario"][st]["level_m"] for st in storms}
    gissr = [st for st in storms if st != GEOCLAW]
    have_b2 = all(js[(c, FULL, s)]["test_per_scenario"][st].get("b2_two_stage") is not None
                  for c in CELLS for s in SEEDS for st in storms)

    # ---- tables -----------------------------------------------------------------------------------------
    rows = []
    for cell in CELLS:
        print(f"\n=== {cell} — soft log loss per held-out storm (mean of seeds 1, 2) ===")
        print(f"{'storm':<26}{'level':>7}{'grp':>7}{'n17':>9}{'n34':>9}{'n50':>9}{'B0 phi':>9}{'B1 look':>9}{'B2 2stg':>9}{'floor':>9}{'cnt96':>8}{'flr96':>8}")
        for st in storms:
            m = {n: arm(js, cell, n, st, "model") for n in SIZES}
            b0 = arm(js, cell, FULL, st, "b0_fragility"); b1 = arm(js, cell, FULL, st, "b1_lookup")
            b2 = arm(js, cell, FULL, st, "b2_two_stage"); fl = arm(js, cell, FULL, st, "floor_split_half")
            c96 = arm(js, cell, FULL, st, "model", "count_err_per_t")[-1]
            f96 = arm(js, cell, FULL, st, "floor_split_half", "count_err_per_t")[-1]
            b2s = f"{b2:9.4f}" if b2 is not None else f"{'—':>9}"
            print(f"{st:<26}{level[st]:7.2f}{group(st, level[st]):>7}{m[17]:9.4f}{m[34]:9.4f}{m[50]:9.4f}{b0:9.4f}{b1:9.4f}{b2s}{fl:9.4f}{c96:8.3f}{f96:8.3f}")
            for n in SIZES:
                for s in SEEDS:
                    r = js[(cell, n, s)]["test_per_scenario"][st]
                    rows.append({"cell": cell, "train_storms": n, "torch_seed": s, "storm": st, "level_m": level[st],
                                 "group": group(st, level[st]), "model_soft_bce": r["model"]["soft_bce"],
                                 "b0_soft_bce": r["b0_fragility"]["soft_bce"], "b1_soft_bce": r["b1_lookup"]["soft_bce"],
                                 "b2_soft_bce": (r["b2_two_stage"] or {}).get("soft_bce"),
                                 "floor_soft_bce": r["floor_split_half"]["soft_bce"],
                                 "model_count_err_96": r["model"]["count_err_per_t"][-1],
                                 "floor_count_err_96": r["floor_split_half"]["count_err_per_t"][-1]})
        means = {n: sum(arm(js, cell, n, st, "model") for st in storms) / len(storms) for n in SIZES}
        mb1 = sum(arm(js, cell, FULL, st, "b1_lookup") for st in storms) / len(storms)
        mb2 = (sum(arm(js, cell, FULL, st, "b2_two_stage") for st in storms) / len(storms)) if have_b2 else None
        mfl = sum(arm(js, cell, FULL, st, "floor_split_half") for st in storms) / len(storms)
        print(f"{'13-storm mean':<26}{'':>14}{means[17]:9.4f}{means[34]:9.4f}{means[50]:9.4f}{'':>9}{mb1:9.4f}"
              f"{(f'{mb2:9.4f}' if mb2 is not None else f'{chr(8212):>9}')}{mfl:9.4f}")

    # ---- D1-D6 -------------------------------------------------------------------------------------------
    def every_cell(fn):
        res = {c: fn(c) for c in CELLS}
        return all(v[0] for v in res.values()), "; ".join(f"{c}: {v[1]}" for c, v in res.items())

    def d1(c):
        k = sum(arm(js, c, FULL, st, "model") < arm(js, c, FULL, st, "b0_fragility") for st in storms)
        return k == 13, f"ahead of B0 on {k}/13"
    verdicts["D1"] = every_cell(d1)

    def d2(c):
        k = sum(arm(js, c, FULL, st, "b1_lookup") < arm(js, c, FULL, st, "model") for st in gissr)
        return k >= 7, f"B1 ahead on {k}/12 GISSR"
    ok2, txt2 = every_cell(d2)
    gc = sum(arm(js, c, FULL, GEOCLAW, "model") < arm(js, c, FULL, GEOCLAW, "b1_lookup") for c in CELLS)
    verdicts["D2"] = (ok2 and gc >= 3, f"{txt2}; model ahead of B1 on geoclaw_2050 in {gc}/4 cells")

    if have_b2:
        def d3(c):
            k = sum(arm(js, c, FULL, st, "b2_two_stage") < arm(js, c, FULL, st, "model") for st in storms)
            mm = sum(arm(js, c, FULL, st, "model") for st in storms) / 13
            m2 = sum(arm(js, c, FULL, st, "b2_two_stage") for st in storms) / 13
            return (k >= 9 and mm <= 1.5 * m2), f"B2 ahead on {k}/13, mean {mm:.4f} vs B2 {m2:.4f} (x{mm / m2:.2f})"
        verdicts["D3"] = every_cell(d3)
    else:
        verdicts["D3"] = (None, "NO VERDICT: B2 (two-stage) scores absent from at least one JSON")

    def d4(c):
        m = {n: sum(arm(js, c, n, st, "model") for st in storms) / 13 for n in SIZES}
        g1, g2 = m[17] - m[34], m[34] - m[50]
        mono = m[17] > m[34] > m[50]
        ratio = (g2 / g1) if g1 > 0 else float("nan")
        note = " [diversity-limited: 34->50 gain < 20% of 17->34]" if (g1 > 0 and g2 < 0.2 * g1) else ""
        return (mono and g1 > 0 and g2 >= 0.5 * g1), f"17/34/50 = {m[17]:.4f}/{m[34]:.4f}/{m[50]:.4f}, gain2/gain1 = {ratio:.2f}{note}"
    verdicts["D4"] = every_cell(d4)

    def d5(c):
        k = sum(abs(arm(js, c, FULL, st, "model", "count_err_per_t")[-1]
                    - arm(js, c, FULL, st, "floor_split_half", "count_err_per_t")[-1]) <= 0.05 for st in storms)
        return k >= 9, f"within 0.05 of the floor at 96 h on {k}/13"
    r5 = {c: d5(c) for c in CELLS if c.startswith("static")}
    verdicts["D5"] = (all(v[0] for v in r5.values()), "; ".join(f"{c}: {v[1]}" for c, v in r5.items()))

    def sign_agree(c, who_a, who_b, n_a=FULL, n_b=FULL):
        k = 0
        for st in storms:
            d = [js[(c, n_a, s)]["test_per_scenario"][st][who_a]["soft_bce"]
                 - js[(c, n_b, s)]["test_per_scenario"][st][who_b]["soft_bce"] for s in SEEDS]
            k += (d[0] > 0) == (d[1] > 0)
        return k

    def d6(c):
        k2 = sign_agree(c, "model", "b1_lookup")
        k3 = sign_agree(c, "model", "b2_two_stage") if have_b2 else None
        k4 = sign_agree(c, "model", "model", 17, FULL)
        ok = k2 >= 11 and k4 >= 11 and (k3 is None or k3 >= 11)
        return ok, f"seeds agree on {k2}/13 (D2), {k3 if k3 is not None else '—'}/13 (D3), {k4}/13 (D4)"
    verdicts["D6"] = every_cell(d6)

    print("\n=== VERDICTS (pre-registered 27 Oct 2026) ===")
    for k in ("D0", "D1", "D2", "D3", "D4", "D5", "D6"):
        ok, txt = verdicts[k]
        tag = "PASS" if ok else ("NO VERDICT" if ok is None else "MISS")
        print(f"  {k}  {tag:<10} {txt}")
    n_pass = sum(1 for v in verdicts.values() if v[0] is True)
    n_dec = sum(1 for v in verdicts.values() if v[0] is not None)
    print(f"  {n_pass} of {n_dec} decided predictions passed")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {a.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
