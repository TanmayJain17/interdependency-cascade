#!/usr/bin/env python3
"""
scripts/week29_numbers_table_v1.py — Week 29 numbers sheet from the eight JSONs written by
hpc/audit_gnn_numbers_v1.sbatch (scripts/week29_numbers_audit_v1.py):

  <dir>/independent_<cell>.json   <dir>/hazard_<cell>.json      cell in the 2x2

Prints (1) the sheet for the hazard head, per cell, validation and each unseen storm;
(2) model against the three baselines; (3) speed; (4) the verdicts below.
Writes analysis/gnn/week29_numbers_v1.csv (one row per head x cell x split x predictor).

PRE-REGISTRATION (written 6 Oct 2026, before the run). "Non-seed" = sites the flood did not
knock out directly; seeds are an input to every predictor.
  P0  gates: every JSON has an empty failure list and is a full (not subset) run.
  P1  validation (storms seen in training): the model's non-seed BCE is below all three
      baselines in all 8 checkpoints.
  P2  unseen storms: the model's non-seed BCE is below the nearest-storm lookup on at least
      3 of 5 storms in each of the 4 hazard cells. Expected to be the closest call: the lookup
      should be strong where an unseen storm sits between two near training storms (914_6, 258_9).
  P3  validation count error at 96 h (WAPE, non-seed, hazard head): at most 0.10 in the static
      cells and 0.20 in the dynamic cells.
  P4  validation calibration (ECE, non-seed, hazard head): at most 0.02 at every horizon.
  P5  speed: median forward pass at most 50 ms.
  No prediction for the failure-window numbers: this is their first measurement.
  Named failure mode: if the lookup beats the model on most unseen storms, the model has learned
  little that transfers beyond storm size, and the new-maps experiment becomes the test of
  whether more storms fix that. It is reported as such, not hidden.
  The sheet may claim: how the existing models score on these measures, one seed each.
  It may not claim anything about models trained on more storms.
"""
import argparse
import csv
import json
from pathlib import Path

CELLS = ["static_mask", "static_timing", "dynamic_mask", "dynamic_timing"]
HEADS = ["independent", "hazard"]
PRED = ["model", "no_cascade", "site_frequency", "nearest_storm"]
SHORT = {"syn_ts_914_6_1p2955": "914_6", "syn_ts_258_9_3p0022": "258_9", "syn_ts_808_27_3p7885": "808_27",
         "geoclaw_2050": "gc_2050", "syn_ts_605_5_3p7563": "605_5"}


def f(x, n=3):
    return "  nan" if x != x else f"{x:.{n}f}"


def row_for(head, cell, split, pred, r, k):
    c, w = r["count_non_seed"], r["windows"]
    d = {"head": head, "cell": cell, "split": split, "predictor": pred, "n_runs": r["n_runs"],
         "loss_type_avg": round(r["loss"], 5), "bce_non_seed": round(r["bce_non_seed"], 5)}
    for j in range(k):
        d[f"bce_non_seed_t{j}"] = round(r["bce_non_seed_per_t"][j], 5)
        d[f"count_true_t{j}"] = round(c["mean_true"][j], 2); d[f"count_pred_t{j}"] = round(c["mean_pred"][j], 2)
        d[f"count_wape_t{j}"] = round(c["wape"][j], 4); d[f"count_bias_t{j}"] = round(c["bias_frac"][j], 4)
        d[f"count_mae_t{j}"] = round(c["mae_sites"][j], 2); d[f"count_corr_t{j}"] = round(c["corr_across_runs"][j], 4)
        d[f"precision_t{j}"] = round(r["flag_at_0p5"]["precision"][j], 4); d[f"recall_t{j}"] = round(r["flag_at_0p5"]["recall"][j], 4)
        d[f"ap_t{j}"] = round(r["ap_non_seed"][j], 4); d[f"auc_t{j}"] = round(r["auc_non_seed"][j], 4)
        d[f"ece_t{j}"] = round(r["calibration"]["ece"][j], 5)
    d.update({"window_given_failure_exact": round(w["given_failure_exact_window"], 4),
              "window_given_failure_within_one": round(w["given_failure_within_one_window"], 4),
              "window_exact_incl_never": round(w["failing_exact_window"], 4),
              "failing_caught_by_last_horizon": round(w["failing_detected_by_last_horizon"], 4),
              "never_failed_correct": round(w["never_failed_correct"], 4)})
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/hpc_results_aug2026/gnn_numbers_v1")
    ap.add_argument("--out", default="analysis/gnn/week29_numbers_v1.csv")
    a = ap.parse_args()

    J, rows = {}, []
    v = {f"P{i}": True for i in range(6)}
    notes = {k_: [] for k_ in v}
    for h in HEADS:
        for c in CELLS:
            j = json.load(open(Path(a.dir) / f"{h}_{c}.json"))
            J[(h, c)] = j
            if j["failures"] or j.get("subset_run"):
                v["P0"] = False; notes["P0"].append(f"{h}_{c}")
            k = len(j["timesteps"])
            splits = [("val", j["val"])] + [(SHORT.get(s, s), r) for s, r in j["test_per_scenario"].items()]
            for name, res in splits:
                for p in PRED:
                    rows.append(row_for(h, c, name, p, res[p], k))
    k = len(J[("hazard", CELLS[0])]["timesteps"])
    ts = J[("hazard", CELLS[0])]["timesteps"]
    last = k - 1

    for c in CELLS:
        j = J[("hazard", c)]
        splits = [("val", j["val"])] + sorted(((SHORT.get(s, s), r) for s, r in j["test_per_scenario"].items()),
                                               key=lambda x: x[1]["model"]["loss"])
        print(f"\n=== hazard head, {c} ===  (non-seed sites; horizons {ts} h)")
        print(f"{'split':8s} {'runs':>5s} {'fail@' + str(ts[last]):>8s} | {'count error (WAPE) per horizon':31s} {'bias@' + str(ts[last]):>8s} | "
              f"{'prec':>5s} {'recall':>6s} {'AP':>6s} | {'window given failure':>22s} | {'ECE':>6s} | {'loss':>7s}")
        for name, res in splits:
            m = res["model"]; cn, w = m["count_non_seed"], m["windows"]
            print(f"{name:8s} {m['n_runs']:5d} {cn['mean_true'][last]:8.0f} | " + " ".join(f(x) for x in cn["wape"]) + "        "
                  f"{cn['bias_frac'][last]:+8.3f} | {f(m['flag_at_0p5']['precision'][last]):>5s} {f(m['flag_at_0p5']['recall'][last]):>6s} "
                  f"{f(m['ap_non_seed'][last]):>6s} | exact {f(w['given_failure_exact_window'])} +-1 {f(w['given_failure_within_one_window'])} "
                  f"| {f(m['calibration']['ece'][last], 4):>6s} | {m['loss']:.5f}")

    print("\n=== model against baselines: non-seed BCE (lower is better), hazard head ===")
    print(f"{'cell':15s} {'split':8s} {'model':>8s} {'no_casc':>8s} {'site_frq':>8s} {'nearest':>8s}  best        nearest storm used")
    for c in CELLS:
        j = J[("hazard", c)]
        wins = 0
        for name, res in [("val", j["val"])] + [(SHORT.get(s, s), r) for s, r in j["test_per_scenario"].items()]:
            b = {p: res[p]["bce_non_seed"] for p in PRED}
            best = min(b, key=b.get)
            used = max(res["nearest_storm_chosen"], key=res["nearest_storm_chosen"].get) if res["nearest_storm_chosen"] else "-"
            print(f"{c:15s} {name:8s} {b['model']:8.4f} {b['no_cascade']:8.4f} {b['site_frequency']:8.4f} {b['nearest_storm']:8.4f}  "
                  f"{best:11s} {SHORT.get(used, used.replace('syn_ts_', ''))}")
            if name != "val" and b["model"] < b["nearest_storm"]:
                wins += 1
        if wins < 3:
            v["P2"] = False; notes["P2"].append(f"{c}: model wins {wins}/5")
    for h in HEADS:
        for c in CELLS:
            res = J[(h, c)]["val"]
            if not all(res["model"]["bce_non_seed"] < res[p]["bce_non_seed"] for p in PRED[1:]):
                v["P1"] = False; notes["P1"].append(f"{h}_{c}")

    print(f"\n=== count error at {ts[last]} h (WAPE, non-seed), hazard head: model | nearest-storm lookup ===")
    for c in CELLS:
        j = J[("hazard", c)]
        parts = []
        for name, res in [("val", j["val"])] + [(SHORT.get(s, s), r) for s, r in j["test_per_scenario"].items()]:
            parts.append(f"{name} {res['model']['count_non_seed']['wape'][last]:.3f}|{res['nearest_storm']['count_non_seed']['wape'][last]:.3f}")
        print(f"{c:15s} " + "   ".join(parts))
        lim = 0.10 if c.startswith("static") else 0.20
        w96 = j["val"]["model"]["count_non_seed"]["wape"][last]
        if not w96 <= lim:
            v["P3"] = False; notes["P3"].append(f"{c}: {w96:.3f} > {lim}")
        ece = j["val"]["model"]["calibration"]["ece"]
        if not max(ece) <= 0.02:
            v["P4"] = False; notes["P4"].append(f"{c}: max ECE {max(ece):.4f}")

    print("\n=== independent head against hazard head, validation ===")
    for c in CELLS:
        i_, h_ = J[("independent", c)]["val"]["model"], J[("hazard", c)]["val"]["model"]
        print(f"{c:15s} count WAPE@{ts[last]} {i_['count_non_seed']['wape'][last]:.3f} | {h_['count_non_seed']['wape'][last]:.3f}   "
              f"window given failure exact {i_['windows']['given_failure_exact_window']:.3f} | {h_['windows']['given_failure_exact_window']:.3f}   "
              f"max ECE {max(i_['calibration']['ece']):.4f} | {max(h_['calibration']['ece']):.4f}")

    sp = [J[(h, c)]["speed"]["forward_ms_median"] for h in HEADS for c in CELLS]
    print(f"\n=== speed === model forward pass, median over checkpoints: {sorted(sp)[len(sp) // 2]:.2f} ms "
          f"(range {min(sp):.2f}-{max(sp):.2f}) on {J[('hazard', CELLS[0])]['speed']['device']}")
    if not max(sp) <= 50:
        v["P5"] = False; notes["P5"].append(f"max {max(sp):.1f} ms")

    names = {"P0": "gates (reproduction, internal rules, full runs)", "P1": "validation: model beats all three baselines (8 checkpoints)",
             "P2": "unseen storms: model beats the lookup on >= 3 of 5 (4 hazard cells)",
             "P3": f"validation count WAPE@{ts[last]} <= 0.10 static, 0.20 dynamic", "P4": "validation ECE <= 0.02 at every horizon",
             "P5": "forward pass <= 50 ms"}
    print("\nPRE-REGISTERED VERDICTS")
    for key in sorted(v):
        print(f"  {key} {names[key]:62s} {'PASS' if v[key] else 'MISS'}" + ("" if v[key] else "   " + "; ".join(notes[key][:6])))
    print("GATES (P0):", "PASS" if v["P0"] else "FAIL")
    print("PREDICTIONS (P1-P5):", "ALL PASS" if all(v[x] for x in ("P1", "P2", "P3", "P4", "P5")) else "SOME MISSED - record them")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("wrote", a.out, f"({len(rows)} rows)")


if __name__ == "__main__":
    main()
