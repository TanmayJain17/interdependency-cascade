#!/usr/bin/env python3
"""
scripts/week29_morestorms_partial_v1.py — interim reading of the more-storms experiment from the two arms
that finished: old17 (the 17 old GISSR storms) and mix57_same (17 old + 40 new storms, same number of
training runs and steps as old17). Reads the JSONs written by hpc/audit_gnn_morestorms_v1.sbatch with
HOLD=all13 (scripts/week29_numbers_audit_v2.py), one per checkpoint:

  <dir>/<arm>_<variant>_<cell>_s<seed>.json   arm old17 | mix57_same;  variant plain | ctx (storm-context input)

WHY THIS FILE EXISTS. The third arm, mix57_full (57 storms, all 45,600 runs), was cancelled by the cluster
on 7 Oct 2026 (Torch job 19358745, tasks 8, 11, 32, 35, all four at 18:01, about 2 h 15 min into a run of
about 4 h). scripts/week29_morestorms_table_v1.py needs all three arms, four cells and two seeds and gives
the pre-registered verdicts M1-M7 and L1. None of those verdicts can be given from what exists, and this
script gives none. It only prints what the finished checkpoints score.

WHAT IT MAY CLAIM: how these checkpoints score on the 13 held-out storms.
WHAT IT MAY NOT CLAIM: a verdict on M1-M7 or L1; anything about seed-to-seed spread of mix57_same when it
has one seed; anything about the cells that were not trained; anything about more runs per storm.
The only yardstick for seed noise printed here is the difference between the existing old17 plain
checkpoint (seed 0, PyTorch seed unrecorded) and its retrain (seed 1).

Vocabulary. "Non-seed BCE": log loss on sites the flood did not knock out directly, mean of the four
horizons, lower is better. "Count bias": (predicted - simulated) / simulated failures among non-seed sites
by the last horizon (96 h), mean over the storm's runs. Storm groups by peak water level, as in the
pre-registration: ABOVE = the 5 held-out storms above every old training storm (old maximum 3.64 m),
INSIDE = the 7 other GISSR held-out storms, OTHER = geoclaw_2050.
"""
import argparse
import csv
import json
from pathlib import Path

CELLS = ["static_mask", "static_timing", "dynamic_mask", "dynamic_timing"]
ARMS = ["old17", "mix57_same"]
VARIANTS = ["plain", "ctx"]
OLD_TRAIN_MAX_M = 3.64


def mean(x):
    x = list(x)
    return sum(x) / len(x) if x else float("nan")


def short(s):
    return s.replace("syn_ts_", "").replace("syn2_", "new ").replace("geoclaw_", "gc_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/hpc_results_aug2026/gnn_morestorms_numbers_v1")
    ap.add_argument("--levels", default="data/flood/timing/node_timing_jesse22_plus_gissr48_v1_summary.csv")
    ap.add_argument("--manifest", default="data/flood/gissr48_manifest_v1.csv")
    ap.add_argument("--out", default="analysis/gnn/week29_morestorms_partial_v1.csv")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--runs_per_storm", type=int, default=1000)
    a = ap.parse_args()

    level = {r["scenario"]: float(r["peak_wl_m"]) for r in csv.DictReader(open(a.levels))}
    lead = {r["tag"]: float(r["lead_median_h"]) for r in csv.DictReader(open(a.manifest))}

    J, bad = {}, []
    for c in CELLS:
        for arm in ARMS:
            for var in VARIANTS:
                for s in (0, 1, 2):
                    p = Path(a.dir) / f"{arm}_{var}_{c}_s{s}.json"
                    if not p.exists():
                        continue
                    j = json.load(open(p))
                    ta = j.get("ckpt_train_args") or {}
                    n = 17 if arm == "old17" else 57
                    want = {"scenario_set": "w29_old" if arm == "old17" else "w29_all", "n_storms": n, "n_split": int(0.8 * n * a.runs_per_storm),
                            "train_subset": int(0.8 * 17 * a.runs_per_storm) if arm == "mix57_same" else None, "torch_seed": s if s > 0 else None,
                            "timing": int(c.endswith("timing")), "context": int(var == "ctx"), "epochs": a.epochs, "head": "hazard"}
                    got = {"scenario_set": j.get("scenario_set"), "n_storms": len(j.get("training_storms", [])), "n_split": j.get("n_train_runs_full_split"),
                           "train_subset": ta.get("train_subset"), "torch_seed": ta.get("torch_seed"), "timing": j.get("timing_features"),
                           "context": j.get("storm_context"), "epochs": ta.get("epochs"), "head": j.get("head")}
                    tps = j.get("test_per_scenario", {})
                    problems = [f"{k}={got[k]} (want {want[k]})" for k in want if got[k] != want[k]]
                    if j["failures"] or j.get("subset_run"):
                        problems.append("failed gates or smoke subset")
                    if len(tps) != 13 or any(r["model"]["n_runs"] != a.runs_per_storm for r in tps.values()):
                        problems.append(f"not 13 held-out storms of {a.runs_per_storm} runs")
                    if problems:
                        bad.append(f"{p.name}: " + ", ".join(problems))
                    else:
                        J[(arm, var, c, s)] = j
    if bad:
        for x in bad:
            print("   bad:", x)
        raise SystemExit("IDENTITY CHECK: FAIL — a JSON does not carry the identity its file name claims; nothing is printed")
    if not J:
        raise SystemExit(f"no JSON found in {a.dir}")
    if len({tuple(sorted(j["test_per_scenario"])) for j in J.values()}) != 1:
        raise SystemExit("the JSONs do not list the same held-out storms")
    print(f"IDENTITY CHECK: PASS — {len(J)} JSONs, each with empty failure list, full run, right storm set, training size, seed, inputs, epochs")
    print("PARTIAL READING — two of three arms. NO PRE-REGISTERED VERDICT (M1-M7, L1) IS GIVEN.")

    storms = sorted(next(iter(J.values()))["test_per_scenario"], key=lambda s: (s.startswith("geoclaw"), level[s]))
    gissr = [s for s in storms if not s.startswith("geoclaw")]
    above = [s for s in gissr if level[s] > OLD_TRAIN_MAX_M + 1e-9]
    inside = [s for s in gissr if s not in above]
    assert (len(storms), len(above), len(inside)) == (13, 5, 7), "storm groups changed"
    grp = lambda s: "ABOVE" if s in above else "INSIDE" if s in inside else "OTHER"
    k_last = len(next(iter(J.values()))["timesteps"]) - 1

    def seeds_of(arm, var, c):
        return [s for s in (1, 2) if (arm, var, c, s) in J]

    def bce(arm, var, c, storm, pred="model"):
        return mean(J[(arm, var, c, s)]["test_per_scenario"][storm][pred]["bce_non_seed"] for s in seeds_of(arm, var, c))

    def cnt(arm, var, c, storm, key, pred="model"):
        return mean(J[(arm, var, c, s)]["test_per_scenario"][storm][pred]["count_non_seed"][key][k_last] for s in seeds_of(arm, var, c))

    rows = []
    for (arm, var, c, s), j in sorted(J.items()):
        for storm in storms:
            m = j["test_per_scenario"][storm]["model"]; nb = j["test_per_scenario"][storm]["nearest_storm"]
            rows.append({"arm": arm, "variant": var, "cell": c, "torch_seed": s, "storm": storm, "level_m": level[storm], "lead_median_h": lead.get(storm, ""),
                         "group": grp(storm), "n_runs": m["n_runs"], "bce_non_seed": round(m["bce_non_seed"], 5),
                         **{f"bce_non_seed_t{t}": round(m["bce_non_seed_per_t"][t], 5) for t in range(k_last + 1)},
                         "count_wape_last": round(m["count_non_seed"]["wape"][k_last], 4), "count_bias_last": round(m["count_non_seed"]["bias_frac"][k_last], 4),
                         "window_given_failure_exact": round(m["windows"]["given_failure_exact_window"], 4),
                         "lookup_bce_non_seed": round(nb["bce_non_seed"], 5), "lookup_count_wape_last": round(nb["count_non_seed"]["wape"][k_last], 4),
                         "saved_val_loss": j.get("saved_val_loss"), "n_training_storms": len(j["training_storms"])})

    cells = [c for c in CELLS if all(seeds_of(arm, var, c) for arm in ARMS for var in VARIANTS)]
    print(f"cells with both arms and both variants: {cells}; seeds used per (arm, variant): "
          f"{ {c: {f'{arm}_{var}': seeds_of(arm, var, c) for arm in ARMS for var in VARIANTS} for c in cells} }")
    cols = [("old17", "plain"), ("mix57_same", "plain"), ("old17", "ctx"), ("mix57_same", "ctx")]
    names = ["17", "57", "17+size", "57+size"]
    for c in cells:
        print(f"\n=== {c}: non-seed BCE on the 13 held-out storms (lower is better) ===")
        print(f"{'storm':22s} {'level':>5s} {'lead':>5s} {'grp':>6s} | {'17 (s0)':>8s} " + " ".join(f"{n:>8s}" for n in names) + f" | {'lookup17':>8s} {'lookup57':>8s}")
        for st in storms:
            s0 = J.get(("old17", "plain", c, 0))
            s0v = f"{s0['test_per_scenario'][st]['model']['bce_non_seed']:.4f}" if s0 else "-"
            print(f"{short(st):22s} {level[st]:5.2f} {str(lead.get(st, '')):>5s} {grp(st):>6s} | {s0v:>8s} " + " ".join(f"{bce(arm, var, c, st):8.4f}" for arm, var in cols)
                  + f" | {bce('old17', 'plain', c, st, 'nearest_storm'):8.4f} {bce('mix57_same', 'plain', c, st, 'nearest_storm'):8.4f}")
        print(f"\n--- {c}: predicted / simulated failures among non-seed sites by 96 h (1.00 = correct), and count error (WAPE) ---")
        print(f"{'storm':22s} {'level':>5s} | " + " ".join(f"{n:>8s}" for n in names) + " | WAPE: " + " ".join(f"{n:>8s}" for n in names) + f" {'lookup57':>8s}")
        for st in storms:
            print(f"{short(st):22s} {level[st]:5.2f} | " + " ".join(f"{1 + cnt(arm, var, c, st, 'bias_frac'):8.2f}" for arm, var in cols) + " |       "
                  + " ".join(f"{cnt(arm, var, c, st, 'wape'):8.3f}" for arm, var in cols) + f" {cnt('mix57_same', 'plain', c, st, 'wape', 'nearest_storm'):8.3f}")

    print("\n=== summary (no verdict) ===")
    for c in cells:
        print(f"{c}")
        s0 = J.get(("old17", "plain", c, 0))
        if s0 and 1 in seeds_of("old17", "plain", c):
            d = [abs(s0["test_per_scenario"][st]["model"]["bce_non_seed"] - J[("old17", "plain", c, 1)]["test_per_scenario"][st]["model"]["bce_non_seed"]) for st in storms]
            print(f"  seed yardstick: old17 plain, existing checkpoint against its retrain: largest BCE difference on a storm {max(d):.4f}, median {sorted(d)[len(d) // 2]:.4f}")
        for var in VARIANTS:
            for name, g in (("ABOVE", above), ("INSIDE", inside), ("OTHER", [s for s in storms if s.startswith("geoclaw")])):
                w = sum(bce("mix57_same", var, c, s) < bce("old17", var, c, s) for s in g)
                r = [(bce("mix57_same", var, c, s) - bce("old17", var, c, s)) / bce("old17", var, c, s) for s in g]
                print(f"  {var:5s} {name:6s}: mix57_same below old17 on {w}/{len(g)} storms; relative change mean {mean(r):+.1%}, range {min(r):+.1%} to {max(r):+.1%}")
        for (arm, var), n in zip(cols, names):
            lk = sum(bce(arm, var, c, s) < bce(arm, var, c, s, "nearest_storm") for s in storms)
            lkg = sum(bce(arm, var, c, s) < bce(arm, var, c, s, "nearest_storm") for s in gissr)
            b = [cnt(arm, var, c, s, "bias_frac") for s in gissr]
            print(f"  {n:8s}: mean BCE over 13 storms {mean(bce(arm, var, c, s) for s in storms):.4f} | beats its own lookup on {lk}/13 ({lkg}/12 GISSR) | "
                  f"count bias on 12 GISSR storms: {min(b):+.2f} to {max(b):+.2f}, storms within +-10%: {sum(abs(x) <= 0.10 for x in b)}/12")
        x = sum(bce("old17", "ctx", c, s) < bce("mix57_same", "plain", c, s) for s in storms)
        print(f"  storm-size input alone (17+size) against more storms alone (57): 17+size has the lower BCE on {x}/13 storms")
    print("\nPARTIAL READING ONLY — the pre-registered verdicts need mix57_full, four cells and two seeds (scripts/week29_morestorms_table_v1.py).")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("wrote", a.out, f"({len(rows)} rows)")


if __name__ == "__main__":
    main()
