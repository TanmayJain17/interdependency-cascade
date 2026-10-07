#!/usr/bin/env python3
"""
scripts/week29_morestorms_table_v1.py — what do 40 further GISSR storms buy? Reads the JSONs written by
hpc/audit_gnn_morestorms_v1.sbatch (scripts/week29_numbers_audit_v2.py), one per checkpoint:

  <dir>/<arm>_<variant>_<cell>_s<seed>.json   arm  in old17 | mix57_same | mix57_full
                                              variant plain (default) or ctx (storm-context input, --variant ctx)
                                              cell in the 2x2;  seed 1, 2 (and 0 = the existing gnn_hazard_v1
                                              checkpoint, old17 plain only, PyTorch seed unrecorded)

Prints, per cell, the 13 held-out storms by water level with the non-seed BCE of each arm, the change
from old17 to mix57_full, the nearest-storm lookup built from 17 and from 57 storms, and the count
error at the last horizon; then the verdicts below. Writes one CSV row per arm x cell x seed x storm.

Vocabulary. "Non-seed BCE": log loss on sites the flood did not knock out directly, mean of the four
horizons, lower is better. Arm value on a storm = mean over PyTorch seeds 1 and 2. "Clear" win of X over
Y on a storm = both seeds of X are below both seeds of Y. Storm groups, by peak water level:
  ABOVE   the 5 held-out storms above every old training storm (old maximum 3.64 m):
          808_27 (3.745), new 3.800, new 4.130, 605_5 (4.458), new 4.722 (also above the new range, 4.572)
  INSIDE  the 7 held-out GISSR storms at or below 3.64 m (914_6, 258_9 and five new ones)
  OTHER   geoclaw_2050, a map from a different flood model

PRE-REGISTRATION (written 6 Oct 2026, before the label campaign and before any training of these arms).
Seen before writing: the old models' held-out losses (worst on 605_5 and 808_27), and a 20-run preview of
the simulator on the 48 new storms (static labels rise smoothly with level; under arrival labels a slow
storm gives a larger cascade than a fast one of similar size at every horizon, e.g. about 3,560 against
3,010 failed sites by 96 h near 4.3-4.5 m). No model output on any new storm was seen.
  G0  gates: all 24 new checkpoints present; every JSON has an empty failure list, is not a smoke subset,
      and carries the right identity for its file name: storm set, 17 or 57 training storms, 13,600 or
      45,600 runs in the full training split, training subset (none / 13,600 / none), PyTorch seed, timing
      inputs on or off, the same 13 held-out storms with 1,000 runs each. If G0 fails no other verdict is given.
  S1  the retrained old17 checkpoints reproduce the existing ones within seed noise: in each cell both
      seeds' saved val loss lies within 0.005 of the existing checkpoint's (0.1080, 0.0980, 0.1880, 0.1651).
  M1  above the old range, more storms help: in each of the 4 cells mix57_full has a lower BCE than old17
      on at least 4 of the 5 ABOVE storms, and the mean relative reduction over the 5 is at least 10%.
  M2  it is the variety, not the number of runs: mix57_same (same number of training runs and steps as
      old17) also beats old17 on at least 4 of the 5 ABOVE storms in each cell.
  M3  inside the old range, static labels: little to gain. In static_mask and static_timing the mean
      relative change of mix57_full against old17 over the 7 INSIDE storms is within +-10%.
  M4  inside the old range, dynamic labels: a gain where timing is new. In dynamic_timing mix57_full beats
      old17 on at least 4 of the 5 NEW INSIDE storms (their leads, 4-11 h, are unlike the old storms of
      that size, which are all fast).
  M5  timing inputs matter once size and timing are decoupled: in the mix57_full arm dynamic_timing beats
      dynamic_mask on at least 10 of the 12 GISSR held-out storms, and its 6-hour BCE is at least 10% lower
      (mean over those 12 storms of the per-storm relative difference, as in M1 and M3).
  M6  a different flood model does not benefit: on geoclaw_2050 there is no clear win in either direction
      between old17 and mix57_full in at least 3 of the 4 cells.
  M7  more runs per storm add little: mix57_full's BCE averaged over the 13 storms is within 10% of
      mix57_same's average in every cell (ratio of the two 13-storm means).
  No prediction for the lookup table. Named failure mode: if the nearest-storm lookup built from 57 storms
  matches the GNN in the static cells, then for GISSR-type maps a table indexed by storm size is enough,
  and the GNN's case rests on timing, on maps from other flood models and on speed. Reported as such.
  ADDED 6 Oct 2026, evening, after the Week 29 numbers sheet (job 19301158) and before any training of
  these arms. The sheet showed the lookup beating the 17-storm model on validation in all 8 checkpoints
  and on 3 of 5 unseen storms, so the named failure mode becomes a prediction:
  L1  with 57 storms the lookup stays ahead on GISSR-type storms: in every cell the lookup built from the
      mix57_full training runs has a lower BCE than the mix57_full model on at least 9 of the 12 GISSR
      held-out storms, and the model beats it on geoclaw_2050 in every cell.
  M1-M7 and L1 are judged on the plain variant. With --variant ctx the same tables are printed for the
  storm-context variant and no verdict is given (its predictions are in scripts/week29_context_table_v1.py).
  The table may claim: how these three arms score on these 13 storms, two seeds each.
  It may not claim: anything about flood maps with new spatial patterns (GISSR maps are nested by level),
  or about seed-to-seed spread beyond the two or three replicates per arm.
"""
import argparse
import csv
import json
from pathlib import Path

CELLS = ["static_mask", "static_timing", "dynamic_mask", "dynamic_timing"]
ARMS = ["old17", "mix57_same", "mix57_full"]
OLD_VAL = {"static_mask": 0.1080, "static_timing": 0.0980, "dynamic_mask": 0.1880, "dynamic_timing": 0.1651}
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
    ap.add_argument("--out", default="analysis/gnn/week29_morestorms_v1.csv")
    ap.add_argument("--variant", choices=["plain", "ctx"], default="plain", help="plain (pre-registered verdicts) or ctx (tables only)")
    ap.add_argument("--partial", action="store_true",
                    help="if the set is incomplete, print the per-storm tables for what is present and clean, without verdicts")
    ap.add_argument("--epochs", type=int, default=6, help="epochs every checkpoint must have been trained for (identity check of G0)")
    ap.add_argument("--runs_per_storm", type=int, default=1000, help="Monte Carlo runs per storm in the label campaigns (identity check of G0)")
    a = ap.parse_args()

    level = {r["scenario"]: float(r["peak_wl_m"]) for r in csv.DictReader(open(a.levels))}
    lead = {r["tag"]: float(r["lead_median_h"]) for r in csv.DictReader(open(a.manifest))}
    new_test = [r["tag"] for r in csv.DictReader(open(a.manifest)) if r["role"] == "test"]

    J, missing, bad = {}, [], []
    for c in CELLS:
        for arm in ARMS:
            for s in (0, 1, 2):
                if s == 0 and (arm != "old17" or a.variant != "plain"):
                    continue
                p = Path(a.dir) / f"{arm}_{a.variant}_{c}_s{s}.json"
                if not p.exists():
                    if s > 0:
                        missing.append(p.name)
                    continue
                j = json.load(open(p)); J[(arm, c, s)] = j
                ta = j.get("ckpt_train_args") or {}
                want = {"scenario_set": "w29_old" if arm == "old17" else "w29_all", "n_storms": 17 if arm == "old17" else 57,
                        "n_split": int(0.8 * (17 if arm == "old17" else 57) * a.runs_per_storm),
                        "train_subset": int(0.8 * 17 * a.runs_per_storm) if arm == "mix57_same" else None,
                        "torch_seed": s if s > 0 else None, "timing": int(c.endswith("timing")), "context": int(a.variant == "ctx"),
                        "epochs": a.epochs, "head": "hazard"}
                got = {"scenario_set": j.get("scenario_set"), "n_storms": len(j.get("training_storms", [])), "n_split": j.get("n_train_runs_full_split"),
                       "train_subset": ta.get("train_subset"), "torch_seed": ta.get("torch_seed"), "timing": j.get("timing_features"),
                       "context": j.get("storm_context"), "epochs": ta.get("epochs"), "head": j.get("head")}
                tps = j.get("test_per_scenario", {})
                problems = [f"{k}={got[k]} (want {want[k]})" for k in want if got[k] != want[k]]
                if j["failures"] or j.get("subset_run"):
                    problems.append("failed gates or smoke subset")
                if len(tps) != 13 or any(r["model"]["n_runs"] != a.runs_per_storm for r in tps.values()):
                    problems.append(f"not 13 held-out storms of {a.runs_per_storm} runs")
                if j.get("saved_val_loss") is None:
                    problems.append("no saved val loss")
                if problems:
                    bad.append(f"{p.name}: " + ", ".join(problems)); del J[(arm, c, s)]
    if not J:
        for x in bad[:30]:
            print("   bad:", x)
        raise SystemExit(f"no usable JSON in {a.dir} ({len(missing)} missing, {len(bad)} failed the identity check)")
    storm_sets = {tuple(sorted(j["test_per_scenario"])) for j in J.values()}
    if len(storm_sets) != 1:
        bad.append("the JSONs do not list the same held-out storms")
    g0 = not missing and not bad
    cells = list(CELLS)
    if not g0:
        print("G0 gates and completeness: FAIL")
        for x in missing[:30]:
            print("   missing:", x)
        for x in bad[:30]:
            print("   bad:", x)
        cells = [c for c in CELLS if all(any((arm, c, s) in J for s in (1, 2)) for arm in ARMS)]
        if not a.partial or len(storm_sets) != 1 or not cells:
            print("GATES (G0): FAIL — no table and no verdict is given until every checkpoint is present and clean (--partial prints what exists)")
            raise SystemExit(1)
        print(f"PARTIAL TABLE — cells with all three arms: {cells}; arm values use the seeds that exist "
              f"({ {c: {arm: [s for s in (1, 2) if (arm, c, s) in J] for arm in ARMS} for c in cells} }). NO VERDICT IS GIVEN.")
    storms = sorted(next(iter(J.values()))["test_per_scenario"], key=lambda s: (s.startswith("geoclaw"), level[s]))
    gissr = [s for s in storms if not s.startswith("geoclaw")]
    above = [s for s in gissr if level[s] > OLD_TRAIN_MAX_M + 1e-9]
    inside = [s for s in gissr if s not in above]
    other = [s for s in storms if s.startswith("geoclaw")]
    new_inside = [s for s in inside if s in new_test]
    k_last = len(next(iter(J.values()))["timesteps"]) - 1
    sizes = (len(storms), len(gissr), len(above), len(inside), len(new_inside), len(other))
    assert sizes == (13, 12, 5, 7, 5, 1), f"storm groups changed: {sizes} (pre-registered: 13 storms = 5 ABOVE + 7 INSIDE of which 5 new + 1 OTHER)"

    def val(arm, c, s, storm, key="bce_non_seed", pred="model", t=None):
        j = J.get((arm, c, s))
        if j is None:
            return None
        v = j["test_per_scenario"][storm][pred][key]
        return v if t is None else v[t]

    def seeds(arm, c, storm, **kw):
        return [v for v in (val(arm, c, s, storm, **kw) for s in (1, 2)) if v is not None]

    def arm_mean(arm, c, storm, **kw):
        return mean(seeds(arm, c, storm, **kw))

    def wape(arm, c, storm):
        return mean(J[(arm, c, s)]["test_per_scenario"][storm]["model"]["count_non_seed"]["wape"][k_last] for s in (1, 2) if (arm, c, s) in J)

    def clear(x_arm, y_arm, c, storm):
        x, y = seeds(x_arm, c, storm), seeds(y_arm, c, storm)
        return bool(len(x) == 2 and len(y) == 2 and max(x) < min(y))      # needs both seeds of both arms

    rows = []
    for (arm, c, s), j in sorted(J.items()):
        for storm in storms:
            m = j["test_per_scenario"][storm]["model"]; nb = j["test_per_scenario"][storm]["nearest_storm"]
            rows.append({"arm": arm, "variant": a.variant, "cell": c, "torch_seed": s, "storm": storm, "level_m": level[storm], "lead_median_h": lead.get(storm, ""),
                         "group": "ABOVE" if storm in above else "INSIDE" if storm in inside else "OTHER", "n_runs": m["n_runs"],
                         "bce_non_seed": round(m["bce_non_seed"], 5), **{f"bce_non_seed_t{t}": round(m["bce_non_seed_per_t"][t], 5) for t in range(k_last + 1)},
                         "loss_type_avg": round(m["loss"], 5), "count_wape_last": round(m["count_non_seed"]["wape"][k_last], 4),
                         "count_bias_last": round(m["count_non_seed"]["bias_frac"][k_last], 4),
                         "window_given_failure_exact": round(m["windows"]["given_failure_exact_window"], 4),
                         "lookup_bce_non_seed": round(nb["bce_non_seed"], 5), "lookup_count_wape_last": round(nb["count_non_seed"]["wape"][k_last], 4),
                         "saved_val_loss": j.get("saved_val_loss"), "n_training_storms": len(j["training_storms"])})

    for c in cells:
        print(f"\n=== {c}: non-seed BCE on the 13 held-out storms (mean of seeds 1, 2; lower is better) ===")
        print(f"{'storm':22s} {'level':>5s} {'lead':>5s} {'grp':>6s} | {'old17':>7s} {'(s0)':>7s} {'57same':>7s} {'57full':>7s} {'change':>7s} {'clear':>5s} | "
              f"{'lookup17':>8s} {'lookup57':>8s} | count error at last horizon: old17 / 57full")
        for storm in storms:
            o, sm, fu = arm_mean("old17", c, storm), arm_mean("mix57_same", c, storm), arm_mean("mix57_full", c, storm)
            s0 = val("old17", c, 0, storm)
            ch = (fu - o) / o if o == o and fu == fu else float("nan")
            cl = "yes" if clear("mix57_full", "old17", c, storm) else "worse" if clear("old17", "mix57_full", c, storm) else "-"
            g = "ABOVE" if storm in above else "INSIDE" if storm in inside else "OTHER"
            print(f"{short(storm):22s} {level[storm]:5.2f} {str(lead.get(storm, '')):>5s} {g:>6s} | {o:7.4f} {(f'{s0:.4f}' if s0 is not None else '-'):>7s} {sm:7.4f} {fu:7.4f} "
                  f"{ch:+7.1%} {cl:>5s} | {arm_mean('old17', c, storm, pred='nearest_storm'):8.4f} {arm_mean('mix57_full', c, storm, pred='nearest_storm'):8.4f} | "
                  f"{wape('old17', c, storm):.3f} / {wape('mix57_full', c, storm):.3f}")

    print("\n=== the lookup against the model, 12 GISSR held-out storms (storms where the lookup has the lower BCE) | geoclaw_2050 winner ===")
    l1 = {}
    for c in cells:
        lw = sum(arm_mean("mix57_full", c, s, pred="nearest_storm") < arm_mean("mix57_full", c, s) for s in gissr)
        gw = bool(other) and arm_mean("mix57_full", c, other[0]) < arm_mean("mix57_full", c, other[0], pred="nearest_storm")
        l1[c] = (lw, gw)
        print(f"{c:15s} mix57_full: lookup ahead on {lw}/{len(gissr)}   geoclaw_2050: {'model' if gw else 'lookup'}")
    if not g0:
        print("\nGATES (G0): FAIL — partial table only, no verdict")
        raise SystemExit(1)
    if a.variant != "plain":
        print(f"\nGATES (G0): PASS — variant {a.variant}: tables only, no pre-registered verdict for this variant")
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print("wrote", a.out, f"({len(rows)} rows)")
        return
    v, notes = {}, {}

    def verdict(key, ok, note=""):
        v[key] = bool(ok); notes[key] = note

    verdict("G0", g0, "24 new checkpoints present and clean" + (f"; {sum(1 for kk in J if kk[2] == 0)} existing old17 checkpoints (s0) also read" if any(kk[2] == 0 for kk in J) else ""))
    s1_bad = [f"{c} s{s}: {J[('old17', c, s)]['saved_val_loss']:.4f}" for c in CELLS for s in (1, 2)
              if ("old17", c, s) in J and abs(J[("old17", c, s)]["saved_val_loss"] - OLD_VAL[c]) > 0.005]
    verdict("S1", not s1_bad and all(("old17", c, s) in J for c in CELLS for s in (1, 2)), "; ".join(s1_bad))

    def rel(x_arm, y_arm, c, group, **kw):
        return mean((arm_mean(x_arm, c, s, **kw) - arm_mean(y_arm, c, s, **kw)) / arm_mean(y_arm, c, s, **kw) for s in group)

    def wins(x_arm, y_arm, c, group):
        return sum(arm_mean(x_arm, c, s) < arm_mean(y_arm, c, s) for s in group)

    m1 = {c: (wins("mix57_full", "old17", c, above), rel("mix57_full", "old17", c, above)) for c in CELLS}
    verdict("M1", all(w >= 4 and r <= -0.10 for w, r in m1.values()), "  ".join(f"{c} {w}/5 {r:+.1%}" for c, (w, r) in m1.items()))
    m2 = {c: wins("mix57_same", "old17", c, above) for c in CELLS}
    verdict("M2", all(w >= 4 for w in m2.values()), "  ".join(f"{c} {w}/5" for c, w in m2.items()))
    m3 = {c: rel("mix57_full", "old17", c, inside) for c in ("static_mask", "static_timing")}
    verdict("M3", all(abs(r) <= 0.10 for r in m3.values()), "  ".join(f"{c} {r:+.1%}" for c, r in m3.items()))
    m4 = wins("mix57_full", "old17", "dynamic_timing", new_inside)
    verdict("M4", m4 >= 4, f"dynamic_timing {m4}/{len(new_inside)}")
    m5w = sum(arm_mean("mix57_full", "dynamic_timing", s) < arm_mean("mix57_full", "dynamic_mask", s) for s in gissr)
    m5r = mean((arm_mean("mix57_full", "dynamic_timing", s, key="bce_non_seed_per_t", t=0) - arm_mean("mix57_full", "dynamic_mask", s, key="bce_non_seed_per_t", t=0))
               / arm_mean("mix57_full", "dynamic_mask", s, key="bce_non_seed_per_t", t=0) for s in gissr)
    verdict("M5", m5w >= 10 and m5r <= -0.10, f"wins {m5w}/{len(gissr)}; 6-hour BCE {m5r:+.1%}")
    m6 = sum(not clear("mix57_full", "old17", c, other[0]) and not clear("old17", "mix57_full", c, other[0]) for c in CELLS) if other else 0
    verdict("M6", m6 >= 3, f"no clear win in {m6}/4 cells")
    m7 = {c: (mean(arm_mean("mix57_full", c, s) for s in storms) - mean(arm_mean("mix57_same", c, s) for s in storms)) / mean(arm_mean("mix57_same", c, s) for s in storms) for c in CELLS}
    verdict("M7", all(abs(r) <= 0.10 for r in m7.values()), "  ".join(f"{c} {r:+.1%}" for c, r in m7.items()))

    print("\n=== GNN against the nearest-storm lookup built from the same training runs (non-seed BCE, storms won by the GNN) ===")
    for c in CELLS:
        parts = []
        for arm in ARMS:
            w = sum(arm_mean(arm, c, s) < arm_mean(arm, c, s, pred="nearest_storm") for s in storms)
            parts.append(f"{arm} {w}/{len(storms)}")
        print(f"{c:15s} " + "   ".join(parts))

    verdict("L1", all(lw >= 9 and gw for lw, gw in l1.values()), "  ".join(f"{c} {lw}/12 gc:{'model' if gw else 'lookup'}" for c, (lw, gw) in l1.items()))
    names = {"G0": "gates and completeness", "S1": "retrained old17 reproduces the existing val loss within 0.005",
             "L1": "lookup from 57 storms ahead on >= 9 of 12 GISSR storms; model ahead on geoclaw_2050 (4 cells)",
             "M1": "ABOVE: mix57_full beats old17 on >= 4 of 5 and by >= 10% on average (4 cells)",
             "M2": "ABOVE: mix57_same beats old17 on >= 4 of 5 (4 cells)", "M3": "INSIDE, static cells: change within +-10%",
             "M4": "INSIDE, dynamic_timing: mix57_full beats old17 on >= 4 of 5 new storms",
             "M5": "mix57_full: timing inputs win on >= 10 of 12 GISSR storms and by >= 10% at 6 h",
             "M6": "geoclaw_2050: no clear win either way in >= 3 of 4 cells", "M7": "mix57_full within 10% of mix57_same (4 cells)"}
    print("\nPRE-REGISTERED VERDICTS")
    for key in ["G0", "S1", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "L1"]:
        print(f"  {key} {names[key]:78s} {'PASS' if v[key] else 'MISS'}   {notes[key]}")
    print("GATES (G0):", "PASS" if v["G0"] else "FAIL")
    print("PREDICTIONS (S1, M1-M7, L1):", "ALL PASS" if all(v[x] for x in v if x != "G0") else "SOME MISSED - record them")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("wrote", a.out, f"({len(rows)} rows)")


if __name__ == "__main__":
    main()
