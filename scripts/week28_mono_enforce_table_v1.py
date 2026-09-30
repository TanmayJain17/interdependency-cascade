#!/usr/bin/env python3
"""
scripts/week28_mono_enforce_table_v1.py — Week 28 verdict table: where to enforce
"P(failed by t) can only go up" — output side (fixes applied to the independent head)
vs model side (hazard head). Reads the eight JSONs written by
hpc/audit_gnn_mono_enforce_v1.sbatch (scripts/week28_mono_enforce_audit_v1.py):

  <dir>/independent_<cell>.json   <dir>/hazard_<cell>.json      cell in the 2x2

Prints the table and the pre-registered verdicts, writes
analysis/gnn/week28_mono_enforce_v1.csv (per-cell block, dip block, held-out block).

PRE-REGISTRATION (written 30 Sep 2026, before the run):
  P0  gates: every JSON has an empty failure list (raw reproduces the saved val loss;
      fixed readings have 0 violations; fixes are no-ops on the hazard head).
  P1  by construction (theorem, asserted per run): sort and isotonic val loss <= raw.
  P2  prediction: the output-side fix is nearly free AND nearly worthless for fit:
      0 <= raw - isotonic <= 0.002 in every cell.
  P3  prediction: model side and output side fit equally well:
      |hazard - (independent + isotonic)| <= 0.002 in every cell.
  P4  prediction: the dips are small: median dip < 0.01 in every cell, and fewer than
      1 % of (node, horizon-pair) entries dip by more than 0.05.
  P5  prediction: the dips sit on flat label rows: more than half of all dips fall on
      seeds, [1111] and [0000] rows in every cell.
  P6  prediction: cascade-only PR-AUC per horizon moves by <= 0.005 under sort and isotonic.
  No prediction is made for cummax beyond "it carries no guarantee" (it can be worse than raw).
  Named failure mode: if raw - isotonic > 0.002 somewhere, the dips were large there, the
  output-side fix buys real fit, and hazard-vs-isotonic (P3) becomes the comparison to report.
  The result may claim: how the two ways of enforcing monotonicity compare in FIT on the four
  fixed horizons, single seed. It may not claim anything about timing accuracy.
"""
import argparse
import csv
import json
from pathlib import Path

CELLS = ["static_mask", "static_timing", "dynamic_mask", "dynamic_timing"]
FIXES = ["cummax", "sort", "isotonic"]


def flat_share(d):
    """Share of all dips that fall on flat label rows (seeds, [1111], [0000])."""
    k = len(d["pair_labels"])
    tot = flat = 0.0
    for name, r in d["by_label_row"].items():
        n = (r["dip_share"] if r["dip_share"] == r["dip_share"] else 0.0) * r["rows"] * k
        tot += n
        if name.startswith("[" + "0" * (k + 1) + "]") or name.startswith("[" + "1" * (k + 1) + "]"):
            flat += n
    s = d["seeds"]
    n = (s["dip_share"] if s["dip_share"] == s["dip_share"] else 0.0) * s["rows"] * k
    return (flat + n) / (tot + n) if (tot + n) else float("nan"), n / (tot + n) if (tot + n) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/hpc_results_aug2026/gnn_mono_enforce_v1")
    ap.add_argument("--out", default="analysis/gnn/week28_mono_enforce_v1.csv")
    a = ap.parse_args()

    rows, dips, hold, verdict = [], [], [], {f"P{i}": True for i in range(7)}
    for c in CELLS:
        ind = json.load(open(Path(a.dir) / f"independent_{c}.json"))
        haz = json.load(open(Path(a.dir) / f"hazard_{c}.json"))
        for j, name in ((ind, "independent"), (haz, "hazard")):
            if j["failures"] or j.get("val_subset") or j.get("holdout_subset"):
                verdict["P0"] = False
                print(f"P0 FAIL {name}_{c}: failures={j['failures'][:3]} subset={j.get('val_subset')}/{j.get('holdout_subset')}")
        v, hz, dv = ind["val"], haz["val"]["raw"], ind["dips_val"]
        raw = v["raw"]["loss"]                      # the production float32 number (reproduces the 2x2)
        # Comparisons between readings use the float64 losses, like with like: the float32 and
        # float64 means of the same run differ in the 9th decimal, which would otherwise decide
        # the sign of a zero difference.
        L = {f: v[f]["loss64"] for f in ["raw"] + FIXES}
        gain_iso = L["raw"] - L["isotonic"]
        d_hz_iso = haz["val"]["raw"]["loss64"] - L["isotonic"]
        pr_d = {f: max(abs(x - y) for x, y in zip(v[f]["cascade_pr_per_t"], v["raw"]["cascade_pr_per_t"])) for f in FIXES}
        flat, seed_share = flat_share(dv)
        th = dv["thresholds"]
        so = dict(zip(th, dv["share_over"]))
        p1 = L["sort"] <= L["raw"] + 1e-10 and L["isotonic"] <= L["raw"] + 1e-10
        p2 = -1e-10 <= gain_iso <= 0.002
        p3 = abs(d_hz_iso) <= 0.002
        p4 = dv["size_quantiles"]["q50"] < 0.01 and so[0.05] < 0.01
        p5 = flat > 0.5
        p6 = pr_d["sort"] <= 0.005 and pr_d["isotonic"] <= 0.005
        for k, ok in zip(("P1", "P2", "P3", "P4", "P5", "P6"), (p1, p2, p3, p4, p5, p6)):
            verdict[k] &= ok
        rows.append({
            "cell": c,
            "indep_raw": round(raw, 5), "indep_cummax": round(v["cummax"]["loss"], 5),
            "indep_sort": round(v["sort"]["loss"], 5), "indep_isotonic": round(v["isotonic"]["loss"], 5),
            "hazard": round(hz["loss"], 5),
            "d_cummax": round(L["cummax"] - L["raw"], 6), "d_sort": round(L["sort"] - L["raw"], 6),
            "d_isotonic": round(L["isotonic"] - L["raw"], 6), "d_hazard": round(hz["loss64"] - L["raw"], 6),
            "hazard_minus_isotonic": round(d_hz_iso, 6),
            "indep_raw_mono_viol": round(v["raw"]["mono_viol"], 5), "hazard_mono_viol": round(hz["mono_viol"], 5),
            "brier_raw": round(v["raw"]["brier"], 5), "brier_isotonic": round(v["isotonic"]["brier"], 5),
            "brier_hazard": round(hz["brier"], 5),
            "cascade_pr_raw": " ".join(f"{x:.3f}" for x in v["raw"]["cascade_pr_per_t"]),
            "cascade_pr_isotonic": " ".join(f"{x:.3f}" for x in v["isotonic"]["cascade_pr_per_t"]),
            "cascade_pr_hazard": " ".join(f"{x:.3f}" for x in hz["cascade_pr_per_t"]),
            "cascade_pr_max_abs_d_cummax": round(pr_d["cummax"], 4), "cascade_pr_max_abs_d_sort": round(pr_d["sort"], 4),
            "cascade_pr_max_abs_d_isotonic": round(pr_d["isotonic"], 4),
            "P1": p1, "P2": p2, "P3": p3, "P4": p4, "P5": p5, "P6": p6,
        })
        dips.append({
            "cell": c, "dips_per_run": round(dv["dips_per_run"], 1),
            **{f"share_pairs_dip_over_{t:g}": round(s, 5) for t, s in zip(th, dv["share_over"])},
            **{f"share_nodes_any_dip_over_{t:g}": round(s, 5) for t, s in zip(th, dv["node_share_any_over"])},
            "size_median": float(f"{dv['size_quantiles']['q50']:.3g}"), "size_p90": float(f"{dv['size_quantiles']['q90']:.3g}"),
            "size_p99": float(f"{dv['size_quantiles']['q99']:.3g}"), "size_mean": float(f"{dv['mean_size']:.3g}"),
            "size_max": float(f"{max(dv['max_size_by_pair']):.3g}"),
            **{f"dip_share_{lab}": round(x[0], 5) for lab, x in zip(dv["pair_labels"], dv["share_over_by_pair"])},
            "share_of_dips_on_flat_rows": round(flat, 4), "share_of_dips_on_seeds": round(seed_share, 4),
            **{f"dip_share_row_{name.split(']')[0][1:]}": round(r["dip_share"], 5) for name, r in dv["by_label_row"].items()},
            "dip_share_seed_rows": round(dv["seeds"]["dip_share"], 5),
        })
        for s in sorted(ind["test_per_scenario"], key=lambda s: ind["test_per_scenario"][s]["raw"]["loss"]):
            t, th_ = ind["test_per_scenario"][s], haz["test_per_scenario"][s]["raw"]
            hold.append({"cell": c, "storm": s, "indep_raw": round(t["raw"]["loss"], 5),
                         "indep_cummax": round(t["cummax"]["loss"], 5), "indep_sort": round(t["sort"]["loss"], 5),
                         "indep_isotonic": round(t["isotonic"]["loss"], 5), "hazard": round(th_["loss"], 5),
                         "hazard_minus_isotonic": round(th_["loss"] - t["isotonic"]["loss"], 5),
                         "indep_raw_mono_viol": round(t["raw"]["mono_viol"], 5)})

    print("\nVAL LOSS (type-averaged BCE, the 2x2 number)")
    print(f"{'cell':15s} {'indep raw':>10s} {'+cummax':>10s} {'+sort':>10s} {'+isotonic':>10s} {'hazard':>10s}   hazard-isotonic")
    for r in rows:
        print(f"{r['cell']:15s} {r['indep_raw']:10.5f} {r['indep_cummax']:10.5f} {r['indep_sort']:10.5f} "
              f"{r['indep_isotonic']:10.5f} {r['hazard']:10.5f}   {r['hazard_minus_isotonic']:+.6f}")
    print("\nCHANGE vs independent raw (negative = better)")
    for r in rows:
        print(f"{r['cell']:15s} cummax {r['d_cummax']:+.6f}  sort {r['d_sort']:+.6f}  isotonic {r['d_isotonic']:+.6f}  "
              f"hazard {r['d_hazard']:+.6f} | mono_viol {r['indep_raw_mono_viol']:.4f} -> 0 | "
              f"cascade PR max|d| cummax {r['cascade_pr_max_abs_d_cummax']:.4f} sort {r['cascade_pr_max_abs_d_sort']:.4f} "
              f"isotonic {r['cascade_pr_max_abs_d_isotonic']:.4f}")
    print("\nTHE DIPS (independent head, val)")
    for d in dips:
        print(f"{d['cell']:15s} {d['dips_per_run']:7.0f}/run | pairs dipping >1e-6 {d['share_pairs_dip_over_1e-06']:.4f}, "
              f">0.01 {d['share_pairs_dip_over_0.01']:.4f}, >0.05 {d['share_pairs_dip_over_0.05']:.4f}, >0.1 {d['share_pairs_dip_over_0.1']:.4f} | "
              f"size median {d['size_median']:.2g} p90 {d['size_p90']:.2g} p99 {d['size_p99']:.2g} max {d['size_max']:.2g} | "
              f"on flat rows {d['share_of_dips_on_flat_rows']:.2f} (seeds {d['share_of_dips_on_seeds']:.2f})")
    print("\nPRE-REGISTERED VERDICTS")
    names = {"P0": "gates (reproduction, invariants)", "P1": "sort / isotonic never worse than raw (theorem)",
             "P2": "0 <= raw - isotonic <= 0.002", "P3": "|hazard - isotonic| <= 0.002",
             "P4": "median dip < 0.01 and < 1 % of pairs dip > 0.05", "P5": "> half of dips on flat label rows",
             "P6": "cascade-only PR moves <= 0.005 under sort / isotonic"}
    for k in sorted(verdict):
        print(f"  {k} {names[k]:52s} {'PASS' if verdict[k] else 'FAIL'}"
              + ("" if k in ("P0", "P1") or verdict[k] else "   cells: " + ", ".join(r["cell"] for r in rows if not r[k])))
    print("GATES AND THEOREMS (P0, P1):", "PASS" if verdict["P0"] and verdict["P1"] else "FAIL")
    print("PREDICTIONS (P2-P6):", "ALL PASS" if all(verdict[k] for k in ("P2", "P3", "P4", "P5", "P6")) else "SOME MISSED — record them")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        for block in (rows, dips, hold):
            w = csv.DictWriter(f, fieldnames=list(block[0].keys()))
            w.writeheader(); w.writerows(block)
            f.write("\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
