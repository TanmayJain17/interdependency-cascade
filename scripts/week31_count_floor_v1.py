#!/usr/bin/env python3
"""
scripts/week31_count_floor_v1.py — how low can the count error go on each held-out storm?

The count error (WAPE at a horizon) is sum_r |c_hat_r - c_r| / sum_r c_r over the runs of a storm, c_r the
simulated number of non-seed failures in run r. A predictor that knows the storm but not the run's random
draws can at best answer the same number every run; the best such constant is the median of c_r, and its
error is the FLOOR for storm-level predictors (the nearest-storm lookup, the proposed direct model). The
GNN also sees the run's seed set, so it may go below the floor; the buffer draws remain hidden from it.

Reads the per-run true counts stored in the score JSONs of hpc/audit_gnn_morestorms_v1.sbatch (field
model.per_run_non_seed_counts.true, 1,000 runs x 4 horizons per held-out storm). Those counts are labels, so
every JSON of a cell must carry the same ones: the script checks this across all arms, variants and seeds
of the cell and stops if they differ. Prints, per cell and storm: mean count at 96 h, run-to-run spread
(SD/mean), the floor (median-constant WAPE), the share of runs below half the mean (a bimodality sign),
and the WAPE of the lookup and of the plain and storm-context models (mix57_full, mean of seeds 1-2).
Writes one CSV row per cell x storm x horizon.

    python scripts/week31_count_floor_v1.py [--dir data/hpc_results_aug2026/gnn_morestorms_numbers_v1]
        [--out analysis/gnn/week31_count_floor_v1.csv]
"""
import argparse, csv, json, statistics as st, sys
from pathlib import Path

CELLS = ("static_mask", "static_timing", "dynamic_mask", "dynamic_timing")
H = (6, 24, 48, 96)


def wape_const(c, k):
    v = [r[k] for r in c]
    med = st.median(v)
    return sum(abs(med - x) for x in v) / sum(v) if sum(v) > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/hpc_results_aug2026/gnn_morestorms_numbers_v1")
    ap.add_argument("--out", default="analysis/gnn/week31_count_floor_v1.csv")
    a = ap.parse_args()
    d = Path(a.dir)
    rows, problems = [], []
    for cell in CELLS:
        files = sorted(d.glob(f"*_{cell}_s*.json"))
        if not files:
            problems.append(f"{cell}: no JSON"); continue
        data = {f.name: json.load(open(f)) for f in files}
        ref_name, ref = next(iter(data.items()))
        storms = sorted(ref["test_per_scenario"])
        # G: every JSON of the cell carries the same per-run true counts for every storm
        for name, dd in data.items():
            if sorted(dd["test_per_scenario"]) != storms:
                problems.append(f"{cell}: {name} has a different storm list"); continue
            for s in storms:
                if dd["test_per_scenario"][s]["model"]["per_run_non_seed_counts"]["true"] != ref["test_per_scenario"][s]["model"]["per_run_non_seed_counts"]["true"]:
                    problems.append(f"{cell}: per-run true counts differ between {ref_name} and {name} on {s}")
        def wape_of(prefix, s, k):
            vals = [dd["test_per_scenario"][s]["model"]["count_non_seed"]["wape"][k] for n, dd in data.items() if n.startswith(prefix) and n.endswith(("_s1.json", "_s2.json"))]
            return sum(vals) / len(vals) if vals else float("nan")
        print(f"\n== {cell}: count error at 96 h on the held-out storms (floor = best constant answer, the median run)")
        print(f"{'storm':24s} {'mean96':>7s} {'SD/mean':>7s} {'floor':>6s} {'runs<half':>9s} | {'lookup57':>8s} {'plain57':>8s} {'ctx57':>8s}   ({len(files)} JSONs agree on the counts)")
        out = []
        for s in storms:
            c = ref["test_per_scenario"][s]["model"]["per_run_non_seed_counts"]["true"]
            v96 = [r[3] for r in c]; m = st.mean(v96); sd = st.pstdev(v96)
            lk = [dd["test_per_scenario"][s]["nearest_storm"]["count_non_seed"]["wape"] for n, dd in data.items() if n.startswith("mix57_full_plain")]
            lk = [sum(x[k] for x in lk) / len(lk) for k in range(4)] if lk else [float("nan")] * 4
            rec = dict(storm=s, mean96=m, sd_over_mean=sd / m if m else float("nan"), frac_below_half=sum(1 for x in v96 if x < 0.5 * m) / len(v96))
            for k, h in enumerate(H):
                rows.append({"cell": cell, "storm": s, "horizon_h": h, "mean_true": st.mean(r[k] for r in c), "floor_wape_median": wape_const(c, k),
                             "lookup57_wape": lk[k], "plain57_wape": wape_of(f"mix57_full_plain_{cell}", s, k), "ctx57_wape": wape_of(f"mix57_full_ctx_{cell}", s, k),
                             "sd_over_mean": st.pstdev(r[k] for r in c) / max(st.mean(r[k] for r in c), 1e-9), "n_runs": len(c)})
            out.append((m, s, rec, wape_const(c, 3), lk[3], wape_of(f"mix57_full_plain_{cell}", s, 3), wape_of(f"mix57_full_ctx_{cell}", s, 3)))
        for m, s, rec, fl, lk3, pl, cx in sorted(out):
            print(f"{s:24s} {m:7.0f} {rec['sd_over_mean']:7.2f} {fl:6.2f} {rec['frac_below_half']:9.2f} | {lk3:8.3f} {pl:8.3f} {cx:8.3f}")
    if problems:
        print("\nPROBLEMS:"); [print("  " + p) for p in problems]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("wrote", a.out, f"({len(rows)} rows)")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
