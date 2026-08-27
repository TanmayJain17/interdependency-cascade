#!/usr/bin/env python3
"""
compare_static_dynamic_v1.py — paired comparison of frozen static runs vs dynamic-forcing runs.

The frozen campaign's first N Monte Carlo runs use the same fragility seeds and the same Weibull
buffer draws as a local N-run dynamic run (proven by the exact-reduction gate), so run i of each
file is the same world with only the seed timing changed. Everything reported here is a paired
difference, dynamic minus static.

Usage:
  python scripts/compare_static_dynamic_v1.py \
      --ref-dir data/hpc_results_aug2026/legacy_v1_n1000 --test-dir data/simulation_dyn_check/arrival \
      --scenarios syn_ts_914_6_1p2955 syn_ts_880_9_2p6632 syn_ts_605_5_3p7563 \
      --out analysis/static_vs_dynamic_v1/summary_v1.csv
Records must have: direct_failures, total_failures, by_timestep{t0,t6,t24,t48,t96}, fail_time_per_node,
cause_counts; dynamic records additionally carry dynamic_forcing.t_peak_h (checked, not required).
"""
import argparse, json, sys
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd

HORIZONS = ["t0", "t6", "t24", "t48", "t96"]     # extended automatically to any horizon present in BOTH files
CRASH_THRESHOLDS = (1000, 1200, 1500)


def ks_stat(a, b):
    a = np.sort(np.asarray(a, float)); b = np.sort(np.asarray(b, float))
    grid = np.concatenate([a, b])
    fa = np.searchsorted(a, grid, side="right") / len(a); fb = np.searchsorted(b, grid, side="right") / len(b)
    return float(np.max(np.abs(fa - fb)))


def compare(ref, test, scenario):
    n = min(len(ref), len(test)); ref = ref[:n]; test = test[:n]
    horizons = [h for h in sorted(set(ref[0]["by_timestep"]) & set(test[0]["by_timestep"]), key=lambda k: int(k[1:]))]
    last = horizons[-1]                       # totals are always compared at the last horizon BOTH files share
    seeds_same = all(r["direct_failures"] == t["direct_failures"] for r, t in zip(ref, test))
    tot_r = np.array([r["by_timestep"][last] for r in ref], float); tot_t = np.array([t["by_timestep"][last] for t in test], float)
    row = dict(scenario=scenario, n_runs=n, seeds_identical=seeds_same, totals_at=last,
               static_total_mean=tot_r.mean(), static_total_sd=tot_r.std(),
               dynamic_total_mean=tot_t.mean(), dynamic_total_sd=tot_t.std(),
               paired_delta_mean=(tot_t - tot_r).mean(), paired_delta_sd=(tot_t - tot_r).std(),
               runs_dynamic_larger=int((tot_t > tot_r).sum()), runs_dynamic_smaller=int((tot_t < tot_r).sum()),
               ks_totals=ks_stat(tot_r, tot_t))
    for h in horizons:
        a = np.array([r["by_timestep"][h] for r in ref], float); b = np.array([t["by_timestep"][h] for t in test], float)
        row[f"static_{h}"] = a.mean(); row[f"dynamic_{h}"] = b.mean(); row[f"delta_{h}"] = (b - a).mean()
    for thr in CRASH_THRESHOLDS:
        row[f"crash_frac_static_{thr}"] = float((tot_r >= thr).mean()); row[f"crash_frac_dynamic_{thr}"] = float((tot_t >= thr).mean())
    # per-node timing shifts (peak-relative hours in both files)
    earlier = later = same = only_dyn = only_stat = pre_peak = 0
    shift = []
    for r, t in zip(ref, test):
        fr, ft = r["fail_time_per_node"], t["fail_time_per_node"]
        for nid in set(fr) | set(ft):
            if nid in fr and nid in ft:
                d = ft[nid] - fr[nid]; shift.append(d)
                earlier += d < 0; later += d > 0; same += d == 0
            elif nid in ft:
                only_dyn += 1
            else:
                only_stat += 1
        pre_peak += sum(1 for v in ft.values() if v < 0)
    tot_pairs = max(earlier + later + same, 1)
    row.update(node_runs_fail_both=earlier + later + same, frac_earlier=earlier / tot_pairs, frac_later=later / tot_pairs,
               frac_same=same / tot_pairs, node_runs_only_dynamic=only_dyn, node_runs_only_static=only_stat,
               median_shift_h=float(np.median(shift)) if shift else np.nan,
               mean_prepeak_failures_dynamic=pre_peak / n)
    cr = Counter(); ct = Counter()
    for r in ref: cr.update(r["cause_counts"])
    for t in test: ct.update(t["cause_counts"])
    for c in sorted(set(cr) | set(ct)):
        row[f"cause_{c}_static"] = cr[c] / n; row[f"cause_{c}_dynamic"] = ct[c] / n
    tp = {t.get("dynamic_forcing", {}).get("t_peak_h") for t in test}
    row["t_peak_h"] = tp.pop() if len(tp) == 1 else None
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-dir", required=True); ap.add_argument("--test-dir", required=True)
    ap.add_argument("--scenarios", nargs="+", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    for sc in a.scenarios:
        rp = Path(a.ref_dir) / f"cascade_results_nyc_{sc}.json"; tp = Path(a.test_dir) / f"cascade_results_nyc_{sc}.json"
        if not rp.exists() or not tp.exists():
            print(f"{sc}: missing {'ref' if not rp.exists() else 'test'} file, skipped"); continue
        rows.append(compare(json.load(open(rp)), json.load(open(tp)), sc))
    if not rows:
        sys.exit("nothing compared")
    df = pd.DataFrame(rows); Path(a.out).parent.mkdir(parents=True, exist_ok=True); df.to_csv(a.out, index=False)
    show = ["scenario", "n_runs", "seeds_identical", "totals_at", "t_peak_h", "static_total_mean", "dynamic_total_mean", "paired_delta_mean",
            "paired_delta_sd", "runs_dynamic_larger", "runs_dynamic_smaller", "ks_totals",
            "static_t0", "dynamic_t0"] + [c for c in df.columns if c.startswith("delta_t")] + [
            "frac_earlier", "frac_later", "frac_same", "median_shift_h", "mean_prepeak_failures_dynamic",
            "crash_frac_static_1200", "crash_frac_dynamic_1200"]
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print(df[show].round(3).T.to_string())
    bad = [r["scenario"] for r in rows if not r["seeds_identical"]]
    if bad:
        print(f"\nWARNING: seeds differ from the frozen campaign for {bad} — comparison is NOT paired there")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
