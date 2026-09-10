#!/usr/bin/env python3
"""
week25_arm_table_v1.py — one table for every Week 25 arm (translation test, surge-phase decomposition,
grid refinement) straight from the result directories, so the memo's arm table is never hand-edited.

Usage (repo root, flood env):
  python scripts/week25_arm_table_v1.py --root data/simulation_dyn_check/w25_translation_v1 \
      --out analysis/static_vs_dynamic_v1/week25_arm_table_v1.csv

Per arm: t360 mean/sd, super-cascade fraction (t360 >= 3000), crash fraction (t360 >= 1000), and the
paired delta against the arm's designated reference (same run index, identical seeds by construction).
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

S9 = "syn_ts_880_9_2p6632"
S6 = "syn_ts_914_6_1p2955"

# name, storm, grid_h, mode, arrival, tpeak, intra, reference arm (or None)
ARMS = [
    ("U",       S9, 6, "arrival",     0,  0,  0, None),
    ("W",       S9, 6, "arrival",    -6,  0,  0, "U"),
    ("R",       S9, 6, "arrival",     6,  6,  0, "U"),
    ("T",       S9, 6, "arrival",    12, 12,  0, "R"),
    ("Pminus",  S9, 6, "arrival",     6,  6, -6, "R"),
    ("Pplus",   S9, 6, "arrival",     6,  6,  6, "R"),
    ("R6",      S6, 6, "arrival",     6,  6,  0, None),
    ("Pminus6", S6, 6, "arrival",     6,  6, -6, "R6"),
    ("Pplus6",  S6, 6, "arrival",     6,  6,  6, "R6"),
    ("S3",      S9, 3, "static_peak", 0,  0,  0, None),
    ("U3",      S9, 3, "arrival",     0,  0,  0, "S3"),
    ("R3",      S9, 3, "arrival",     6,  6,  0, "U3"),
    ("Pplus3",  S9, 3, "arrival",     6,  6,  6, "R3"),
    ("Pminus3", S9, 3, "arrival",     6,  6, -6, "R3"),
    ("W3",      S9, 3, "arrival",    -6,  0,  0, "U3"),
    ("V3",      S9, 3, "arrival",     6,  0,  0, "U3"),   # pre-registered: flood +6 h, peak fixed, dense grid
]


def totals(root: Path, arm: str, storm: str, key="t360"):
    f = root / arm / f"cascade_results_nyc_{storm}.json"
    if not f.exists():
        return None
    recs = json.load(open(f))
    return np.array([r["by_timestep"][key] for r in recs], float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/simulation_dyn_check/w25_translation_v1")
    ap.add_argument("--out", default="analysis/static_vs_dynamic_v1/week25_arm_table_v1.csv")
    a = ap.parse_args()
    root = Path(a.root)
    rows, cache = [], {}
    for name, storm, grid, mode, arr, tp, io, ref in ARMS:
        t = totals(root, name, storm)
        if t is None:
            print(f"  {name:8s} missing — skipped")
            continue
        cache[name] = t
        row = dict(arm=name, storm=storm, grid_h=grid, mode=mode, arrival_shift_h=arr, tpeak_shift_h=tp,
                   intra_origin_shift_h=io, n=len(t), t360_mean=round(t.mean(), 2), t360_sd=round(t.std(), 2),
                   super_cascade_frac=round(float((t >= 3000).mean()), 2), crash1000_frac=round(float((t >= 1000).mean()), 2),
                   ref=ref or "")
        if ref and ref in cache:
            n = min(len(t), len(cache[ref])); d = t[:n] - cache[ref][:n]
            row.update(paired_delta_mean=round(d.mean(), 2), paired_delta_sd=round(d.std(), 2),
                       runs_larger=int((d > 0).sum()), runs_smaller=int((d < 0).sum()))
        rows.append(row)
    df = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(df.to_string(index=False))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
