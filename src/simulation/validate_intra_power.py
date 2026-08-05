#!/usr/bin/env python3
"""
validate_intra_power.py — correctness test for the intra_power resampler.

Reproduces Jesse's per-OSI-node pct_failed by replaying his own 1,000-run
library through our decoding rules, for BOTH tied-bus rules. Because index
pairing makes our marginals a deterministic function of his run set, the
reproduced values must equal his published pct up to the name-level ambiguity
(12 of 50 Ward bus names are shared by two pp idxs).

This also empirically ADJUDICATES the B2 question (does his pct count the
matched bus only, or any tied bus?): whichever rule reproduces his numbers is
the rule he used.

Usage (project root):  python src/simulation/validate_intra_power.py
Env: JESSE_DIR (default data/external/jesse_power)
"""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from intra_power import load_power_library  # noqa: E402

JESSE_DIR = os.environ.get("JESSE_DIR", "data/external/jesse_power")
TOL_PP = 0.05  # percentage-point tolerance for an "exact" reproduction


def published_pct(lib, jesse_scen):
    path = os.path.join(lib.jesse_dir, jesse_scen, f"osi_node_failure_pct_{jesse_scen}.csv")
    out = {}
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            out[r["osi_node_id"]] = float(r["pct_failed"])
    return out


def main():
    lib = load_power_library(JESSE_DIR)
    print(f"library scenarios: {sorted(lib.scenarios.values())}")
    verdict = {}
    for jesse_scen in sorted(lib.scenarios.values()):
        runs = lib.runs(jesse_scen)
        n = len(runs)
        idx_col = lib.used_idx_column[jesse_scen]
        print(f"\n=== {jesse_scen}: {n} runs | decode mode: "
              f"{'final_bus_oos_idx (exact)' if idx_col else 'NAME-level fallback'} ===")
        pub = published_pct(lib, jesse_scen)
        for rule in ("matched", "any_tied"):
            deltas = []
            worst = []
            for osi_id, row in lib.osi.items():
                dead = sum(1 for run in runs if lib.osi_dead(run, row, tied_rule=rule))
                repro = 100.0 * dead / n
                d = repro - pub.get(osi_id, 0.0)
                deltas.append(abs(d))
                if abs(d) > TOL_PP:
                    worst.append((osi_id, row["osi_name"], row["ward_bus_name"],
                                  pub.get(osi_id, 0.0), round(repro, 2), round(d, 2)))
            exact = sum(1 for d in deltas if d <= TOL_PP)
            mean_d = sum(deltas) / len(deltas)
            max_d = max(deltas)
            print(f"  rule={rule:9s}: exact {exact}/175  mean|delta|={mean_d:.3f}pp  "
                  f"max|delta|={max_d:.2f}pp")
            worst.sort(key=lambda w: -abs(w[5]))
            for w in worst[:8]:
                print(f"      {w[0]:6s} {w[1][:28]:28s} bus={w[2][:18]:18s} "
                      f"pub={w[3]:6.2f} repro={w[4]:6.2f} delta={w[5]:+6.2f}")
            verdict.setdefault(jesse_scen, {})[rule] = (exact, round(mean_d, 4), round(max_d, 2))

    print("\n=== B2 adjudication summary (exact-count / mean|delta| / max|delta|) ===")
    for scen, res in verdict.items():
        m, a = res["matched"], res["any_tied"]
        better = "matched" if (m[0], -m[1]) >= (a[0], -a[1]) else "any_tied"
        print(f"  {scen}: matched={m}  any_tied={a}  ->  Jesse's rule looks like: {better}")


if __name__ == "__main__":
    main()
