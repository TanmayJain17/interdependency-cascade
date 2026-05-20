#!/usr/bin/env python3
"""
calibrate_surge_scaling.py
==========================
Calibrates the SURGE_DEPTH_SCALING constant in src/simulation/fragility.py
against the Sandy 2012 four-hospital evacuation pattern.

Method:
  1. Sweep candidate scaling factors [1.0, 1.1, 1.2, 1.3, 1.4, 1.5].
  2. For each candidate, monkey-patch the constant and run the inference
     wrapper on geoclaw_2026 (Sandy proxy) with 100 MC samples.
  3. Count how many of the Sandy target hospitals reach P(t=96) > 0.5 and > 0.9.
  4. Recommend the smallest factor that satisfies all four targets at P > 0.5.

This script does NOT mutate the source file. It prints a recommendation;
the actual constant update is a manual decision by the researcher.

Usage:
    python scripts/calibrate_surge_scaling.py
    python scripts/calibrate_surge_scaling.py --candidates 1.0 1.2 1.4 1.6
    python scripts/calibrate_surge_scaling.py --n-mc 200   # tighter estimates
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import argparse
import importlib

from src.simulation import fragility
from src.inference.predict_cascade import predict_cascade_for_scenario



# Substring patterns for the four Sandy evacuation hospitals.
# Each entry is (label, list_of_substrings_any_of_which_matches).
SANDY_TARGETS = [
    ("Bellevue",          ["bellevue"]),
    ("NYU Langone",       ["nyu_langone_hospital", "nyu_langone_hospitals"]),
    ("Mt Sinai Beth Israel", ["mount_sinai_beth_israel", "beth_israel"]),
    ("Mt Sinai NYEE",     ["nyee", "eye_and_ear", "eye_ear", "infirmary"]),
]


def find_target_hospitals(hospital_records):
    """Match Sandy target hospitals by substring. Returns dict: label -> (node_id, p96).

    If a target matches multiple node_ids, returns the one with the highest p96
    (this handles cases where a hospital has multiple sub-facilities in the data).
    """
    matches = {}
    for label, patterns in SANDY_TARGETS:
        best = None
        for rec in hospital_records:
            nid_lower = rec["node_id"].lower()
            if any(p in nid_lower for p in patterns):
                p96 = rec["p_fail"][-1]
                if best is None or p96 > best[1]:
                    best = (rec["node_id"], p96)
        matches[label] = best  # None if no match found
    return matches


def run_calibration_step(scaling_factor, scenario, n_mc, seed):
    """Monkey-patch SURGE_DEPTH_SCALING, run inference, restore original."""
    original = fragility.SURGE_DEPTH_SCALING
    try:
        fragility.SURGE_DEPTH_SCALING = scaling_factor
        result = predict_cascade_for_scenario(
            scenario=scenario,
            n_mc_samples=n_mc,
            seed=seed,
            verbose=False,
        )
    finally:
        fragility.SURGE_DEPTH_SCALING = original
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=float, nargs="+",
                        default=[1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
                        help="Scaling factors to test")
    parser.add_argument("--scenario", default="geoclaw_2026",
                        help="Scenario used as Sandy proxy (default: geoclaw_2026)")
    parser.add_argument("--n-mc", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold-low", type=float, default=0.5,
                        help="Lower P(t=96) threshold counting as 'predicted to fail'")
    parser.add_argument("--threshold-high", type=float, default=0.9,
                        help="Higher P(t=96) threshold for 'strongly predicted to fail'")
    args = parser.parse_args()

    print(f"Calibrating SURGE_DEPTH_SCALING against Sandy four-hospital pattern")
    print(f"Scenario: {args.scenario}    n_mc: {args.n_mc}    "
          f"current constant: {fragility.SURGE_DEPTH_SCALING}")
    print(f"Targets: {', '.join(t[0] for t in SANDY_TARGETS)}")
    print("=" * 110)

    sweep_results = []

    header = (f"{'Factor':>8}  {'Total_failures_t96':>20}  "
              f"{'Targets_P>'+str(args.threshold_low):>18}  "
              f"{'Targets_P>'+str(args.threshold_high):>18}  "
              "  Target details")
    print(header)
    print("-" * 110)

    for factor in args.candidates:
        result = run_calibration_step(factor, args.scenario, args.n_mc, args.seed)
        hospital_records = result["per_node_probabilities"]["hospital"]
        matches = find_target_hospitals(hospital_records)

        # Total expected failures at t=96
        total_t96 = result["summary"]["expected_failures_per_timestep"]["96"]

        # Count targets crossing thresholds
        n_low = sum(1 for m in matches.values() if m is not None and m[1] > args.threshold_low)
        n_high = sum(1 for m in matches.values() if m is not None and m[1] > args.threshold_high)
        n_matched = sum(1 for m in matches.values() if m is not None)

        # Compact per-target column
        target_str = "  ".join(
            f"{lbl[:12]}={m[1]:.2f}" if m is not None else f"{lbl[:12]}=N/A"
            for lbl, m in matches.items()
        )
        print(f"{factor:>8.2f}  {total_t96:>20.1f}  "
              f"{n_low:>4} / {n_matched:<11}  "
              f"{n_high:>4} / {n_matched:<11}    {target_str}")

        sweep_results.append({
            "factor": factor,
            "total_t96": total_t96,
            "matches": matches,
            "n_low": n_low,
            "n_high": n_high,
            "n_matched": n_matched,
        })

    # Recommend the smallest factor that gets all matched targets above the LOW threshold.
    print("\n" + "=" * 110)
    print("RECOMMENDATION")
    print("=" * 110)

    qualified = [r for r in sweep_results
                 if r["n_matched"] > 0 and r["n_low"] == r["n_matched"]]
    if qualified:
        best = qualified[0]   # smallest factor in sorted order
        print(f"Smallest factor satisfying all {best['n_matched']} matched targets "
              f"at P(t=96) > {args.threshold_low}: {best['factor']:.2f}")
        print(f"  (current value in code: {fragility.SURGE_DEPTH_SCALING})")
        if abs(best["factor"] - fragility.SURGE_DEPTH_SCALING) > 0.05:
            print(f"  CONSIDER updating SURGE_DEPTH_SCALING in "
                  f"src/simulation/fragility.py from "
                  f"{fragility.SURGE_DEPTH_SCALING} to {best['factor']:.2f}")
        else:
            print(f"  Current value is close to recommended. No change needed.")
    else:
        print("WARNING: No tested factor satisfied all matched Sandy targets at "
              f"P > {args.threshold_low}.")
        print("This could mean:")
        print("  1. The target hospital substring matchers missed some nodes — "
              "check the matches column above for N/A entries.")
        print("  2. The largest factor tested is still too small.")
        print("  3. The model has a structural bias that depth scaling alone can't fix")
        print("     (probably the case — points toward Option B learnable fragility).")

    # Always print the matched node_ids so substring matching can be debugged
    print("\nMatched target node_ids (from last sweep step):")
    last_matches = sweep_results[-1]["matches"]
    for label, m in last_matches.items():
        if m is None:
            print(f"  {label:<25} NO MATCH (check substring patterns in script)")
        else:
            print(f"  {label:<25} {m[0]}")


if __name__ == "__main__":
    main()