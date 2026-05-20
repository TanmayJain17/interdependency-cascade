#!/usr/bin/env python3
"""
demo_inference.py
=================
Runs the cascade inference wrapper across all six scenarios, prints a
comparison table, and saves per-scenario JSON results to data/inference/.

Usage:
    python scripts/demo_inference.py                    # n_mc=100, default output dir
    python scripts/demo_inference.py --n-mc 200         # more MC samples for stability
    python scripts/demo_inference.py --output-dir /tmp  # custom output dir
    python scripts/demo_inference.py --scenarios extreme_2080 geoclaw_2026
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import argparse
import json
import time
from pathlib import Path

from src.inference.predict_cascade import (
    SCENARIO_REGIME_MAP,
    predict_cascade_for_scenario,
    _to_json_safe,
)


ALL_SCENARIOS = list(SCENARIO_REGIME_MAP.keys())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-mc", type=int, default=100,
                        help="MC samples per scenario (default: 100)")
    parser.add_argument("--output-dir", default="data/inference",
                        help="Directory to save per-scenario JSON results")
    parser.add_argument("--scenarios", nargs="+", default=ALL_SCENARIOS,
                        choices=ALL_SCENARIOS,
                        help="Scenarios to run (default: all 6)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--top-k", type=int, default=10,
                        help="Top-K vulnerable nodes to report (default: 10)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running inference on {len(args.scenarios)} scenario(s) "
          f"with n_mc={args.n_mc}, device={args.device}")
    print(f"Output directory: {out_dir.resolve()}")
    print("=" * 100)

    summary_rows = []
    total_t0 = time.time()

    for scenario in args.scenarios:
        print(f"\n>>> Scenario: {scenario} ({SCENARIO_REGIME_MAP[scenario]})")
        result = predict_cascade_for_scenario(
            scenario=scenario,
            n_mc_samples=args.n_mc,
            seed=args.seed,
            device=args.device,
            verbose=False,
        )

        # Save JSON
        out_path = out_dir / f"{scenario}_predictions.json"
        with open(out_path, "w") as f:
            json.dump(_to_json_safe(result), f, indent=2)

        s = result["summary"]["expected_failures_per_timestep"]
        runtime = result["metadata"]["runtime_seconds"]
        top1 = result["summary"][f"top_{args.top_k}_vulnerable_nodes_t96"][0]

        summary_rows.append({
            "scenario": scenario,
            "regime": SCENARIO_REGIME_MAP[scenario],
            "t6": s["6"], "t24": s["24"], "t48": s["48"], "t96": s["96"],
            "runtime_s": runtime,
            "top1_id": top1["node_id"],
            "top1_p": top1["p_fail"],
            "out_path": str(out_path),
        })
        print(f"    t=6:{s['6']:7.1f}  t=24:{s['24']:7.1f}  "
              f"t=48:{s['48']:7.1f}  t=96:{s['96']:7.1f}   "
              f"({runtime:.1f}s)")
        print(f"    Top: {top1['node_id']} (P={top1['p_fail']:.3f})")
        print(f"    Saved -> {out_path}")

    # Final table
    print("\n" + "=" * 100)
    print("COMPARISON TABLE")
    print("=" * 100)
    print(f"{'Scenario':<20} {'Regime':<8} {'t=6':>8} {'t=24':>8} {'t=48':>8} "
          f"{'t=96':>8} {'sec':>6}   {'Top vulnerable (t=96)':<45}")
    print("-" * 100)
    for r in summary_rows:
        top_label = f"{r['top1_id'][:35]}... P={r['top1_p']:.3f}" \
            if len(r['top1_id']) > 35 else f"{r['top1_id']} P={r['top1_p']:.3f}"
        print(f"{r['scenario']:<20} {r['regime']:<8} "
              f"{r['t6']:>8.1f} {r['t24']:>8.1f} "
              f"{r['t48']:>8.1f} {r['t96']:>8.1f} "
              f"{r['runtime_s']:>6.1f}   {top_label}")

    total_elapsed = time.time() - total_t0
    print(f"\nTotal runtime: {total_elapsed:.1f}s across {len(args.scenarios)} scenarios")
    print(f"All outputs in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()