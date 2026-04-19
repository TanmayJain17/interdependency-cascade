#!/usr/bin/env python3
"""
Task 5: Generate summary output for Monday meeting.
"""

import json
import os
import numpy as np


def generate_summary():
    """Load all simulation results and print/save a clean summary."""

    print("=== Loading results ===")

    # Load scenario comparison
    with open("data/simulation/scenario_comparison.json") as f:
        scenarios = json.load(f)

    # Load amplifier nodes
    with open("data/simulation/amplifier_nodes.json") as f:
        amplifiers = json.load(f)

    # Load cascade results for sandy to get time-step details
    with open("data/simulation/cascade_results_sandy_actual.json") as f:
        sandy_cascade = json.load(f)

    # Compute centrality stats from amplifiers
    amp_bc = [a["betweenness_centrality"] for a in amplifiers]
    # Load all node centralities from cascade results if needed
    # For non-amplifier centrality, we recompute from the amplifier file
    mean_amp_bc = np.mean(amp_bc) if amp_bc else 0

    # Build summary text
    lines = []
    lines.append("=" * 65)
    lines.append("FLOOD SCENARIO CASCADE ANALYSIS — LOWER MANHATTAN")
    lines.append("=" * 65)
    lines.append("")
    lines.append("Graph: 347 nodes, 6 infrastructure types, 684 cascade edges")
    lines.append("")

    # Scenario table
    lines.append(f"{'Scenario':<14} | {'Direct Failures':>16} | {'Total Failures (t=96h)':>22} | {'Amplification':>14}")
    lines.append("-" * 75)
    for name in ["sandy_actual", "500yr"]:
        if name not in scenarios:
            continue
        r = scenarios[name]
        lines.append(
            f"{name:<14} | "
            f"{r['direct_mean']:>7.1f} +/- {r['direct_std']:<5.1f} | "
            f"{r['total_mean']:>9.1f} +/- {r['total_std']:<8.1f} | "
            f"{r['amplification_mean']:>8.2f}x"
        )
    lines.append("")

    # Amplifier nodes
    lines.append(f"Cascade Amplifier Nodes (fail despite no direct flood exposure): {len(amplifiers)}")
    if amplifiers:
        lines.append(f"  {'Node ID':<40} | {'Type':<10} | {'Betweenness':>11} | Upstream Trigger")
        lines.append(f"  {'-'*40}-+-{'-'*10}-+-{'-'*11}-+-{'-'*30}")
        for a in amplifiers[:15]:
            lines.append(
                f"  {a['node_id']:<40} | {a['infra_type']:<10} | "
                f"{a['betweenness_centrality']:>11.6f} | {a['upstream_trigger']}"
            )
        if len(amplifiers) > 15:
            lines.append(f"  ... and {len(amplifiers) - 15} more (see amplifier_nodes.csv)")
    lines.append("")

    # Centrality analysis
    lines.append("Centrality Analysis:")
    lines.append(f"  Mean betweenness (amplifiers):     {mean_amp_bc:.6f}")
    # Note: non-amplifier centrality computed during scenario_analysis run
    lines.append(f"  (See scenario_analysis output for full comparison)")
    lines.append("")

    # Time-step progression
    sandy = scenarios.get("sandy_actual", {})
    ts = sandy.get("by_timestep", {})
    lines.append("Time-step progression (Sandy scenario, mean across 1000 MC runs):")
    prev = 0
    for t_key, t_label in [("t0", "0"), ("t6", "6"), ("t24", "24"), ("t48", "48"), ("t96", "96")]:
        if t_key in ts:
            mean_val = ts[t_key]["mean"]
            delta = mean_val - prev
            delta_str = f"(+{delta:.0f})" if prev > 0 else "(direct flood)"
            lines.append(f"  t={t_label:>3}h: {mean_val:>6.1f} failures {delta_str}")
            prev = mean_val
    lines.append("")
    lines.append("=" * 65)

    summary = "\n".join(lines)

    # Print
    print(summary)

    # Save
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/week5_summary.txt", "w") as f:
        f.write(summary)
    print(f"\nSaved to outputs/week5_summary.txt")

    return summary


if __name__ == "__main__":
    os.chdir(os.path.expanduser("~/Desktop/RA"))
    generate_summary()
