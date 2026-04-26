#!/usr/bin/env python3
"""
verify_and_plot_loss_exceedance.py


1. Verify the direct flood counts are ordered correctly across scenarios
   (Extreme 2080 must have >= Moderate 2050 >= Moderate Current). The Week 6
   presentation slide had stale pre-bugfix numbers showing the wrong ordering
   (346 > 629 > 298). This script verifies the corrected numbers from the
   final simulation output.

2. Plot loss exceedance probability (EP) curves — direct vs cascading — for
   all three scenarios. The EP curve is standard in risk analysis: for each
   possible loss magnitude L on the x-axis, the y-axis shows P(Loss >= L)
   across the 1000 Monte Carlo runs.

Outputs:
    outputs/direct_flood_verification.txt     — written sanity-check report
    outputs/loss_exceedance_curves.png        — slide-ready figure
    outputs/loss_exceedance_curves.pdf        — vector version
    outputs/loss_exceedance_data.csv          — the underlying per-scenario
                                                 loss arrays, for reference

Run from project root:
    python3 src/analysis/verify_and_plot_loss_exceedance.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

NODES_PATH = Path("data/flood/nyc_infra_nodes_dep_flood.geojson")

SCENARIOS = {
    "moderate_current": {
        "label":        "Moderate / Current",
        "depth_col":    "flood_moderate_current_depth_m",
        "mc_path":      Path("data/simulation/monte_carlo_failures_nyc_moderate_current.json"),
        "cascade_path": Path("data/simulation/cascade_results_nyc_moderate_current.json"),
        "color":        "#2b7fb8",   # blue
    },
    "moderate_2050": {
        "label":        "Moderate / 2050",
        "depth_col":    "flood_moderate_2050_depth_m",
        "mc_path":      Path("data/simulation/monte_carlo_failures_nyc_moderate_2050.json"),
        "cascade_path": Path("data/simulation/cascade_results_nyc_moderate_2050.json"),
        "color":        "#d97f2c",   # orange
    },
    "extreme_2080": {
        "label":        "Extreme / 2080",
        "depth_col":    "flood_extreme_2080_depth_m",
        "mc_path":      Path("data/simulation/monte_carlo_failures_nyc_extreme_2080.json"),
        "cascade_path": Path("data/simulation/cascade_results_nyc_extreme_2080.json"),
        "color":        "#b03030",   # dark red
    },
}

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Part 1: Verify direct flood counts
# -----------------------------------------------------------------------------

def verify_direct_flood_counts():
    """
    Cross-check direct flood counts three ways:
      (a) Count nodes with flood_depth > 0 in the geojson (the exposure baseline)
      (b) Count nodes in each MC run's initial failure set (fragility sample)
      (c) Count nodes in each cascade run at t=0 (what the simulator reports)

    All three should be increasing from moderate_current to extreme_2080.
    If the presentation slide is wrong, the numbers here will show it.
    """
    print("=" * 75)
    print("PART 1: Direct flood count verification")
    print("=" * 75)

    report_lines = []
    report_lines.append("DIRECT FLOOD COUNT VERIFICATION")
    report_lines.append("=" * 75)
    report_lines.append("")
    report_lines.append("Question: do the direct flood counts increase with scenario severity?")
    report_lines.append("(Expected: Moderate/Current < Moderate/2050 < Extreme/2080)")
    report_lines.append("")

    # Load nodes geojson
    if not NODES_PATH.exists():
        print(f"ERROR: {NODES_PATH} not found")
        return None
    nodes = gpd.read_file(NODES_PATH)

    # --- Check (a): raw exposure from geojson ---
    report_lines.append("(a) Raw flood exposure in the geojson (nodes with flood_depth > 0):")
    report_lines.append(f"    {'Scenario':<22} {'Exposed':>10} {'% of 6231':>12}")
    report_lines.append("    " + "-" * 46)
    exposure_counts = {}
    for key, cfg in SCENARIOS.items():
        col = cfg["depth_col"]
        if col not in nodes.columns:
            report_lines.append(f"    {cfg['label']:<22} MISSING COLUMN {col}")
            continue
        exposed = int((nodes[col].fillna(0.0) > 0.0).sum())
        pct = 100.0 * exposed / len(nodes)
        exposure_counts[key] = exposed
        report_lines.append(f"    {cfg['label']:<22} {exposed:>10,d} {pct:>11.2f}%")
    report_lines.append("")

    # --- Check (b): mean initial failures in Monte Carlo runs ---
    report_lines.append("(b) Initial flood failures from Monte Carlo fragility sampling:")
    report_lines.append(f"    {'Scenario':<22} {'Mean':>8} {'Std':>8} {'Min':>6} {'Max':>6} {'(1000 runs)':>14}")
    report_lines.append("    " + "-" * 64)
    mc_arrays = {}
    for key, cfg in SCENARIOS.items():
        if not cfg["mc_path"].exists():
            report_lines.append(f"    {cfg['label']:<22} MISSING FILE {cfg['mc_path']}")
            continue
        with open(cfg["mc_path"]) as f:
            mc_runs = json.load(f)
        counts = np.array([r["n_failed"] for r in mc_runs])
        mc_arrays[key] = counts
        report_lines.append(f"    {cfg['label']:<22} {counts.mean():>8.1f} "
                            f"{counts.std():>8.1f} {counts.min():>6d} {counts.max():>6d}")
    report_lines.append("")

    # --- Check (c): cascade sim's t=0 count (confirms simulator agrees) ---
    report_lines.append("(c) Cascade simulator's t=0 failures (what the simulator reports):")
    report_lines.append(f"    {'Scenario':<22} {'Mean':>8} {'Std':>8} {'(1000 runs)':>14}")
    report_lines.append("    " + "-" * 50)
    t0_arrays = {}
    for key, cfg in SCENARIOS.items():
        if not cfg["cascade_path"].exists():
            report_lines.append(f"    {cfg['label']:<22} MISSING FILE {cfg['cascade_path']}")
            continue
        with open(cfg["cascade_path"]) as f:
            cascade_runs = json.load(f)
        direct = np.array([r["direct_failures"] for r in cascade_runs])
        t0_arrays[key] = direct
        report_lines.append(f"    {cfg['label']:<22} {direct.mean():>8.1f} "
                            f"{direct.std():>8.1f}")
    report_lines.append("")

    # --- Verdict ---
    report_lines.append("VERDICT")
    report_lines.append("-" * 75)
    keys_in_order = ["moderate_current", "moderate_2050", "extreme_2080"]
    exposure_vals = [exposure_counts.get(k, -1) for k in keys_in_order]
    mc_means = [mc_arrays[k].mean() if k in mc_arrays else -1 for k in keys_in_order]

    exposure_ok = all(
        exposure_vals[i] <= exposure_vals[i + 1] for i in range(len(exposure_vals) - 1)
    )
    mc_ok = all(mc_means[i] <= mc_means[i + 1] for i in range(len(mc_means) - 1))

    if exposure_ok and mc_ok:
        report_lines.append("Direct flood counts are correctly ordered:")
        report_lines.append(f"  Moderate/Current ({mc_means[0]:.0f})  <  "
                            f"Moderate/2050 ({mc_means[1]:.0f})  <  "
                            f"Extreme/2080 ({mc_means[2]:.0f})")
        report_lines.append("")
        report_lines.append("The presentation slide (which showed 346 / 629 / 298) was using")
        report_lines.append("stale numbers from the pre-bugfix simulator run. Those bugs were:")
        report_lines.append("  - t=0 propagation through zero-buffer edges")
        report_lines.append("  - Structural edges (subway_line, power_line) treated as")
        report_lines.append("    failure-propagating")
        report_lines.append("Both were fixed in the final Week 6 simulator.")
        report_lines.append("")
        report_lines.append(f"Correct numbers: {mc_means[0]:.0f} / {mc_means[1]:.0f} / "
                            f"{mc_means[2]:.0f} direct flood failures.")
    else:
        report_lines.append("WARNING: Direct flood counts are NOT correctly ordered.")
        report_lines.append("This indicates a genuine data issue — investigate before proceeding.")
        report_lines.append(f"Raw exposure: {exposure_vals}")
        report_lines.append(f"MC means:     {mc_means}")

    report_text = "\n".join(report_lines)
    print(report_text)

    # Save report
    report_path = OUT_DIR / "direct_flood_verification.txt"
    report_path.write_text(report_text)
    print(f"\nReport saved: {report_path}")

    return {
        "exposure": exposure_counts,
        "mc_arrays": mc_arrays,
        "t0_arrays": t0_arrays,
    }


# -----------------------------------------------------------------------------
# Part 2: Loss Exceedance Probability curves
# -----------------------------------------------------------------------------

def compute_ep_curve(loss_values):
    """
    Empirical loss exceedance probability curve.
    For each sorted loss L, compute P(X >= L) across the N MC runs.
    Returns (sorted_losses, exceedance_probs) — both arrays of length N.
    """
    n = len(loss_values)
    # Sort ascending
    sorted_losses = np.sort(loss_values)
    # Exceedance probability: rank i (0-indexed from top) / N
    # The largest value has EP = 1/N (only itself exceeds it)
    # The smallest value has EP = N/N = 1.0 (all runs exceed-or-equal it)
    ep = np.arange(n, 0, -1) / n
    return sorted_losses, ep


def plot_loss_exceedance(results):
    """Produce the loss exceedance curve figure."""
    print("\n" + "=" * 75)
    print("PART 2: Loss Exceedance Probability curves")
    print("=" * 75)

    # Collect per-scenario loss arrays
    # direct loss = initial flood failures
    # cascade loss = total failures at t=96 minus initial (pure cascade delta)
    # total loss   = total failures at t=96

    all_data = {}  # key -> dict with direct/cascade/total arrays
    csv_rows = []  # for CSV export

    for key, cfg in SCENARIOS.items():
        if not cfg["cascade_path"].exists():
            print(f"  Skipping {key} (no cascade file)")
            continue

        with open(cfg["cascade_path"]) as f:
            runs = json.load(f)

        direct = np.array([r["direct_failures"] for r in runs])
        total = np.array([r["total_failures"] for r in runs])
        cascade = total - direct   # the pure cascade delta

        all_data[key] = {
            "direct": direct,
            "cascade": cascade,
            "total": total,
            "label": cfg["label"],
            "color": cfg["color"],
        }

        for i, (d, c, t) in enumerate(zip(direct, cascade, total)):
            csv_rows.append({
                "scenario": key,
                "run_id": i,
                "direct_loss": int(d),
                "cascade_loss": int(c),
                "total_loss": int(t),
            })

        print(f"  {cfg['label']}: "
              f"direct mean={direct.mean():.1f}, "
              f"cascade mean={cascade.mean():.1f}, "
              f"total mean={total.mean():.1f}")

    # Save raw data
    csv_path = OUT_DIR / "loss_exceedance_data.csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f"\nRaw per-run data saved: {csv_path}")

    # Build figure: 3 panels (direct, cascade, total) side by side
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    panel_titles = ["Direct Loss (t=0 flood failures)",
                    "Cascade Loss (t=0 to t=96h delta)",
                    "Total Loss (t=96h failures)"]
    panel_keys = ["direct", "cascade", "total"]

    for ax, title, loss_key in zip(axes, panel_titles, panel_keys):
        for _, data in all_data.items():
            losses = data[loss_key]
            x, y = compute_ep_curve(losses)
            ax.plot(x, y, color=data["color"], linewidth=2.2,
                    label=data["label"], alpha=0.9)

            # Add markers at key exceedance probabilities (1%, 10%, 50%)
            for p in [0.01, 0.10, 0.50]:
                # Find the loss value with closest EP to p
                idx = int(np.argmin(np.abs(y - p)))
                ax.scatter([x[idx]], [y[idx]], color=data["color"],
                           s=35, zorder=5, edgecolor="white", linewidth=0.8)

        ax.set_xlabel("Number of failed nodes (loss)", fontsize=11)
        ax.set_ylabel("Exceedance probability  P(Loss ≥ L)", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_yscale("log")   # log scale for tail visibility
        ax.set_ylim(0.0005, 1.5)
        ax.grid(True, alpha=0.3, which="both", linestyle=":")
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

        # Reference lines at 1%, 10%, 50% EP
        for p, style in [(0.01, ":"), (0.10, "--"), (0.50, "-.")]:
            ax.axhline(p, color="gray", linewidth=0.8, linestyle=style, alpha=0.5)

    fig.suptitle("Loss Exceedance Probability — NYC Citywide Cascade (1000 MC runs per scenario)",
                 fontsize=13, fontweight="bold", y=1.02)

    plt.tight_layout()

    # Save
    png_path = OUT_DIR / "loss_exceedance_curves.png"
    pdf_path = OUT_DIR / "loss_exceedance_curves.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    print(f"\nFigure saved:")
    print(f"  {png_path}")
    print(f"  {pdf_path}")

    plt.close(fig)

    # Also print a small summary table of key tail probabilities
    print("\nKey tail losses by scenario (for quick reference):")
    print(f"  {'Scenario':<22} {'P(L>=X) for':>14} {'direct':>9} {'cascade':>9} {'total':>8}")
    for key, data in all_data.items():
        for p in [0.50, 0.10, 0.01]:
            dv = np.percentile(data["direct"], 100 * (1 - p))
            cv = np.percentile(data["cascade"], 100 * (1 - p))
            tv = np.percentile(data["total"], 100 * (1 - p))
            label = data["label"] if p == 0.50 else ""
            print(f"  {label:<22} {f'EP={p:.2%}':>14} "
                  f"{dv:>9.0f} {cv:>9.0f} {tv:>8.0f}")
        print()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    results = verify_direct_flood_counts()
    if results is None:
        print("\nSkipping EP curves — verification did not complete.")
        return 1
    plot_loss_exceedance(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())