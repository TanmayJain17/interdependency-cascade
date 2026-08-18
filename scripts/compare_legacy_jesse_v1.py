#!/usr/bin/env python
"""
compare_legacy_jesse_v1.py — Twin-campaign comparison (frozen v1 graph, N=1000).

Reads per-run distributions from cascade_results_nyc_<scen>.json in the legacy
and jesse result directories and produces the canonical December figure set:

  1. comparison_summary_v1.csv   per-scenario stats side by side (mean/sigma/
                                 P50/P90/P99, crash fractions at 3 thresholds,
                                 KS statistic+p, sigma ratio, amplification)
  2. cause_counts_v1.csv         mean failures per mechanism, legacy vs jesse
  3. fig_variance_amplification  sigma_jesse / sigma_legacy per scenario
  4. fig_histograms_grid         per-scenario overlaid total-failure histograms
  5. fig_scurve_overlay          crash probability vs wet-node count, both
                                 campaigns, logistic fits + 50% crossings
  6. fig_quantile_shift          P50/P90/P99 legacy -> jesse arrows
  7. fig_timestep_curves         mean cumulative failures vs t for the gc trio
  8. key_numbers_v1.json         machine-readable summary (memo source)

HARD GUARD (identical-sigma fingerprint, made permanent): if any scenario's
sorted per-run totals are identical between campaigns, the script prints a
loud error and exits nonzero — that fingerprint means power coupling was
silently inactive and the jesse campaign is invalid.

Run on the Mac (conda env `flood`), AFTER rsync-ing results off scratch:

    python scripts/compare_legacy_jesse_v1.py

Optional geoclaw wet-node counts for the S-curve (syn counts come from the
temp_nodes_*.geojson files automatically):

    python scripts/compare_legacy_jesse_v1.py \
        --gc-wet nyc_geoclaw_2026=NNN nyc_geoclaw_2050=NNN nyc_geoclaw_2080=NNN
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit

# ---------------------------------------------------------------- house style
C_LEGACY = "#9CA3AF"   # gray
C_JESSE = "#4F46E5"    # indigo
C_AMBER = "#F59E0B"
C_GREEN = "#10B981"
C_INK = "#111827"

plt.rcParams.update({
    "font.family": "Arial",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#D1D5DB",
    "axes.labelcolor": C_INK,
    "text.color": C_INK,
    "xtick.color": "#4B5563",
    "ytick.color": "#4B5563",
    "axes.grid": True,
    "grid.color": "#E5E7EB",
    "grid.linewidth": 0.6,
})

CRASH_THRESHOLDS = [1000, 1200, 1500]
CRASH_DEFAULT = 1200      # primary threshold; sensitivity via the other two
WET_EPS = 0.01            # metres; a node is "wet" if flood_depth_m > WET_EPS


# ------------------------------------------------------------------- loading
def scenario_tags(dirpath):
    files = sorted(glob.glob(os.path.join(dirpath, "cascade_results_*.json")))
    tags = []
    for f in files:
        base = os.path.basename(f)
        tags.append(base[len("cascade_results_"):-len(".json")])
    return tags


def load_runs(dirpath, tag):
    """Return dict of numpy arrays: totals, directs, cause_counts (per mech)."""
    path = os.path.join(dirpath, f"cascade_results_{tag}.json")
    with open(path) as f:
        runs = json.load(f)
    totals = np.array([r["total_failures"] for r in runs], dtype=float)
    directs = np.array([r["direct_failures"] for r in runs], dtype=float)
    mechs = {}
    for r in runs:
        for k, v in r.get("cause_counts", {}).items():
            mechs.setdefault(k, []).append(v)
    mech_mean = {k: float(np.mean(v)) for k, v in mechs.items()}
    return {"totals": totals, "directs": directs, "mech_mean": mech_mean,
            "n": len(runs)}


def wet_counts_from_temp_nodes(dirpath):
    """Wet-node count per syn scenario from temp_nodes_*.geojson (legacy dir).
    Depth resampling is deterministic, so legacy's copies are authoritative."""
    out = {}
    pat = re.compile(r"temp_nodes_(nyc_syn_ts_.+)\.geojson$")
    for f in glob.glob(os.path.join(dirpath, "temp_nodes_*.geojson")):
        m = pat.search(os.path.basename(f))
        if not m:
            continue
        tag = m.group(1)
        with open(f) as fh:
            gj = json.load(fh)
        wet = sum(1 for feat in gj["features"]
                  if (feat["properties"].get("flood_depth_m") or 0) > WET_EPS)
        out[tag] = wet
    return out


def load_timesteps(dirpath):
    """Per-scenario by_timestep dict from nyc_scenario_comparison.json."""
    path = os.path.join(dirpath, "nyc_scenario_comparison.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        comp = json.load(f)
    out = {}
    for key, blob in comp.items():
        ts = blob.get("by_timestep")
        if ts:
            out[key] = ts
    return out


# -------------------------------------------------------------------- guards
def identical_distribution_guard(tag, L, J):
    if len(L) == len(J) and np.array_equal(np.sort(L), np.sort(J)):
        return True
    return False


# ------------------------------------------------------------------ analysis
def per_scenario_stats(arr):
    return {
        "mean": float(np.mean(arr)),
        "sigma": float(np.std(arr, ddof=1)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def logistic(w, k, w0):
    return 1.0 / (1.0 + np.exp(-k * (w - w0)))


def fit_scurve(wets, probs):
    """Fit crash-prob vs wet-count logistic; return (k, w0) or None."""
    wets = np.asarray(wets, dtype=float)
    probs = np.asarray(probs, dtype=float)
    if len(wets) < 4 or probs.max() - probs.min() < 0.2:
        return None
    try:
        popt, _ = curve_fit(logistic, wets, probs,
                            p0=[0.05, float(np.median(wets))],
                            maxfev=20000)
        return tuple(float(x) for x in popt)
    except Exception:
        return None


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    home = os.path.expanduser("~")
    ap.add_argument("--legacy-dir", default=os.path.join(
        home, "Desktop/RA/data/hpc_results_aug2026/legacy_v1_n1000"))
    ap.add_argument("--jesse-dir", default=os.path.join(
        home, "Desktop/RA/data/hpc_results_aug2026/jesse_v1_n1000"))
    ap.add_argument("--out-dir", default=os.path.join(
        home, "Desktop/RA/analysis/legacy_vs_jesse_v1"))
    ap.add_argument("--gc-wet", nargs="*", default=[],
                    help="e.g. nyc_geoclaw_2026=315 (v1-graph wet counts)")
    args = ap.parse_args()

    for d in (args.legacy_dir, args.jesse_dir):
        if not os.path.isdir(d):
            sys.exit(f"ERROR: result dir not found: {d}")
    os.makedirs(args.out_dir, exist_ok=True)

    tags_L = set(scenario_tags(args.legacy_dir))
    tags_J = set(scenario_tags(args.jesse_dir))
    tags = sorted(tags_L & tags_J)
    print(f"legacy scenarios: {len(tags_L)} | jesse: {len(tags_J)} "
          f"| common: {len(tags)}")
    if len(tags) != 23:
        print(f"WARNING: expected 23 common scenarios, found {len(tags)} — "
              f"missing from jesse: {sorted(tags_L - tags_J)} | "
              f"missing from legacy: {sorted(tags_J - tags_L)}")
    if not tags:
        sys.exit("ERROR: no common scenarios.")

    gc_wet = {}
    for item in args.gc_wet:
        k, v = item.split("=")
        gc_wet[k] = int(v)
    wet = wet_counts_from_temp_nodes(args.legacy_dir)
    wet.update(gc_wet)

    rows, cause_rows, identical = [], [], []
    totals_store = {}

    for tag in tags:
        L = load_runs(args.legacy_dir, tag)
        J = load_runs(args.jesse_dir, tag)
        if L["n"] != 1000 or J["n"] != 1000:
            print(f"WARNING {tag}: run counts legacy={L['n']} jesse={J['n']}")

        if identical_distribution_guard(tag, L["totals"], J["totals"]):
            identical.append(tag)

        ks = stats.ks_2samp(L["totals"], J["totals"])
        sL = per_scenario_stats(L["totals"])
        sJ = per_scenario_stats(J["totals"])
        row = {"scenario": tag, "wet_nodes": wet.get(tag, np.nan),
               "ks_stat": float(ks.statistic), "ks_p": float(ks.pvalue),
               "sigma_ratio": sJ["sigma"] / sL["sigma"] if sL["sigma"] else np.nan,
               "direct_mean_legacy": float(np.mean(L["directs"])),
               "direct_mean_jesse": float(np.mean(J["directs"]))}
        for name, s in (("legacy", sL), ("jesse", sJ)):
            for k, v in s.items():
                row[f"{k}_{name}"] = v
        for thr in CRASH_THRESHOLDS:
            row[f"crash_frac_{thr}_legacy"] = float(np.mean(L["totals"] >= thr))
            row[f"crash_frac_{thr}_jesse"] = float(np.mean(J["totals"] >= thr))
        rows.append(row)
        totals_store[tag] = (L["totals"], J["totals"])

        mechs = sorted(set(L["mech_mean"]) | set(J["mech_mean"]))
        for m in mechs:
            cause_rows.append({"scenario": tag, "mechanism": m,
                               "legacy_mean": L["mech_mean"].get(m, 0.0),
                               "jesse_mean": J["mech_mean"].get(m, 0.0)})

    # ---------------------------------------------------------------- guard
    if identical:
        print("\n" + "=" * 70)
        print("FATAL — IDENTICAL-DISTRIBUTION GUARD TRIPPED")
        print("These scenarios have byte-identical run distributions across")
        print("campaigns, the fingerprint of power coupling being silently")
        print("inactive (the run_synthetic20 missing-4th-arg bug):")
        for t in identical:
            print(f"    {t}")
        print("The jesse campaign is INVALID for these scenarios. Fix and rerun")
        print("before any downstream analysis or GNN labeling.")
        print("=" * 70)
        sys.exit(2)
    print("identical-distribution guard: PASS "
          f"(all {len(tags)} scenarios differ; min KS stat "
          f"{min(r['ks_stat'] for r in rows):.4f})")

    df = pd.DataFrame(rows).sort_values("wet_nodes")
    df.to_csv(os.path.join(args.out_dir, "comparison_summary_v1.csv"),
              index=False)
    pd.DataFrame(cause_rows).to_csv(
        os.path.join(args.out_dir, "cause_counts_v1.csv"), index=False)

    syn = df[df["scenario"].str.contains("syn_ts")].dropna(subset=["wet_nodes"])

    # ------------------------------------------------- fig 1: sigma ratio
    fig, ax = plt.subplots(figsize=(11, 5))
    d1 = df.sort_values("wet_nodes")
    labels = [t.replace("nyc_", "").replace("syn_ts_", "s")
              for t in d1["scenario"]]
    colors = [C_AMBER if r >= 2 else C_JESSE for r in d1["sigma_ratio"]]
    ax.bar(range(len(d1)), d1["sigma_ratio"], color=colors)
    ax.axhline(1.0, color=C_INK, lw=1, ls="--")
    ax.set_xticks(range(len(d1)))
    ax.set_xticklabels(labels, rotation=75, fontsize=8)
    ax.set_ylabel("sigma jesse / sigma legacy")
    ax.set_title("Variance amplification under physics-coupled power seeding "
                 "(frozen v1 graph, N=1000)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "fig_variance_amplification.png"),
                dpi=200)
    plt.close(fig)

    # ------------------------------------------- fig 2: histogram grid
    n = len(tags)
    ncol = 5
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.6 * nrow))
    order = list(d1["scenario"])
    for ax, tag in zip(axes.flat, order):
        L, J = totals_store[tag]
        lo, hi = min(L.min(), J.min()), max(L.max(), J.max())
        bins = np.linspace(lo, hi + 1, 40)
        ax.hist(L, bins=bins, color=C_LEGACY, alpha=0.65, label="legacy")
        ax.hist(J, bins=bins, color=C_JESSE, alpha=0.55, label="jesse")
        w = wet.get(tag)
        ax.set_title(tag.replace("nyc_", "") +
                     (f"  (wet {w})" if w is not None else ""), fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axes.flat[n:]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Total-failure distributions per scenario, legacy vs jesse",
                 y=1.002)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "fig_histograms_grid.png"), dpi=170)
    plt.close(fig)

    # ------------------------------------------- fig 3: S-curve overlay
    fits = {}
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for name, col, color in (("legacy", f"crash_frac_{CRASH_DEFAULT}_legacy",
                              C_LEGACY),
                             ("jesse", f"crash_frac_{CRASH_DEFAULT}_jesse",
                              C_JESSE)):
        pts = df.dropna(subset=["wet_nodes"])
        ax.scatter(pts["wet_nodes"], pts[col], color=color, s=45, zorder=3,
                   label=f"{name} (per scenario)")
        fit = fit_scurve(pts["wet_nodes"], pts[col])
        fits[name] = fit
        if fit:
            k, w0 = fit
            wgrid = np.linspace(pts["wet_nodes"].min() * 0.8,
                                pts["wet_nodes"].max() * 1.1, 300)
            ax.plot(wgrid, logistic(wgrid, k, w0), color=color, lw=2)
            ax.axvline(w0, color=color, lw=1, ls=":")
            ax.annotate(f"{name} 50% @ {w0:.0f}", xy=(w0, 0.5),
                        xytext=(w0 + 8, 0.55 if name == "legacy" else 0.4),
                        fontsize=9, color=color)
    ax.set_xlabel(f"wet-node count (flood_depth_m > {WET_EPS})")
    ax.set_ylabel(f"P(total failures >= {CRASH_DEFAULT})")
    ax.set_title("Crash-probability S-curve vs storm footprint — "
                 "does the threshold move under physics seeding?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "fig_scurve_overlay.png"), dpi=200)
    plt.close(fig)

    # ------------------------------------------- fig 4: quantile shift
    fig, ax = plt.subplots(figsize=(11, 5))
    d2 = d1.reset_index(drop=True)
    for i, (_, r) in enumerate(d2.iterrows()):
        for q, color in (("p50", C_INK), ("p90", C_AMBER), ("p99", C_JESSE)):
            ax.annotate("", xy=(i, r[f"{q}_jesse"]),
                        xytext=(i, r[f"{q}_legacy"]),
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.4))
    handles = [plt.Line2D([0], [0], color=c, lw=2, label=l) for c, l in
               ((C_INK, "P50"), (C_AMBER, "P90"), (C_JESSE, "P99"))]
    ax.legend(handles=handles)
    ax.set_xticks(range(len(d2)))
    ax.set_xticklabels([t.replace("nyc_", "").replace("syn_ts_", "s")
                        for t in d2["scenario"]], rotation=75, fontsize=8)
    ax.set_ylabel("total failures")
    ax.set_title("Quantile shift legacy -> jesse (arrow tail = legacy, "
                 "head = jesse)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "fig_quantile_shift.png"), dpi=200)
    plt.close(fig)

    # ------------------------------------------- fig 5: timestep curves
    tsL = load_timesteps(args.legacy_dir)
    tsJ = load_timesteps(args.jesse_dir)
    gc_keys = [k for k in ("geoclaw_2026", "geoclaw_2050", "geoclaw_2080",
                           "nyc_geoclaw_2026", "nyc_geoclaw_2050",
                           "nyc_geoclaw_2080")
               if k in tsL and k in tsJ]
    if gc_keys:
        fig, ax = plt.subplots(figsize=(8, 5))
        tgrid = [0, 6, 24, 48, 96]
        for key in gc_keys:
            for src, color, ls in ((tsL, C_LEGACY, "--"), (tsJ, C_JESSE, "-")):
                ys = [src[key].get(f"t{t}", np.nan) for t in tgrid]
                ax.plot(tgrid, ys, color=color, ls=ls, marker="o", ms=4)
            ax.annotate(key.replace("nyc_", ""), xy=(96, tsJ[key]["t96"]),
                        fontsize=9, xytext=(97, tsJ[key]["t96"]))
        ax.plot([], [], color=C_LEGACY, ls="--", label="legacy")
        ax.plot([], [], color=C_JESSE, ls="-", label="jesse")
        ax.set_xlabel("hours after landfall")
        ax.set_ylabel("mean cumulative failures")
        ax.set_title("Cascade propagation, GeoClaw trio")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "fig_timestep_curves.png"),
                    dpi=200)
        plt.close(fig)

    # --------------------------------------------------- key numbers dump
    key = {
        "n_scenarios": len(tags),
        "guard": "PASS",
        "crash_threshold_primary": CRASH_DEFAULT,
        "scurve_fits": {k: (dict(k_slope=v[0], w0_50pct=v[1]) if v else None)
                        for k, v in fits.items()},
        "max_sigma_ratio": {
            "scenario": str(df.loc[df["sigma_ratio"].idxmax(), "scenario"]),
            "ratio": float(df["sigma_ratio"].max())},
        "median_sigma_ratio": float(df["sigma_ratio"].median()),
        "direct_drop_pct_mean": float(np.mean(
            100 * (df["direct_mean_jesse"] / df["direct_mean_legacy"] - 1))),
        "transition_band": df[df["scenario"].isin(
            ["nyc_syn_ts_321_19", "nyc_syn_ts_463_29",
             "nyc_syn_ts_436_11", "nyc_syn_ts_156_15"])][
            ["scenario", "sigma_legacy", "sigma_jesse", "sigma_ratio",
             f"crash_frac_{CRASH_DEFAULT}_legacy",
             f"crash_frac_{CRASH_DEFAULT}_jesse"]].to_dict("records"),
    }
    with open(os.path.join(args.out_dir, "key_numbers_v1.json"), "w") as f:
        json.dump(key, f, indent=2)

    # ------------------------------------------------------------- verdict
    print("\n================ VERDICT ================")
    print(f"scenarios analysed : {len(tags)}")
    print(f"guard              : PASS (no identical distributions)")
    print(f"direct-failure shift (mean over scenarios): "
          f"{key['direct_drop_pct_mean']:+.1f}%")
    print(f"sigma ratio        : median {key['median_sigma_ratio']:.2f}, "
          f"max {key['max_sigma_ratio']['ratio']:.2f} "
          f"({key['max_sigma_ratio']['scenario']})")
    for name, fit in fits.items():
        if fit:
            print(f"S-curve {name:6s}    : 50% crossing at "
                  f"{fit[1]:.0f} wet nodes (slope {fit[0]:.3f})")
        else:
            print(f"S-curve {name:6s}    : fit not converged / insufficient "
                  f"spread")
    print(f"outputs in         : {args.out_dir}")
    print("=========================================")


if __name__ == "__main__":
    main()
