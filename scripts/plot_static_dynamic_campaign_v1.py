#!/usr/bin/env python3
"""
plot_static_dynamic_campaign_v1.py — campaign view of the paired static-vs-dynamic comparison.

Inputs : the CSV written by compare_static_dynamic_v1.py (one row per scenario) and the resolved-map table
         (scenario_tag, peak_water_level_m_MSL, wet_nodes_v1). GeoClaw scenarios get peak 3.48 m (Sandy NOAA) and
         their wet counts from the timing summary if present.
Outputs: <out>/campaign_table_v1.md, <out>/fig_timing_effect_vs_level_v1.png, <out>/fig_crash_fraction_v1.png

Usage:
  python scripts/plot_static_dynamic_campaign_v1.py --compare analysis/static_vs_dynamic_v1/campaign_arrival_vs_staticpeak_h360_n1000_v1.csv \
      --maps analysis/TwinCampaign_20maps_resolved_v1.csv --timing-summary data/flood/timing/node_timing_jesse22_v1_summary.csv \
      --out analysis/static_vs_dynamic_v1
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

GC_PEAK = 3.48


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", required=True); ap.add_argument("--maps", required=True)
    ap.add_argument("--timing-summary", default=None); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cmp_ = pd.read_csv(a.compare)
    maps = pd.read_csv(a.maps)
    maps["scenario"] = maps["depth_column"].str.replace("^flood_", "", regex=True).str.replace("_depth_m$", "", regex=True)
    lvl = dict(zip(maps["scenario"], maps["peak_water_level_m_MSL"])); wet = dict(zip(maps["scenario"], maps["wet_nodes_v1"]))
    if a.timing_summary:
        ts = pd.read_csv(a.timing_summary)
        for _, r in ts.iterrows():
            wet.setdefault(r["scenario"], int(r["wet_nodes"]))
    cmp_["peak_level_m"] = cmp_["scenario"].map(lambda s: lvl.get(s, GC_PEAK if s.startswith("geoclaw") else np.nan))
    cmp_["wet_nodes"] = cmp_["scenario"].map(wet)
    cmp_["is_geoclaw"] = cmp_["scenario"].str.startswith("geoclaw")
    for h in (96, 360):
        cmp_[f"rel_delta_t{h}_pct"] = 100 * cmp_[f"delta_t{h}"] / cmp_[f"static_t{h}"]
    cmp_ = cmp_.sort_values("peak_level_m")
    cols = ["scenario", "peak_level_m", "wet_nodes", "t_peak_h", "mean_prepeak_failures_dynamic", "static_t0", "dynamic_t0",
            "delta_t6", "delta_t96", "rel_delta_t96_pct", "delta_t360", "rel_delta_t360_pct", "median_shift_h",
            "runs_dynamic_larger", "crash_frac_static_1200", "crash_frac_dynamic_1200"]
    cols = [c for c in cols if c in cmp_.columns]
    tbl = cmp_[cols].copy()
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in tbl.iterrows():
        md.append("| " + " | ".join(f"{r[c]:.1f}" if isinstance(r[c], (float, np.floating)) else str(r[c]) for c in cols) + " |")
    (out / "campaign_table_v1.md").write_text("\n".join(md)); print("\n".join(md))
    cmp_.to_csv(out / "campaign_table_v1.csv", index=False)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        syn = cmp_[~cmp_.is_geoclaw]; gc = cmp_[cmp_.is_geoclaw]
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
        ax = axes[0]
        ax.plot(syn.peak_level_m, syn.rel_delta_t96_pct, "o-", color="#4F46E5", label="peak + 96 h")
        ax.plot(syn.peak_level_m, syn.rel_delta_t360_pct, "s--", color="#F59E0B", label="peak + 360 h")
        if len(gc):
            ax.scatter(gc.peak_level_m, gc.rel_delta_t96_pct, marker="^", color="#4F46E5", s=60, label="GeoClaw trio (t96)")
            ax.scatter(gc.peak_level_m, gc.rel_delta_t360_pct, marker="^", color="#F59E0B", s=60, label="GeoClaw trio (t360)")
        for x, lab in ((2.17, "156 wet"), (2.45, "218"), (3.08, "390")):
            ax.axvline(x, color="grey", lw=0.8, ls=":"); ax.text(x, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 1, lab, fontsize=7, rotation=90, va="top", ha="right", color="grey")
        ax.axhline(0, color="k", lw=0.6); ax.set_xlabel("peak water level at The Battery (m, MSL)"); ax.set_ylabel("failures, dynamic − static (% of static)")
        ax.set_title("timing effect vs storm size", fontsize=10); ax.grid(alpha=.3); ax.legend(fontsize=7)
        ax = axes[1]
        ax.plot(syn.peak_level_m, syn.crash_frac_static_1200, "o-", color="#4F46E5", label="static at peak")
        ax.plot(syn.peak_level_m, syn.crash_frac_dynamic_1200, "s--", color="#F59E0B", label="arrival-ordered")
        if len(gc):
            ax.scatter(gc.peak_level_m, gc.crash_frac_static_1200, marker="^", color="#4F46E5", s=60); ax.scatter(gc.peak_level_m, gc.crash_frac_dynamic_1200, marker="^", color="#F59E0B", s=60)
        ax.set_xlabel("peak water level at The Battery (m, MSL)"); ax.set_ylabel("P(crash): ≥1,200 failures by peak + 96 h")
        ax.set_title("crash decision", fontsize=10); ax.grid(alpha=.3); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(out / "fig_timing_effect_vs_level_v1.png", dpi=200); plt.close(fig)
        fig, ax = plt.subplots(figsize=(5, 3.4))
        ax.plot(syn.peak_level_m, syn.mean_prepeak_failures_dynamic, "o-", color="#F59E0B", label="failures before the peak (dynamic)")
        ax.plot(syn.peak_level_m, syn.static_t0, "o-", color="#4F46E5", label="direct failures (static seeds)")
        if len(gc):
            ax.scatter(gc.peak_level_m, gc.mean_prepeak_failures_dynamic, marker="^", color="#F59E0B", s=60); ax.scatter(gc.peak_level_m, gc.static_t0, marker="^", color="#4F46E5", s=60)
        ax.set_xlabel("peak water level at The Battery (m, MSL)"); ax.set_ylabel("nodes per run"); ax.grid(alpha=.3); ax.legend(fontsize=7)
        ax.set_title("what has already failed when the static model starts", fontsize=9)
        fig.tight_layout(); fig.savefig(out / "fig_prepeak_failures_v1.png", dpi=200); plt.close(fig)
        print(f"figures written to {out}")
    except ImportError:
        print("matplotlib not available: table written, figures skipped")


if __name__ == "__main__":
    main()
