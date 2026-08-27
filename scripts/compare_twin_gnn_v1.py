#!/usr/bin/env python3
"""
compare_twin_gnn_v1.py — legacy vs jesse CascadeGNN twins from their history.json files.

Reads  <results>/{legacy,jesse}/history.json (schema: epochs[], test_per_scenario{}, holdout_scenarios, scenario_set)
Writes <out>/twin_heldout_v1.csv, <out>/twin_heldout_v1.md, <out>/twin_val_curves_v1.csv,
       <out>/fig_twin_cascade_pr_v1.png, <out>/fig_twin_val_loss_v1.png

Usage: python scripts/compare_twin_gnn_v1.py --results data/hpc_results_aug2026/gnn_twin_v1 --out analysis/gnn_twin_v1
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

HORIZONS = [6, 24, 48, 96]
ARMS = ["legacy", "jesse"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--sigma", default=None, help="optional csv with columns scenario,sigma_ratio to join (Week 22 variance amplification)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    hist = {arm: json.load(open(Path(a.results) / arm / "history.json")) for arm in ARMS}

    # ---- validation curves
    vrows = []
    for arm in ARMS:
        for e in hist[arm]["epochs"]:
            r = dict(arm=arm, epoch=e["epoch"], train_loss=e["train_loss"], val_loss=e["val_loss"], elapsed_s=e.get("elapsed_s"))
            for i, t in enumerate(HORIZONS):
                r[f"val_cascade_pr_t{t}"] = e["val_cascade_pr_per_t"][i]; r[f"val_pr_t{t}"] = e["val_pr_per_t"][i]
            vrows.append(r)
    val = pd.DataFrame(vrows); val.to_csv(out / "twin_val_curves_v1.csv", index=False)
    best = {arm: val[val.arm == arm].sort_values("val_loss").iloc[0] for arm in ARMS}

    # ---- held-out table
    rows = []
    scen = hist["legacy"]["holdout_scenarios"]
    for sc in scen:
        r = dict(scenario=sc, n_runs=hist["legacy"]["test_per_scenario"][sc]["n_runs"])
        for arm in ARMS:
            m = hist[arm]["test_per_scenario"][sc]
            r[f"loss_{arm}"] = m["loss"]
            for i, t in enumerate(HORIZONS):
                r[f"cascade_pr_t{t}_{arm}"] = m["cascade_pr_per_t"][i]; r[f"pr_t{t}_{arm}"] = m["pr_per_t"][i]
                r[f"cascade_auc_t{t}_{arm}"] = m["cascade_auc_per_t"][i]
        r["loss_ratio_jesse_over_legacy"] = r["loss_jesse"] / r["loss_legacy"]
        for t in HORIZONS:
            r[f"cascade_pr_t{t}_delta_pp"] = 100 * (r[f"cascade_pr_t{t}_jesse"] - r[f"cascade_pr_t{t}_legacy"])
        r["cascade_pr_mean_legacy"] = np.mean([r[f"cascade_pr_t{t}_legacy"] for t in HORIZONS])
        r["cascade_pr_mean_jesse"] = np.mean([r[f"cascade_pr_t{t}_jesse"] for t in HORIZONS])
        rows.append(r)
    df = pd.DataFrame(rows)
    if a.sigma:
        s = pd.read_csv(a.sigma); df = df.merge(s, on="scenario", how="left")
    df.to_csv(out / "twin_heldout_v1.csv", index=False)

    # ---- markdown table (what goes on the slide)
    md = ["| held-out | loss legacy | loss jesse | ratio | cascade PR-AUC legacy (t6/24/48/96) | cascade PR-AUC jesse | Δ pp (t6/24/48/96) |", "|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        L = "/".join(f"{r[f'cascade_pr_t{t}_legacy']:.3f}" for t in HORIZONS); J = "/".join(f"{r[f'cascade_pr_t{t}_jesse']:.3f}" for t in HORIZONS)
        D = "/".join(f"{r[f'cascade_pr_t{t}_delta_pp']:+.1f}" for t in HORIZONS)
        md.append(f"| {r.scenario} | {r.loss_legacy:.4f} | {r.loss_jesse:.4f} | {r.loss_ratio_jesse_over_legacy:.2f} | {L} | {J} | {D} |")
    md.append("")
    md.append(f"Best validation loss: legacy {best['legacy'].val_loss:.5f} (epoch {int(best['legacy'].epoch)}), jesse {best['jesse'].val_loss:.5f} (epoch {int(best['jesse'].epoch)}). "
              f"Task: given the direct (t=0) failure mask as input, predict failure by t in {HORIZONS}; cascade-only metrics score nodes that were not direct failures. "
              f"Scenario set {hist['legacy']['scenario_set']}, 5-way scenario holdout, identical seed/architecture/graph across arms.")
    (out / "twin_heldout_v1.md").write_text("\n".join(md))
    print("\n".join(md))

    # ---- figures
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(scen), figsize=(3.2 * len(scen), 3.4), sharey=True)
        for ax, sc in zip(np.atleast_1d(axes), scen):
            r = df[df.scenario == sc].iloc[0]
            ax.plot(HORIZONS, [r[f"cascade_pr_t{t}_legacy"] for t in HORIZONS], "o-", color="#4F46E5", label="legacy (HAZUS)")
            ax.plot(HORIZONS, [r[f"cascade_pr_t{t}_jesse"] for t in HORIZONS], "s--", color="#F59E0B", label="jesse (grid physics)")
            ax.set_title(sc.replace("syn_ts_", "syn ").replace("_1p", " 1.").replace("_3p", " 3."), fontsize=9); ax.set_xlabel("horizon (h)"); ax.set_xticks(HORIZONS); ax.grid(alpha=.3)
        np.atleast_1d(axes)[0].set_ylabel("cascade-only PR-AUC (held-out)"); np.atleast_1d(axes)[0].legend(fontsize=8, loc="lower left")
        fig.tight_layout(); fig.savefig(out / "fig_twin_cascade_pr_v1.png", dpi=200); plt.close(fig)
        fig, ax = plt.subplots(figsize=(5, 3.4))
        for arm, c in zip(ARMS, ["#4F46E5", "#F59E0B"]):
            v = val[val.arm == arm]; ax.plot(v.epoch, v.val_loss, "o-", color=c, label=f"{arm} val"); ax.plot(v.epoch, v.train_loss, ":", color=c, label=f"{arm} train")
        ax.set_xlabel("epoch"); ax.set_ylabel("BCE loss"); ax.grid(alpha=.3); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(out / "fig_twin_val_loss_v1.png", dpi=200); plt.close(fig)
        print(f"figures written to {out}")
    except ImportError:
        print("matplotlib not available: tables written, figures skipped")


if __name__ == "__main__":
    main()
