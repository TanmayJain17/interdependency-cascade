#!/usr/bin/env python3
"""
scripts/week27_phase0a_table_v1.py — Phase 0a verdict table: hazard head (gnn_hazard_v1) vs independent head
(2x2 v1 checkpoints re-evaluated by audit_gnn_2x2_mono_v1) against the pre-registration.

Reads (repo root, after mirroring from Torch):
  data/hpc_results_aug2026/gnn_hazard_v1/<cell>/history.json
  data/hpc_results_aug2026/gnn_audit_mono_v1/<cell>.eval.json
Writes analysis/gnn/week27_phase0a_hazard_vs_independent_v1.csv (per-cell block, then held-out per-storm block)
and prints the same. P1 |delta best val| <= 0.005; P2 hazard mono_viol == 0; P3 held-out order unchanged;
P4 cascade-only PR-AUC per horizon |delta| <= 0.01.
"""
import argparse, csv, json
from pathlib import Path

CELLS = ["static_mask", "static_timing", "dynamic_mask", "dynamic_timing"]
BASE_BEST = {"static_mask": 0.1082, "static_timing": 0.0981, "dynamic_mask": 0.1877, "dynamic_timing": 0.1653}  # 2x2 v1 / twin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hazard", default="data/hpc_results_aug2026/gnn_hazard_v1")
    ap.add_argument("--audit", default="data/hpc_results_aug2026/gnn_audit_mono_v1")
    ap.add_argument("--out", default="analysis/gnn/week27_phase0a_hazard_vs_independent_v1.csv")
    a = ap.parse_args()

    rows, hold = [], []
    for c in CELLS:
        hz = json.load(open(Path(a.hazard) / c / "history.json"))
        au = json.load(open(Path(a.audit) / f"{c}.eval.json"))
        best = min(hz["epochs"], key=lambda e: e["val_loss"])
        d = best["val_loss"] - au["val"]["loss"]
        pr_d = max(abs(h - i) for h, i in zip(best["val_cascade_pr_per_t"], au["val"]["cascade_pr_per_t"]))
        t_h, t_i = hz["test_per_scenario"], au["test_per_scenario"]
        order_same = sorted(t_h, key=lambda s: t_h[s]["loss"]) == sorted(t_i, key=lambda s: t_i[s]["loss"])
        rows.append({
            "cell": c, "indep_best_val": round(au["val"]["loss"], 5), "indep_best_val_2x2": BASE_BEST[c],
            "hazard_best_val": round(best["val_loss"], 5), "hazard_best_epoch": best["epoch"],
            "delta": round(d, 5), "P1_no_harm": abs(d) <= 0.005,
            "indep_mono_viol": round(au["val"]["mono_viol"], 5), "hazard_mono_viol": round(best["val_mono_viol"], 5),
            "P2_monotone": best["val_mono_viol"] == 0.0,
            "P3_holdout_order_same": order_same,
            "cascade_pr_max_abs_delta": round(pr_d, 4), "P4_pr_within_0p01": pr_d <= 0.01,
            "hazard_cascade_pr": " ".join(f"{x:.3f}" for x in best["val_cascade_pr_per_t"]),
            "indep_cascade_pr": " ".join(f"{x:.3f}" for x in au["val"]["cascade_pr_per_t"]),
        })
        for s in sorted(t_h, key=lambda s: t_h[s]["loss"]):
            hold.append({"cell": c, "storm": s, "hazard_loss": round(t_h[s]["loss"], 5), "indep_loss": round(t_i[s]["loss"], 5),
                         "delta": round(t_h[s]["loss"] - t_i[s]["loss"], 5),
                         "hazard_mono_viol": round(t_h[s]["mono_viol"], 5), "indep_mono_viol": round(t_i[s]["mono_viol"], 5),
                         "cascade_pr_max_abs_delta": round(max(abs(h - i) for h, i in zip(t_h[s]["cascade_pr_per_t"], t_i[s]["cascade_pr_per_t"])), 4)})

    for r in rows:
        print(f"{r['cell']:15s} indep {r['indep_best_val']:.5f} hazard {r['hazard_best_val']:.5f} delta {r['delta']:+.5f} "
              f"P1 {r['P1_no_harm']} | mono {r['indep_mono_viol']:.4f} -> {r['hazard_mono_viol']:.4f} P2 {r['P2_monotone']} | "
              f"P3 {r['P3_holdout_order_same']} | PR max|d| {r['cascade_pr_max_abs_delta']:.4f} P4 {r['P4_pr_within_0p01']}")
    allpass = all(r["P1_no_harm"] and r["P2_monotone"] and r["P3_holdout_order_same"] and r["P4_pr_within_0p01"] for r in rows)
    print("ALL PRE-REGISTERED CRITERIA:", "PASS" if allpass else "FAIL")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        f.write("\n")
        w2 = csv.DictWriter(f, fieldnames=list(hold[0].keys())); w2.writeheader(); w2.writerows(hold)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
