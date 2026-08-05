#!/usr/bin/env python
"""
scripts/validate_handoff3.py — Phase-0 gate for Jesse's Tanmay_Handoff (23 surge scenarios).

Six gates, all must pass before the resampler rebuild touches this data:
  G1  Structure    : 23 scenario folders, 4 files each, mc_ prefixed names
  G2  Pct schema   : 12 expected columns, 175 rows/scenario, pct_failed in [0,100]
  G3  Run schema   : osi_node_failed_final present, 1,000 rows, convergence report
  G4  Consistency  : per-node failure counts recomputed from osi_node_failed_final
                     must equal n_times_failed EXACTLY (the 175/175 gate)
  G5  Crosswalk    : gwyn_<A>_<B> dirs pair 1:1 with flood_syn_ts_<A>_<B>_* columns
                     in synthetic20_node_depths.csv; gc trio matches flood_source.yaml
  G6  GOETHALS     : rescue columns present; per-scenario rescue counts reported;
                     S141 name printed (naming-collision awareness)

Usage (Mac):
  conda activate flood
  python scripts/validate_handoff3.py --handoff ~/Desktop/RA/Tanmay_Handoff \
      --repo ~/Desktop/RA/interdependency-cascade
Writes: <handoff>/validation_report_handoff3.csv and prints PASS/FAIL per gate.
"""
import argparse, os, re, sys
import pandas as pd

PCT_COLS = ["osi_node_id","osi_name","ward_pp_bus_idx","ward_tied_pp_bus_idxs",
            "own_hazus_failure_pct","flood_depth_m","n_times_failed",
            "n_converged_runs","pct_failed","n_times_own_flood_only",
            "n_times_ward_bus_only","n_times_both"]
RUN_REQUIRED = ["run_id","converged_overall","n_osi_node_failed_final",
                "osi_node_failed_final","goethals_power_model_failure",
                "goethals_tie_draw_failed","goethals_rescued"]
N_NODES, N_RUNS = 175, 1000

def fail(msg):
    print(f"  FAIL  {msg}")
    return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--handoff", required=True)
    ap.add_argument("--repo", required=True)
    args = ap.parse_args()
    H = os.path.expanduser(args.handoff)
    R = os.path.expanduser(args.repo)

    ok = {f"G{i}": True for i in range(1, 7)}
    rows = []

    # ---- G1: structure -------------------------------------------------------
    dirs = sorted(d for d in os.listdir(H) if os.path.isdir(os.path.join(H, d))
                  and (d.startswith("gwyn_") or d.startswith("gc_")))
    print(f"G1 structure: {len(dirs)} scenario folders")
    if len(dirs) != 23:
        ok["G1"] = fail(f"expected 23 scenario folders, found {len(dirs)}")
    for d in dirs:
        expect = {f"mc_element_failure_pct_{d}.csv", f"mc_failure_map_{d}.html",
                  f"mc_osi_node_failure_pct_{d}.csv", f"mc_run_summary_{d}.csv"}
        have = set(os.listdir(os.path.join(H, d)))
        if not expect <= have:
            ok["G1"] = fail(f"{d}: missing {sorted(expect - have)}")

    # ---- G2-G4: per-scenario checks -----------------------------------------
    for d in dirs:
        pct = pd.read_csv(os.path.join(H, d, f"mc_osi_node_failure_pct_{d}.csv"))
        run = pd.read_csv(os.path.join(H, d, f"mc_run_summary_{d}.csv"))

        # G2
        if list(pct.columns) != PCT_COLS:
            ok["G2"] = fail(f"{d}: pct columns differ: {list(pct.columns)}")
        if len(pct) != N_NODES:
            ok["G2"] = fail(f"{d}: {len(pct)} pct rows (expected {N_NODES})")
        if not pct["pct_failed"].between(0, 100).all():
            ok["G2"] = fail(f"{d}: pct_failed outside [0,100]")
        if (pct["flood_depth_m"] < 0).any():
            ok["G2"] = fail(f"{d}: negative flood_depth_m")

        # G3
        missing = [c for c in RUN_REQUIRED if c not in run.columns]
        if missing:
            ok["G3"] = fail(f"{d}: run summary missing {missing}")
            continue
        if len(run) != N_RUNS:
            ok["G3"] = fail(f"{d}: {len(run)} runs (expected {N_RUNS})")
        conv = run["converged_overall"].astype(str).str.lower().eq("true")
        n_conv = int(conv.sum())
        if n_conv < N_RUNS:
            print(f"  note  {d}: {N_RUNS - n_conv} non-converged runs")

        # G4: recompute counts from per-run lists over CONVERGED runs
        counts = {}
        for lst in run.loc[conv, "osi_node_failed_final"].fillna(""):
            for nid in str(lst).split(";"):
                nid = nid.strip()
                if nid:
                    counts[nid] = counts.get(nid, 0) + 1
        pct_idx = pct.set_index("osi_node_id")
        mism = 0
        for nid in pct_idx.index:
            expect_n = int(pct_idx.at[nid, "n_times_failed"])
            got_n = counts.get(nid, 0)
            if expect_n != got_n:
                mism += 1
                if mism <= 3:
                    print(f"  FAIL  {d}: {nid} recomputed {got_n} vs published {expect_n}")
        extra = set(counts) - set(pct_idx.index)
        if extra:
            ok["G4"] = fail(f"{d}: run lists contain unknown node ids {sorted(extra)[:5]}")
        if mism:
            ok["G4"] = fail(f"{d}: {mism}/{N_NODES} nodes mismatch")
        else:
            print(f"  G4 ok {d}: {N_NODES}/{N_NODES} exact  "
                  f"(conv {n_conv}/{N_RUNS}, mean pct {pct['pct_failed'].mean():.2f}, "
                  f"max {pct['pct_failed'].max():.1f})")

        rows.append(dict(scenario=d, n_converged=n_conv,
                         mean_pct=pct["pct_failed"].mean(),
                         max_pct=pct["pct_failed"].max(),
                         nodes_ge50=int((pct["pct_failed"] >= 50).sum()),
                         goethals_rescued=int(run["goethals_rescued"].astype(str)
                                              .str.lower().eq("true").sum())))

    # ---- G5: crosswalk -------------------------------------------------------
    depths = os.path.join(R, "data/flood/synthetic20_node_depths.csv")
    syn_pairs = set()
    with open(depths) as f:
        for c in f.readline().strip().split(","):
            m = re.match(r"flood_syn_ts_(\d+)_(\d+)_", c)
            if m:
                syn_pairs.add((m.group(1), m.group(2)))
    gwyn_pairs = {tuple(d.split("_")[1:3]) for d in dirs if d.startswith("gwyn_")}
    if len(gwyn_pairs) != 20:
        ok["G5"] = fail(f"{len(gwyn_pairs)} gwyn pairs (expected 20)")
    if gwyn_pairs != syn_pairs:
        ok["G5"] = fail(f"crosswalk mismatch: jesse-only {sorted(gwyn_pairs - syn_pairs)}, "
                        f"repo-only {sorted(syn_pairs - gwyn_pairs)}")
    else:
        print(f"G5 crosswalk: 20/20 gwyn<->syn pairs match 1:1")
    fs = open(os.path.join(R, "config/flood_source.yaml")).read()
    for gc in ["gc_2026", "gc_2050", "gc_2080"]:
        if gc not in dirs or gc not in fs:
            ok["G5"] = fail(f"gc scenario {gc} missing from handoff or flood_source.yaml")
    print("G5 gc trio: gc_2026/gc_2050/gc_2080 present in handoff and config")

    # ---- G6: GOETHALS + S141 awareness --------------------------------------
    rep = pd.DataFrame(rows)
    print("\nG6 GOETHALS rescues per scenario (informational):")
    print(rep[["scenario", "goethals_rescued"]].to_string(index=False))
    any_pct = pd.read_csv(os.path.join(H, dirs[0], f"mc_osi_node_failure_pct_{dirs[0]}.csv"))
    s141 = any_pct[any_pct.osi_node_id == "S141"]
    if len(s141):
        print(f"S141 osi_name = '{s141.iloc[0].osi_name}' "
              f"(ward bus 'Unknown129920' collision noted in README - resampler must join by osi_node_id)")

    out = os.path.join(H, "validation_report_handoff3.csv")
    rep.to_csv(out, index=False)
    print(f"\nreport -> {out}")
    print("\n" + " ".join(f"{g}:{'PASS' if v else 'FAIL'}" for g, v in ok.items()))
    sys.exit(0 if all(ok.values()) else 1)

if __name__ == "__main__":
    main()
