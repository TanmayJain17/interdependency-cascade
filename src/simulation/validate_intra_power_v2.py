#!/usr/bin/env python3
"""
validate_intra_power_v2.py — proves the UPDATED intra_power module reproduces
Jesse's published per-node failure rates exactly, for all 23 handoff-3 scenarios,
through the module's own decode path (not raw file reads — this catches module
bugs, not just data bugs).

Checks:
  V1  resolver: syn_ts_<A>_<B>_x -> gwyn_<A>_<B> for all 20 synthetics,
      geoclaw_2026/2050/2080 -> gc trio
  V2  exact-path flag: every scenario decodes via osi_node_failed_final
  V3  per-node frequency: for each scenario, count osi_dead() over all 1,000
      runs and compare with Jesse's n_times_failed — expect 175/175 exact
  V4  (optional, --nodes) join coverage: build_power_join stats on our graph

Usage (Mac):
  python src/simulation/validate_intra_power_v2.py \
      --handoff ~/Desktop/RA/Tanmay_Handoff \
      [--nodes data/graph/nyc_infra_nodes.geojson]
"""
import argparse, csv, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.simulation.intra_power import load_power_library, build_power_join  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--handoff", required=True)
    ap.add_argument("--nodes", default=None)
    args = ap.parse_args()
    H = os.path.expanduser(args.handoff)

    lib = load_power_library(H)
    ok = True

    # V1 — resolver
    gwyns = sorted(d for d in os.listdir(H) if d.startswith("gwyn_"))
    for d in gwyns:
        a, b = d.split("_")[1:3]
        got = lib.resolve(f"syn_ts_{a}_{b}_9p9999")
        if got != d:
            ok = False
            print(f"V1 FAIL resolver: syn_ts_{a}_{b}_* -> {got} (expected {d})")
    for ours, theirs in [("geoclaw_2026", "gc_2026"), ("geoclaw_2050", "gc_2050"),
                         ("geoclaw_2080", "gc_2080")]:
        if lib.resolve(ours) != theirs:
            ok = False
            print(f"V1 FAIL resolver: {ours} -> {lib.resolve(ours)}")
    if lib.resolve("dep_pluvial_extreme") is not None:
        ok = False
        print("V1 FAIL: unknown scenario should resolve to None")
    print(f"V1 resolver: 20 synthetics + gc trio + negative case "
          f"{'PASS' if ok else 'FAIL'}")

    # V2 + V3 — exact path + per-node frequency through the module
    all_scen = gwyns + ["gc_2026", "gc_2050", "gc_2080"]
    total_exact = 0
    for js in sorted(all_scen):
        runs = lib.runs(js)
        if not lib.used_osi_column.get(js):
            ok = False
            print(f"V2 FAIL {js}: osi_node_failed_final not detected")
            continue
        counts = {}
        for run in runs:
            for r in lib.crosswalk:
                if lib.osi_dead(run, r):
                    counts[r["osi_node_id"]] = counts.get(r["osi_node_id"], 0) + 1
        pct_path = os.path.join(H, js, f"mc_osi_node_failure_pct_{js}.csv")
        mism = 0
        with open(pct_path, newline="", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                if counts.get(row["osi_node_id"], 0) != int(row["n_times_failed"]):
                    mism += 1
        n = 175 - mism
        total_exact += n
        tag = "ok " if mism == 0 else "FAIL"
        if mism:
            ok = False
        print(f"V3 {tag} {js}: {n}/175 exact via module decode "
              f"({len(runs)} runs, exact-path={lib.used_osi_column[js]})")

    print(f"\nV3 total: {total_exact}/{23 * 175} node-scenario cells exact")

    # V4 — optional join stats
    if args.nodes:
        join = build_power_join(args.nodes, lib)
        uniq = len(set(join.values()))
        print(f"V4 join: {len(join)} power nodes -> {uniq} distinct OSI nodes "
              f"(of 175 in crosswalk)")

    print("VERDICT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
