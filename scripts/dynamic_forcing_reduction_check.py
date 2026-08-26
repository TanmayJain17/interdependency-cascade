#!/usr/bin/env python3
"""
dynamic_forcing_reduction_check.py — the exact-reduction gate for dynamic forcing.

A run with DYNAMIC_FORCING=1 DYNAMIC_MODE=static_peak must reproduce the frozen static campaign
record-for-record on every key the frozen schema has. Any difference is fatal (exit 2), in the
spirit of the identical-distribution guard in compare_legacy_jesse_v1.py.

Usage:
  python scripts/dynamic_forcing_reduction_check.py \
      --ref  data/hpc_results_aug2026/legacy_v1_n1000/cascade_results_nyc_syn_ts_914_6_1p2955.json \
      --test data/simulation/cascade_results_nyc_syn_ts_914_6_1p2955.json --first 20
"""
import argparse, json, sys

KEYS = ["direct_failures", "total_failures", "by_timestep", "failed_nodes_t96", "fail_time_per_node", "cause_counts"]


def norm(rec, k):
    v = rec[k]
    if k == "failed_nodes_t96":
        return sorted(v)
    if k == "fail_time_per_node":
        return {str(n): int(t) for n, t in v.items()}
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="frozen campaign cascade_results json")
    ap.add_argument("--test", required=True, help="static_peak dynamic-forcing cascade_results json")
    ap.add_argument("--first", type=int, default=None, help="compare only the first N runs")
    a = ap.parse_args()
    ref = json.load(open(a.ref)); test = json.load(open(a.test))
    n = min(len(ref), len(test)) if a.first is None else min(a.first, len(ref), len(test))
    if n == 0:
        sys.exit("no runs to compare")
    bad = 0
    for i in range(n):
        r, t = ref[i], test[i]
        for k in KEYS:
            if norm(r, k) != norm(t, k):
                bad += 1
                if bad <= 10:
                    if k == "fail_time_per_node":
                        rf, tf = norm(r, k), norm(t, k)
                        diff = [(nid, rf.get(nid), tf.get(nid)) for nid in set(rf) | set(tf) if rf.get(nid) != tf.get(nid)]
                        print(f"run {i}: {k} differs on {len(diff)} nodes, e.g. {diff[:4]}")
                    else:
                        print(f"run {i}: {k} ref={norm(r, k) if k != 'failed_nodes_t96' else len(norm(r, k))} "
                              f"test={norm(t, k) if k != 'failed_nodes_t96' else len(norm(t, k))}")
    fp_ref = sorted(r["total_failures"] for r in ref[:n]); fp_test = sorted(t["total_failures"] for t in test[:n])
    print(f"compared {n} runs | fingerprint (sorted totals) identical: {fp_ref == fp_test} | mismatching fields: {bad}")
    if bad:
        print("EXACT REDUCTION FAILED — dynamic code does not reduce to the frozen static run"); sys.exit(2)
    print("EXACT REDUCTION OK — static_peak reproduces the frozen campaign record-for-record")


if __name__ == "__main__":
    main()
