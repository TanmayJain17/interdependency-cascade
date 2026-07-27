#!/usr/bin/env python3
"""
compare_power_coupling.py — paired side-by-side: legacy power layer vs
Jesse-coupled (Week 19 advisor packet, companion to the Phase-0 table).

Both archives must come from runs with FRAGILITY_SEED=42 / BUFFER_SEED=43
(the runner defaults), which makes run i in the baseline and run i in the
coupled world share identical flood draws and buffer streams — every delta is
a deterministic consequence of the power coupling, so paired statistics apply.

The DEP scenarios double as a built-in identity check: the coupling never
touches them, so their per-run results must be EXACTLY equal across archives.
A nonzero DEP delta means the two archives were not produced under paired
seeds (or code drifted between runs) and every other number is suspect.

Usage (project root):
    python src/analysis/compare_power_coupling.py \
        data/simulation_side_by_side_baseline_100 \
        data/simulation_side_by_side_coupled_100
Writes: data/analysis/power_coupling_side_by_side.json
"""

import json
import math
import os
import sys
from collections import Counter, defaultdict

SCENARIOS_GC = ["geoclaw_2026", "geoclaw_2050", "geoclaw_2080"]
SCENARIOS_DEP = ["moderate_current", "moderate_2050", "extreme_2080"]
TYPES = ["power", "telecom", "hospital", "subway", "water", "fuel"]
OUT_PATH = "data/analysis/power_coupling_side_by_side.json"


def node_type(nid):
    return nid.split("_", 1)[0]


def load(dirpath, scen):
    p = os.path.join(dirpath, f"cascade_results_nyc_{scen}.json")
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return json.load(fh)


def per_type_t96(run):
    c = Counter(node_type(n) for n in run["failed_nodes_t96"])
    return {ty: c.get(ty, 0) for ty in TYPES}


def power_marginals(runs):
    n = len(runs)
    seed, t6, t96 = Counter(), Counter(), Counter()
    for r in runs:
        for nid, t in r["fail_time_per_node"].items():
            if not nid.startswith("power_"):
                continue
            t96[nid] += 1
            if t == 0:
                seed[nid] += 1
            if t <= 6:
                t6[nid] += 1
    return {"n": n,
            "seed": {k: v / n for k, v in seed.items()},
            "t6": {k: v / n for k, v in t6.items()},
            "t96": {k: v / n for k, v in t96.items()}}


def mean_std(xs):
    n = len(xs)
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / n
    return m, math.sqrt(var)


def compare_scenario(scen, base_runs, coup_runs):
    n = min(len(base_runs), len(coup_runs))
    base_runs, coup_runs = base_runs[:n], coup_runs[:n]

    d_direct = [c["direct_failures"] - b["direct_failures"]
                for b, c in zip(base_runs, coup_runs)]
    d_total = [c["total_failures"] - b["total_failures"]
               for b, c in zip(base_runs, coup_runs)]

    # per-cause mean counts
    cause_b, cause_c = Counter(), Counter()
    for b in base_runs:
        cause_b.update(b.get("cause_counts", {}))
    for c in coup_runs:
        cause_c.update(c.get("cause_counts", {}))
    causes = sorted(set(cause_b) | set(cause_c))
    cause_tbl = {cz: {"baseline": cause_b.get(cz, 0) / n,
                      "coupled": cause_c.get(cz, 0) / n,
                      "delta": (cause_c.get(cz, 0) - cause_b.get(cz, 0)) / n}
                 for cz in causes}

    # per-type t96 mean deltas (paired)
    type_d = defaultdict(list)
    for b, c in zip(base_runs, coup_runs):
        tb, tc = per_type_t96(b), per_type_t96(c)
        for ty in TYPES:
            type_d[ty].append(tc[ty] - tb[ty])
    type_tbl = {ty: dict(zip(("mean", "std"), (round(x, 2) for x in mean_std(type_d[ty]))),
                         up=sum(1 for x in type_d[ty] if x > 0),
                         down=sum(1 for x in type_d[ty] if x < 0))
                for ty in TYPES}

    # power-node marginal movers
    pm_b, pm_c = power_marginals(base_runs), power_marginals(coup_runs)
    all_p = set(pm_b["t96"]) | set(pm_c["t96"])
    movers = sorted(
        ({"node": p,
          "base_seed": round(pm_b["seed"].get(p, 0.0), 3),
          "coup_seed": round(pm_c["seed"].get(p, 0.0), 3),
          "base_t96": round(pm_b["t96"].get(p, 0.0), 3),
          "coup_t96": round(pm_c["t96"].get(p, 0.0), 3)}
         for p in all_p),
        key=lambda r: abs(r["coup_t96"] - r["base_t96"]), reverse=True)

    md, sd = mean_std(d_direct)
    mt, st = mean_std(d_total)
    out = {
        "n_pairs": n,
        "direct_delta": {"mean": round(md, 2), "std": round(sd, 2)},
        "total_delta": {"mean": round(mt, 2), "std": round(st, 2),
                        "up": sum(1 for x in d_total if x > 0),
                        "down": sum(1 for x in d_total if x < 0),
                        "zero": sum(1 for x in d_total if x == 0)},
        "per_cause": cause_tbl,
        "per_type_t96_delta": type_tbl,
        "power_movers_top15": movers[:15],
        "power_mean_dead_t96": {
            "baseline": round(sum(pm_b["t96"].values()), 2),
            "coupled": round(sum(pm_c["t96"].values()), 2)},
    }

    print(f"\n=== {scen} ({n} paired runs) ===")
    print(f"  direct: {md:+.2f} ± {sd:.2f}   total: {mt:+.2f} ± {st:.2f}  "
          f"(up {out['total_delta']['up']} / down {out['total_delta']['down']} "
          f"/ zero {out['total_delta']['zero']})")
    print(f"  power dead@t96 (mean/run): {out['power_mean_dead_t96']['baseline']} "
          f"-> {out['power_mean_dead_t96']['coupled']}")
    print(f"  {'cause':16s}{'base':>10s}{'coupled':>10s}{'delta':>10s}")
    for cz in causes:
        r = cause_tbl[cz]
        print(f"  {cz:16s}{r['baseline']:10.1f}{r['coupled']:10.1f}{r['delta']:+10.1f}")
    print(f"  per-type t96 delta (mean ± std, paired):")
    for ty in TYPES:
        r = type_tbl[ty]
        print(f"    {ty:9s}{r['mean']:+8.2f} ± {r['std']:6.2f}   "
              f"(up {r['up']}/dn {r['down']})")
    print(f"  top power movers (t96 marginal):")
    for r in movers[:8]:
        print(f"    {r['node']:38s} seed {r['base_seed']:.2f}->{r['coup_seed']:.2f}"
              f"  t96 {r['base_t96']:.2f}->{r['coup_t96']:.2f}")
    return out


def identity_check(scen, base_runs, coup_runs):
    n = min(len(base_runs), len(coup_runs))
    diffs = sum(1 for b, c in zip(base_runs[:n], coup_runs[:n])
                if b["fail_time_per_node"] != c["fail_time_per_node"])
    ok = diffs == 0
    print(f"  [{'PASS' if ok else 'FAIL'}] {scen}: "
          f"{diffs}/{n} paired runs differ (must be 0)")
    return {"n": n, "runs_differing": diffs, "pass": ok}


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    base_dir, coup_dir = sys.argv[1], sys.argv[2]
    report = {"baseline_dir": base_dir, "coupled_dir": coup_dir,
              "identity_check_dep": {}, "gc": {}}

    print("=== DEP identity check (coupling never touches these; "
          "any diff = unpaired archives) ===")
    all_ok = True
    for scen in SCENARIOS_DEP:
        b, c = load(base_dir, scen), load(coup_dir, scen)
        if b is None or c is None:
            print(f"  [SKIP] {scen}: missing in one archive")
            continue
        res = identity_check(scen, b, c)
        report["identity_check_dep"][scen] = res
        all_ok &= res["pass"]
    if not all_ok:
        print("\nHARD STOP: archives are not seed-paired — "
              "gc deltas would be uninterpretable.")
        sys.exit(1)

    for scen in SCENARIOS_GC:
        b, c = load(base_dir, scen), load(coup_dir, scen)
        if b is None or c is None:
            print(f"\n[SKIP] {scen}: missing in one archive")
            continue
        report["gc"][scen] = compare_scenario(scen, b, c)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as fh:
        json.dump(report, fh, indent=1)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
