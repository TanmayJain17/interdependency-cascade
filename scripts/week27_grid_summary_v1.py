#!/usr/bin/env python3
"""
scripts/week27_grid_summary_v1.py — Week 27 grid-convergence summary (880_9, 1-h vs 3-h vs 6-h).

Reads cascade_results_nyc_<scenario>.json from named result dirs and prints, per arm:
  N runs, t360 mean ± sd, super-cascade share (t360 >= 3000), contained-mode mean (runs < 3000),
  crash-decision counts (>= 1000 / 1200 / 1500), t0 mean
and, per requested pair test:ref, the PAIRED delta (same run index = same seeds): mean ± sd, up/down counts.

Usage (repo root):
  python scripts/week27_grid_summary_v1.py --scenario syn_ts_880_9_2p6632 \
      --arm S3=data/simulation_dyn_check/w25_translation_v1/S3 --arm U3=data/simulation_dyn_check/w25_translation_v1/U3 \
      --arm S1=data/simulation_dyn_check/w27_grid_v1/S1 --arm U1=data/simulation_dyn_check/w27_grid_v1/U1 \
      --pair U1:S1 --pair S1:S3 --pair U1:U3 --pair U3:S3 [--out analysis/.../week27_grid_table_v1.csv]
"""
import argparse, csv, json, statistics as st
from pathlib import Path

SUPER = 3000
CRASH = (1000, 1200, 1500)


def load(path, scenario):
    fp = Path(path) / f"cascade_results_nyc_{scenario}.json"
    runs = json.load(open(fp))
    runs = runs if isinstance(runs, list) else runs.get("runs", runs)
    return runs


def t_at(run, key):
    bt = run.get("by_timestep", {})
    if key in bt:
        return float(bt[key])
    if key == "t360":
        return float(run.get("total_failures_last_horizon", run.get("total_failures")))
    raise KeyError(key)


def summarise(name, runs):
    end = [t_at(r, "t360") for r in runs]
    t0 = [t_at(r, "t0") for r in runs]
    contained = [e for e in end if e < SUPER]
    row = {
        "arm": name, "n": len(runs),
        "t360_mean": st.mean(end), "t360_sd": st.pstdev(end) if len(end) > 1 else 0.0,
        "super_share": sum(e >= SUPER for e in end) / len(end),
        "contained_mean": st.mean(contained) if contained else float("nan"),
        "t0_mean": st.mean(t0),
    }
    for c in CRASH:
        row[f"crash_ge{c}"] = sum(e >= c for e in end)
    return row, end


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--arm", action="append", required=True, help="NAME=path (repeatable)")
    ap.add_argument("--pair", action="append", default=[], help="TEST:REF (repeatable)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    ends, rows = {}, []
    for spec in a.arm:
        name, path = spec.split("=", 1)
        runs = load(path, a.scenario)
        row, end = summarise(name, runs)
        rows.append(row); ends[name] = end

    print(f"{'arm':<6}{'n':>5}{'t360 mean':>12}{'sd':>9}{'super':>8}{'contained':>11}{'t0':>8}   crash>=1000/1200/1500")
    for r in rows:
        print(f"{r['arm']:<6}{r['n']:>5}{r['t360_mean']:>12.2f}{r['t360_sd']:>9.2f}{r['super_share']:>8.2f}"
              f"{r['contained_mean']:>11.1f}{r['t0_mean']:>8.1f}   {r['crash_ge1000']}/{r['crash_ge1200']}/{r['crash_ge1500']}")

    pair_rows = []
    for spec in a.pair:
        test, ref = spec.split(":")
        n = min(len(ends[test]), len(ends[ref]))
        d = [ends[test][i] - ends[ref][i] for i in range(n)]
        up, down = sum(x > 0 for x in d), sum(x < 0 for x in d)
        pr = {"pair": f"{test} - {ref}", "n": n, "delta_mean": st.mean(d), "delta_sd": st.pstdev(d) if n > 1 else 0.0,
              "up": up, "down": down, "same": n - up - down}
        pair_rows.append(pr)
        print(f"paired {pr['pair']:<10} n={n:<4} delta = {pr['delta_mean']:+8.2f} ± {pr['delta_sd']:7.2f}   up/down/same = {up}/{down}/{pr['same']}")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["arm", "n", "t360_mean", "t360_sd", "super_share", "contained_mean", "t0_mean", "crash_ge1000", "crash_ge1200", "crash_ge1500"])
            for r in rows:
                w.writerow([r[k] for k in ["arm", "n", "t360_mean", "t360_sd", "super_share", "contained_mean", "t0_mean", "crash_ge1000", "crash_ge1200", "crash_ge1500"]])
            w.writerow([])
            w.writerow(["pair", "n", "delta_mean", "delta_sd", "up", "down", "same"])
            for r in pair_rows:
                w.writerow([r[k] for k in ["pair", "n", "delta_mean", "delta_sd", "up", "down", "same"]])
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
