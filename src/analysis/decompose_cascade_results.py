#!/usr/bin/env python3
"""
decompose_cascade_results.py — Week 17 post-hoc analysis (no re-simulation).

Two jobs in one pass over the saved Monte Carlo outputs:

1. TELECOM SANITY (flood-realistic validation of the Week-17 intra-telecom
   mechanism): per scenario, telecom deaths split by cause — flood seed,
   inter (power/backhaul dependency), intra crash — and as % of the 4,150-
   site layer. Sandy anchor: FCC reported ~25% of cell sites down across
   the whole 158-county reporting area (worst counties >50%); geoclaw_2026
   is the closest Sandy analog, so its telecom share should sit near that
   band, and rain (DEP) scenarios should sit well below it.

2. AMPLIFICATION DECOMPOSITION (Dr. Miura's trend question): amplification
   A = total/seeds = 1 + cascade/seeds is a RATIO, so it conflates seed
   growth with cascade propensity. Report the pieces separately:
     N_seed, N_cascade, N_total, A, and
     cascade_fraction = N_cascade / (N_seedable - N_seed)
   i.e. of the nodes the flood spared, what share the cascade still claimed.
   cascade_fraction has no denominator artifact and is the honest
   cross-severity comparison.

Reads:  data/simulation/monte_carlo_failures_nyc_{scenario}.json  (seeds/run)
        data/simulation/cascade_results_nyc_{scenario}.json       (t96, causes)
        data/graph/nyc_infra_nodes.geojson                        (node types)
Run:    python src/analysis/decompose_cascade_results.py
"""

import json
import sys
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[2]
SIM = ROOT / "data/simulation"
NODES = ROOT / "data/graph/nyc_infra_nodes.geojson"

SCENARIOS = ["moderate_current", "moderate_2050", "extreme_2080",
             "geoclaw_2026", "geoclaw_2050", "geoclaw_2080"]
CAUSE_KEYS = ["inter", "intra_subway", "intra_telecom", "intra_fuel",
              "intra_water", "intra_hospital"]


def load_types():
    with open(NODES) as f:
        feats = json.load(f)["features"]
    types = {f["properties"]["node_id"]: f["properties"]["infra_type"]
             for f in feats}
    external = {f["properties"]["node_id"] for f in feats
                if f["properties"].get("external")}
    return types, external


def main():
    types, external = load_types()
    n_total_nodes = len(types)
    n_seedable = n_total_nodes - len(external)   # NJ/terminal nodes can't seed
    n_telecom = sum(1 for t in types.values() if t == "telecom")
    print(f"Nodes: {n_total_nodes} total, {n_seedable} seedable "
          f"(external excluded), {n_telecom} telecom\n")

    hdr = (f"{'scenario':<17} {'seeds':>7} {'cascade':>8} {'total':>7} "
           f"{'A':>6} {'casc_frac':>9} | {'inter':>6} {'i_sub':>6} "
           f"{'i_tel':>6} {'i_fuel':>6} {'i_wat':>6} {'i_hosp':>6}")
    print(hdr)
    print("-" * len(hdr))

    tel_rows = []
    for sc in SCENARIOS:
        mc_p = SIM / f"monte_carlo_failures_nyc_{sc}.json"
        cr_p = SIM / f"cascade_results_nyc_{sc}.json"
        if not (mc_p.exists() and cr_p.exists()):
            print(f"{sc:<17} MISSING — run multi_scenario_runner first")
            continue
        with open(mc_p) as f:
            mc = json.load(f)
        with open(cr_p) as f:
            cr = json.load(f)
        if len(mc) != len(cr):
            sys.exit(f"STOP: {sc} run-count mismatch mc={len(mc)} cr={len(cr)}")

        seeds_l, casc_l, tot_l, cf_l = [], [], [], []
        cause_l = {k: [] for k in CAUSE_KEYS}
        tel_seed_l, tel_intra_l, tel_inter_l, tel_dead_l = [], [], [], []
        for run_mc, run_cr in zip(mc, cr):
            seeds = set(run_mc.get("failed_nodes",
                                   run_mc.get("initial_failures", [])))
            dead = set(run_cr["failed_nodes_t96"])
            n_seed, n_tot = len(seeds), len(dead)
            n_casc = n_tot - n_seed
            seeds_l.append(n_seed); tot_l.append(n_tot); casc_l.append(n_casc)
            cf_l.append(n_casc / (n_seedable - n_seed))
            cc = run_cr.get("cause_counts", {})
            for k in CAUSE_KEYS:
                cause_l[k].append(cc.get(k, 0))
            tel_dead = sum(1 for n in dead if types.get(n) == "telecom")
            tel_seed = sum(1 for n in seeds if types.get(n) == "telecom")
            tel_intra = cc.get("intra_telecom", 0)
            tel_seed_l.append(tel_seed); tel_intra_l.append(tel_intra)
            tel_inter_l.append(tel_dead - tel_seed - tel_intra)
            tel_dead_l.append(tel_dead)

        A = mean(tot_l) / mean(seeds_l)
        print(f"{sc:<17} {mean(seeds_l):>7.1f} {mean(casc_l):>8.1f} "
              f"{mean(tot_l):>7.1f} {A:>6.2f} {mean(cf_l):>9.4f} | "
              + " ".join(f"{mean(cause_l[k]):>6.1f}" for k in CAUSE_KEYS))
        tel_rows.append((sc, mean(tel_seed_l), mean(tel_inter_l),
                         mean(tel_intra_l), mean(tel_dead_l),
                         stdev(tel_dead_l) if len(tel_dead_l) > 1 else 0.0))

    print(f"\nTELECOM BY CAUSE (mean/run; layer = {n_telecom} sites) — "
          f"Sandy anchor: FCC ~25% region-wide, worst counties >50%; "
          f"geoclaw_2026 ~= Sandy analog")
    print(f"{'scenario':<17} {'seed':>6} {'inter':>7} {'crash':>7} "
          f"{'dead':>7} {'sd':>6} {'% layer':>8}")
    print("-" * 62)
    for sc, s, i, c, d, sd in tel_rows:
        flag = ""
        if "geoclaw" in sc and d / n_telecom > 0.40:
            flag = "  <- above Sandy worst-county band, check calibration"
        elif "geoclaw" not in sc and d / n_telecom > 0.20:
            flag = "  <- rain scenario above Sandy REGIONAL average: too hot"
        print(f"{sc:<17} {s:>6.1f} {i:>7.1f} {c:>7.1f} {d:>7.1f} {sd:>6.1f} "
              f"{100*d/n_telecom:>7.1f}%{flag}")

    print("""
READING GUIDE (for the Dr. Miura slide):
  A            = total/seeds. Falls mechanically when seeds grow faster than
                 cascade (denominator effect) — NOT a statement that bigger
                 floods cascade 'less'.
  casc_frac    = cascade / (seedable - seeds): of the nodes the flood spared,
                 the share the cascade still claimed. No denominator artifact;
                 compare THIS across severities and hazard families.
  If casc_frac rises with severity while A falls (GeoClaw), the A-trend is
  pure saturation/direct-damage substitution. If casc_frac is non-monotonic
  across DEP scenarios, seed LOCATION (which critical nodes each flood map
  touches), not seed count, is driving outcomes.""")


if __name__ == "__main__":
    main()
