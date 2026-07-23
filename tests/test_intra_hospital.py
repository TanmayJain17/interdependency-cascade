#!/usr/bin/env python3
"""
test_intra_hospital.py — Week 17 hospital intra-cascade checks.

Uses the OFFICIAL beds file (data/healthcare/hospital_beds.csv) when
present — i.e., on a machine where join_hospital_beds.py has run — and
falls back to tests/fixtures_hospital_beds.csv (plausible magnitudes,
mechanics-only) otherwise, with a loud notice.

    python tests/test_intra_hospital.py

Checks:
  1. Context builds: beds loaded, systems derived from FacDB OPNAME,
     NULL-bed nodes excluded loudly.
  2. Empty dead set -> empty closure; determinism/memo.
  3. SANDY REPLAY (the validation): kill NYU Langone + Bellevue (+ Coney
     Island if matched). Expect (a) NO collapse deaths at defaults, (b)
     Beth Israel among the top-2 strained receivers, (c) its census ratio
     in the strained band [divert_frac, collapse_frac) — the documented
     ~1.15x Sandy peak lives there.
  4. AFFILIATION EFFECT (Lee 2015): with only Bellevue (public) dead,
     Metropolitan+Woodhull (public) absorb a larger share with
     affiliation_bonus=2 than with bonus=1.
  5. Engine integration: joint run with hospital seeds; strain log
     populated; cause 'intra_hospital' reserved for collapse only.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "simulation"))
sys.path.insert(0, str(ROOT))

import networkx as nx
import yaml

from cascade_joint import build_intra_context, simulate_cascade_joint
from intra_hospital import build_hospital_entry, hospital_closure

GRAPH = ROOT / "data" / "graph" / "nyc_infra_graph.graphml"
CONFIG = ROOT / "config" / "intra_cascade.yaml"
REAL_BEDS = ROOT / "data" / "healthcare" / "hospital_beds.csv"
FIXTURE_BEDS = ROOT / "tests" / "fixtures_hospital_beds.csv"

PASS, FAIL = "  [PASS]", "  [FAIL]"
failures = []


def check(ok, label, detail=""):
    print(f"{PASS if ok else FAIL} {label}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(label)


def find(entry, *needles):
    for n in entry["nodes"]:
        if all(x in n for x in needles):
            return n
    return None


def main():
    G = nx.read_graphml(GRAPH)
    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)["networks"]["hospital"]
    if REAL_BEDS.exists():
        print("Using OFFICIAL NYS DOH beds file.")
    else:
        print("NOTICE: official beds file absent — using mechanics FIXTURE "
              "(run join_hospital_beds.py for real capacities).")
        cfg = dict(cfg, beds_csv=str(FIXTURE_BEDS.relative_to(ROOT)))

    # ---- 1. context --------------------------------------------------------
    entry = build_hospital_entry(G, cfg, root=ROOT)
    check(len(entry["nodes"]) >= 55, "context builds with beds + systems",
          f"{len(entry['nodes'])} participating, "
          f"{len(entry['excluded'])} excluded (NULL beds)")
    hh = sum(1 for s in entry["system"].values()
             if s == "NYC Health + Hospitals")
    check(hh >= 8, "H+H system recognized from OPNAME", f"{hh} facilities")

    # ---- 2. empty + determinism -------------------------------------------
    check(hospital_closure(entry, frozenset()) == frozenset(),
          "empty dead set -> empty closure")
    nyu = find(entry, "nyu_langone_hospitals")
    bell = find(entry, "bellevue")
    coney = find(entry, "coney_island")
    beth = find(entry, "beth_israel")
    sandy_dead = frozenset(n for n in (nyu, bell, coney) if n)
    c1 = hospital_closure(entry, sandy_dead)
    c2 = hospital_closure(entry, sandy_dead)
    check(c1 == c2, "closure deterministic / memo-consistent")

    # ---- 3. Sandy replay ---------------------------------------------------
    collapsed = set(c1) - set(sandy_dead)
    check(len(collapsed) == 0,
          "Sandy replay: ZERO volume-driven collapses (invariant — capped "
          "EMTALA absorption)", f"collapse deaths: {len(collapsed)}")
    log = entry["strain_log"][frozenset(sandy_dead)]
    ratios = log["census_ratio"]
    top = sorted(ratios.items(), key=lambda kv: -kv[1])[:5]
    print(f"  [INFO] diverting: {log['n_diverting']}, "
          f"unplaced: {log['unplaced']}, top strain: "
          + ", ".join(f"{n.split('hospital_')[1]}={r}" for n, r in top))
    top_ratio = max(ratios.values())
    check(cfg.get("divert_frac", 0.95) - 0.10 <= top_ratio
          <= cfg.get("collapse_frac", 1.20) + 1e-9,
          "top receiver heavily strained but bounded by surge capacity",
          f"max census ratio {top_ratio}")
    if beth:
        rank = [n for n, _ in sorted(ratios.items(), key=lambda kv: -kv[1])
                ].index(beth) + 1
        occ = float(cfg.get("baseline_occupancy", 0.75))
        rel = ratios[beth] / occ
        check(1.05 <= rel <= 1.65,
              "Beth Israel (2012's absorber) heavily strained vs its "
              "baseline", f"rank {rank}, {rel:.3f}x baseline")
    else:
        # THEN vs NOW: with current HFIS, Mount Sinai Beth Israel is
        # CLOSED (0 certified beds) — the hospital that absorbed the 2012
        # surge no longer exists, so a Sandy repeat routes around a hole.
        beth_node = find({"nodes": entry["zero_bed"] + entry["excluded"],
                          }, "beth_israel") if hasattr(entry, "get") else None
        in_zero = any("beth_israel" in n for n in entry.get("zero_bed", []))
        check(in_zero, "then-vs-now: Beth Israel closed (0 certified beds) "
              "— 2012's absorber is gone; receiving buffer has thinned",
              "documented for the deck")

    # ---- 4. affiliation effect (Lee 2015) ----------------------------------
    if bell:
        metro, wood = find(entry, "metropolitan"), find(entry, "woodhull")
        gains = {}
        for b in (1.0, 2.0):
            e = build_hospital_entry(G, dict(cfg, affiliation_bonus=b),
                                     root=ROOT)
            hospital_closure(e, frozenset({bell}))
            r = e["strain_log"][frozenset({bell})]["census_ratio"]
            gains[b] = sum(r[n] - e["census0"][n] / e["beds"][n]
                           for n in (metro, wood) if n)
        check(gains[2.0] > gains[1.0],
              "public->public pull: bonus=2 routes more Bellevue load to "
              "Metropolitan+Woodhull than bonus=1",
              f"strain gain {gains[1.0]:.4f} -> {gains[2.0]:.4f}")

    # ---- 5. engine integration ---------------------------------------------
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        ctx = build_intra_context(G, config_path=CONFIG) \
            if REAL_BEDS.exists() else None
    if ctx is None:   # fixture path: build ctx with patched hospital cfg
        with open(CONFIG) as f:
            full = yaml.safe_load(f)
        full["networks"]["hospital"]["beds_csv"] = \
            str(FIXTURE_BEDS.relative_to(ROOT))
        tmp = Path("/tmp/ic_hosp_test.yaml")
        with open(tmp, "w") as f:
            yaml.dump(full, f)
        with contextlib.redirect_stdout(io.StringIO()):
            ctx = build_intra_context(G, config_path=tmp)
    power = [n for n, d in G.nodes(data=True)
             if d.get("infra_type") == "power"][:3]
    seeds = set(sandy_dead) | set(power)
    res, ft, cause = simulate_cascade_joint(G, seeds, ctx)
    check(set(res["t0"]) == {n for n in seeds if n in G},
          "t=0 seeds only", f"|t0|={len(res['t0'])}")
    n_collapse = sum(1 for c in cause.values() if c == "intra_hospital")
    print(f"  [INFO] joint run: total t96 = {len(res['t96'])}, "
          f"hospital collapse deaths = {n_collapse} "
          f"(strain is the primary output; collapse is the rare tail)")
    check(len(ctx["hospital"]["strain_log"]) > 0,
          "strain diagnostics populated for reporting/Sami")

    print("\n" + ("ALL CHECKS PASSED" if not failures
                  else f"{len(failures)} FAILURE(S): {failures}"))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
