#!/usr/bin/env python3
"""
test_intra_telecom.py — Week 17 intra-telecom cascade checks.

Runs on the citywide base graph (data/graph/nyc_infra_graph.graphml) with
synthetic seeds — no flood data required, so it runs anywhere the repo does.

    python3 tests/test_intra_telecom.py

Checks:
  1. Context builds from node attrs alone (no within-type edges consumed).
  2. Empty dead set  -> empty closure (no spontaneous failures at default
     surge/rho, i.e. defaults sit inside the cascade regime m(t) < 1/rho).
  3. Closure is a superset of the seeds and deterministic (memo-consistent).
  4. Carrier isolation: killing only AT&T sites never fails a T-Mobile site
     when roaming is off.
  5. Surge monotonicity: loads scale linearly with m(t) against fixed
     capacity, so closure(t=6) ⊇ closure(t=96) for the same dead set.
  6. Roaming comparison (report, not assert): merged-layer cascade vs
     carrier-segmented cascade for the same seeds.
  7. Engine integration: simulate_cascade_joint runs end-to-end with the
     telecom mode registered, telecom kills carry cause 'intra_telecom',
     and t=0 semantics hold (seeds only, no propagation at t0).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "simulation"))
sys.path.insert(0, str(ROOT))

import networkx as nx
import yaml

from cascade_joint import build_intra_context, simulate_cascade_joint, intra_closure
from intra_telecom import build_telecom_entry, telecom_closure, surge_multiplier

GRAPH = ROOT / "data" / "graph" / "nyc_infra_graph.graphml"
CONFIG = ROOT / "config" / "intra_cascade.yaml"

PASS, FAIL = "  [PASS]", "  [FAIL]"
failures = []


def check(ok, label, detail=""):
    print(f"{PASS if ok else FAIL} {label}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(label)


def dense_seeds(entry, G, carrier, k=12):
    """k sites of one carrier in the densest area (min mean neighbor dist)."""
    cands = [n for n in entry["nodes"]
             if entry["layer_of"][n] == carrier and entry["adj"][n]]
    cands.sort(key=lambda n: sum(d for _, d in entry["adj"][n]) / len(entry["adj"][n]))
    return set(cands[:k])


def main():
    print(f"Loading {GRAPH.name} ...")
    G = nx.read_graphml(GRAPH)
    with open(CONFIG) as f:
        tel_cfg = yaml.safe_load(f)["networks"]["telecom"]

    # ---- 1. context build ------------------------------------------------
    entry = build_telecom_entry(G, tel_cfg)
    n_sites = len(entry["nodes"])
    n_edges = sum(len(v) for v in entry["adj"].values()) // 2
    check(n_sites > 4000 and n_edges > n_sites,
          "context builds with derived failover adjacency",
          f"{n_sites} sites, {n_edges} edges")
    layers = set(entry["layer_of"].values())
    check(layers == {"AT&T", "T-Mobile"},
          "carrier layers = {AT&T, T-Mobile} (Metro aliased, Verizon absent by data gap)",
          f"layers={sorted(layers)}")

    # ---- 2. no spontaneous failures at defaults --------------------------
    m6 = surge_multiplier(6, entry["surge_peak"], entry["surge_tau_h"])
    check(m6 < 1.0 / entry["rho"],
          "defaults inside cascade regime: m(6) < 1/rho",
          f"m(6)={m6:.3f} < {1/entry['rho']:.3f}")
    check(telecom_closure(entry, frozenset(), 6) == frozenset(),
          "empty dead set -> empty closure")

    # ---- 3. superset + determinism ---------------------------------------
    seeds = dense_seeds(entry, G, "AT&T", k=12)
    c1 = telecom_closure(entry, frozenset(seeds), 6)
    c2 = telecom_closure(entry, frozenset(seeds), 6)
    check(seeds <= set(c1), "closure contains its seeds")
    check(c1 == c2, "closure deterministic / memo-consistent",
          f"{len(seeds)} seeds -> {len(c1)} failed "
          f"(+{len(c1) - len(seeds)} crash kills)")

    # ---- 4. carrier isolation --------------------------------------------
    spilled = {n for n in c1 if entry["layer_of"][n] != "AT&T"}
    check(not spilled, "no cross-carrier spill with roaming off",
          f"T-Mobile casualties: {len(spilled)}")

    # ---- 5. surge monotonicity: closure(6) ⊇ closure(96) -----------------
    c96 = telecom_closure(entry, frozenset(seeds), 96)
    check(set(c96) <= set(c1),
          "surge monotone: closure(t=96) ⊆ closure(t=6)",
          f"t=6: {len(c1)} vs t=96: {len(c96)}")

    # ---- 6. roaming comparison (report) -----------------------------------
    roam_cfg = dict(tel_cfg, roaming_enabled=True)
    entry_r = build_telecom_entry(G, roam_cfg)
    cr = telecom_closure(entry_r, frozenset(seeds), 6)
    print(f"  [INFO] roaming OFF: {len(c1)} failed | roaming ON: {len(cr)} failed "
          f"(MDRI effect on this seed set: {len(c1) - len(cr):+d})")

    # ---- 7. engine integration (PLUMBING check, not calibration) ----------
    # Deliberately uses crash_factor=1.5 (known contagion regime) so the
    # intra_telecom cause label is guaranteed to be exercised end-to-end;
    # the calibrated production value (3.0) intentionally yields zero
    # crashes on these 12 toy seeds. Also self-contained: hospital beds
    # fall back to the fixture when the official NYS DOH join is absent
    # (production hard-stops by design; tests should not).
    full = yaml.safe_load(open(CONFIG))
    full["networks"]["telecom"]["crash_factor"] = 1.5
    if not (ROOT / "data/healthcare/hospital_beds.csv").exists():
        if "hospital" in full.get("networks", {}):
            full["networks"]["hospital"]["beds_csv"] = \
                "tests/fixtures_hospital_beds.csv"
    cfg_path = Path("/tmp/ic_tel_test.yaml")
    with open(cfg_path, "w") as f:
        yaml.dump(full, f)
    intra_ctx = build_intra_context(G, config_path=cfg_path)
    # seeds: telecom cluster + a couple of power nodes to exercise inter+intra
    power = [n for n, d in G.nodes(data=True) if d.get("infra_type") == "power"][:2]
    init = set(seeds) | set(power)
    results, fail_time, cause = simulate_cascade_joint(G, init, intra_ctx)
    check(set(results["t0"]) == {n for n in init if n in G},
          "t=0 is seeds only (no propagation)", f"|t0|={len(results['t0'])}")
    tel_intra = [n for n, c in cause.items() if c == "intra_telecom"]
    check(len(tel_intra) > 0, "engine produces cause='intra_telecom' kills",
          f"{len(tel_intra)} telecom overload kills, total dead t96 = "
          f"{len(results['t96'])}")
    cum_ok = all(set(results[a]) <= set(results[b])
                 for a, b in zip(["t0", "t6", "t24", "t48"],
                                 ["t6", "t24", "t48", "t96"]))
    check(cum_ok, "timestep results are cumulative/monotone")

    print("\n" + ("ALL CHECKS PASSED" if not failures
                  else f"{len(failures)} FAILURE(S): {failures}"))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
