"""
Phase 3 — NYC-trained model, zero-shot transfer to Boston.

DESCRIPTIVE prediction demonstration, NOT a validated accuracy claim.
We have no real Boston cascade labels; this shows what the NYC v1 model
predicts when run on Boston's schema-matched graph + CRB flood scenarios.

Pipeline (per CRB scenario):
  1. Point-in-polygon: which Boston nodes fall inside the CRB extent.
  2. Severity-ladder depth (SLR sets base in NYC's {0.2,0.6,0.8} regime;
     AEP=1% adds a modest +0.1; capped at 0.9 to stay in-distribution).
  3. predict_cascade(): HAZUS fragility → initial-failure mask → GNN forward,
     averaged over Monte Carlo draws → per-node P(fail) at t={6,24,48,96}.

Framing caveats (must accompany every number):
  * Leaky v1: the t=0 initial-failure mask is a model input, so predictions
    are inflated — treat as directional, not calibrated.
  * Extent→discrete-depth ladder (CRB is extent-only; NYC never used true
    continuous depth either — it mapped categories to {0.2,0.6,0.8} m).
  * No real Boston labels — validated accuracy awaits the NOAA label set.
  * Regime = pluvial: CRB is physically coastal/SLR, but the ladder depths
    (0.2-0.9 m) sit in the pluvial DEP range NYC trained on, so pluvial
    fragility keeps us in-distribution (vs the 3-4 m surge saturation).
"""
from __future__ import annotations

import logging
import os

os.environ.setdefault("OGR_GEOJSON_MAX_OBJ_SIZE", "0")

import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from shapely.geometry import Point  # noqa: E402

from src.cities import boston  # noqa: E402
from src.inference.predict_cascade import predict_cascade  # noqa: E402

log = logging.getLogger(__name__)

BOSTON_GRAPH = str(boston.GRAPH_DIR / "boston_infra_heterodata.pt")
V1_CHECKPOINT = "data/gnn_checkpoints/best.pt"
TIMESTEPS = (6, 24, 48, 96)
INFRA_TYPES = ("power", "telecom", "hospital", "subway", "water", "fuel")

# Severity ladder: SLR base depth + AEP bump (meters). Stays in NYC's regime.
SLR_BASE_DEPTH = {9: 0.2, 21: 0.6, 36: 0.8}
AEP_BUMP = {10: 0.0, 1: 0.1}


def _node_coords(base) -> dict[str, tuple[float, float]]:
    """node_id -> (lon, lat) from x_raw cols [1]=lon, [0]=lat."""
    out = {}
    for nt in base.node_types:
        xr = base[nt].x_raw
        for nid, lat, lon in zip(base[nt].node_ids, xr[:, 0].tolist(), xr[:, 1].tolist()):
            out[nid] = (lon, lat)
    return out


def _flood_depths_for_scenario(scn: dict, node_coords: dict) -> dict[str, float]:
    """Point-in-polygon node flooding → ladder depth per node."""
    depth = SLR_BASE_DEPTH[scn["slr_in"]] + AEP_BUMP[scn["aep_pct"]]
    path = boston.FLOOD_DIR / f"crb_{scn['name']}.geojson"
    gdf = gpd.read_file(path)
    poly = gdf.geometry.union_all()
    out = {}
    n_flooded = 0
    for nid, (lon, lat) in node_coords.items():
        if poly.intersects(Point(lon, lat)):
            out[nid] = depth
            n_flooded += 1
        else:
            out[nid] = 0.0
    return out, depth, n_flooded


def run_all(n_mc: int = 100, seed: int = 42):
    base = torch.load(BOSTON_GRAPH, weights_only=False)
    node_coords = _node_coords(base)
    type_of = {nid: nt for nt in base.node_types for nid in base[nt].node_ids}

    results = {}
    print("=" * 100)
    print("PHASE 3 — NYC v1 model, ZERO-SHOT on Boston (DIRECTIONAL — no real labels; leaky v1)")
    print("=" * 100)
    print("'nodes in extent' = geometric point-in-polygon count (model-independent).")
    print("'E[fail t0]' = expected initial failures after HAZUS fragility (the amplification denominator).")
    print("Few in-extent nodes actually fail at t0 in shallow scenarios (fragility median > ladder depth).")
    print("-" * 100)
    print(f"{'scenario':14s} {'ladder_m':>8s} {'nodes_in_extent':>16s} {'E[fail t0]':>11s} "
          f"{'E[fail t96]':>11s} {'amplification':>13s}")
    print("-" * 100)

    for scn in boston.CRB_SCENARIOS:
        depths, ladder_m, n_extent = _flood_depths_for_scenario(scn, node_coords)
        res = predict_cascade(
            flood_depths=depths,
            base_graph_path=BOSTON_GRAPH,
            checkpoint_path=V1_CHECKPOINT,
            n_mc_samples=n_mc,
            hazard_regime="pluvial",
            seed=seed,
            device="cpu",
            verbose=False,
        )
        probs = res["probabilities"]  # {nt: [N,4]}
        e_fail_t96 = sum(float(probs[nt][:, -1].sum()) for nt in probs)
        e_initial = sum(float(p.sum()) for p in res["initial_probabilities"].values())
        amp = e_fail_t96 / e_initial if e_initial > 0 else float("nan")
        results[scn["name"]] = dict(res=res, depths=depths, ladder_m=ladder_m,
                                    n_extent=n_extent, e_fail_t96=e_fail_t96,
                                    e_initial=e_initial, amp=amp)
        print(f"{scn['name']:14s} {ladder_m:8.1f} {n_extent:16d} {e_initial:11.2f} "
              f"{e_fail_t96:11.1f} {amp:13.2f}")

    # Per-type failure fraction at t96, worst case
    print("\n" + "=" * 90)
    print("Predicted failure fraction by infrastructure type (E[fail t96] / N), per scenario")
    print("=" * 90)
    header = f"{'type':10s}" + "".join(f"{s['name'][3:]:>12s}" for s in boston.CRB_SCENARIOS)
    print(header)
    for nt in INFRA_TYPES:
        row = f"{nt:10s}"
        N = base[nt].num_nodes
        for scn in boston.CRB_SCENARIOS:
            p = results[scn["name"]]["res"]["probabilities"][nt]
            frac = float(p[:, -1].sum()) / N
            row += f"{frac*100:11.1f}%"
        print(row)

    return base, node_coords, type_of, results


def report_hospitals(base, results):
    worst = results["slr36_aep01"]["res"]
    probs = worst["probabilities"]["hospital"]
    ids = base["hospital"].node_ids
    depths = results["slr36_aep01"]["depths"]
    ranked = sorted(
        [(nid, float(probs[i, -1]), depths.get(nid, 0.0)) for i, nid in enumerate(ids)],
        key=lambda x: x[1], reverse=True,
    )
    print("\n" + "=" * 90)
    print("HOSPITAL ranking by predicted P(fail @ t96), worst case slr36_aep01")
    print("(direct flood @ t0 shown; non-flooded hospitals can only fail via cascade)")
    print("=" * 90)
    print(f"{'rank':>4s}  {'P(fail)':>8s}  {'depth@t0':>8s}  hospital")
    n_flooded_hosp = sum(1 for _, _, d in ranked if d > 0)
    print(f"  [hospitals directly flooded at t0: {n_flooded_hosp}/{len(ids)}]")
    for rank, (nid, p, d) in enumerate(ranked[:12], 1):
        name = nid.replace("hospital_", "").replace("_", " ").title()
        flood_tag = f"{d:.1f}m" if d > 0 else "dry"
        print(f"{rank:>4d}  {p:8.4f}  {flood_tag:>8s}  {name[:54]}")


def anchor_check(base, node_coords, results, scenario_name):
    """Jan 4 2018 nor'easter anchor: Aquarium Blue Line station + East Boston /
    Chelsea Creek waterfront flooded. The 2018 event was at PRESENT-DAY sea
    level, so the fair near-term test is slr09_aep01 (nearest-term rare storm);
    slr36_aep01 is the end-century worst case for contrast."""
    worst = results[scenario_name]
    res = worst["res"]
    depths = worst["depths"]

    print("\n" + "=" * 90)
    print(f"ANCHOR CHECK — Jan 4 2018 nor'easter — scenario {scenario_name} (ladder {worst['ladder_m']}m)")
    print("2018 event was present-day SL → slr09_aep01 is the present-comparable test.")
    print("Sanity check vs documented real flooding, NOT an accuracy metric.")
    print("=" * 90)

    # Build node_id -> P(fail t96)
    pfail = {}
    for nt in base.node_types:
        for i, nid in enumerate(base[nt].node_ids):
            pfail[nid] = float(res["probabilities"][nt][i, -1])

    # 1. Aquarium Blue Line station
    aqua = [nid for nid in base["subway"].node_ids if "aquarium" in nid.lower()]
    print("\n[1] Aquarium station (Blue Line — flooded Jan 2018):")
    if aqua:
        for nid in aqua:
            print(f"    {nid}: depth@t0={depths.get(nid,0):.1f}m  P(fail t96)={pfail[nid]:.4f}")
    else:
        print("    NOT FOUND in subway nodes")

    # 2. East Boston waterfront (lat ~42.37-42.39, lon ~ -71.00 to -71.04)
    def region(name, latlo, lathi, lonlo, lonhi):
        ids = [nid for nid, (lon, lat) in node_coords.items()
               if latlo <= lat <= lathi and lonlo <= lon <= lonhi]
        if not ids:
            print(f"\n[{name}] no nodes in window")
            return
        flooded = [nid for nid in ids if depths.get(nid, 0) > 0]
        ps = [pfail[nid] for nid in ids]
        print(f"\n[{name}] {len(ids)} nodes, {len(flooded)} flooded@t0, "
              f"mean P(fail t96)={np.mean(ps):.3f}, max={np.max(ps):.3f}")
        top = sorted(ids, key=lambda n: pfail[n], reverse=True)[:4]
        for nid in top:
            nm = nid.split("_", 1)[1][:38]
            print(f"      {pfail[nid]:.3f}  ({'flood' if depths.get(nid,0)>0 else 'dry'})  {nid.split('_')[0]:8s} {nm}")

    region("2] East Boston waterfront", 42.36, 42.40, -71.04, -70.99)
    region("3] Chelsea Creek (fuel terminals)", 42.38, 42.40, -71.05, -71.00)


def trace_hospital_cascade(base, results, scenario_name, name_substrings):
    """For dry-but-failing hospitals, trace the power_dependency edge that drives
    them: which substation feeds it, is that substation flooded, and is the edge
    a sensible short hop or a long one riding the cap?"""
    worst = results[scenario_name]
    depths = worst["depths"]
    res = worst["res"]
    rel = ("power", "feeds", "hospital")
    ei = base[rel].edge_index
    dist = base[rel].edge_attr_raw[:, 1] if hasattr(base[rel], "edge_attr_raw") else None
    power_ids = base["power"].node_ids
    hosp_ids = base["hospital"].node_ids
    power_pfail = {pid: float(res["probabilities"]["power"][i, -1]) for i, pid in enumerate(power_ids)}

    print("\n" + "=" * 90)
    print(f"CASCADE TRACE — dry-but-failing hospitals, scenario {scenario_name}")
    print("Confirms whether the failure rides a real flooded substation vs a long cap-edge.")
    print("=" * 90)
    for sub in name_substrings:
        h_idx = next((i for i, nid in enumerate(hosp_ids) if sub.lower() in nid.lower()), None)
        if h_idx is None:
            print(f"\n  '{sub}': not found")
            continue
        hid = hosp_ids[h_idx]
        h_depth = depths.get(hid, 0.0)
        h_pfail = float(res["probabilities"]["hospital"][h_idx, -1])
        print(f"\n  {hid.replace('hospital_','').replace('_',' ').title()[:46]}  "
              f"(depth@t0={h_depth:.1f}m, P(fail t96)={h_pfail:.3f})")
        # incoming feeds edges
        feeders = [(int(ei[0, k]), float(dist[k]) if dist is not None else None)
                   for k in range(ei.shape[1]) if int(ei[1, k]) == h_idx]
        if not feeders:
            print("    no power_dependency feeder (would fail only via subway/fuel/water)")
        for pidx, d_m in feeders:
            pid = power_ids[pidx]
            p_depth = depths.get(pid, 0.0)
            print(f"    fed by substation: {pid.replace('power_','')[:34]:34s}  "
                  f"dist={d_m/1000:.2f}km  flooded@t0={'YES' if p_depth>0 else 'no':3s}  "
                  f"P(sub fail)={power_pfail[pid]:.3f}")


def save_artifact(base, results, path=None):
    """Write a lightweight, tracked JSON summary (no huge per-node arrays)."""
    import json
    if path is None:
        path = boston.SIMULATION_DIR / "zeroshot_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "_framing": "Directional zero-shot, leaky v1 (t=0 mask inflation), "
                    "extent→discrete-depth ladder, pluvial regime, NO real Boston "
                    "labels. Not a validated accuracy claim.",
        "ladder_depths_m": {s["name"]: SLR_BASE_DEPTH[s["slr_in"]] + AEP_BUMP[s["aep_pct"]]
                            for s in boston.CRB_SCENARIOS},
        "scenarios": {},
    }
    for scn in boston.CRB_SCENARIOS:
        r = results[scn["name"]]
        per_type = {nt: round(float(r["res"]["probabilities"][nt][:, -1].sum())
                              / base[nt].num_nodes, 4) for nt in INFRA_TYPES}
        out["scenarios"][scn["name"]] = {
            "nodes_in_extent": r["n_extent"],
            "E_fail_t0": round(r["e_initial"], 2),
            "E_fail_t96": round(r["e_fail_t96"], 2),
            "amplification": round(r["amp"], 3),
            "fail_fraction_by_type_t96": per_type,
        }
    # Hospital ranking (worst case)
    worst = results["slr36_aep01"]
    hp = worst["res"]["probabilities"]["hospital"]
    hids = base["hospital"].node_ids
    out["hospital_ranking_slr36_aep01"] = [
        {"hospital": hids[i].replace("hospital_", ""),
         "p_fail_t96": round(float(hp[i, -1]), 4),
         "flooded_t0": worst["depths"].get(hids[i], 0.0) > 0}
        for i in sorted(range(len(hids)), key=lambda i: float(hp[i, -1]), reverse=True)
    ]
    # Hospital direct-flood gradient (model-independent geometry)
    out["hospitals_in_extent_by_scenario"] = {
        s["name"]: sum(1 for i, nid in enumerate(base["hospital"].node_ids)
                       if results[s["name"]]["depths"].get(nid, 0.0) > 0)
        for s in boston.CRB_SCENARIOS
    }
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved artifact → {path}")

    # Per-node per-timestep predictions (for downstream post-processing, e.g.
    # the interactive map). Lightweight: 483 nodes x 6 scenarios x 4 timesteps.
    # node_id -> [depth_m, p_t6, p_t24, p_t48, p_t96]; deterministic at seed=42.
    pernode_path = boston.SIMULATION_DIR / "zeroshot_per_node.json"
    TS = list(TIMESTEPS)
    pn = {"timesteps": TS, "scenarios": {}}
    for scn in boston.CRB_SCENARIOS:
        r = results[scn["name"]]
        depths = r["depths"]
        recs = {}
        for nt in INFRA_TYPES:
            P = r["res"]["probabilities"][nt]
            for i, nid in enumerate(base[nt].node_ids):
                recs[nid] = [round(depths.get(nid, 0.0), 3)] + \
                            [round(float(P[i, t]), 4) for t in range(len(TS))]
        pn["scenarios"][scn["name"]] = recs
    with open(pernode_path, "w") as f:
        json.dump(pn, f)
    print(f"Saved per-node artifact → {pernode_path}")
    return path


if __name__ == "__main__":
    from src.cities.boston_ingest._logging import configure as configure_logging
    configure_logging()
    logging.getLogger().setLevel(logging.WARNING)  # quiet the per-run logs
    base, node_coords, type_of, results = run_all(n_mc=100)
    report_hospitals(base, results)
    # Anchor: present-comparable (near-term rare storm) AND end-century worst case
    anchor_check(base, node_coords, results, "slr09_aep01")
    anchor_check(base, node_coords, results, "slr36_aep01")
    # Trace the two dry-but-cascading hospitals
    trace_hospital_cascade(base, results, "slr36_aep01", ["elizabeth", "mount_auburn"])
    save_artifact(base, results)
    print("\n" + "=" * 90)
    print("FRAMING: directional zero-shot, leaky v1 (t=0 mask inflation), extent→ladder depth,")
    print("no real Boston labels. Validated transfer awaits the decoupled model + NOAA labels.")
    print("=" * 90)
