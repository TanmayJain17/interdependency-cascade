"""
build_handshake.py - the two-file economic handshake to Sami (Week-14 Step 2).

Replaces Sami's flat theta = 30% shock with a real, infrastructure-derived,
per-node inoperability (q0) + downtime (T), mapped into his census-block x
CNS-sector world. Single timestep (t_step = 0) this week; the t_step dimension
is present so it expands to a 6 h trajectory next week with NO schema change.

Outputs (outputs/economic/)
---------------------------
A) node_failure_downtime.csv   one row per node per timestep (one step now)
     node_id, infra_type, t_step, t_hours, p_fail, q0, functional_frac,
     downtime_days_total
B) node_to_block_sector.csv    static crosswalk, long/tidy form
     node_id, infra_type, GEOID20, cns_sector, dependence_weight
C) handshake_metadata.json     provenance + the PROVISIONAL flag (travels with
     the data, not only the README)
D) README.md                   so Sami wires it in without a meeting

PROVISIONAL: q0/T derive from the gc_2026 GeoClaw overlay, which over-floods
~6x vs the official DEP map (likely eta-not-differenced-against-DEM). Treat as
an UPPER BOUND until the hazard footprint is fixed / Gwen's dynamic maps land.
The depth(node,t) seam makes that a one-config-line swap; this file is
flood-agnostic and rebuilds unchanged on corrected depths.
"""

from __future__ import annotations

import datetime
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.flood.depth_interface import REPO_ROOT

# gc_2026 depth scenario == "geoclaw_2026" cascade scenario id (same physical run).
CASCADE_SCENARIO = "geoclaw_2026"
DEPTH_SCENARIO = "gc_2026"
N_RUNS_FALLBACK = 1000

NODES_METADATA = REPO_ROOT / "exports" / "economic_impact" / "nodes_metadata.csv"
NODE_DOWNTIME = REPO_ROOT / "outputs" / "recovery" / "node_downtime.csv"
# POST-cascade results (failed_nodes_t96 = final failed set per run). NOT
# monte_carlo_failures_*, which is initial/direct failures only - verified its
# union (365) == the flooded set with 0 cascade spread, so it would erase the
# whole cascade-driven shock that is the point of replacing Sami's flat theta.
CASCADE_JSON = REPO_ROOT / "data" / "simulation" / f"cascade_results_nyc_{CASCADE_SCENARIO}.json"
BLOCKS_SHP = REPO_ROOT / "data" / "raw" / "census" / "tl_2023_36_tabblock20.shp"
SECTOR_MAP = REPO_ROOT / "config" / "infra_sector_map.yaml"
OUT_DIR = REPO_ROOT / "outputs" / "economic"

# Bronx, Kings(Brooklyn), New York(Manhattan), Queens, Richmond(Staten Island)
NYC_COUNTY_FIPS = {"005", "047", "061", "081", "085"}


# --------------------------------------------------------------------------- #
# q0  (expected inoperability = Monte-Carlo post-cascade failure frequency)
# --------------------------------------------------------------------------- #
def aggregate_q0(cascade_json: Path = CASCADE_JSON) -> tuple[pd.DataFrame, int]:
    """q0 per node = fraction of MC runs in which the node is in the FINAL
    post-cascade failed set (failed_nodes_t96 = direct flood + dependency
    propagation), already a fraction in [0,1] = a valid DIIM q0."""
    runs = json.load(open(cascade_json))
    n_runs = len(runs)
    counts: Counter = Counter()
    for r in runs:
        counts.update(set(r.get("failed_nodes_t96", [])))  # set(): one count per run
    df = pd.DataFrame({"node_id": list(counts.keys()),
                       "q0": [counts[n] / n_runs for n in counts]})
    return df, n_runs


# --------------------------------------------------------------------------- #
# node -> GEOID20  (TIGER tabblock20 point-in-polygon)
# --------------------------------------------------------------------------- #
def build_node_to_geoid(nodes_df: pd.DataFrame, blocks_shp: Path = BLOCKS_SHP) -> pd.DataFrame:
    """Point-in-polygon node -> GEOID20 against ALL NY-state blocks in the metro
    bbox (NYC + Nassau/Westchester). Plain 'within', no county pre-filter and no
    nearest-snap: a node gets its TRUE containing block or stays unmatched.
    NJ nodes fall in-bbox but NJ blocks are FIPS 34 (not in this NY file), so
    they correctly stay null and are handled via dependency, not spatially.
    """
    import geopandas as gpd

    if not blocks_shp.exists():
        raise FileNotFoundError(
            f"census blocks not found at {blocks_shp}. Download TIGER 2023 NY: "
            "https://www2.census.gov/geo/tiger/TIGER2023/TABBLOCK20/tl_2023_36_tabblock20.zip"
        )
    bbox = (-74.30, 40.45, -73.65, 41.00)  # NY metro window around the node extent
    blocks = gpd.read_file(blocks_shp, columns=["GEOID20", "geometry"], bbox=bbox).to_crs(4326)

    pts = gpd.GeoDataFrame(
        nodes_df[["node_id"]].copy(),
        geometry=gpd.points_from_xy(nodes_df["lon"], nodes_df["lat"]),
        crs=4326,
    )
    joined = gpd.sjoin(pts, blocks[["GEOID20", "geometry"]], predicate="within", how="left")
    out = joined[["node_id", "GEOID20"]].drop_duplicates("node_id")
    out["GEOID20"] = out["GEOID20"].astype("string")
    return out


# --------------------------------------------------------------------------- #
# infra -> CNS sectors served (long form, from the overridable config)
# --------------------------------------------------------------------------- #
def sector_rows_for_infra(infra_type: str, sector_cfg: dict) -> list[tuple[str, float]]:
    dep = sector_cfg["infra_sector_dependence"].get(infra_type, {})
    base = dep.get("ALL")
    out = []
    for s in sector_cfg["cns_sectors"]:
        w = dep.get(s, base)
        if w is not None and w > 0:
            out.append((s, float(w)))
    return out


# --------------------------------------------------------------------------- #
# assemble both files
# --------------------------------------------------------------------------- #
def build(out_dir: Path = OUT_DIR) -> dict:
    nodes = pd.read_csv(NODES_METADATA)
    nodes["node_id"] = nodes["node_id"].astype(str)
    if "external" not in nodes.columns:
        nodes["external"] = False

    q0_df, n_runs = aggregate_q0()
    down = pd.read_csv(NODE_DOWNTIME)[["node_id", "T_total_days"]]
    down["node_id"] = down["node_id"].astype(str)
    sector_cfg = yaml.safe_load(open(SECTOR_MAP))

    # ---- file A: node_failure_downtime (single t_step=0 slice) ----
    a = nodes[["node_id", "infra_type"]].merge(q0_df, on="node_id", how="left").merge(
        down, on="node_id", how="left")
    a["q0"] = a["q0"].fillna(0.0)
    a["p_fail"] = a["q0"]
    a["downtime_days_total"] = a["T_total_days"].fillna(0.0)
    a["t_step"] = 0
    a["t_hours"] = 0.0
    # t=0 impact slice: a failing node is fully down (functional_frac=0); a node
    # that never fails is fully functional (1). Sami: q(t)=q0*(1-functional_frac)
    # -> q(0)=q0. Next week functional_frac becomes restoration_fraction(t,T,shape).
    a["functional_frac"] = np.where(a["q0"] > 0, 0.0, 1.0)
    file_a = a[["node_id", "infra_type", "t_step", "t_hours", "p_fail", "q0",
                "functional_frac", "downtime_days_total"]]

    # ---- file B: node_to_block_sector (static crosswalk, long form) ----
    geoid = build_node_to_geoid(nodes)
    rows = []
    sector_cache = {it: sector_rows_for_infra(it, sector_cfg) for it in nodes["infra_type"].unique()}
    geoid_map = dict(zip(geoid["node_id"], geoid["GEOID20"]))
    for nid, it in zip(nodes["node_id"], nodes["infra_type"]):
        g = geoid_map.get(nid, np.nan)
        for sector, w in sector_cache.get(it, []):
            rows.append((nid, it, g, sector, w))
    file_b = pd.DataFrame(rows, columns=["node_id", "infra_type", "GEOID20",
                                         "cns_sector", "dependence_weight"])

    out_dir.mkdir(parents=True, exist_ok=True)
    file_a.to_csv(out_dir / "node_failure_downtime.csv", index=False)
    file_b.to_csv(out_dir / "node_to_block_sector.csv", index=False)

    # ---- C: provenance sidecar (the provisional flag travels with the data) ----
    meta = {
        "scenario": DEPTH_SCENARIO,
        "cascade_scenario": CASCADE_SCENARIO,
        "provisional_upper_bound": True,
        "reason": ("gc_2026 over-floods ~6x vs the official DEP map (373 vs 61 wet nodes); "
                   "eta-vs-DEM suspected. q0/downtime are an UPPER BOUND until the hazard "
                   "footprint is corrected or Gwen's dynamic maps replace it."),
        "date": datetime.date.today().isoformat(),
        "n_timesteps": 1,
        "t_steps": [0],
        "n_mc_runs": n_runs,
        "q0_definition": ("Expected POST-cascade inoperability in [0,1] = fraction of MC runs in "
                          "which the node is in the final failed set (failed_nodes_t96); direct "
                          "flood AND dependency cascade."),
        "q0_source": str(CASCADE_JSON.relative_to(REPO_ROOT)),
        "downtime_source": "outputs/recovery/node_downtime.csv (HAZUS-FL placeholder anchors)",
        "downtime_caveat": ("downtime_days_total is own-flood-restoration only this week; "
                            "dependency-failed (non-flooded) nodes have q0>0 but a downtime "
                            "FLOOR of 0 (recover-gating dormant) until the dynamic timeline."),
        "sector_vocabulary": "LODES WAC CNS01..CNS20",
        "sector_map_source": "config/infra_sector_map.yaml (modeling judgment, overridable)",
        "block_geometries": "TIGER 2023 tl_2023_36_tabblock20 (GEOID20)",
        "crosswalk_note": ("Filter file B to GEOID20 prefix in {36005,36047,36061,36081,36085} for "
                           "the five NYC counties. Null GEOID20 = out-of-NY-state suppliers (NJ "
                           "power/fuel/telecom, entering via dependency not spatially) plus ~89 "
                           "telecom-cluster nodes with coarse/offset coordinates (~1 mi from any "
                           "block; a pre-existing telecom-layer geocoding artifact, not a join bug)."),
    }
    (out_dir / "handshake_metadata.json").write_text(json.dumps(meta, indent=2))

    _write_readme(out_dir, meta)
    return {"file_a": file_a, "file_b": file_b, "geoid": geoid, "meta": meta,
            "sector_cfg": sector_cfg, "nodes": nodes}


def _write_readme(out_dir: Path, meta: dict) -> None:
    txt = f"""# Economic handshake (Week-14 Step 2) - for Sami

Drop-in replacement for the flat theta = 30% shock: a per-node, infrastructure-
derived inoperability (q0) and downtime (T), mapped to census block x CNS sector.

**PROVISIONAL ({meta['date']}):** {meta['reason']}
See handshake_metadata.json (provisional_upper_bound = true). The pipeline is
flood-agnostic and rebuilds unchanged once depths are corrected.

## Files
- **node_failure_downtime.csv** - one row per node per timestep (this week: one
  step, t_step=0). `q0` = expected inoperability in [0,1] ({meta['q0_definition']}).
  `functional_frac` = fraction functional at the step (0 at impact). Use
  `q(t) = q0 * (1 - functional_frac)` -> at t=0, q = q0. `downtime_days_total` =
  expected recovery time (days). The t_step/t_hours columns expand to a 6 h
  trajectory next week with no schema change.
- **node_to_block_sector.csv** - static crosswalk, long form: one row per
  (node, CNS sector served). `GEOID20` = census block (point-in-polygon); null
  for external NJ / offshore nodes. `dependence_weight` in [0,1] = how strongly
  that sector leans on this infra (config/infra_sector_map.yaml - tune freely).
- **handshake_metadata.json** - provenance + the provisional flag.

## To aggregate to block x sector (your units)
Join file A (q0) with file B on node_id; for each (GEOID20, cns_sector):
`q_block_sector = 1 - prod_over_nodes(1 - q0 * dependence_weight)` (illustrative
probabilistic-OR; pick your own aggregator). That replaces theta in `f_s`.

## Caveats to know
- {meta['downtime_caveat']}
- q0 covers all hazard+cascade failures; downtime currently only the flooded set.
- dependence_weight defaults are modeling placeholders (Week-14 brief), not yet
  reconciled with your interdependence ratios.
- GEOID20 coverage: {meta['crosswalk_note']}
"""
    (out_dir / "README.md").write_text(txt)


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
def _validate(res: dict) -> None:
    nodes, geoid, file_a, file_b = res["nodes"], res["geoid"], res["file_a"], res["file_b"]
    g = dict(zip(geoid["node_id"], geoid["GEOID20"]))
    nodes = nodes.copy()
    nodes["GEOID20"] = nodes["node_id"].map(g).astype("string")
    nodes["county5"] = nodes["GEOID20"].str[:5]
    NYC5 = {"36005", "36047", "36061", "36081", "36085"}  # Bronx, Kings, NY, Queens, Richmond

    n = len(nodes)
    is_nj = nodes["borough"] == "NJ (external)"
    matched = nodes["GEOID20"].notna()
    in_nyc = matched & nodes["county5"].isin(NYC5)
    other_ny = matched & ~nodes["county5"].isin(NYC5)
    unm_nj = (~matched) & is_nj
    unm_other = (~matched) & (~is_nj)

    print("=" * 72)
    print(f"economic handshake  |  scenario = {DEPTH_SCENARIO} (PROVISIONAL)  |  nodes = {n}")
    print("=" * 72)
    print(f"GEOID20 matched {int(matched.sum())}/{n}  -> NYC-county {int(in_nyc.sum())}, "
          f"other-NY (Nassau/Westchester) {int(other_ny.sum())}")
    west = int((unm_other & (nodes["lon"] <= -73.99)).sum())   # west of Hudson = unlabeled NJ
    east = int((unm_other & (nodes["lon"] > -73.99)).sum())     # east = telecom coarse-coord artifact
    print(f"unmatched: NJ-external {int(unm_nj.sum())} (expected, FIPS 34 - via dependency)"
          f"  | other {int(unm_other.sum())} = {west} west-of-Hudson (unlabeled NJ) + "
          f"{east} east (telecom-cluster geocoding artifact, ~1 mi offset)")
    dup = file_b.dropna(subset=["GEOID20"]).groupby("node_id")["GEOID20"].nunique()
    print(f"nodes mapping to >1 GEOID20: {int((dup > 1).sum())} (should be 0)")
    print(f"file A rows: {len(file_a)} (== nodes)  | file B rows: {len(file_b)} (nodes x sectors)")
    print(f"q0 range {file_a['q0'].min():.3f}-{file_a['q0'].max():.3f}  "
          f"mean(failed)={file_a.loc[file_a.q0>0,'q0'].mean():.3f}  n(q0>0)={int((file_a.q0>0).sum())}")
    direct = int(((file_a["q0"] > 0) & (file_a["downtime_days_total"] > 0)).sum())
    casc = int(((file_a["q0"] > 0) & (file_a["downtime_days_total"] == 0)).sum())
    print(f"  -> direct-flood {direct} | cascade-driven {casc} (q0>0 beyond the flood; "
          f"downtime floor 0 pending the gated clock)")
    if int(unm_other.sum()):
        print("  sample other-unmatched:", list(nodes[unm_other]["node_id"].head(5)))

    # aggregate q0 to block x sector for ONE flooded Manhattan (county 36061) block
    nb = nodes.merge(file_a[["node_id", "q0"]], on="node_id")
    lm = nb[(nb["county5"] == "36061") & (nb["lat"] < 40.73) & (nb["q0"] > 0)]
    print("\nBlock x sector inoperability - one flooded Lower-Manhattan (36061) block:")
    if len(lm):
        blk = lm["GEOID20"].value_counts().index[0]
        bnodes = nb[nb["GEOID20"] == blk]
        wmap = res["sector_cfg"]["infra_sector_dependence"]
        sectors = res["sector_cfg"]["cns_sectors"]
        print(f"  GEOID20 {blk}: {len(bnodes)} node(s) -> "
              f"{dict(bnodes['infra_type'].value_counts())}")
        def w(it, s):
            d = wmap.get(it, {}); return d.get(s, d.get("ALL", 0.0))
        shown = 0
        for s in sectors:
            prod = 1.0
            for _, r in bnodes.iterrows():
                prod *= (1 - r["q0"] * w(r["infra_type"], s))
            q = 1 - prod
            if q > 0 and shown < 6:
                print(f"    {s}: block inoperability q = {q:.3f}")
                shown += 1
        print("  (illustrative probabilistic-OR; becomes the q0(t) trajectory next week)")
    else:
        print("  (no flooded Manhattan block found)")
    print("-" * 72)
    for f in ["node_failure_downtime.csv", "node_to_block_sector.csv",
              "handshake_metadata.json", "README.md"]:
        print(f"saved: {OUT_DIR / f}")
    print("=" * 72)


if __name__ == "__main__":
    _validate(build())
