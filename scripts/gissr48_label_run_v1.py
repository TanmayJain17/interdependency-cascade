#!/usr/bin/env python3
"""
gissr48_label_run_v1.py — Monte Carlo cascade labels for one storm whose node depths come from a
depth table, in one of the two label modes the GNN is trained on:

  static   every flood seed at t = 0 on the frozen grid [0, 6, 24, 48, 96]  (as legacy_v1_n1000)
  arrival  each flood seed lands when the hydrograph first wets it           (as dynamic_v1/arrival_h360)

Written for the 48 GISSR storms of Week 29 (data/flood/gissr48_manifest_v1.csv). It is the same thin
wrapper as run_synthetic20.py and dynamic_forcing_reduction_run.py: it sets the production runner's
module globals and calls its run_scenario() verbatim. No file under src/ is changed. Two differences
from dynamic_forcing_reduction_run.py:
  * depths come from --depth-csv only (the new storms have no campaign folder to read them from);
  * the timing tables come from --dyn-config (default: the gissr48 config), by pointing the runner's
    load_dynamic_forcing at that file.

The wrapper owns every setting the labels depend on, so a run by hand gives the same labels as the
sbatch: the frozen v1 graph (NODES_IN / GRAPH_IN default to data/graph_v1_frozen), the arrival-mode
post-peak offsets (--offsets, default "0 6 24 48 96 144 240 360"), no shift knobs, and in arrival mode
a timing row for every wet site. A conflicting environment variable stops the run.

Identity gate (--check-dir): run an OLD storm through this driver and compare the records with the
frozen campaign file for the same storm, record for record. Exit code 2 on any difference. The old
storm's depths must be the campaign's own (--depth-geojson <campaign folder>/temp_nodes_nyc_<storm>.geojson),
as in dynamic_forcing_reduction_run.py: the first gate run (job 19301966, depths from
data/flood/jesse22_node_depths_v1.csv) failed in 1 run of 20 on one site, telecom_cluster_03042, whose
depth in that table is 0.105 m above the depth the frozen campaign sampled; lowering that one depth by
0.105 m reproduces the frozen record exactly. The gate tests the code, so it gets the campaign's inputs.

  python scripts/gissr48_label_run_v1.py --scenario syn_ts_914_6_1p2955 --mode static --n 20 \
      --depth-geojson $SCRATCH/results/legacy_v1_n1000/temp_nodes_nyc_syn_ts_914_6_1p2955.geojson \
      --out $SCRATCH/results/gissr48_v1_smoke/identity_static --check-dir $SCRATCH/results/legacy_v1_n1000

Production (one storm, one mode):
  python scripts/gissr48_label_run_v1.py --scenario syn2_t589w12_wl2p614 --mode arrival --n 1000 \
      --out $SCRATCH/results/gissr48_v1/arrival_h360
"""
import argparse, os, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO)
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src" / "simulation"))      # same import path as run_synthetic20.py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--mode", required=True, choices=["static", "arrival"])
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--depth-csv", default="data/flood/gissr48_node_depths_v1.csv")
    ap.add_argument("--depth-geojson", default=None,
                    help="read depths from a campaign temp_nodes geojson (property flood_depth_m) instead of --depth-csv; "
                         "for the identity gate on an old storm")
    ap.add_argument("--dyn-config", default="config/dynamic_forcing_gissr48_v1.yaml")
    ap.add_argument("--offsets", default="0 6 24 48 96 144 240 360",
                    help="arrival mode: post-peak evaluation offsets in hours (the dynamic_v1 campaign's)")
    ap.add_argument("--check-dir", default=None,
                    help="folder holding a frozen cascade_results_nyc_<scenario>.json to compare against (identity gate)")
    a = ap.parse_args()

    os.environ["POWER_COUPLING"] = "0"                      # legacy arm only, as in both v1 label campaigns
    for var, frozen in (("NODES_IN", "data/graph_v1_frozen/nyc_infra_nodes_all_flood.geojson"),
                        ("GRAPH_IN", "data/graph_v1_frozen/nyc_infra_graph_all_flood.graphml")):
        if os.environ.setdefault(var, frozen) != frozen:
            sys.exit(f"{var}={os.environ[var]} is not the frozen v1 graph ({frozen}) — stop")
    for knob in ("DYNAMIC_ARRIVAL_SHIFT_H", "DYNAMIC_TPEAK_SHIFT_H", "DYNAMIC_INTRA_ORIGIN_SHIFT_H", "DYNAMIC_PRE_PEAK_STEP_H", "DYNAMIC_INTRA_CLOCK"):
        if os.environ.get(knob):
            sys.exit(f"{knob}={os.environ[knob]} is set in the environment; label runs use the campaign defaults — unset it")
    if a.mode == "static":
        os.environ["DYNAMIC_FORCING"] = "0"
    else:
        os.environ["DYNAMIC_FORCING"] = "1"; os.environ["DYNAMIC_MODE"] = "arrival"
        if os.environ.get("DYNAMIC_OFFSETS", a.offsets).split() != a.offsets.split():
            sys.exit(f"DYNAMIC_OFFSETS='{os.environ['DYNAMIC_OFFSETS']}' differs from --offsets '{a.offsets}' — stop")
        os.environ["DYNAMIC_OFFSETS"] = a.offsets
    if a.check_dir and not (Path(a.check_dir) / f"cascade_results_nyc_{a.scenario}.json").exists():
        sys.exit(f"identity gate: missing frozen file {Path(a.check_dir) / f'cascade_results_nyc_{a.scenario}.json'}")

    import pandas as pd
    import geopandas as gpd
    import multi_scenario_runner as msr
    import dynamic_forcing as dfm
    from src.cascade.stochastic_buffer import load_buffer_config

    cfg = Path(a.dyn_config)
    if not cfg.exists():
        sys.exit(f"missing dynamic-forcing config: {cfg}")
    msr.load_dynamic_forcing = lambda: dfm.load_dynamic_forcing(cfg)      # the runner reads the timing tables named here
    if a.mode == "arrival":
        dyn = dfm.load_dynamic_forcing(cfg)
        # never let a dynamic run silently fall back to static forcing
        if dyn is None or not dyn.has_scenario(a.scenario) or a.scenario not in dyn.timing:
            sys.exit(f"mode=arrival requested but {a.scenario} is not in the timing tables of {cfg} — refusing to fall back to static")
        want = [float(x) for x in a.offsets.split()]
        if dyn.mode != "arrival" or dyn.intra_clock != "peak" or dyn.post_peak_offsets_h != want or dyn.pre_peak_step_h != 6.0 \
                or dyn.tpeak_shift_h or dyn.intra_origin_shift_h:
            sys.exit(f"unexpected dynamic settings: mode={dyn.mode} intra_clock={dyn.intra_clock} offsets={dyn.post_peak_offsets_h} "
                     f"pre_peak_step={dyn.pre_peak_step_h} (campaign uses arrival / peak / {want} / 6)")
        print(f"timing: {len(dyn.timing[a.scenario])} wet sites, t_peak {dyn.t_peak[a.scenario]:.1f} h, offsets {dyn.post_peak_offsets_h} (from {cfg})")

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    result = out / f"cascade_results_nyc_{a.scenario}.json"
    if result.exists():
        result.unlink(); print(f"removed stale {result}")

    nodes_gdf = gpd.read_file(msr.NODES_IN)
    col = f"flood_{a.scenario}_depth_m"
    if a.depth_geojson:
        import json
        if not Path(a.depth_geojson).exists():
            sys.exit(f"missing depth geojson: {a.depth_geojson}")
        feats = json.load(open(a.depth_geojson))["features"]
        depth = {f["properties"]["node_id"]: (f["properties"].get("flood_depth_m") or 0.0) for f in feats}
        src_name = Path(a.depth_geojson).name
    else:
        dcsv = pd.read_csv(a.depth_csv, usecols=lambda c: c in ("node_id", col), float_precision="round_trip")
        if col not in dcsv.columns:
            sys.exit(f"{a.depth_csv} has no column {col}")
        depth = dict(zip(dcsv["node_id"], dcsv[col].fillna(0.0)))
        src_name = f"{Path(a.depth_csv).name}:{col}"
    missing = int(nodes_gdf["node_id"].map(depth).isna().sum())
    if missing or len(depth) != len(nodes_gdf):
        sys.exit(f"node ids do not match: {missing} of {len(nodes_gdf)} nodes missing from {src_name} ({len(depth)} rows) — graph/nodes mismatch, stop")
    nodes_gdf[col] = nodes_gdf["node_id"].map(depth)
    wet = int((nodes_gdf[col] > 0.01).sum())
    print(f"nodes {len(nodes_gdf):,} from {msr.NODES_IN} | graph {msr.GRAPH_IN} | depths from {src_name}: wet {wet}")
    if wet == 0:
        sys.exit("no wet node in this storm — wrong column or wrong table, stop")
    if a.mode == "arrival":
        # a wet site without a timing row would be placed at the peak without a word: refuse instead
        no_row = set(nodes_gdf.loc[nodes_gdf[col] > 0.01, "node_id"]) - set(dyn.timing[a.scenario])
        if no_row:
            sys.exit(f"{len(no_row)} wet sites of {a.scenario} have no timing row (e.g. {sorted(no_row)[:3]}) — depth table and timing table disagree, stop")

    msr.SIM_DIR = out
    msr.N_MONTE_CARLO = a.n
    msr.SCENARIO_DEPTH_COL[a.scenario] = col
    buffer_config = load_buffer_config("config/buffer_distributions.yaml")
    power_coupling = msr.load_power_coupling()
    if power_coupling is not None:
        sys.exit("power coupling came back enabled despite POWER_COUPLING=0 — stop")
    print(f"SIM_DIR -> {msr.SIM_DIR} | N_MC -> {msr.N_MONTE_CARLO} | mode -> {a.mode} | time steps {msr.TIME_STEPS} | "
          f"seeds fragility {msr.FRAGILITY_SEED} buffer {msr.BUFFER_SEED}")

    t1 = time.time()
    _mc, runs = msr.run_scenario(a.scenario, nodes_gdf, buffer_config, power_coupling)
    print(f"scenario wall time: {(time.time() - t1) / 60:.1f} min | runs: {len(runs)}")

    # the record must be of the mode that was asked for
    is_dyn = "dynamic_forcing" in runs[0]
    if is_dyn != (a.mode == "arrival"):
        sys.exit(f"mode={a.mode} but the records are {'dynamic' if is_dyn else 'static'} — stop")
    if len(runs) != a.n:
        sys.exit(f"expected {a.n} runs, got {len(runs)}")
    print(f"LABEL RUN OK: {a.scenario} mode={a.mode} runs={len(runs)} -> {result}")

    if a.check_dir:
        ref = Path(a.check_dir) / f"cascade_results_nyc_{a.scenario}.json"
        if not ref.exists():
            sys.exit(f"identity gate: missing frozen file {ref}")
        rc = subprocess.call([sys.executable, "scripts/dynamic_forcing_reduction_check.py",
                              "--ref", str(ref), "--test", str(result), "--first", str(a.n)])
        print("IDENTITY GATE:", "PASS — this driver reproduces the frozen campaign record for record" if rc == 0 else "FAIL")
        sys.exit(rc)


if __name__ == "__main__":
    main()
