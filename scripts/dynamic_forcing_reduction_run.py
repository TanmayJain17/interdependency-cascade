#!/usr/bin/env python3
"""
dynamic_forcing_reduction_run.py — run one synthetic scenario through the production runner with the
campaign's OWN node depths (from its temp_nodes geojson, no raster re-sampling), in a chosen forcing
mode, and compare the result against the frozen campaign file with the fatal reduction checker.

Mirrors run_synthetic20.py's setup exactly; only the depth source and the single-scenario call differ.

Usage (repo root, flood env):
  python scripts/dynamic_forcing_reduction_run.py --scenario syn_ts_914_6_1p2955 --n 20 --mode static_peak
  python scripts/dynamic_forcing_reduction_run.py --scenario syn_ts_914_6_1p2955 --n 20 --mode static
Outputs go to data/simulation_dyn_check/<mode>/ ; the frozen reference is
data/hpc_results_aug2026/legacy_v1_n1000/cascade_results_nyc_<scenario>.json (override with --ref-dir).
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO)
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src" / "simulation"))      # same trick as run_synthetic20.py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--mode", default="static_peak", choices=["static", "static_peak", "arrival", "threshold"])
    ap.add_argument("--ref-dir", default="data/hpc_results_aug2026/legacy_v1_n1000")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-check", action="store_true")
    a = ap.parse_args()

    # forcing mode via the same env knobs the sbatch will use; legacy arm only
    os.environ["POWER_COUPLING"] = "0"
    if a.mode == "static":
        os.environ["DYNAMIC_FORCING"] = "0"
    else:
        os.environ["DYNAMIC_FORCING"] = "1"; os.environ["DYNAMIC_MODE"] = a.mode

    import geopandas as gpd
    import multi_scenario_runner as msr
    from src.cascade.stochastic_buffer import load_buffer_config

    ref_dir = Path(a.ref_dir)
    ref_nodes = ref_dir / f"temp_nodes_nyc_{a.scenario}.geojson"
    ref_runs = ref_dir / f"cascade_results_nyc_{a.scenario}.json"
    for p in (ref_nodes, ref_runs):
        if not p.exists():
            sys.exit(f"missing campaign file: {p}")

    out = Path(a.out) if a.out else Path("data/simulation_dyn_check") / a.mode
    out.mkdir(parents=True, exist_ok=True)
    stale = out / f"cascade_results_nyc_{a.scenario}.json"
    if stale.exists():
        stale.unlink(); print(f"removed stale {stale}")

    nodes_gdf = gpd.read_file(msr.NODES_IN)
    gj = json.load(open(ref_nodes))
    depth = {f["properties"]["node_id"]: (f["properties"].get("flood_depth_m") or 0.0) for f in gj["features"]}
    col = f"flood_{a.scenario}_depth_m"
    nodes_gdf[col] = nodes_gdf["node_id"].map(depth).fillna(0.0)
    missing = int(nodes_gdf["node_id"].map(depth).isna().sum())
    wet = int((nodes_gdf[col] > 0.01).sum())
    print(f"nodes {len(nodes_gdf):,} from {msr.NODES_IN} | depths from {ref_nodes.name}: wet {wet}, unmatched ids {missing}")
    if missing:
        sys.exit("node ids in NODES_IN do not all appear in the campaign geojson — graph/nodes mismatch, stop")

    msr.SIM_DIR = out
    msr.N_MONTE_CARLO = a.n
    msr.SCENARIO_DEPTH_COL[a.scenario] = col
    buffer_config = load_buffer_config("config/buffer_distributions.yaml")
    power_coupling = msr.load_power_coupling()
    if power_coupling is not None:
        sys.exit("power coupling came back enabled despite POWER_COUPLING=0 — stop")
    print(f"SIM_DIR -> {msr.SIM_DIR} | N_MC -> {msr.N_MONTE_CARLO} | mode -> {a.mode}")

    t1 = time.time()
    _mc, runs = msr.run_scenario(a.scenario, nodes_gdf, buffer_config, power_coupling)
    print(f"scenario wall time: {(time.time() - t1) / 60:.1f} min | runs: {len(runs)}")

    if a.no_check or a.mode in ("arrival", "threshold"):
        print("no reduction check for this mode (only static/static_peak reduce to the frozen run)"); return
    rc = subprocess.call([sys.executable, "scripts/dynamic_forcing_reduction_check.py",
                          "--ref", str(ref_runs), "--test", str(stale), "--first", str(a.n)])
    sys.exit(rc)


if __name__ == "__main__":
    main()
