#!/bin/bash
#SBATCH --job-name=dyn-v1
#SBATCH --account=torch_pr_1108_tandon_advanced
#SBATCH --partition=cpu_short
#SBATCH --array=0-45%16
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --time=01:30:00
#SBATCH --requeue
#SBATCH --output=logs/dyn-v1-%A_%a.out
#
# Dynamic-forcing campaign v1 (Week 23): every jesse22 scenario, N=1000 Monte Carlo, legacy arm,
# two modes on the 360-h post-peak horizon:
#   static_peak : all flood seeds at the storm peak (pure time translation of the frozen campaign;
#                 the driver checks it record-for-record against $SCRATCH/results/legacy_v1_n1000)
#   arrival     : each flood seed lands when the hydrograph first wets it (World 1 dynamic labels)
# Task i -> mode = i / 23, scenario = i % 23.  Outputs: $SCRATCH/results/dynamic_v1/<mode>_h360/
# Acceptance: 46 COMPLETED; 23 "EXACT REDUCTION OK" lines across the static_peak logs;
#             46 cascade_results_nyc_*.json files.
# Cheap first:  sbatch --test-only hpc/campaign_dynamic_v1.sbatch ; SMOKE=1 sbatch --array=0,23 hpc/campaign_dynamic_v1.sbatch  (N=20)

set -euo pipefail
export PYTHONUNBUFFERED=1
mkdir -p logs

SCENARIOS=(
  syn_ts_173_2_0p8859 syn_ts_662_3_0p9973 syn_ts_192_7_1p0240 syn_ts_156_15_1p066 syn_ts_810_14_1p1460
  syn_ts_708_13_1p1567 syn_ts_914_6_1p2955 syn_ts_436_11_1p3524 syn_ts_570_2_1p5351 syn_ts_463_29_1p6644
  syn_ts_321_19_1p7614 syn_ts_831_27_1p9452 syn_ts_192_15_2p5752 syn_ts_880_9_2p6632 syn_ts_514_13_2p7829
  syn_ts_258_9_3p0022 syn_ts_914_26_3p4047 syn_ts_999_12_3p44 syn_ts_605_5_3p7563 syn_ts_808_27_3p7885
  geoclaw_2026 geoclaw_2050 geoclaw_2080
)
MODES=(static_peak arrival)
N_SC=${#SCENARIOS[@]}
MODE=${MODES[$(( SLURM_ARRAY_TASK_ID / N_SC ))]}
SC=${SCENARIOS[$(( SLURM_ARRAY_TASK_ID % N_SC ))]}
N_MC=${SMOKE:+20}; N_MC=${N_MC:-1000}

source "$SCRATCH/miniforge3/bin/activate" flood
cd "$SCRATCH/interdependency-cascade"

# ---- fail-fast gates (graph identity + inputs) -------------------------------------------------
export NODES_IN=data/graph_v1_frozen/nyc_infra_nodes_all_flood.geojson
export GRAPH_IN=data/graph_v1_frozen/nyc_infra_graph_all_flood.graphml
for f in "$NODES_IN" "$GRAPH_IN" data/flood/timing/node_timing_jesse22_v1.csv config/dynamic_forcing.yaml \
         "$SCRATCH/results/legacy_v1_n1000/cascade_results_nyc_${SC}.json"; do
  [[ -f "$f" ]] || { echo "GATE FAIL: missing $f"; exit 2; }
done
python - <<'PY' || exit 2
import geopandas as gpd, os
n = len(gpd.read_file(os.environ["NODES_IN"]))
assert n == 6231, f"GATE FAIL: NODES_IN has {n} nodes, expected 6231 (v1 frozen)"
print("gate ok: 6231 nodes (v1 frozen)")
PY

export DYNAMIC_OFFSETS="0 6 24 48 96 144 240 360"
export POWER_COUPLING=0
OUT="$SCRATCH/results/dynamic_v1/${MODE}_h360"
echo "task=$SLURM_ARRAY_TASK_ID mode=$MODE scenario=$SC n_mc=$N_MC out=$OUT commit=$(git rev-parse --short HEAD)"

python scripts/dynamic_forcing_reduction_run.py --scenario "$SC" --n "$N_MC" --mode "$MODE" \
    --ref-dir "$SCRATCH/results/legacy_v1_n1000" --out "$OUT"
