#!/bin/bash
#SBATCH --job-name=gissr48-v1
#SBATCH --account=torch_pr_1108_tandon_advanced
#SBATCH --partition=cpu_short
#SBATCH --array=0-97%16
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --time=01:30:00
#SBATCH --requeue
#SBATCH --output=logs/gissr48-v1-%A_%a.out
#
# Label campaign gissr48_v1 (Week 29): the 48 new GISSR storms of data/flood/gissr48_manifest_v1.csv
# (40 for training, 8 held out), N=1000 Monte Carlo, legacy arm, in the two label modes the GNN uses:
#   static   seeds at t = 0, frozen grid [0, 6, 24, 48, 96]        -> $SCRATCH/results/gissr48_v1/static
#   arrival  seeds at hydrograph arrival, 360-h post-peak horizon   -> $SCRATCH/results/gissr48_v1/arrival_h360
# Same engine, seeds (fragility 42, buffer 43), graph, clock (peak) and offsets as legacy_v1_n1000 and
# dynamic_v1/arrival_h360. No file under src/ is changed; scripts/gissr48_label_run_v1.py is a thin wrapper.
#
# Task i in 0..95 -> mode = i / 48 (0 static, 1 arrival), storm = line (i % 48) + 1 of hpc/scenarios_gissr48.txt.
# Tasks 96, 97 = identity gate: the OLD storm syn_ts_914_6_1p2955 through the same wrapper, static and
# arrival, compared record for record with the frozen campaign files (exit 2 on any difference).
#
# Acceptance: 98 COMPLETED; 96 "LABEL RUN OK" lines; 2 "IDENTITY GATE: PASS" lines;
#             48 cascade_results_nyc_*.json files in each of the two output folders.
# Cheap first:  sbatch --test-only hpc/campaign_gissr48_v1.sbatch
#               SMOKE=1 sbatch --array=96,97,10,58,47,95 hpc/campaign_gissr48_v1.sbatch     (N=20)
# Production:   sbatch hpc/campaign_gissr48_v1.sbatch

set -euo pipefail
export PYTHONUNBUFFERED=1
mkdir -p logs

source "$SCRATCH/miniforge3/bin/activate" flood
cd "$SCRATCH/interdependency-cascade"

N_NEW=48
I=$SLURM_ARRAY_TASK_ID
N_MC=${SMOKE:+20}; N_MC=${N_MC:-1000}
ROOT_OUT="$SCRATCH/results/gissr48_v1"; [[ "${SMOKE:-0}" == "1" ]] && ROOT_OUT="$SCRATCH/results/gissr48_v1_smoke"

export NODES_IN=data/graph_v1_frozen/nyc_infra_nodes_all_flood.geojson
export GRAPH_IN=data/graph_v1_frozen/nyc_infra_graph_all_flood.graphml
export DYNAMIC_OFFSETS="0 6 24 48 96 144 240 360"      # read only in arrival mode; same as dynamic_v1
export POWER_COUPLING=0
DYN_CFG=config/dynamic_forcing_gissr48_v1.yaml

# ---- fail-fast gates ---------------------------------------------------------------------------
for f in "$NODES_IN" "$GRAPH_IN" "$DYN_CFG" hpc/scenarios_gissr48.txt scripts/gissr48_label_run_v1.py \
         data/flood/gissr48_node_depths_v1.csv data/flood/jesse22_node_depths_v1.csv \
         data/flood/timing/node_timing_jesse22_plus_gissr48_v1.csv data/flood/timing/node_timing_jesse22_plus_gissr48_v1_summary.csv; do
  [[ -f "$f" ]] || { echo "GATE FAIL: missing $f"; exit 2; }
done
[[ "$(grep -c . hpc/scenarios_gissr48.txt)" == "$N_NEW" ]] || { echo "GATE FAIL: scenario list is not $N_NEW lines"; exit 2; }
python - <<'PY' || exit 2
import geopandas as gpd, os
n = len(gpd.read_file(os.environ["NODES_IN"]))
assert n == 6231, f"GATE FAIL: NODES_IN has {n} nodes, expected 6231 (v1 frozen)"
print("gate ok: 6231 nodes (v1 frozen)")
PY

if (( I < 2 * N_NEW )); then
  MODES=(static arrival)
  MODE=${MODES[$(( I / N_NEW ))]}
  SC=$(sed -n "$(( I % N_NEW + 1 ))p" hpc/scenarios_gissr48.txt)
  [[ -n "$SC" ]] || { echo "GATE FAIL: empty scenario for task $I"; exit 2; }
  OUT="$ROOT_OUT/static"; [[ "$MODE" == "arrival" ]] && OUT="$ROOT_OUT/arrival_h360"
  echo "task=$I mode=$MODE scenario=$SC n_mc=$N_MC out=$OUT commit=$(git rev-parse --short HEAD)"
  python scripts/gissr48_label_run_v1.py --scenario "$SC" --mode "$MODE" --n "$N_MC" --out "$OUT" \
      --depth-csv data/flood/gissr48_node_depths_v1.csv --dyn-config "$DYN_CFG"
elif (( I == 2 * N_NEW )); then
  SC=syn_ts_914_6_1p2955
  echo "task=$I IDENTITY static scenario=$SC n_mc=$N_MC commit=$(git rev-parse --short HEAD)"
  python scripts/gissr48_label_run_v1.py --scenario "$SC" --mode static --n "$N_MC" --out "$ROOT_OUT/identity_static" \
      --depth-csv data/flood/jesse22_node_depths_v1.csv --dyn-config "$DYN_CFG" --check-dir "$SCRATCH/results/legacy_v1_n1000"
elif (( I == 2 * N_NEW + 1 )); then
  SC=syn_ts_914_6_1p2955
  echo "task=$I IDENTITY arrival scenario=$SC n_mc=$N_MC commit=$(git rev-parse --short HEAD)"
  python scripts/gissr48_label_run_v1.py --scenario "$SC" --mode arrival --n "$N_MC" --out "$ROOT_OUT/identity_arrival" \
      --depth-csv data/flood/jesse22_node_depths_v1.csv --dyn-config "$DYN_CFG" --check-dir "$SCRATCH/results/dynamic_v1/arrival_h360"
else
  echo "GATE FAIL: task id $I out of range"; exit 2
fi
