#!/bin/bash
#SBATCH --job-name=gnn-frag-audit
#SBATCH --account=torch_pr_1108_tandon_advanced
#SBATCH --array=0-35
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:l40s:1
#SBATCH --partition=l40s_public
#SBATCH --time=00:30:00
#SBATCH --output=logs/gnn-frag-audit-%A_%a.out
# End-to-end fragility, Week 33: score the 32 checkpoints of hpc/campaign_gnn_fragility_v1.sbatch and re-audit the
# 4 Week 32 checkpoints that form the a0_formula arm, with scripts/week33_fragility_audit_v1.py (E1-E5 against the
# simulator's storm-level probabilities, the formula baseline B0, the split-half floor, the learned fragility
# parameters and curve error, the decomposition error against g_hat, the monotonicity violation rate, the head's
# seed-frequency fit). Refuses a checkpoint whose training has not finished.
#
#   Task i in 0..31: same mapping as the campaign (arm = i % 8, cell = (i / 8) % 2, seed = 1 + i / 16).
#   Task 32..35:     a0_formula = Week 32 checkpoint direct_<cell>_n50_s<seed> (cell = (i - 32) % 2, seed = 1 + (i - 32) / 2),
#                    compared with its Week 32 audit JSON in $SCRATCH/results/gnn_direct_numbers_v1/.
#
#   sbatch hpc/audit_gnn_fragility_v1.sbatch            then
#   python scripts/week33_fragility_table_v1.py --dir $SCRATCH/results/gnn_fragility_numbers_v1 \
#       --fingerprint analysis/gnn/week33_fragility_fingerprint_v1.json
# Lines to read in each log: "G6 identity OK", "[gate] G2 ... OK", "[gate] G3 ... OK", "G4 ... OK; G5 ... OK",
# "learned fragility: ..." (lognormal arms), "[gate] G7 Week 32 match ... OK" (tasks 32-35), "ALL GATES PASS".

source $SCRATCH/miniforge3/bin/activate flood
cd $SCRATCH/interdependency-cascade
mkdir -p logs
export PYTHONUNBUFFERED=1
unset GNN_STORM_CONTEXT GNN_TIMING_FEATURES GNN_TIMING_CSV GNN_TARGET SCENARIO_SET CASCADE_RESULTS_DIR SYN_RESULTS_DIR GNN_CKPT_DIR HETERODATA

[[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] || { echo "[gate] FAIL: run as an array task"; exit 1; }
I=$SLURM_ARRAY_TASK_ID
(( I >= 0 && I < 36 )) || { echo "[gate] FAIL: task id $I out of range"; exit 1; }
ARMS=(a1a_lognormal_l1 a1b_lognormal_l0 a1c_lognormal_l0p1 a1d_lognormal_l10 a1e_lognormal_nodetach a4_sequential a2_mixture a3_blackbox)
CELLS=(static_plain arrival_timing)
MODES=(static arrival)
TIMING_FLAGS=(0 1)
W32=""
if (( I < 32 )); then
  ARM_I=$(( I % 8 )); CELL_I=$(( (I / 8) % 2 )); TSEED=$(( 1 + I / 16 ))
  ARM=${ARMS[$ARM_I]}
  CKPT=$SCRATCH/results/gnn_fragility_v1/e2e_${ARM}_${CELLS[$CELL_I]}_s${TSEED}/best.pt
else
  J=$(( I - 32 )); CELL_I=$(( J % 2 )); TSEED=$(( 1 + J / 2 ))
  ARM=a0_formula
  CKPT=$SCRATCH/results/gnn_direct_v1/direct_${CELLS[$CELL_I]}_n50_s${TSEED}/best.pt
  W32=$SCRATCH/results/gnn_direct_numbers_v1/direct_${CELLS[$CELL_I]}_n50_s${TSEED}.json
fi
CELL=${CELLS[$CELL_I]}; MODE=${MODES[$CELL_I]}; TIMING=${TIMING_FLAGS[$CELL_I]}
NAME=e2e_${ARM}_${CELL}_s${TSEED}
CACHE=$SCRATCH/results/gnn_direct_targets_v1/$MODE
OUT=$SCRATCH/results/gnn_fragility_numbers_v1/$NAME.json
mkdir -p "$(dirname "$OUT")"

export GNN_TARGET=marginal
export SCENARIO_SET=w29_all
export HETERODATA=data/graph/nyc_infra_heterodata_v1_frozen_failover_edges.pt
export GNN_TIMING_CSV=data/flood/timing/node_timing_jesse22_plus_gissr48_v1.csv
[[ "$TIMING" == "1" ]] && export GNN_TIMING_FEATURES=1
echo "task=$I name=$NAME arm=$ARM cell=$CELL ckpt=$CKPT week32_json=${W32:-none} cache=$CACHE out=$OUT commit=$(git rev-parse --short HEAD) branch=$(git rev-parse --abbrev-ref HEAD)"

# ---- gates --------------------------------------------------------------------------------------------
[[ -f "$CKPT" ]] || { echo "[gate] FAIL: no checkpoint $CKPT"; exit 1; }
[[ -f "$(dirname "$CKPT")/history.json" ]] || { echo "[gate] FAIL: training of $NAME not finished (no history.json)"; exit 1; }
[[ -z "$W32" || -f "$W32" ]] || { echo "[gate] FAIL: Week 32 audit JSON $W32 missing"; exit 1; }
[[ $(ls "$CACHE"/*.pt 2>/dev/null | wc -l) -ge 70 ]] || { echo "[gate] FAIL: target cache $CACHE incomplete"; exit 1; }
python -c "import torch; assert torch.cuda.is_available(); print('[gate] CUDA OK:', torch.cuda.get_device_name(0))" || exit 1

# ---- audit --------------------------------------------------------------------------------------------
if [[ -n "$W32" ]]; then
  python scripts/week33_fragility_audit_v1.py --ckpt "$CKPT" --cache "$CACHE" --arm "$ARM" --out "$OUT" --week32_json "$W32" --device cuda
else
  python scripts/week33_fragility_audit_v1.py --ckpt "$CKPT" --cache "$CACHE" --arm "$ARM" --out "$OUT" --device cuda
fi
