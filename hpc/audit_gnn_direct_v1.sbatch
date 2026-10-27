#!/bin/bash
#SBATCH --job-name=gnn-direct-audit
#SBATCH --account=torch_pr_1108_tandon_advanced
#SBATCH --array=0-23
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:l40s:1
#SBATCH --partition=l40s_public
#SBATCH --time=00:40:00
#SBATCH --output=logs/gnn-direct-audit-%A_%a.out
# Direct model, Week 32: score the 24 checkpoints of hpc/campaign_gnn_direct_v1.sbatch with
# scripts/week31_direct_audit_v1.py (E1-E5 against the simulator's storm-level probabilities, baselines B0 fragility,
# B1 lookup, B2 the two-stage storm-summary checkpoint of the same cell and PyTorch seed over 100 seed sets, and the
# split-half floor). Same task mapping as the campaign. Refuses a checkpoint whose training has not finished.
#
#   sbatch hpc/audit_gnn_direct_v1.sbatch            then
#   python scripts/week31_direct_table_v1.py --dir $SCRATCH/results/gnn_direct_numbers_v1 \
#       --fingerprint analysis/gnn/week31_direct_fingerprint_static_v1.json,analysis/gnn/week31_direct_fingerprint_arrival_v1.json
# Lines to read in each log: "[gate] G2 ... OK", "G6 inputs match data.py: True", "ALL GATES PASS".

source $SCRATCH/miniforge3/bin/activate flood
cd $SCRATCH/interdependency-cascade
mkdir -p logs
export PYTHONUNBUFFERED=1
unset GNN_STORM_CONTEXT GNN_TIMING_FEATURES GNN_TIMING_CSV GNN_TARGET SCENARIO_SET CASCADE_RESULTS_DIR SYN_RESULTS_DIR GNN_CKPT_DIR HETERODATA

[[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] || { echo "[gate] FAIL: run as an array task"; exit 1; }
I=$SLURM_ARRAY_TASK_ID
(( I >= 0 && I < 24 )) || { echo "[gate] FAIL: task id $I out of range"; exit 1; }
CELLS=(static_plain static_timing arrival_plain arrival_timing)
TS_CELLS=(static_mask static_timing dynamic_mask dynamic_timing)     # the two-stage cell that B2 comes from
MODES=(static static arrival arrival)
TIMING_FLAGS=(0 1 0 1)
SIZES=(17 34 50)
CELL_I=$(( I % 4 )); SIZE_I=$(( (I / 4) % 3 )); TSEED=$(( 1 + I / 12 ))
CELL=${CELLS[$CELL_I]}; MODE=${MODES[$CELL_I]}; TIMING=${TIMING_FLAGS[$CELL_I]}; SIZE=${SIZES[$SIZE_I]}
NAME=direct_${CELL}_n${SIZE}_s${TSEED}
CKPT=$SCRATCH/results/gnn_direct_v1/$NAME/best.pt
TS=$SCRATCH/results/gnn_morestorms_v1/mix57_full_ctx_${TS_CELLS[$CELL_I]}_s${TSEED}/best.pt
CACHE=$SCRATCH/results/gnn_direct_targets_v1/$MODE
OUT=$SCRATCH/results/gnn_direct_numbers_v1/$NAME.json
mkdir -p "$(dirname "$OUT")"

export GNN_TARGET=marginal
export SCENARIO_SET=w29_all
export HETERODATA=data/graph/nyc_infra_heterodata_v1_frozen_failover_edges.pt
export GNN_TIMING_CSV=data/flood/timing/node_timing_jesse22_plus_gissr48_v1.csv
[[ "$TIMING" == "1" ]] && export GNN_TIMING_FEATURES=1
echo "task=$I name=$NAME ckpt=$CKPT two_stage=$TS cache=$CACHE out=$OUT commit=$(git rev-parse --short HEAD) branch=$(git rev-parse --abbrev-ref HEAD)"

# ---- gates --------------------------------------------------------------------------------------------
[[ -f "$CKPT" ]] || { echo "[gate] FAIL: no checkpoint $CKPT"; exit 1; }
[[ -f "$(dirname "$CKPT")/history.json" ]] || { echo "[gate] FAIL: training of $NAME not finished (no history.json)"; exit 1; }
[[ -f "$TS" ]] || { echo "[gate] FAIL: two-stage checkpoint for B2 missing: $TS"; exit 1; }
[[ -f "$(dirname "$TS")/history.json" ]] || { echo "[gate] FAIL: two-stage checkpoint $TS has no history.json"; exit 1; }
[[ $(ls "$CACHE"/*.pt 2>/dev/null | wc -l) -ge 70 ]] || { echo "[gate] FAIL: target cache $CACHE incomplete"; exit 1; }
python -c "import torch; assert torch.cuda.is_available(); print('[gate] CUDA OK:', torch.cuda.get_device_name(0))" || exit 1

python scripts/week31_direct_audit_v1.py --ckpt "$CKPT" --cache "$CACHE" --two-stage-ckpt "$TS" --seed-sets 100 \
    --device cuda --out "$OUT"
