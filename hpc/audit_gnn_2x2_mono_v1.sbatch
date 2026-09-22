#!/bin/bash
#SBATCH --job-name=gnn-audit-mono
#SBATCH --account=torch_pr_1108_tandon_advanced
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --gres=gpu:l40s:1
#SBATCH --partition=l40s_public
#SBATCH --time=01:00:00
#SBATCH --output=logs/gnn-audit-mono-%A_%a.out
# Week 27 audit: re-evaluate the four 2x2 v1 checkpoints (independent head)
# with --mode evalckpt and report the monotonicity-violation fraction
# (share of (node, horizon) pairs where P(failed by t_{k+1}) < P(failed by t_k))
# on the SAME val split and held-out scenarios as training. No training.
# The val loss printed here must reproduce the 2x2 v1 best val losses
# (0.1082 / 0.0981 / 0.1877 / 0.1653) - that is the gate that the split and
# inputs were rebuilt correctly.
#
# TWIN_CKPT: the (static, mask) twin checkpoint. Fill in before submitting.
TWIN_CKPT="$SCRATCH/results/gnn_twin_v1/legacy/best.pt"   # <-- verify with: ls $SCRATCH/results/gnn_twin_v1/

source $SCRATCH/miniforge3/bin/activate flood
cd $SCRATCH/interdependency-cascade
mkdir -p logs
export PYTHONUNBUFFERED=1

ARMS=(static_mask static_timing dynamic_mask dynamic_timing)
CKPTS=("$TWIN_CKPT" "$SCRATCH/results/gnn_2x2_v1/static_timing/best.pt" "$SCRATCH/results/gnn_2x2_v1/dynamic_mask/best.pt" "$SCRATCH/results/gnn_2x2_v1/dynamic_timing/best.pt")
LABEL_DIRS=("$SCRATCH/results/legacy_v1_n1000" "$SCRATCH/results/legacy_v1_n1000" "$SCRATCH/results/dynamic_v1/arrival_h360" "$SCRATCH/results/dynamic_v1/arrival_h360")
TIMING_FLAGS=(0 1 0 1)

ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}
CKPT=${CKPTS[$SLURM_ARRAY_TASK_ID]}
LABELS=${LABEL_DIRS[$SLURM_ARRAY_TASK_ID]}
TIMING=${TIMING_FLAGS[$SLURM_ARRAY_TASK_ID]}
[[ -s "$CKPT" ]] || { echo "[gate] FAIL: checkpoint missing: $CKPT"; exit 1; }

export SCENARIO_SET=jesse22
export HETERODATA=data/graph/nyc_infra_heterodata_v1_frozen_failover_edges.pt
export CASCADE_RESULTS_DIR=$LABELS
export SYN_RESULTS_DIR=$LABELS
if [[ "$TIMING" == "1" ]]; then
  export GNN_TIMING_FEATURES=1
  export GNN_TIMING_CSV=data/flood/timing/node_timing_jesse22_v1.csv
fi
HOLDOUT="geoclaw_2050,syn_ts_914_6_1p2955,syn_ts_258_9_3p0022,syn_ts_605_5_3p7563,syn_ts_808_27_3p7885"
mkdir -p "$SCRATCH/results/gnn_audit_mono_v1"
echo "arm=$ARM ckpt=$CKPT labels=$LABELS timing=${GNN_TIMING_FEATURES:-unset}"

python -m src.gnn.train --mode evalckpt --device cuda --seed 42 \
  --ckpt "$CKPT" --holdout_scenario "$HOLDOUT" \
  --eval_out "$SCRATCH/results/gnn_audit_mono_v1/${ARM}.eval.json"
