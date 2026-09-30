#!/bin/bash
#SBATCH --job-name=gnn-mono-enforce
#SBATCH --account=torch_pr_1108_tandon_advanced
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --gres=gpu:l40s:1
#SBATCH --partition=l40s_public
#SBATCH --time=01:30:00
#SBATCH --output=logs/gnn-mono-enforce-%A_%a.out
# Week 28 audit: WHERE to enforce "P(failed by t) can only go up" - inside the model
# (hazard head, Week 27) or on the output side (running max / sort / isotonic regression
# applied to the independent head's outputs)? No training. Eight saved checkpoints:
#   tasks 0-3  independent head: the 2x2 v1 checkpoints (same four as audit_gnn_2x2_mono_v1)
#   tasks 4-7  hazard head:      $SCRATCH/results/gnn_hazard_v1/<arm>/best.pt (job 18286484)
# Each task rebuilds the exact val split and held-out evaluation of `train.py --mode evalckpt`
# and scores four readings of the same outputs (raw, cummax, sort, isotonic), plus the size
# distribution of the dips. src/gnn/model.py and src/gnn/train.py are NOT modified.
#
# Gates inside scripts/week28_mono_enforce_audit_v1.py (exit 1 on any failure):
#   G1 raw reading == train.eval_step; G2 raw val loss reproduces the checkpoint's saved
#   val_loss (and, for tasks 0-3, the Week 27 audit JSON tracked in the repo);
#   invariants: fixed readings have 0 violations, sort/isotonic never raise BCE or Brier,
#   and all three fixes are no-ops on a hazard-head checkpoint.
#
# Smoke first (cheap-before-expensive):   SMOKE=1 sbatch hpc/audit_gnn_mono_enforce_v1.sbatch
# Production:                             sbatch hpc/audit_gnn_mono_enforce_v1.sbatch
# Priority if GPUs are scarce:            sbatch --array=3,7,1,5 hpc/audit_gnn_mono_enforce_v1.sbatch
TWIN_CKPT="$SCRATCH/results/gnn_twin_v1/legacy/best.pt"

source $SCRATCH/miniforge3/bin/activate flood
cd $SCRATCH/interdependency-cascade
mkdir -p logs
export PYTHONUNBUFFERED=1

ARMS=(static_mask static_timing dynamic_mask dynamic_timing)
LABEL_DIRS=("$SCRATCH/results/legacy_v1_n1000" "$SCRATCH/results/legacy_v1_n1000" "$SCRATCH/results/dynamic_v1/arrival_h360" "$SCRATCH/results/dynamic_v1/arrival_h360")
TIMING_FLAGS=(0 1 0 1)
INDEP_CKPTS=("$TWIN_CKPT" "$SCRATCH/results/gnn_2x2_v1/static_timing/best.pt" "$SCRATCH/results/gnn_2x2_v1/dynamic_mask/best.pt" "$SCRATCH/results/gnn_2x2_v1/dynamic_timing/best.pt")

CELL=$(( SLURM_ARRAY_TASK_ID % 4 ))
ARM=${ARMS[$CELL]}
LABELS=${LABEL_DIRS[$CELL]}
TIMING=${TIMING_FLAGS[$CELL]}
if (( SLURM_ARRAY_TASK_ID < 4 )); then
  HEAD=independent
  CKPT=${INDEP_CKPTS[$CELL]}
  REF="data/hpc_results_aug2026/gnn_audit_mono_v1/${ARM}.eval.json"   # tracked since 9fc6c08
  [[ -s "$REF" ]] || { echo "[gate] FAIL: Week 27 audit JSON missing: $REF (git pull?)"; exit 1; }
  REF_ARG="--ref_eval $REF"
else
  HEAD=hazard
  CKPT="$SCRATCH/results/gnn_hazard_v1/${ARM}/best.pt"
  REF_ARG=""
fi
[[ -s "$CKPT" ]] || { echo "[gate] FAIL: checkpoint missing: $CKPT"; exit 1; }
[[ -s scripts/week28_mono_enforce_audit_v1.py ]] || { echo "[gate] FAIL: audit script missing"; exit 1; }

export SCENARIO_SET=jesse22
export HETERODATA=data/graph/nyc_infra_heterodata_v1_frozen_failover_edges.pt
export CASCADE_RESULTS_DIR=$LABELS
# Same dir on purpose (twin rule): a missing scenario must RAISE, never fall back.
export SYN_RESULTS_DIR=$LABELS
if [[ "$TIMING" == "1" ]]; then
  export GNN_TIMING_FEATURES=1
  export GNN_TIMING_CSV=data/flood/timing/node_timing_jesse22_v1.csv
  [[ -s "$GNN_TIMING_CSV" ]] || { echo "[gate] FAIL: $GNN_TIMING_CSV missing or empty"; exit 1; }
fi
NRES=$(ls "$CASCADE_RESULTS_DIR"/cascade_results_*.json 2>/dev/null | wc -l)
echo "[gate] $CASCADE_RESULTS_DIR cascade_results files: $NRES (need 23)"
[[ "$NRES" == "23" ]] || exit 1
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('[gate] CUDA OK:', torch.cuda.get_device_name(0))" || exit 1

HOLDOUT="geoclaw_2050,syn_ts_914_6_1p2955,syn_ts_258_9_3p0022,syn_ts_605_5_3p7563,syn_ts_808_27_3p7885"
OUT_DIR="$SCRATCH/results/gnn_mono_enforce_v1"
[[ "${SMOKE:-0}" == "1" ]] && OUT_DIR="$SCRATCH/results/gnn_mono_enforce_v1_smoke"
mkdir -p "$OUT_DIR"
echo "arm=$ARM head=$HEAD ckpt=$CKPT labels=$LABELS timing=${GNN_TIMING_FEATURES:-unset} smoke=${SMOKE:-0} out=$OUT_DIR"

if [[ "${SMOKE:-0}" == "1" ]]; then
  python scripts/week28_mono_enforce_audit_v1.py --ckpt "$CKPT" --device cuda --seed 42 \
    --holdout_scenario "$HOLDOUT" --val_subset 20 --holdout_subset 5 \
    --out "$OUT_DIR/${HEAD}_${ARM}.json"
else
  python scripts/week28_mono_enforce_audit_v1.py --ckpt "$CKPT" --device cuda --seed 42 \
    --holdout_scenario "$HOLDOUT" $REF_ARG \
    --out "$OUT_DIR/${HEAD}_${ARM}.json"
fi
