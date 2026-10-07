#!/bin/bash
#SBATCH --job-name=gnn-morestorms-numbers
#SBATCH --account=torch_pr_1108_tandon_advanced
#SBATCH --array=0-51
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96GB
#SBATCH --gres=gpu:l40s:1
#SBATCH --partition=l40s_public
#SBATCH --time=04:00:00
#SBATCH --output=logs/gnn-morestorms-numbers-%A_%a.out
# Week 29: scores the checkpoints of hpc/campaign_gnn_morestorms_v1.sbatch with
# scripts/week29_numbers_audit_v2.py. No training.
#   tasks 0-47   same index as the training array (variant, seed, arm, cell)
#   tasks 48-51  the four existing hazard checkpoints of gnn_hazard_v1 (17 old storms, plain, PyTorch seed
#                not recorded): a third replicate of old17 plain, called seed 0
# HOLD chooses the held-out storms:
#   HOLD=old5   (old17 tasks only; needs only the frozen labels)  the 5 old held-out storms, storm set
#               jesse22, exactly the setting of the Week 29 numbers sheet -> $SCRATCH/results/gnn_morestorms_numbers_v1_old5
#   HOLD=all13  (default; needs the gissr48 labels)  the 5 old + 8 new held-out storms; old17 checkpoints
#               are scored under storm set w29_old, whose train/val split equals jesse22's
#               -> $SCRATCH/results/gnn_morestorms_numbers_v1
# Each checkpoint is scored with the train/val subsets it was trained with (--design_subsets), so the
# reproduction gate G2 applies to every task. Baselines are built from that checkpoint's training runs.
#
# Storm-context question, old labels only:  HOLD=old5 sbatch --export=ALL,HOLD=old5 --array=0-3,12-15,24-27,36-39,48-51 hpc/audit_gnn_morestorms_v1.sbatch
# More-storms question, after the labels:   sbatch --array=<the training task ids that finished>,48-51 hpc/audit_gnn_morestorms_v1.sbatch
# Smoke (reads the training smoke's checkpoints):  SMOKE=1 HOLD=old5 sbatch --export=ALL,SMOKE=1,HOLD=old5 --array=3,27,51 hpc/audit_gnn_morestorms_v1.sbatch

source $SCRATCH/miniforge3/bin/activate flood
cd $SCRATCH/interdependency-cascade
mkdir -p logs
export PYTHONUNBUFFERED=1

# sbatch passes the submitting shell's environment along: clear every switch this script sets, so a
# variable left over in the login shell cannot turn a plain or mask task into something else
unset GNN_STORM_CONTEXT GNN_TIMING_FEATURES GNN_TIMING_CSV SCENARIO_SET CASCADE_RESULTS_DIR SYN_RESULTS_DIR GNN_CKPT_DIR HETERODATA

CELLS=(static_mask static_timing dynamic_mask dynamic_timing)
OLD_DIRS=("$SCRATCH/results/legacy_v1_n1000" "$SCRATCH/results/legacy_v1_n1000" "$SCRATCH/results/dynamic_v1/arrival_h360" "$SCRATCH/results/dynamic_v1/arrival_h360")
NEW_DIRS=("$SCRATCH/results/gissr48_v1/static" "$SCRATCH/results/gissr48_v1/static" "$SCRATCH/results/gissr48_v1/arrival_h360" "$SCRATCH/results/gissr48_v1/arrival_h360")
TIMING_FLAGS=(0 1 0 1)
ARMS=(old17 mix57_same mix57_full)
VARIANTS=(plain ctx)

[[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] || { echo "[gate] FAIL: run as an array task (sbatch --array=...)"; exit 1; }
case "${SMOKE:-0}" in 0|1) ;; *) echo "[gate] FAIL: SMOKE must be 0 or 1"; exit 1 ;; esac
HOLD=${HOLD:-all13}
case "$HOLD" in old5|all13) ;; *) echo "[gate] FAIL: HOLD must be old5 or all13"; exit 1 ;; esac
I=$SLURM_ARRAY_TASK_ID
(( I >= 0 && I < 52 )) || { echo "[gate] FAIL: task id $I out of range"; exit 1; }
CELL_I=$(( I % 4 )); CELL=${CELLS[$CELL_I]}; TIMING=${TIMING_FLAGS[$CELL_I]}
CK_ROOT=$SCRATCH/results/gnn_morestorms_v1; OUT_DIR=$SCRATCH/results/gnn_morestorms_numbers_v1
[[ "$HOLD" == "old5" ]] && OUT_DIR=${OUT_DIR}_old5
if [[ "${SMOKE:-0}" == "1" ]]; then CK_ROOT=${CK_ROOT}_smoke; OUT_DIR=${OUT_DIR}_smoke; fi
if (( I < 48 )); then
  VAR_I=$(( I / 24 )); J=$(( I % 24 )); ARM_I=$(( (J / 4) % 3 )); TSEED=$(( 1 + J / 12 ))
  ARM=${ARMS[$ARM_I]}; VAR=${VARIANTS[$VAR_I]}
  NAME=${ARM}_${VAR}_${CELL}_s${TSEED}
  CKPT=$CK_ROOT/$NAME/best.pt
else
  ARM=old17; VAR=plain
  NAME=old17_plain_${CELL}_s0
  CKPT=$SCRATCH/results/gnn_hazard_v1/${CELL}/best.pt
fi
[[ -s "$CKPT" ]] || { echo "[gate] FAIL: checkpoint missing: $CKPT"; exit 1; }
# train.py writes history.json only after the last epoch and the held-out evaluation: without it the
# training is still running, was requeued or timed out, and best.pt is from an early epoch
[[ -s "$(dirname "$CKPT")/history.json" ]] || { echo "[gate] FAIL: training not finished (no history.json beside $CKPT)"; exit 1; }
[[ -s scripts/week29_numbers_audit_v2.py ]] || { echo "[gate] FAIL: audit script missing"; exit 1; }
if [[ "$HOLD" == "old5" && "$ARM" != "old17" ]]; then echo "[gate] FAIL: HOLD=old5 applies to old17 checkpoints only (task $I is $ARM)"; exit 1; fi

export HETERODATA=data/graph/nyc_infra_heterodata_v1_frozen_failover_edges.pt
export CASCADE_RESULTS_DIR=${OLD_DIRS[$CELL_I]}
OLD_HOLDOUT="geoclaw_2050,syn_ts_914_6_1p2955,syn_ts_258_9_3p0022,syn_ts_605_5_3p7563,syn_ts_808_27_3p7885"
if [[ "$HOLD" == "old5" ]]; then
  export SCENARIO_SET=jesse22
  export SYN_RESULTS_DIR=$CASCADE_RESULTS_DIR
  HOLDOUT="$OLD_HOLDOUT"
else
  export SCENARIO_SET=w29_all; [[ "$ARM" == "old17" ]] && export SCENARIO_SET=w29_old
  export SYN_RESULTS_DIR=${NEW_DIRS[$CELL_I]}
  NEW_HOLDOUT=$(python -c "import csv; print(','.join(r['tag'] for r in csv.DictReader(open('data/flood/gissr48_manifest_v1.csv')) if r['role'] == 'test'))")
  HOLDOUT="$OLD_HOLDOUT,$NEW_HOLDOUT"
fi
if [[ "$TIMING" == "1" ]]; then
  export GNN_TIMING_FEATURES=1
  export GNN_TIMING_CSV=data/flood/timing/node_timing_jesse22_plus_gissr48_v1.csv
  [[ -s "$GNN_TIMING_CSV" ]] || { echo "[gate] FAIL: $GNN_TIMING_CSV missing or empty"; exit 1; }
fi
if [[ "$VAR" == "ctx" ]]; then
  export GNN_STORM_CONTEXT=1
fi
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('[gate] CUDA OK:', torch.cuda.get_device_name(0))" || exit 1

mkdir -p "$OUT_DIR"
echo "task=$I name=$NAME hold=$HOLD set=$SCENARIO_SET ckpt=$CKPT labels=$CASCADE_RESULTS_DIR labels2=$SYN_RESULTS_DIR timing=${GNN_TIMING_FEATURES:-unset} context=${GNN_STORM_CONTEXT:-unset} smoke=${SMOKE:-0} out=$OUT_DIR commit=$(git rev-parse --short HEAD) uncommitted_files=$(git status --porcelain --untracked-files=no | wc -l)"

if [[ "${SMOKE:-0}" == "1" ]]; then
  python scripts/week29_numbers_audit_v2.py --ckpt "$CKPT" --device cuda --seed 42 \
    --holdout_scenario "$HOLDOUT" --val_subset 20 --holdout_subset 5 --train_subset 400 \
    --out "$OUT_DIR/$NAME.json"
else
  python scripts/week29_numbers_audit_v2.py --ckpt "$CKPT" --device cuda --seed 42 \
    --holdout_scenario "$HOLDOUT" --design_subsets \
    --out "$OUT_DIR/$NAME.json"
fi
