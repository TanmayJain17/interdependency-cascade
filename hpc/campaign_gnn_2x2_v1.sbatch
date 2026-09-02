#!/bin/bash
#SBATCH --job-name=gnn-2x2-v1
#SBATCH --account=torch_pr_1108_tandon_advanced
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --gres=gpu:l40s:1
#SBATCH --partition=l40s_public
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=logs/gnn-2x2-%A_%a.out
# CascadeGNN 2x2: labels {static, dynamic-arrival} x inputs {mask, mask+timing}.
# The trained twin (gnn_twin_v1/legacy) IS the (static, mask) cell; this array
# fills the other three:
#   task 0  static_timing   legacy_v1_n1000 labels + timing features
#   task 1  dynamic_mask    dynamic_v1/arrival_h360 labels, mask only
#   task 2  dynamic_timing  dynamic_v1/arrival_h360 labels + timing features
# Identical config, seed, and 5-way holdout across all four cells - the only
# differences are the label directory and the input features. Questions isolated:
#   (a) do arrival/duration/time-to-peak input features add skill on static labels?
#   (b) are arrival-ordered labels harder to fit than static ones?
#   (c) do timing features close whatever gap (b) opens?
#
# Smoke first (cheap-before-expensive):   SMOKE=1 sbatch hpc/campaign_gnn_2x2_v1.sbatch
# Production:                             sbatch hpc/campaign_gnn_2x2_v1.sbatch

source $SCRATCH/miniforge3/bin/activate flood
cd $SCRATCH/interdependency-cascade
mkdir -p logs
export PYTHONUNBUFFERED=1

ARMS=(static_timing dynamic_mask dynamic_timing)
LABEL_DIRS=("$SCRATCH/results/legacy_v1_n1000" "$SCRATCH/results/dynamic_v1/arrival_h360" "$SCRATCH/results/dynamic_v1/arrival_h360")
TIMING_FLAGS=(1 0 1)

ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}
LABELS=${LABEL_DIRS[$SLURM_ARRAY_TASK_ID]}
TIMING=${TIMING_FLAGS[$SLURM_ARRAY_TASK_ID]}

export SCENARIO_SET=jesse22
export HETERODATA=data/graph/nyc_infra_heterodata_v1_frozen_failover_edges.pt
export CASCADE_RESULTS_DIR=$LABELS
# Same dir on purpose (twin rule): a scenario missing from the campaign must
# RAISE, never silently fall back to stale synthetic outputs.
export SYN_RESULTS_DIR=$LABELS
export GNN_CKPT_DIR=$SCRATCH/results/gnn_2x2_v1/${ARM}
mkdir -p "$GNN_CKPT_DIR"

# Timing features: exported ONLY in the timing arms. The mask arm leaves the
# env var UNSET so it runs the exact default build already proven byte-identical
# to the trained twin - stronger than trusting a "0" code path.
if [[ "$TIMING" == "1" ]]; then
  export GNN_TIMING_FEATURES=1
  export GNN_TIMING_CSV=data/flood/timing/node_timing_jesse22_v1.csv
fi

HOLDOUT="geoclaw_2050,syn_ts_914_6_1p2955,syn_ts_258_9_3p0022,syn_ts_605_5_3p7563,syn_ts_808_27_3p7885"
echo "arm=$ARM  labels=$CASCADE_RESULTS_DIR  timing=${GNN_TIMING_FEATURES:-unset}  ckpt=$GNN_CKPT_DIR  smoke=${SMOKE:-0}"

# ---- fail-fast gates -------------------------------------------------------
# Gate 1: CUDA actually available. Gate 2: frozen v1 heterodata has EXACTLY
# the v1 shape (6,231 total / telecom 4,150) - a v2 graph would not crash,
# it would silently mislabel.
python - <<'PYEOF'
import os, sys, torch
assert torch.cuda.is_available(), "CUDA not available on this node"
print(f"[gate] CUDA OK: {torch.cuda.get_device_name(0)}")
d = torch.load(os.environ["HETERODATA"], weights_only=False)
counts = {nt: d[nt].num_nodes for nt in d.node_types}
total = sum(counts.values())
ok = total == 6231 and counts.get("telecom") == 4150
print(f"[gate] heterodata {os.environ['HETERODATA']}: {counts} "
      f"total={total} -> {'OK' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
PYEOF

# Gate 3: the label campaign is complete - exactly 23 per-scenario result files
# (20 synthetic + GeoClaw trio). Catches pointing at a partial or wrong dir.
NRES=$(ls "$CASCADE_RESULTS_DIR"/cascade_results_*.json 2>/dev/null | wc -l)
echo "[gate] $CASCADE_RESULTS_DIR cascade_results files: $NRES (need 23)"
[[ "$NRES" == "23" ]] || exit 1

# Gate 4 (timing arms only): the timing table exists and carries the feature columns.
if [[ "${GNN_TIMING_FEATURES:-0}" == "1" ]]; then
  [[ -s "$GNN_TIMING_CSV" ]] || { echo "[gate] FAIL: $GNN_TIMING_CSV missing or empty"; exit 1; }
  head -1 "$GNN_TIMING_CSV" | grep -q "arrival_h" || { echo "[gate] FAIL: arrival_h column not in $GNN_TIMING_CSV"; exit 1; }
  echo "[gate] timing table OK: $(wc -l < "$GNN_TIMING_CSV") rows"
fi

# ---- training --------------------------------------------------------------
if [[ "${SMOKE:-0}" == "1" ]]; then
  python -m src.gnn.train --mode train --device cuda --seed 42 \
    --epochs 1 --train_subset 50 --val_subset 20 \
    --holdout_scenario "$HOLDOUT"
else
  python -m src.gnn.train --mode train --device cuda --seed 42 \
    --epochs 6 \
    --holdout_scenario "$HOLDOUT"
fi
