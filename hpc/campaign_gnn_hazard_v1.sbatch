#!/bin/bash
#SBATCH --job-name=gnn-hazard-v1
#SBATCH --account=torch_pr_1108_tandon_advanced
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --gres=gpu:l40s:1
#SBATCH --partition=l40s_public
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=logs/gnn-hazard-%A_%a.out
# CascadeGNN Phase 0a (Week 27): the 2x2 of campaign_gnn_2x2_v1.sbatch re-run
# with the HAZARD output head (--head hazard, see src/gnn/model.py). Everything
# else - labels, inputs, holdout, seed, epochs, config - is identical to the
# 2x2 v1 cells, so the only difference per cell is the head:
#   task 0  static_mask     legacy_v1_n1000 labels, mask only        (twin cell: 0.1082)
#   task 1  static_timing   legacy_v1_n1000 labels + timing features (2x2 v1: 0.0981)
#   task 2  dynamic_mask    dynamic_v1/arrival_h360 labels, mask only (2x2 v1: 0.1877)
#   task 3  dynamic_timing  dynamic_v1/arrival_h360 labels + timing   (2x2 v1: 0.1653)
# Question isolated: does forcing P(failed by t) to be monotone in t (chained
# per-window hazards, same parameters) change fit? Pre-registered: no harm
# (|delta best val loss| <= 0.005 per cell), mono_viol == 0 by construction.
#
# Smoke first (cheap-before-expensive):   SMOKE=1 sbatch hpc/campaign_gnn_hazard_v1.sbatch
# Production:                             sbatch hpc/campaign_gnn_hazard_v1.sbatch
# Priority if GPUs are scarce:            sbatch --array=3,1 hpc/campaign_gnn_hazard_v1.sbatch

source $SCRATCH/miniforge3/bin/activate flood
cd $SCRATCH/interdependency-cascade
mkdir -p logs
export PYTHONUNBUFFERED=1

ARMS=(static_mask static_timing dynamic_mask dynamic_timing)
LABEL_DIRS=("$SCRATCH/results/legacy_v1_n1000" "$SCRATCH/results/legacy_v1_n1000" "$SCRATCH/results/dynamic_v1/arrival_h360" "$SCRATCH/results/dynamic_v1/arrival_h360")
TIMING_FLAGS=(0 1 0 1)

ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}
LABELS=${LABEL_DIRS[$SLURM_ARRAY_TASK_ID]}
TIMING=${TIMING_FLAGS[$SLURM_ARRAY_TASK_ID]}

export SCENARIO_SET=jesse22
export HETERODATA=data/graph/nyc_infra_heterodata_v1_frozen_failover_edges.pt
export CASCADE_RESULTS_DIR=$LABELS
# Same dir on purpose (twin rule): a scenario missing from the campaign must
# RAISE, never silently fall back to stale synthetic outputs.
export SYN_RESULTS_DIR=$LABELS
export GNN_CKPT_DIR=$SCRATCH/results/gnn_hazard_v1/${ARM}
mkdir -p "$GNN_CKPT_DIR"

# Timing features: exported ONLY in the timing arms (mask arms leave it UNSET,
# exactly as in the 2x2 v1 / twin builds).
if [[ "$TIMING" == "1" ]]; then
  export GNN_TIMING_FEATURES=1
  export GNN_TIMING_CSV=data/flood/timing/node_timing_jesse22_v1.csv
fi

HOLDOUT="geoclaw_2050,syn_ts_914_6_1p2955,syn_ts_258_9_3p0022,syn_ts_605_5_3p7563,syn_ts_808_27_3p7885"
echo "arm=$ARM  head=hazard  labels=$CASCADE_RESULTS_DIR  timing=${GNN_TIMING_FEATURES:-unset}  ckpt=$GNN_CKPT_DIR  smoke=${SMOKE:-0}"

# ---- fail-fast gates (identical to campaign_gnn_2x2_v1.sbatch) ---------------
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

NRES=$(ls "$CASCADE_RESULTS_DIR"/cascade_results_*.json 2>/dev/null | wc -l)
echo "[gate] $CASCADE_RESULTS_DIR cascade_results files: $NRES (need 23)"
[[ "$NRES" == "23" ]] || exit 1

if [[ "${GNN_TIMING_FEATURES:-0}" == "1" ]]; then
  [[ -s "$GNN_TIMING_CSV" ]] || { echo "[gate] FAIL: $GNN_TIMING_CSV missing or empty"; exit 1; }
  head -1 "$GNN_TIMING_CSV" | grep -q "arrival_h" || { echo "[gate] FAIL: arrival_h column not in $GNN_TIMING_CSV"; exit 1; }
  echo "[gate] timing table OK: $(wc -l < "$GNN_TIMING_CSV") rows"
fi

# Gate 5 (Week 27): the code on Torch carries the head switch.
python -c "from src.gnn.model import HEADS; assert 'hazard' in HEADS; print('[gate] head switch present:', HEADS)" || exit 1

# ---- training --------------------------------------------------------------
if [[ "${SMOKE:-0}" == "1" ]]; then
  python -m src.gnn.train --mode train --device cuda --seed 42 --head hazard \
    --epochs 1 --train_subset 50 --val_subset 20 \
    --holdout_scenario "$HOLDOUT"
else
  python -m src.gnn.train --mode train --device cuda --seed 42 --head hazard \
    --epochs 6 \
    --holdout_scenario "$HOLDOUT"
fi
