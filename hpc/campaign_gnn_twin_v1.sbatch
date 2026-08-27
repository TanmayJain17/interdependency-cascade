#!/bin/bash
#SBATCH --job-name=gnn-twin-v1
#SBATCH --account=torch_pr_1108_tandon_advanced
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --gres=gpu:l40s:1
#SBATCH --partition=l40s_public
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=logs/gnn-twin-%A_%a.out

# Twin CascadeGNN retrain on the frozen v1 graph.
# Task 0 = legacy labels, task 1 = jesse (physics-coupled) labels.
# Byte-identical config, seed, and 5-way holdout across arms — the ONLY
# difference is which campaign supplies the labels. Isolates the causal
# question: do physics-realistic labels change surrogate skill?
#
# Smoke first (cheap-before-expensive):   SMOKE=1 sbatch hpc/campaign_gnn_twin_v1.sbatch
# Production:                             sbatch hpc/campaign_gnn_twin_v1.sbatch

source $SCRATCH/miniforge3/bin/activate flood
cd $SCRATCH/interdependency-cascade
mkdir -p logs
export PYTHONUNBUFFERED=1

ARMS=(legacy jesse)
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}

export SCENARIO_SET=jesse22
export HETERODATA=data/graph/nyc_infra_heterodata_v1_frozen_failover_edges.pt
export CASCADE_RESULTS_DIR=$SCRATCH/results/${ARM}_v1_n1000
# Same dir on purpose: a scenario missing from the campaign must RAISE,
# never silently fall back to stale v1-era synthetic outputs.
export SYN_RESULTS_DIR=$SCRATCH/results/${ARM}_v1_n1000
export GNN_CKPT_DIR=$SCRATCH/results/gnn_twin_v1/${ARM}
mkdir -p "$GNN_CKPT_DIR"

HOLDOUT="geoclaw_2050,syn_ts_914_6_1p2955,syn_ts_258_9_3p0022,syn_ts_605_5_3p7563,syn_ts_808_27_3p7885"

echo "arm=$ARM  labels=$CASCADE_RESULTS_DIR  ckpt=$GNN_CKPT_DIR  smoke=${SMOKE:-0}"

# ---- fail-fast gates (the lessons of this week, automated) ----------------
# Gate 1: CUDA actually available — otherwise we'd silently train on CPU
# while a GPU idles (the --device default is cpu; we pass cuda explicitly,
# but the assert catches a broken driver/partition too).
# Gate 2: the frozen v1 heterodata exists and has EXACTLY the v1 shape.
# v1 telecom cluster ids shifted in v2 — a v2 graph would not crash, it
# would silently mislabel. 6,231 total / telecom 4,150 or we stop.
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

# ---- training -------------------------------------------------------------
if [[ "${SMOKE:-0}" == "1" ]]; then
  python -m src.gnn.train --mode train --device cuda --seed 42 \
    --epochs 1 --train_subset 50 --val_subset 20 \
    --holdout_scenario "$HOLDOUT"
else
  python -m src.gnn.train --mode train --device cuda --seed 42 \
    --epochs 6 \
    --holdout_scenario "$HOLDOUT"
fi
