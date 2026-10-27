#!/bin/bash
#SBATCH --job-name=gnn-direct
#SBATCH --account=torch_pr_1108_tandon_advanced
#SBATCH --array=0-23
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:l40s:1
#SBATCH --partition=l40s_public
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --output=logs/gnn-direct-%A_%a.out
# Direct model, Week 32: flood map -> per-site failure probability in one pass, trained on the storm-level
# probabilities of the simulator (pre-registration scripts/week31_direct_table_v1.py, commit 3f305d7).
# 24 tasks: cell x training size x PyTorch seed. One example = one storm; targets come from the cache written by
# hpc/fingerprint_gnn_direct_v1.sbatch, never from the label files. Hazard head, Adam 1e-3, wd 1e-5, dropout 0.1,
# up to 600 epochs, early stopping on the 7 validation storms (patience 50), split seed 42.
#
#   Task i in 0..23:  cell = i % 4 (static_plain, static_timing, arrival_plain, arrival_timing)
#                     size = (i / 4) % 3 -> 17, 34, 50 training storms;  seed = 1 + i / 12
#   "plain" here = no per-site timing columns (the expected-seed summary is always on; there is no no-summary arm).
#
# Smoke first:   SMOKE=1 sbatch --array=0,3,11 hpc/campaign_gnn_direct_v1.sbatch   (5 epochs, patience 3, _smoke root)
# Production:    sbatch hpc/campaign_gnn_direct_v1.sbatch
# Lines to read in each log: every "[gate]" line ends in OK; "Train: 17|34|50 storms  Val: 7  Test: 13 | runs per
# storm [1000]"; "input width 17" (plain) or "23" (timing); "Model parameters: ..."; "[early stop] ..." or
# "done: 600 epochs (max epochs)"; 13 "[held-out ...]" lines.

source $SCRATCH/miniforge3/bin/activate flood
cd $SCRATCH/interdependency-cascade
mkdir -p logs
export PYTHONUNBUFFERED=1
unset GNN_STORM_CONTEXT GNN_TIMING_FEATURES GNN_TIMING_CSV GNN_TARGET SCENARIO_SET CASCADE_RESULTS_DIR SYN_RESULTS_DIR GNN_CKPT_DIR HETERODATA

[[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] || { echo "[gate] FAIL: run as an array task (sbatch --array=...)"; exit 1; }
case "${SMOKE:-0}" in 0|1) ;; *) echo "[gate] FAIL: SMOKE must be 0 or 1"; exit 1 ;; esac
I=$SLURM_ARRAY_TASK_ID
(( I >= 0 && I < 24 )) || { echo "[gate] FAIL: task id $I out of range"; exit 1; }
CELLS=(static_plain static_timing arrival_plain arrival_timing)
MODES=(static static arrival arrival)
TIMING_FLAGS=(0 1 0 1)
SIZES=(17 34 50)
CELL_I=$(( I % 4 )); SIZE_I=$(( (I / 4) % 3 )); TSEED=$(( 1 + I / 12 ))
CELL=${CELLS[$CELL_I]}; MODE=${MODES[$CELL_I]}; TIMING=${TIMING_FLAGS[$CELL_I]}; SIZE=${SIZES[$SIZE_I]}

export GNN_TARGET=marginal
export SCENARIO_SET=w29_all
export HETERODATA=data/graph/nyc_infra_heterodata_v1_frozen_failover_edges.pt
export GNN_TIMING_CSV=data/flood/timing/node_timing_jesse22_plus_gissr48_v1.csv   # read in every cell: arrival needs the arrival hour
[[ "$TIMING" == "1" ]] && export GNN_TIMING_FEATURES=1
CACHE=$SCRATCH/results/gnn_direct_targets_v1/$MODE
FP=analysis/gnn/week31_direct_fingerprint_${MODE}_v1.json
CK_ROOT=$SCRATCH/results/gnn_direct_v1; [[ "${SMOKE:-0}" == "1" ]] && CK_ROOT=$SCRATCH/results/gnn_direct_v1_smoke
export GNN_CKPT_DIR=$CK_ROOT/direct_${CELL}_n${SIZE}_s${TSEED}
mkdir -p "$GNN_CKPT_DIR"
echo "task=$I cell=$CELL mode=$MODE size=$SIZE torch_seed=$TSEED timing=${GNN_TIMING_FEATURES:-unset} cache=$CACHE ckpt=$GNN_CKPT_DIR smoke=${SMOKE:-0} commit=$(git rev-parse --short HEAD) branch=$(git rev-parse --abbrev-ref HEAD) uncommitted_files=$(git status --porcelain --untracked-files=no | wc -l)"

# ---- fail-fast gates -----------------------------------------------------------------------------------
[[ "$(git rev-parse --abbrev-ref HEAD)" == "feat/direct-model" ]] || { echo "[gate] FAIL: not on feat/direct-model"; exit 1; }
python - <<'PYEOF' || exit 1
import os, sys, torch
assert torch.cuda.is_available(), "CUDA not available on this node"
print(f"[gate] CUDA OK: {torch.cuda.get_device_name(0)}")
d = torch.load(os.environ["HETERODATA"], weights_only=False)
total = sum(d[nt].num_nodes for nt in d.node_types)
ok = total == 6231 and d["telecom"].num_nodes == 4150
print(f"[gate] heterodata {os.environ['HETERODATA']}: total={total} -> {'OK' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
PYEOF
python - <<PYEOF || exit 1
# the fingerprint of this label mode passed, F1-F5 all decided; the target cache holds all 70 storms
import json, sys
from pathlib import Path
fp = Path("$FP")
if not fp.exists():
    print(f"[gate] FAIL: no fingerprint result {fp}; run hpc/fingerprint_gnn_direct_v1.sbatch first"); sys.exit(1)
r = json.load(open(fp))
checks = {k: r.get(k, {}).get("pass") for k in ("F1", "F2", "F3", "F4", "F5")}
ok = all(v is True for v in checks.values())
print(f"[gate] fingerprint {fp}: {checks} -> {'OK' if ok else 'FAIL'}")
from src.gnn import data
sp = data.load_direct_split()
storms = sp["train50"] + sp["val7"] + sp["test13"]
missing = [s for s in storms if not (Path("$CACHE") / f"{s}.pt").exists()]
print(f"[gate] targets in $CACHE: {len(storms) - len(missing)}/{len(storms)}; missing {missing[:5]} -> {'OK' if not missing else 'FAIL'}")
ok2 = data.TARGET == "marginal" and not data.STORM_CONTEXT and data.TIMING_FEATURES == ("$TIMING" == "1")
print(f"[gate] switches: target {data.TARGET}, storm context {'on' if data.STORM_CONTEXT else 'off'}, timing inputs {'on' if data.TIMING_FEATURES else 'off'} -> {'OK' if ok2 else 'FAIL'}")
sys.exit(0 if ok and not missing and ok2 else 1)
PYEOF
python -m src.gnn.train --help | grep -q -- "--target" || { echo "[gate] FAIL: train.py has no --target"; exit 1; }

# ---- training ------------------------------------------------------------------------------------------
if [[ "${SMOKE:-0}" == "1" ]]; then
  python -m src.gnn.train --mode train --target marginal --device cuda --seed 42 --torch_seed "$TSEED" --head hazard \
    --label_mode "$MODE" --train_storms "$SIZE" --marginal_cache "$CACHE" --max_epochs 5 --patience 3
else
  python -m src.gnn.train --mode train --target marginal --device cuda --seed 42 --torch_seed "$TSEED" --head hazard \
    --label_mode "$MODE" --train_storms "$SIZE" --marginal_cache "$CACHE" --max_epochs 600 --patience 50
fi
