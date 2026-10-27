#!/bin/bash
#SBATCH --job-name=gnn-direct-fp
#SBATCH --account=torch_pr_1108_tandon_advanced
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --gres=gpu:l40s:1
#SBATCH --partition=l40s_public
#SBATCH --time=02:00:00
#SBATCH --output=logs/gnn-direct-fp-%A_%a.out
# Direct model, Week 32, step before any training: build the storm-level targets and fingerprint them.
# Task 0 = static labels, task 1 = arrival labels. Each task reads the 70 storms' label files (old storms from
# the frozen folders, new storms from gissr48_v1), writes one target file per storm to
# $SCRATCH/results/gnn_direct_targets_v1/<mode>/<storm>.pt (pi_hat, seed frequency, the two half-sample
# estimates), and runs scripts/week31_direct_fingerprint_v1.py F1-F5 on them. CPU work only; the GPU
# reservation is there because this is the partition every job of this project runs on. The campaign script
# refuses to train unless this job's JSON says F1-F5 passed for its label mode.
#
#   sbatch hpc/fingerprint_gnn_direct_v1.sbatch
# Lines to read in each log: "[F1] PASS", "[F2] PASS", "[F3] PASS ... telecom_cluster_03042 flagged in [...]",
# "[F4] PASS", "[F5] PASS", "FINGERPRINT PASSED (F1, F2, F3, F4, F5 decided)", and "targets: 70 files".

source $SCRATCH/miniforge3/bin/activate flood
cd $SCRATCH/interdependency-cascade
mkdir -p logs analysis/gnn
export PYTHONUNBUFFERED=1
unset GNN_STORM_CONTEXT GNN_TIMING_FEATURES GNN_TIMING_CSV GNN_TARGET SCENARIO_SET CASCADE_RESULTS_DIR SYN_RESULTS_DIR GNN_CKPT_DIR HETERODATA

[[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] || { echo "[gate] FAIL: run as an array task"; exit 1; }
I=$SLURM_ARRAY_TASK_ID
MODES=(static arrival)
OLD_DIRS=("$SCRATCH/results/legacy_v1_n1000" "$SCRATCH/results/dynamic_v1/arrival_h360")
NEW_DIRS=("$SCRATCH/results/gissr48_v1/static" "$SCRATCH/results/gissr48_v1/arrival_h360")
MODE=${MODES[$I]}
LABELS="${OLD_DIRS[$I]},${NEW_DIRS[$I]}"
CACHE=$SCRATCH/results/gnn_direct_targets_v1/$MODE
OUT=analysis/gnn/week31_direct_fingerprint_${MODE}_v1.json
mkdir -p "$CACHE"
echo "task=$I mode=$MODE labels=$LABELS cache=$CACHE out=$OUT commit=$(git rev-parse --short HEAD) branch=$(git rev-parse --abbrev-ref HEAD) uncommitted_files=$(git status --porcelain --untracked-files=no | wc -l)"

# ---- gates --------------------------------------------------------------------------------------------
[[ "$(git rev-parse --abbrev-ref HEAD)" == "feat/direct-model" ]] || { echo "[gate] FAIL: not on feat/direct-model"; exit 1; }
for d in ${LABELS//,/ }; do [[ -d "$d" ]] || { echo "[gate] FAIL: label folder missing: $d"; exit 1; }; done
python - <<'PYEOF' || exit 1
import os, sys, torch
d = torch.load("data/graph/nyc_infra_heterodata_v1_frozen_failover_edges.pt", weights_only=False)
total = sum(d[nt].num_nodes for nt in d.node_types)
ok = total == 6231 and d["telecom"].num_nodes == 4150
print(f"[gate] heterodata total={total} -> {'OK' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
PYEOF
python - <<PYEOF || exit 1
# every storm of the split has exactly one label file across the two folders
import csv, sys
from pathlib import Path
import yaml
sp = yaml.safe_load(open("config/gnn_direct_split_v1.yaml"))
storms = sp["train17"] + sp["new33"] + sp["val7"] + sp["test13"]
dirs = [Path(x) for x in "$LABELS".split(",")]
bad = [s for s in storms if sum((d / f"cascade_results_nyc_{s}.json").exists() for d in dirs) != 1]
print(f"[gate] labels for {len(storms)} storms in {dirs}: storms without exactly one file {bad} -> {'OK' if not bad else 'FAIL'}")
sys.exit(0 if not bad else 1)
PYEOF

# ---- fingerprint + targets -----------------------------------------------------------------------------
python scripts/week31_direct_fingerprint_v1.py --base-commit f45a7b2 --labels-dir "$LABELS" --label-mode "$MODE" \
    --storms all --cache-dir "$CACHE" --out "$OUT"
RC=$?
echo "targets: $(ls "$CACHE"/*.pt 2>/dev/null | wc -l) files in $CACHE"
echo "fingerprint exit=$RC"
exit $RC
