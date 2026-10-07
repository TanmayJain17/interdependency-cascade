#!/bin/bash
#SBATCH --job-name=gnn-morestorms
#SBATCH --account=torch_pr_1108_tandon_advanced
#SBATCH --array=0-47
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96GB
#SBATCH --gres=gpu:l40s:1
#SBATCH --partition=l40s_public
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=logs/gnn-morestorms-%A_%a.out
# CascadeGNN, Week 29: two questions, one array. Hazard head, 6 epochs, split seed 42, PyTorch seeds 1 and 2,
# every other setting identical to campaign_gnn_hazard_v1.sbatch.
#
#   Question 1, more storms (the advisors' ask): what do 40 further GISSR storms buy?
#     old17       trains on the 17 storms of the jesse22 campaigns (13,600 runs; the jesse22 split)
#     mix57_same  17 old + 40 new storms, SAME number of training runs and steps (13,600 runs drawn at
#                 random over the 57 storms, about 239 per storm) -> isolates storm variety
#     mix57_full  all 45,600 training runs of the 57 storms (3.35x the steps)
#   Question 2, storm context (from the Week 29 numbers sheet): the model counts badly because a site with
#     no flooded site within two links gets the same input in every storm. Variant "ctx" adds a storm-level
#     summary to every site's input (GNN_STORM_CONTEXT=1, src/gnn/data.py); variant "plain" does not.
#
# Task i in 0..47:  variant = i / 24 (0 plain, 1 ctx);  j = i % 24;  seed = 1 + j / 12;  arm = (j / 4) % 3;
#                   cell = j % 4 (static_mask, static_timing, dynamic_mask, dynamic_timing)
#     plain: 0-3 old17 s1 | 4-7 mix57_same s1 | 8-11 mix57_full s1 | 12-15, 16-19, 20-23 the same with seed 2
#     ctx:   the same + 24
# old17 trains under SCENARIO_SET=jesse22 on the frozen label folders only, so it does NOT need the new
# labels and can start at once. The mix arms need $SCRATCH/results/gissr48_v1/{static,arrival_h360}.
# Wall time on one L40S: old17 and mix57_same about 1.5 h, mix57_full about 5.5 h.
# Predictions: scripts/week29_context_table_v1.py (question 2), scripts/week29_morestorms_table_v1.py (question 1).
#
# Smoke first (cheap-before-expensive):   SMOKE=1 sbatch --array=3,27 hpc/campaign_gnn_morestorms_v1.sbatch
# Tonight, old labels only (16 tasks):    sbatch --array=0-3,12-15,24-27,36-39 hpc/campaign_gnn_morestorms_v1.sbatch
# After the label campaign, in priority:  sbatch --array=8-11,32-35,4-7 hpc/campaign_gnn_morestorms_v1.sbatch
#                                         (smoke the mix arms first: SMOKE=1 sbatch --array=7,35 ...)
#                                         then seed 2 and the rest: --array=20-23,44-47,16-19,28-31,40-43
# Lines to read in each log: every "[gate]" line ends in OK; old17 "Train: 13600  Val: 3400"; mix arms "Train: 45600
# Val: 11400", then "Subsampled train to 13600" (mix57_same only) and "Subsampled val to 3400";
# "torch seed set: 1|2"; ctx tasks "Model parameters: 698,648" (mask cells) or "700,952" (timing cells).

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
I=$SLURM_ARRAY_TASK_ID
(( I >= 0 && I < 48 )) || { echo "[gate] FAIL: task id $I out of range"; exit 1; }
VAR_I=$(( I / 24 )); J=$(( I % 24 ))
CELL_I=$(( J % 4 )); ARM_I=$(( (J / 4) % 3 )); TSEED=$(( 1 + J / 12 ))
CELL=${CELLS[$CELL_I]}; ARM=${ARMS[$ARM_I]}; VAR=${VARIANTS[$VAR_I]}; TIMING=${TIMING_FLAGS[$CELL_I]}

export HETERODATA=data/graph/nyc_infra_heterodata_v1_frozen_failover_edges.pt
OLD_HOLDOUT="geoclaw_2050,syn_ts_914_6_1p2955,syn_ts_258_9_3p0022,syn_ts_605_5_3p7563,syn_ts_808_27_3p7885"
export CASCADE_RESULTS_DIR=${OLD_DIRS[$CELL_I]}
if [[ "$ARM" == "old17" ]]; then
  # same folder twice (twin rule): a missing storm RAISES
  export SCENARIO_SET=jesse22
  export SYN_RESULTS_DIR=$CASCADE_RESULTS_DIR
  HOLDOUT="$OLD_HOLDOUT"
else
  # old storms from the frozen folder, new storms from the gissr48 folder; the two folders share no
  # file name (gate below), so a missing storm still RAISES instead of falling back
  export SCENARIO_SET=w29_all
  export SYN_RESULTS_DIR=${NEW_DIRS[$CELL_I]}
  NEW_HOLDOUT=$(python -c "import csv; print(','.join(r['tag'] for r in csv.DictReader(open('data/flood/gissr48_manifest_v1.csv')) if r['role'] == 'test'))")
  HOLDOUT="$OLD_HOLDOUT,$NEW_HOLDOUT"
fi
CK_ROOT=$SCRATCH/results/gnn_morestorms_v1; [[ "${SMOKE:-0}" == "1" ]] && CK_ROOT=$SCRATCH/results/gnn_morestorms_v1_smoke
export GNN_CKPT_DIR=$CK_ROOT/${ARM}_${VAR}_${CELL}_s${TSEED}
mkdir -p "$GNN_CKPT_DIR"

if [[ "$TIMING" == "1" ]]; then
  export GNN_TIMING_FEATURES=1
  export GNN_TIMING_CSV=data/flood/timing/node_timing_jesse22_plus_gissr48_v1.csv     # old rows byte-identical to node_timing_jesse22_v1.csv
fi
# Storm context: exported ONLY in the ctx variant (the plain variant leaves it UNSET: the default build).
if [[ "$VAR" == "ctx" ]]; then
  export GNN_STORM_CONTEXT=1
fi
export HOLDOUT VAR TIMING
echo "task=$I arm=$ARM variant=$VAR cell=$CELL torch_seed=$TSEED set=$SCENARIO_SET head=hazard labels=$CASCADE_RESULTS_DIR labels2=$SYN_RESULTS_DIR timing=${GNN_TIMING_FEATURES:-unset} context=${GNN_STORM_CONTEXT:-unset} ckpt=$GNN_CKPT_DIR smoke=${SMOKE:-0} commit=$(git rev-parse --short HEAD) uncommitted_files=$(git status --porcelain --untracked-files=no | wc -l)"

# ---- fail-fast gates -----------------------------------------------------------------------------
python - <<'PYEOF' || exit 1
import os, sys, torch
assert torch.cuda.is_available(), "CUDA not available on this node"
print(f"[gate] CUDA OK: {torch.cuda.get_device_name(0)}")
d = torch.load(os.environ["HETERODATA"], weights_only=False)
counts = {nt: d[nt].num_nodes for nt in d.node_types}
total = sum(counts.values())
ok = total == 6231 and counts.get("telecom") == 4150
print(f"[gate] heterodata {os.environ['HETERODATA']}: total={total} -> {'OK' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
PYEOF

python - <<'PYEOF' || exit 1
# every storm of the set has exactly one label file; held-out storms are in the set; timing rows exist;
# the context switch is what the variant says
import os, sys
from pathlib import Path
from src.gnn import data
a, b = Path(os.environ["CASCADE_RESULTS_DIR"]), Path(os.environ["SYN_RESULTS_DIR"])
name = lambda s: f"cascade_results_nyc_{s}.json"
both = [] if a == b else [s for s in data.SCENARIOS if (a / name(s)).exists() and (b / name(s)).exists()]
none = [s for s in data.SCENARIOS if not (a / name(s)).exists() and not (b / name(s)).exists()]
hold = os.environ["HOLDOUT"].split(",")
n_train = len(data.SCENARIOS) - len(hold)
want = {"jesse22": (22, 17, 5), "w29_all": (70, 57, 13)}[os.environ["SCENARIO_SET"]]
ok = not both and not none and all(h in data.SCENARIOS for h in hold) and (len(data.SCENARIOS), n_train, len(hold)) == want
print(f"[gate] set {os.environ['SCENARIO_SET']}: {len(data.SCENARIOS)} storms, {n_train} for training, {len(hold)} held out; "
      f"in both folders {both}; missing {none} -> {'OK' if ok else 'FAIL'}")
if data.TIMING_FEATURES:
    import pandas as pd
    have = set(pd.read_csv(data.TIMING_CSV, usecols=["scenario"]).scenario)
    miss = [s for s in data.SCENARIOS if s not in have]
    print(f"[gate] timing table {data.TIMING_CSV}: storms without rows {miss} -> {'OK' if not miss else 'FAIL'}")
    ok = ok and not miss
ctx_ok = data.STORM_CONTEXT == (os.environ["VAR"] == "ctx") and data.TIMING_FEATURES == (os.environ["TIMING"] == "1")
print(f"[gate] variant {os.environ['VAR']}: storm context {'on' if data.STORM_CONTEXT else 'off'}, timing inputs {'on' if data.TIMING_FEATURES else 'off'} -> {'OK' if ctx_ok else 'FAIL'}")
sys.exit(0 if ok and ctx_ok else 1)
PYEOF

python -c "from src.gnn.model import HEADS; assert 'hazard' in HEADS; print('[gate] head switch present:', HEADS)" || exit 1
python -m src.gnn.train --help | grep -q -- "--torch_seed" || { echo "[gate] FAIL: train.py has no --torch_seed"; exit 1; }

# ---- training ------------------------------------------------------------------------------------
SUBSETS=()
[[ "$ARM" == "mix57_same" ]] && SUBSETS=(--train_subset 13600 --val_subset 3400)
[[ "$ARM" == "mix57_full" ]] && SUBSETS=(--val_subset 3400)
if [[ "${SMOKE:-0}" == "1" ]]; then
  python -m src.gnn.train --mode train --device cuda --seed 42 --torch_seed "$TSEED" --head hazard \
    --epochs 1 --train_subset 50 --val_subset 20 \
    --holdout_scenario "$HOLDOUT"
else
  python -m src.gnn.train --mode train --device cuda --seed 42 --torch_seed "$TSEED" --head hazard \
    --epochs 6 "${SUBSETS[@]}" \
    --holdout_scenario "$HOLDOUT"
fi
