#!/bin/bash
#SBATCH --job-name=gnn-morestorms-resume
#SBATCH --account=torch_pr_1108_tandon_advanced
#SBATCH --array=0-47
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96GB
#SBATCH --gres=gpu:l40s:1
#SBATCH --partition=l40s_public
#SBATCH --time=02:00:00
#SBATCH --output=logs/gnn-morestorms-resume-%A_%a.out
# CascadeGNN, Week 30: the same 48 training tasks as hpc/campaign_gnn_morestorms_v1.sbatch (same task -> arm,
# variant, cell, seed mapping; same data, gates and settings), run as a CHAIN OF SHORT JOBS instead of one
# long job. Why: on 7 Oct the cluster cancelled all four mix57_full jobs of 19358745 at the same minute,
# 2 h 14-19 min after they started (CANCELLED by 0, "DUE to SIGNAL Terminated"); the jobs of the same
# array that finished within 1 h 38 min were never touched. Cause not confirmed; a per-job time or GPU-use
# rule fits. Each job here trains EPOCHS_PER_JOB epochs (default 1, about 40 min for mix57_full), saves
# last.pt with the full training state, and submits the next job for the same task. The job that
# completes epoch 6 runs the held-out evaluation and writes history.json, exactly as an unbroken run.
# scripts/week30_resume_fingerprint_v1.py shows the split run equals the unbroken run bit for bit on CPU.
#
# Task i in 0..47:  variant = i / 24 (0 plain, 1 ctx);  j = i % 24;  seed = 1 + j / 12;  arm = (j / 4) % 3;
#                   cell = j % 4 (static_mask, static_timing, dynamic_mask, dynamic_timing)
# Finished on 7 Oct (do not resubmit): old17 0-3,12-15,24-27,36-39; mix57_same 4,7,28,31.
# Remaining 28:  sbatch --array=5,6,8-11,16-23,29,30,32-35,40-47 hpc/campaign_gnn_morestorms_resume_v1.sbatch
# The four cancelled mix57_full folders (8, 11, 32, 35) hold a last.pt WITHOUT training state: train.py
# refuses to resume from them. Move them aside before submitting (the submission block does this).
# A task whose folder already has history.json exits at once ("already finished"): resubmitting is safe.
#
# Smoke first:  SMOKE=1 sbatch --array=11,35 hpc/campaign_gnn_morestorms_resume_v1.sbatch
#   (2 epochs on 50 runs, 1 epoch per job -> two chained jobs; the second log must show
#    "[resume] continuing from" and end with "[chain] finished"; checkpoints go to a _smoke folder)
# Lines to read: every "[gate]" line ends in OK; "[chain] task=.. depth=.. epochs_done_before=..";
# first job "[resume] no .../last.pt: starting afresh", later jobs "[resume] continuing from ...";
# each job but the last ends "[chain] submitted next job <id>"; the last ends "[chain] finished".

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
case "${EPOCHS_PER_JOB:-1}" in 1|2|3) ;; *) echo "[gate] FAIL: EPOCHS_PER_JOB must be 1, 2 or 3"; exit 1 ;; esac
DEPTH=${CHAIN_DEPTH:-0}
(( DEPTH <= 8 )) || { echo "[gate] FAIL: chain depth $DEPTH > 8; something is resubmitting without progress"; exit 1; }
I=$SLURM_ARRAY_TASK_ID
(( I >= 0 && I < 48 )) || { echo "[gate] FAIL: task id $I out of range"; exit 1; }
VAR_I=$(( I / 24 )); J=$(( I % 24 ))
CELL_I=$(( J % 4 )); ARM_I=$(( (J / 4) % 3 )); TSEED=$(( 1 + J / 12 ))
CELL=${CELLS[$CELL_I]}; ARM=${ARMS[$ARM_I]}; VAR=${VARIANTS[$VAR_I]}; TIMING=${TIMING_FLAGS[$CELL_I]}

export HETERODATA=data/graph/nyc_infra_heterodata_v1_frozen_failover_edges.pt
OLD_HOLDOUT="geoclaw_2050,syn_ts_914_6_1p2955,syn_ts_258_9_3p0022,syn_ts_605_5_3p7563,syn_ts_808_27_3p7885"
export CASCADE_RESULTS_DIR=${OLD_DIRS[$CELL_I]}
if [[ "$ARM" == "old17" ]]; then
  export SCENARIO_SET=jesse22
  export SYN_RESULTS_DIR=$CASCADE_RESULTS_DIR
  HOLDOUT="$OLD_HOLDOUT"
else
  export SCENARIO_SET=w29_all
  export SYN_RESULTS_DIR=${NEW_DIRS[$CELL_I]}
  NEW_HOLDOUT=$(python -c "import csv; print(','.join(r['tag'] for r in csv.DictReader(open('data/flood/gissr48_manifest_v1.csv')) if r['role'] == 'test'))")
  HOLDOUT="$OLD_HOLDOUT,$NEW_HOLDOUT"
fi
# same checkpoint root as the v1 campaign, so finished and resumed runs sit side by side
CK_ROOT=$SCRATCH/results/gnn_morestorms_v1; [[ "${SMOKE:-0}" == "1" ]] && CK_ROOT=$SCRATCH/results/gnn_morestorms_resume_v1_smoke
export GNN_CKPT_DIR=$CK_ROOT/${ARM}_${VAR}_${CELL}_s${TSEED}
mkdir -p "$GNN_CKPT_DIR"

if [[ "$TIMING" == "1" ]]; then
  export GNN_TIMING_FEATURES=1
  export GNN_TIMING_CSV=data/flood/timing/node_timing_jesse22_plus_gissr48_v1.csv
fi
if [[ "$VAR" == "ctx" ]]; then
  export GNN_STORM_CONTEXT=1
fi
export HOLDOUT VAR TIMING
DONE_BEFORE=$(python -c "
import sys, torch
from pathlib import Path
d = Path(sys.argv[1])
if (d / 'history.json').exists(): print('finished')
elif (d / 'last.pt').exists():
    ck = torch.load(d / 'last.pt', map_location='cpu', weights_only=False)
    print(ck['epoch'] + 1 if 'optimizer_state' in ck else 'NOT-RESUMABLE')
else: print(0)" "$GNN_CKPT_DIR")
echo "task=$I arm=$ARM variant=$VAR cell=$CELL torch_seed=$TSEED set=$SCENARIO_SET head=hazard labels=$CASCADE_RESULTS_DIR labels2=$SYN_RESULTS_DIR timing=${GNN_TIMING_FEATURES:-unset} context=${GNN_STORM_CONTEXT:-unset} ckpt=$GNN_CKPT_DIR smoke=${SMOKE:-0} commit=$(git rev-parse --short HEAD) uncommitted_files=$(git status --porcelain --untracked-files=no | wc -l)"
echo "[chain] task=$I depth=$DEPTH epochs_per_job=${EPOCHS_PER_JOB:-1} epochs_done_before=$DONE_BEFORE job=$SLURM_JOB_ID"
[[ "$DONE_BEFORE" != "NOT-RESUMABLE" ]] || { echo "[gate] FAIL: $GNN_CKPT_DIR/last.pt has no training state (a job cancelled before Week 30); move the folder aside and resubmit"; exit 1; }

# ---- fail-fast gates (as in campaign_gnn_morestorms_v1.sbatch) ------------------------------------
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
python -m src.gnn.train --help | grep -q -- "--resume" || { echo "[gate] FAIL: train.py has no --resume (pull the Week 30 commit)"; exit 1; }

# ---- training: this job's share of the epochs ------------------------------------------------------
SUBSETS=()
[[ "$ARM" == "mix57_same" ]] && SUBSETS=(--train_subset 13600 --val_subset 3400)
[[ "$ARM" == "mix57_full" ]] && SUBSETS=(--val_subset 3400)
if [[ "${SMOKE:-0}" == "1" ]]; then
  python -m src.gnn.train --mode train --device cuda --seed 42 --torch_seed "$TSEED" --head hazard \
    --epochs 2 --train_subset 50 --val_subset 20 --epochs_this_job 1 --resume \
    --holdout_scenario "$HOLDOUT" || exit 1
else
  python -m src.gnn.train --mode train --device cuda --seed 42 --torch_seed "$TSEED" --head hazard \
    --epochs 6 "${SUBSETS[@]}" --epochs_this_job "${EPOCHS_PER_JOB:-1}" --resume \
    --holdout_scenario "$HOLDOUT" || exit 1
fi

# ---- chain: finished, or submit the next job for this task --------------------------------------------
if [[ -f "$GNN_CKPT_DIR/history.json" ]]; then
  echo "[chain] finished: $GNN_CKPT_DIR/history.json written after $(python -c "import json,sys; print(len(json.load(open(sys.argv[1]))['epochs']))" "$GNN_CKPT_DIR/history.json") epochs"
  exit 0
fi
[[ -f "$GNN_CKPT_DIR/last.pt" ]] || { echo "[chain] FAIL: no last.pt and no history.json after training"; exit 1; }
NEXT=$(CHAIN_DEPTH=$(( DEPTH + 1 )) sbatch --parsable --dependency=afterok:$SLURM_JOB_ID --array="$I" "$SCRATCH/interdependency-cascade/hpc/campaign_gnn_morestorms_resume_v1.sbatch") || { echo "[chain] FAIL: could not submit the next job"; exit 1; }
echo "[chain] submitted next job $NEXT for task $I (depth $(( DEPTH + 1 )))"
