#!/bin/bash
#SBATCH --job-name=gnn-frag
#SBATCH --account=torch_pr_1108_tandon_advanced
#SBATCH --array=0-31
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:l40s:1
#SBATCH --partition=l40s_public
#SBATCH --time=01:30:00
#SBATCH --requeue
#SBATCH --output=logs/gnn-frag-%A_%a.out
# End-to-end fragility, Week 33: the fragility curve is a learnable head inside the network (depth in, probability
# out, nothing computed from the formula at run time). Pre-registration scripts/week33_fragility_table_v1.py.
# 32 tasks: 8 arms x 2 cells x 2 PyTorch seeds. The seed also fixes the head's starting point: seed 1 starts the
# lognormal head "up" (medians x2, betas x0.5), seed 2 "down" (medians x0.5, betas x2), so the two runs of a cell
# approach the assumed values from opposite sides. Targets: the Week 32 cache (hpc/fingerprint_gnn_direct_v1.sbatch),
# 50 training storms, hazard head, GNN Adam 1e-3 wd 1e-5, head Adam 5e-3 wd 0, dropout 0.1, <= 600 epochs, early
# stopping on the 7 validation storms (patience 50), split seed 42.
#
#   Task i in 0..31:  arm  = i % 8        (a1a_lognormal_l1, a1b_lognormal_l0, a1c_lognormal_l0p1, a1d_lognormal_l10,
#                                           a1e_lognormal_nodetach, a4_sequential, a2_mixture, a3_blackbox)
#                     cell = (i / 8) % 2  (static_plain, arrival_timing)
#                     seed = 1 + i / 16
#   a0_formula is not trained here: the audit re-scores the Week 32 checkpoints direct_<cell>_n50_s<seed>.
#
# Smoke first:   SMOKE=1 sbatch --array=0,5,7,14 hpc/campaign_gnn_fragility_v1.sbatch   (5 epochs, patience 3, _smoke root)
# Production:    sbatch hpc/campaign_gnn_fragility_v1.sbatch
# Lines to read in each log: every "[gate]" line ends in OK; "Train: 50 storms  Val: 7  Test: 13 | runs per storm [1000]";
# "input width 17" (static_plain) or "23" (arrival_timing; 16/22 for a3_blackbox); "Model parameters: ... (head 12)"
# (54 for a2_mixture, 0 for a3_blackbox); "head at start: ..."; for a4_sequential "[sequential] stage 1 done in N steps";
# "[early stop] ..." or "done: 600 epochs (max epochs)"; 13 "[held-out ...]" lines; "head at best epoch: ...".

source $SCRATCH/miniforge3/bin/activate flood
cd $SCRATCH/interdependency-cascade
mkdir -p logs
export PYTHONUNBUFFERED=1
unset GNN_STORM_CONTEXT GNN_TIMING_FEATURES GNN_TIMING_CSV GNN_TARGET SCENARIO_SET CASCADE_RESULTS_DIR SYN_RESULTS_DIR GNN_CKPT_DIR HETERODATA

[[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] || { echo "[gate] FAIL: run as an array task (sbatch --array=...)"; exit 1; }
case "${SMOKE:-0}" in 0|1) ;; *) echo "[gate] FAIL: SMOKE must be 0 or 1"; exit 1 ;; esac
I=$SLURM_ARRAY_TASK_ID
(( I >= 0 && I < 32 )) || { echo "[gate] FAIL: task id $I out of range"; exit 1; }
ARMS=(a1a_lognormal_l1 a1b_lognormal_l0 a1c_lognormal_l0p1 a1d_lognormal_l10 a1e_lognormal_nodetach a4_sequential a2_mixture a3_blackbox)
CELLS=(static_plain arrival_timing)
MODES=(static arrival)
TIMING_FLAGS=(0 1)
ARM_I=$(( I % 8 )); CELL_I=$(( (I / 8) % 2 )); TSEED=$(( 1 + I / 16 ))
ARM=${ARMS[$ARM_I]}; CELL=${CELLS[$CELL_I]}; MODE=${MODES[$CELL_I]}; TIMING=${TIMING_FLAGS[$CELL_I]}
INIT=perturb_up; [[ "$TSEED" == "2" ]] && INIT=perturb_down
case "$ARM" in
  a1a_lognormal_l1)       EXTRA="--fragility lognormal --frag_init $INIT --lambda_seed 1.0  --frag_regime joint      --detach_phi 1" ;;
  a1b_lognormal_l0)       EXTRA="--fragility lognormal --frag_init $INIT --lambda_seed 0.0  --frag_regime joint      --detach_phi 1" ;;
  a1c_lognormal_l0p1)     EXTRA="--fragility lognormal --frag_init $INIT --lambda_seed 0.1  --frag_regime joint      --detach_phi 1" ;;
  a1d_lognormal_l10)      EXTRA="--fragility lognormal --frag_init $INIT --lambda_seed 10.0 --frag_regime joint      --detach_phi 1" ;;
  a1e_lognormal_nodetach) EXTRA="--fragility lognormal --frag_init $INIT --lambda_seed 1.0  --frag_regime joint      --detach_phi 0" ;;
  a4_sequential)          EXTRA="--fragility lognormal --frag_init $INIT --lambda_seed 1.0  --frag_regime sequential --detach_phi 1 --stage1_steps 2000" ;;
  a2_mixture)             EXTRA="--fragility mixture   --frag_init hazus --lambda_seed 1.0  --frag_regime joint      --detach_phi 1" ;;
  a3_blackbox)            EXTRA="--fragility none      --lambda_seed 0.0 --frag_regime joint" ;;
  *) echo "[gate] FAIL: unknown arm $ARM"; exit 1 ;;
esac

export GNN_TARGET=marginal
export SCENARIO_SET=w29_all
export HETERODATA=data/graph/nyc_infra_heterodata_v1_frozen_failover_edges.pt
export GNN_TIMING_CSV=data/flood/timing/node_timing_jesse22_plus_gissr48_v1.csv   # read in every cell: arrival needs the arrival hour
[[ "$TIMING" == "1" ]] && export GNN_TIMING_FEATURES=1
CACHE=$SCRATCH/results/gnn_direct_targets_v1/$MODE
FP=analysis/gnn/week33_fragility_fingerprint_v1.json
FP32=analysis/gnn/week31_direct_fingerprint_${MODE}_v1.json
CK_ROOT=$SCRATCH/results/gnn_fragility_v1; [[ "${SMOKE:-0}" == "1" ]] && CK_ROOT=$SCRATCH/results/gnn_fragility_v1_smoke
NAME=e2e_${ARM}_${CELL}_s${TSEED}
export GNN_CKPT_DIR=$CK_ROOT/$NAME
mkdir -p "$GNN_CKPT_DIR"
echo "task=$I name=$NAME arm=$ARM cell=$CELL mode=$MODE torch_seed=$TSEED init=$INIT timing=${GNN_TIMING_FEATURES:-unset} extra=[$EXTRA] cache=$CACHE ckpt=$GNN_CKPT_DIR smoke=${SMOKE:-0} commit=$(git rev-parse --short HEAD) branch=$(git rev-parse --abbrev-ref HEAD) uncommitted_files=$(git status --porcelain --untracked-files=no | wc -l)"

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
# the Week 33 fingerprint passed F1-F6 (and the Week 32 one of this label mode F1-F5); the target cache holds all 70 storms
import json, sys
from pathlib import Path
ok = True
for fp, keys in ((Path("$FP"), ("F1", "F2", "F3", "F4", "F5", "F6")), (Path("$FP32"), ("F1", "F2", "F3", "F4", "F5"))):
    if not fp.exists():
        print(f"[gate] FAIL: no fingerprint result {fp}"); sys.exit(1)
    r = json.load(open(fp))
    checks = {k: r.get(k, {}).get("pass") for k in keys}
    ok_ = all(v is True for v in checks.values()); ok = ok and ok_
    print(f"[gate] fingerprint {fp}: {checks} -> {'OK' if ok_ else 'FAIL'}")
from src.gnn import data
sp = data.load_direct_split()
storms = sp["train50"] + sp["val7"] + sp["test13"]
missing = [s for s in storms if not (Path("$CACHE") / f"{s}.pt").exists()]
print(f"[gate] targets in $CACHE: {len(storms) - len(missing)}/{len(storms)}; missing {missing[:5]} -> {'OK' if not missing else 'FAIL'}")
ok2 = data.TARGET == "marginal" and not data.STORM_CONTEXT and data.TIMING_FEATURES == ("$TIMING" == "1")
print(f"[gate] switches: target {data.TARGET}, storm context {'on' if data.STORM_CONTEXT else 'off'}, timing inputs {'on' if data.TIMING_FEATURES else 'off'} -> {'OK' if ok2 else 'FAIL'}")
from src.simulation.fragility import FRAGILITY_PARAMS
want = {"power": (0.6, 0.5), "telecom": (0.3, 0.6), "hospital": (0.6, 0.5), "subway": (0.1, 0.4), "water": (0.5, 0.5), "fuel": (0.5, 0.6)}
ok3 = {k: tuple(float(x) for x in v) for k, v in FRAGILITY_PARAMS.items()} == want
print(f"[gate] assumed fragility parameters unchanged: {'OK' if ok3 else 'FAIL'}")
sys.exit(0 if ok and not missing and ok2 and ok3 else 1)
PYEOF
python -m src.gnn.train --help | grep -q -- "--fragility" || { echo "[gate] FAIL: train.py has no --fragility"; exit 1; }

# ---- training ------------------------------------------------------------------------------------------
if [[ "${SMOKE:-0}" == "1" ]]; then
  python -m src.gnn.train --mode train --target marginal --device cuda --seed 42 --torch_seed "$TSEED" --head hazard \
    --label_mode "$MODE" --train_storms 50 --marginal_cache "$CACHE" --max_epochs 5 --patience 3 --arm "$ARM" $EXTRA
else
  python -m src.gnn.train --mode train --target marginal --device cuda --seed 42 --torch_seed "$TSEED" --head hazard \
    --label_mode "$MODE" --train_storms 50 --marginal_cache "$CACHE" --max_epochs 600 --patience 50 --arm "$ARM" $EXTRA
fi
