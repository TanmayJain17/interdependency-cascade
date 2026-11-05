#!/bin/bash
#SBATCH --job-name=gnn-frag-fp
#SBATCH --account=torch_pr_1108_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:l40s:1
#SBATCH --partition=l40s_public
#SBATCH --time=01:00:00
#SBATCH --output=logs/gnn-frag-fp-%j.out
# End-to-end fragility, Week 33, the step before any training: scripts/week33_fragility_fingerprint_v1.py F1-F6 on
# the real targets. F3 (the oracle: a probit maximum-likelihood fit of seed frequency on log depth over the 50
# training storms, per type, with storm bootstrap) needs the Week 32 target cache, which lives here; the other five
# checks are the same ones the Mac run decides. Writes analysis/gnn/week33_fragility_fingerprint_v1.json, which
# hpc/campaign_gnn_fragility_v1.sbatch refuses to train without. CPU work only; the GPU reservation is there because
# this is the partition every job of this project runs on.
#
#   sbatch hpc/fingerprint_gnn_fragility_v1.sbatch
# Lines to read in the log: "[F1] PASS" ... "[F6] PASS", the six per-type oracle lines of F3 (each "ok", with the
# 95 % bootstrap intervals), "stage-1 Adam fit within x.xx% of the MLE" (x.xx <= 0.50), and
# "FINGERPRINT PASSED (F1, F2, F3, F4, F5, F6 decided)".

source $SCRATCH/miniforge3/bin/activate flood
cd $SCRATCH/interdependency-cascade
mkdir -p logs analysis/gnn
export PYTHONUNBUFFERED=1
unset GNN_STORM_CONTEXT GNN_TIMING_FEATURES GNN_TIMING_CSV GNN_TARGET SCENARIO_SET CASCADE_RESULTS_DIR SYN_RESULTS_DIR GNN_CKPT_DIR HETERODATA

CACHE=$SCRATCH/results/gnn_direct_targets_v1/static      # seeds are identical in both label modes; static holds every storm
echo "commit=$(git rev-parse --short HEAD) branch=$(git rev-parse --abbrev-ref HEAD) uncommitted_files=$(git status --porcelain --untracked-files=no | wc -l) cache=$CACHE"
[[ "$(git rev-parse --abbrev-ref HEAD)" == "feat/direct-model" ]] || { echo "[gate] FAIL: not on feat/direct-model"; exit 1; }
[[ $(ls "$CACHE"/*.pt 2>/dev/null | wc -l) -ge 70 ]] || { echo "[gate] FAIL: target cache $CACHE incomplete (run hpc/fingerprint_gnn_direct_v1.sbatch first)"; exit 1; }
git cat-file -e f1a1e48^{commit} || { echo "[gate] FAIL: base commit f1a1e48 not in this clone"; exit 1; }

python scripts/week33_fragility_fingerprint_v1.py --base-commit f1a1e48 --cache "$CACHE" --bootstrap 100 \
    --out analysis/gnn/week33_fragility_fingerprint_v1.json
echo "exit=$?"
