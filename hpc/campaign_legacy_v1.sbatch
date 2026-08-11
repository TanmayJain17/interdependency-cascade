#!/bin/bash
#SBATCH --job-name=cascade-legacy-v1
#SBATCH --account=torch_pr_1108_tandon_advanced
#SBATCH --array=0-22
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --time=08:00:00
#SBATCH --output=logs/legacy-%A_%a.out

source $SCRATCH/miniforge3/bin/activate flood
cd $SCRATCH/interdependency-cascade
mkdir -p logs

# pin to the FROZEN v1 graph (abstract-comparable; committed in git)
export NODES_IN=data/graph_v1_frozen/nyc_infra_nodes_all_flood.geojson
export GRAPH_IN=data/graph_v1_frozen/nyc_infra_graph_all_flood.graphml
export POWER_COUPLING=0


SCEN=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" hpc/scenarios23.txt)
OUTDIR=$SCRATCH/results/legacy_v1_n1000
echo "task $SLURM_ARRAY_TASK_ID -> $SCEN -> $OUTDIR (POWER_COUPLING=0)"
if [[ $SCEN == syn_* ]]; then
  SCENARIO_ONLY=$SCEN N_MC=1000 OUT_SIM=$OUTDIR python src/simulation/run_synthetic20.py
else
  SCENARIO_ONLY=$SCEN N_MC=1000 SIM_DIR=$OUTDIR python src/simulation/multi_scenario_runner.py
fi
