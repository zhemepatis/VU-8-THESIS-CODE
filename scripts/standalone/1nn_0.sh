#!/bin/bash
#SBATCH -p main
#SBATCH --output=/dev/null
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --array=0-14

RESULTS_FILE="output/raw_results.csv"
LOCK_FILE="output/raw_results.lock"

BENCHMARK_FUNCS=(0 1 2)
DATASET_SIZES=(1000 10000 100000 1000000 10000000)

BENCHMARK_FUNC=${BENCHMARK_FUNCS[$(($SLURM_ARRAY_TASK_ID / 5))]}
SIZE=${DATASET_SIZES[$(($SLURM_ARRAY_TASK_ID % 5))]}

singularity run ./containers/torch.sif knn_experiment.py \
    --benchmark-func $BENCHMARK_FUNC \
    --processes 10 \
    --neighbors 1 \
    --data-set-size $SIZE \
| flock "$LOCK_FILE" tee -a "$RESULTS_FILE"