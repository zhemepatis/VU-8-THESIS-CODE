#!/bin/bash
#SBATCH -p main
#SBATCH --output=/dev/null
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --array=0-4

RESULTS_FILE="output/raw/1nn_0p5.csv"
LOCK_FILE="output/raw/1nn_0p5.lock"

DATASET_SIZES=(1000 10000 100000 1000000 10000000)
SIZE=${DATASET_SIZES[$SLURM_ARRAY_TASK_ID]}

singularity run ./containers/torch.sif knn_experiment.py \
    --processes 10 \
    --neighbors 1 \
    --data-set-size $SIZE \
    --noise-mean 0 \
    --noise-std 0.5 \
| flock "$LOCK_FILE" tee -a "$RESULTS_FILE"