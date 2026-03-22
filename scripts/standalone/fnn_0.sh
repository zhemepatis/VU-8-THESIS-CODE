#!/bin/bash
#SBATCH -p main
#SBATCH --output=/dev/null
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --array=0-3

RESULTS_FILE="output/raw/fnn_0.csv"
LOCK_FILE="output/raw/fnn_0.lock"

DATASET_SIZES=(1000 10000 100000 1000000)
SIZE=${DATASET_SIZES[$SLURM_ARRAY_TASK_ID]}

singularity run ./containers/torch.sif fnn_experiment.py \
    --processes 10 \
    --data-set-size $SIZE \
| flock "$LOCK_FILE" tee -a "$RESULTS_FILE"