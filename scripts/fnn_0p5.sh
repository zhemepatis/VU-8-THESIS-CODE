#!/bin/bash
#SBATCH -p main
#SBATCH --output=output/slurm-%A_%a.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --array=0-3

DATASET_SIZES=(1000 10000 100000 1000000)
SIZE=${DATASET_SIZES[$SLURM_ARRAY_TASK_ID]}   

singularity run ./containers/torch.sif fnn_experiment.py \
    --processes 10 \
    --data-set-size $SIZE \
    --noise-mean 0 \
    --noise-std 0.5