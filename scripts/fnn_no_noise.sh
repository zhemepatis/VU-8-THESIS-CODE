#!/bin/bash
#SBATCH -p main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10

singularity run ./containers/torch.sif fnn_experiment.py \
    --processes 10 \
    --data-set-size 1000

singularity run ./containers/torch.sif fnn_experiment.py \
    --processes 10 \
    --data-set-size 10000

singularity run ./containers/torch.sif fnn_experiment.py \
    --processes 10 \
    --data-set-size 100000

singularity run ./containers/torch.sif fnn_experiment.py \
    --processes 10 \
    --data-set-size 1000000