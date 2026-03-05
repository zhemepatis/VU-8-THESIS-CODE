#!/bin/bash
#SBATCH -p main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

singularity run ./containers/torch.sif fnn_experiment.py \
    --processes 10 \
    --data-set-size 1000 \
    --noise-mean 0 \
    --noise-std 5

# singularity run ./containers/torch.sif fnn_experiment.py \
#     --processes 10 \
#     --data-set-size 10000 \
#     --noise-mean 0 \
#     --noise-std 5

# singularity run ./containers/torch.sif fnn_experiment.py \
#     --processes 10 \
#     --data-set-size 100000 \
#     --noise-mean 0 \
#     --noise-std 5

# singularity run ./containers/torch.sif fnn_experiment.py \
#     --processes 10 \
#     --data-set-size 1000000 \
#     --noise-mean 0 \
#     --noise-std 5