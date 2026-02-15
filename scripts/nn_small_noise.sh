#!/bin/bash
#SBATCH -p main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

singularity run ./containers/torch.sif knn_experiment.py \
    --processes 4 \
    --neighbors 1 \
    --data-set-size 1000 \
    --noise-mean 0 \
    --noise-std 0.5

singularity run ./containers/torch.sif knn_experiment.py \
    --processes 4 \
    --neighbors 1 \
    --data-set-size 10000 \
    --noise-mean 0 \
    --noise-std 0.5

singularity run ./containers/torch.sif knn_experiment.py \
    --processes 4 \
    --neighbors 1 \
    --data-set-size 100000 \
    --noise-mean 0 \
    --noise-std 0.5

singularity run ./containers/torch.sif knn_experiment.py \
    --processes 4 \
    --neighbors 1 \
    --data-set-size 1000000 \
    --noise-mean 0 \
    --noise-std 0.5

singularity run ./containers/torch.sif knn_experiment.py \
    --processes 4 \
    --neighbors 1 \
    --data-set-size 10000000 \
    --noise-mean 0 \
    --noise-std 0.5