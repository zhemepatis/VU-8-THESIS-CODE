#!/bin/bash
#SBATCH -p main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

singularity run ./containers/torch.sif knn_experiment.py \
    --processes 8 \
    --neighbors 3 \
    --data-set-size 1000

singularity run ./containers/torch.sif knn_experiment.py \
    --processes 8 \
    --neighbors 3 \
    --data-set-size 10000

singularity run ./containers/torch.sif knn_experiment.py \
    --processes 8 \
    --neighbors 3 \
    --data-set-size 100000

singularity run ./containers/torch.sif knn_experiment.py \
    --processes 8 \
    --neighbors 3 \
    --data-set-size 1000000 

singularity run ./containers/torch.sif knn_experiment.py \
    --processes 8 \
    --neighbors 3 \
    --data-set-size 10000000 