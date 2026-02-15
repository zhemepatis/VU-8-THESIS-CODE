#!/bin/bash
#SBATCH -p main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

singularity run ./containers/torch.sif knn_experiment.py