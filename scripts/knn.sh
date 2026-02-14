#!/bin/bash
#SBATCH -p main
#SBATCH -n4

singularity run ./containers/torch.sif knn_experiment.py