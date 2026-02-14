#!/bin/bash
#SBATCH -p main
#SBATCH -n4

singularity build --fakeroot torch_env.sif torch_env.def
singularity exec torch_env.sif python train.py