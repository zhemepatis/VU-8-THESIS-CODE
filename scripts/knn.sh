#!/bin/bash
#SBATCH -p main
#SBATCH -n4

singularity build --fakeroot ./containers/torch_env.sif ./containers/torch_env.def
singularity exec ./containers/torch_env.sif python train.py