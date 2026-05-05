#!/bin/bash
#SBATCH -p main
#SBATCH --output=/dev/null
#SBATCH --cpus-per-task=10

# general setup
BENCHMARK_FUNC=$1
DATA_SET_SIZE=$2

# noise setup
NOISE_STD=$3

NOISE_MEAN_ARG=""
NOISE_STD_ARG=""
if [ "$NOISE_STD" != "None" ]; then
    NOISE_MEAN_ARG="--noise-mean 0"
    NOISE_STD_ARG="--noise-std $NOISE_STD"
fi

# output files setup
RESULTS_FILE="$4"
LOCK_FILE="$RESULTS_FILE.lock"

# run experiment
singularity run ./containers/torch.sif src/fnn_experiment.py \
    --processes 10 \
    --benchmark-func $BENCHMARK_FUNC \
    --data-set-size $DATA_SET_SIZE \
    $NOISE_MEAN_ARG \
    $NOISE_STD_ARG \
| flock "$LOCK_FILE" tee -a "$RESULTS_FILE"