#!/bin/bash
#SBATCH -p main
#SBATCH --output=/dev/null
#SBATCH --cpus-per-task=10

# general setup
NEIGHBOR_COUNT=$1

BENCHMARK_FUNC=$2
DATA_SET_SIZE=$3

# noise setup
NOISE_STD=$4

NOISE_MEAN_ARG=""
NOISE_STD_ARG=""
if [ "$NOISE_STD" != "None" ]; then
    NOISE_MEAN_ARG="--noise-mean 0"
    NOISE_STD_ARG="--noise-std $NOISE_STD"
fi

# output files setup
RESULTS_FILE="$5"
LOCK_FILE="$RESULTS_FILE.lock"

# run experiment
singularity run ./containers/torch.sif src/knn_experiment.py \
    --processes 10 \
    --neighbors $NEIGHBOR_COUNT \
    --benchmark-func $BENCHMARK_FUNC \
    --data-set-size $DATA_SET_SIZE \
    $NOISE_MEAN_ARG \
    $NOISE_STD_ARG \
| flock "$LOCK_FILE" tee -a "$RESULTS_FILE"