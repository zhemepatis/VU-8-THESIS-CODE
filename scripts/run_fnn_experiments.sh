#!/bin/bash

mkdir -p output/raw

BENCHMARK_FUNCS=("sphere_func" "rosenbrock_func" "rastrigin_func")
DATA_SET_SIZES=(1000 10000 100000 1000000)
NOISE=("None" 5)

JOB_IDS=()
for BENCHMARK_FUNC in "${BENCHMARK_FUNCS[@]}"; do
    for SIZE in "${DATA_SET_SIZES[@]}"; do
        for NOISE_STD in "${NOISE[@]}"; do
            NOISE_STD_INT=0
            if [ $NOISE_STD != "None" ]; then
                NOISE_STD_INT=$NOISE_STD
            fi
            
            JOB_ID=$(sbatch --parsable scripts/fnn_experiment.sh $BENCHMARK_FUNC $SIZE $NOISE_STD "output/raw/fnn_${BENCHMARK_FUNC}_${NOISE_STD_INT}.csv")
            echo "$JOB_ID has been queued"

            JOB_IDS+=($JOB_ID)
        done
    done
done