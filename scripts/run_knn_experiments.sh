#!/bin/bash

mkdir -p output/raw

BENCHMARK_FUNCS=("sphere_func" "rosenbrock_func" "rastrigin_func")
DATA_SET_SIZES=(1000 10000 100000 1000000 10000000)
NOISE=("None" 5)

NEIGHBOR_COUNTS=(1 2 4 8 16 32)

JOB_IDS=()
for NEIGHBOR_COUNT in "${NEIGHBOR_COUNTS[@]}"; do
    for BENCHMARK_FUNC in "${BENCHMARK_FUNCS[@]}"; do
        for SIZE in "${DATA_SET_SIZES[@]}"; do
            for NOISE_STD in "${NOISE[@]}"; do
                NOISE_STD_INT=0
                if [ $NOISE_STD != "None" ]; then
                    NOISE_STD_INT=$NOISE_STD
                fi 

                FUNC_NAME=${BENCHMARK_FUNC_NAMES[$BENCHMARK_FUNC]}

                JOB_ID=$(sbatch --parsable scripts/knn_experiment.sh $NEIGHBOR_COUNT $BENCHMARK_FUNC $SIZE $NOISE_STD "output/raw/${NEIGHBOR_COUNT}nn_${BENCHMARK_FUNC}_${NOISE_STD_INT}.csv")
                echo "$JOB_ID has been queued"

                JOB_IDS+=($JOB_ID)
            done
        done
    done
done