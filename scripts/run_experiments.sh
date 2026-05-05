#!/bin/bash

# wait for all experiments to finish
# combine csvs
# rm raw folder

mkdir -p output/raw
JOB_IDS=()

# setup experiments
BENCHMARK_FUNCS=(0 1 2)
DATA_SET_SIZES=(1000 10000 100000 1000000)
NOISE=("None" 5)

# queue fnn experiments
for BENCHMARK_FUNC in "${BENCHMARK_FUNCS[@]}"; do
    for SIZE in "${DATA_SET_SIZES[@]}"; do
        for NOISE_STD in "${NOISE[@]}"; do
            NOISE_STD_INT=0
            if [ $NOISE_STD != "None" ]; then
                NOISE_STD_INT=$NOISE_STD
            fi 

            JOB_ID=$(sbatch  --parsable scripts/standalone/fnn.sh $BENCHMARK_FUNC $SIZE $NOISE_STD "output/raw/fnn_${NOISE_STD_INT}.csv")
            echo "[$BENCHMARK_FUNC | size=$SIZE | noise=$NOISE_STD] queued as job $JOB_ID"

            JOB_IDS+=($JOB_ID)
        done
    done
done

# setup experiments
NEIGHBOR_COUNTS=(1 2 4 8 16 32)
DATA_SET_SIZES+=(10000000)

# queue knn experiments
for NEIGHBOR_COUNT in "${NEIGHBOR_COUNTS[@]}"; do
    for BENCHMARK_FUNC in "${BENCHMARK_FUNCS[@]}"; do
        for SIZE in "${DATA_SET_SIZES[@]}"; do
            for NOISE_STD in "${NOISE[@]}"; do
                NOISE_STD_INT=0
                if [ NOISE_STD != "None" ]; then
                    NOISE_STD_INT=$NOISE_STD
                fi 

                JOB_ID=$(sbatch --parsable scripts/standalone/knn.sh $NEIGHBOR_COUNT $BENCHMARK_FUNC $SIZE $NOISE_STD "output/raw/${NEIGHBOR_COUNT}nn_${NOISE_STD_INT}.csv")
                echo "[$BENCHMARK_FUNC | size=$SIZE | noise=$NOISE_STD] queued as job $JOB_ID"

                JOB_IDS+=($JOB_ID)
            done
        done
    done
done

# wait for all the tasks to finish
DEPS=$(IFS=:; echo "${JOB_IDS[*]}")
sbatch --dependency=afterok:$DEPS --wrap "echo all jobs done"

# combine output data
OUTPUT_FILE="output/raw_$(date +%Y%m%d_%H%M%S).csv"

for FILE in output/raw/*.csv; do
    cat "$FILE" >> "$OUTPUT_FILE"
done

rm -rf output/raw
echo "Output file: $OUTPUT_FILE"