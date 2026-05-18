#!/bin/bash

INPUT_FILE="${1:-output/raw_data.csv}"
OUTPUT_DIR="${2:-output/graphs}"
mkdir -p $OUTPUT_DIR

BENCHMARK_FUNCS=("sphere_func" "rastrigin_func" "rosenbrock_func")
NOISE_STD_VALUES=(0 5)
NOISE_STD_NAMES=("0" "5")
METRICS=("mean" "std")

for i in "${!BENCHMARK_FUNCS[@]}"; do
    for j in "${!NOISE_STD_VALUES[@]}"; do
        for metric in "${METRICS[@]}"; do
            OUTPUT_FILE="${OUTPUT_DIR}/knn_${BENCHMARK_FUNCS[$i]}_${NOISE_STD_NAMES[$j]}_${metric}.png"

            python src/generate_condensed_results_graph.py \
                --data-src-filename $INPUT_FILE \
                --graph-filename $OUTPUT_FILE \
                --benchmark-func "${BENCHMARK_FUNCS[$i]}" \
                --noise-std "${NOISE_STD_VALUES[$j]}" \
                --metric "$metric"
        done
    done
done