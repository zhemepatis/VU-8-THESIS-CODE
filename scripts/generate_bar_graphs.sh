#!/bin/bash

INPUT_FILE="${1:-output/raw_data.csv}"
OUTPUT_DIR="${2:-output/graphs}"

BENCHMARK_FUNCS=("sphere_func" "rastrigin_func" "rosenbrock_func")
NOISE_STD_VALUES=(0 5)
NOISE_STD_NAMES=("0" "5")

mkdir -p $OUTPUT_DIR

for i in "${!BENCHMARK_FUNCS[@]}"; do
    for j in "${!NOISE_STD_VALUES[@]}"; do
        OUTPUT_FILE="${OUTPUT_DIR}/bar_${BENCHMARK_FUNCS[$i]}_${NOISE_STD_NAMES[$j]}.png"

        python src/generate_bar_graph.py \
            --data-src-filename $INPUT_FILE \
            --graph-filename $OUTPUT_FILE \
            --benchmark-func "${BENCHMARK_FUNCS[$i]}" \
            --noise-std "${NOISE_STD_VALUES[$j]}"
    done
done