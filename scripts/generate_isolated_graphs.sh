#!/bin/bash

INPUT_FILE="${1:-output/raw_data.csv}"
OUTPUT_DIR="${2:-output/graphs}"
mkdir -p "$OUTPUT_DIR"

METHODS=("fnn")
# METHODS=("1nn" "2nn" "4nn" "8nn" "16nn" "32nn" "fnn")
NOISE_STDS=(0 5)
ERROR_TYPES=("absolute" "normalized" "relative")
STATS=("std" "mean")
DATA_FUNCTIONS=("sphere_func" "rosenbrock_func" "rastrigin_func")

for method in "${METHODS[@]}"; do
    for noise_std in "${NOISE_STDS[@]}"; do
        for error_type in "${ERROR_TYPES[@]}"; do
            for stat in "${STATS[@]}"; do
                for data_function in "${DATA_FUNCTIONS[@]}"; do
                    OUTPUT_FILE="${OUTPUT_DIR}/${method}_${data_function}_${noise_std}_${error_type}_${stat}.png"

                    python src/generate_isolated_graph.py \
                        --data-src-filename "$INPUT_FILE" \
                        --graph-filename "$OUTPUT_FILE" \
                        --method "$method" \
                        --noise-std "$noise_std" \
                        --error-type "$error_type" \
                        --stat "$stat" \
                        --data-function "$data_function"
                done
            done
        done
    done
done