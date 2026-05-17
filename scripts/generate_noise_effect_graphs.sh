#!/bin/bash

INPUT_FILE="${1:-output/raw_data.csv}"
OUTPUT_DIR="${2:-output/graphs}"
mkdir -p "$OUTPUT_DIR"

METHODS=("fnn")
# METHODS=("1nn" "2nn" "4nn" "8nn" "16nn" "32nn" "fnn")
DATA_FUNCTIONS=("sphere_func" "rosenbrock_func" "rastrigin_func")
STATS=("std" "mean")

for method in "${METHODS[@]}"; do
    for data_function in "${DATA_FUNCTIONS[@]}"; do
        for stat in "${STATS[@]}"; do
            OUTPUT_FILE="${OUTPUT_DIR}/${method}_${data_function}_${stat}.png"

            python src/generate_noise_effect_graph.py \
                --data-src-filename "$INPUT_FILE" \
                --graph-filename "$OUTPUT_FILE" \
                --method "$method" \
                --metric "$stat" \
                --data-function "$data_function"
        done
    done
done