#!/bin/bash

INPUT_FILE="${1:-output/raw_data.csv}"
OUTPUT_DIR="${2:-output/graphs}"
mkdir -p "output/graphs"

METHODS=("1nn" "2nn" "4nn" "8nn" "16nn" "32nn" "fnn")

for i in "${!METHODS[@]}"; do
    OUTPUT_FILE="${OUTPUT_DIR}/noise_${METHODS[$i]}.png"


    python src/generate_noise_graph.py \
        --data-src-filename $INPUT_FILE \
        --graph-filename $OUTPUT_FILE \
        --method "${METHODS[$i]}"
done

