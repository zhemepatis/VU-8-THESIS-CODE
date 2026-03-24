#!/bin/bash

METHODS=("1nn" "2nn" "4nn" "8nn" "16nn" "32nn" "fnn")

mkdir -p "output/graphs"

for i in "${!METHODS[@]}"; do
    python noise_graph.py \
        --data-src-filename "output/raw_data.csv" \
        --graph-filename "output/graphs/noise_${METHODS[$i]}.png" \
        --method "${METHODS[$i]}"
done

