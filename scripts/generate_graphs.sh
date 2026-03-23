#!/bin/bash

BENCHMARK_FUNCS=("sphere_func" "rastrigin_func")
TITLES=("Modelių rezultatai su sferos funkcijos generuotais duomenim" "Modelių rezultatai su Rastrigin'o funkcijos generuotais duomenim")

NOISE_STD_VALUES=(0 0.5 5)
NOISE_STD_NAMES=("0" "0p5" "5")

mkdir -p "output/graphs"

for i in "${!BENCHMARK_FUNCS[@]}"; do
    for j in "${!NOISE_STD_VALUES[@]}"; do
        python bar_graph.py \
            --data-src-filename "output/raw_data.csv" \
            --graph-filename "output/graphs/bar_${BENCHMARK_FUNCS[$i]}_${NOISE_STD_NAMES[$j]}.png" \
            --graph-title "${TITLES[$i]}" \
            --benchmark-func "${BENCHMARK_FUNCS[$i]}" \
            --noise-std "${NOISE_STD_VALUES[$j]}"
    done
done