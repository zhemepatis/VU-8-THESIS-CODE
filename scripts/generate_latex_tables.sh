#!/bin/bash

INPUT_FILE="${1:-output/raw_data.csv}"
OUTPUT_DIR="${2:-output/tables}"
mkdir -p $OUTPUT_DIR

METHODS=("1nn" "2nn" "4nn" "8nn" "16nn" "32nn" "fnn")
METHOD_NAMES=("\$k\$ artimiausių kaimynų metodo" "\$k\$ artimiausių kaimynų metodo" "\$k\$ artimiausių kaimynų metodo" "\$k\$ artimiausių kaimynų metodo" "\$k\$ artimiausių kaimynų metodo" "\$k\$ artimiausių kaimynų metodo" "Dirbtinio neuroninio tinklo" )
NEIGHBOUR_COUNT=(1 2 4 8 16 32 0)

BENCHMARK_FUNCS=("sphere_func" "rastrigin_func" "rosenbrock_func")
BENCHMARK_FUNC_NAMES=("sferos" "Rastrigin'o" "Rosenbrock'o")

NOISE_STD_VALUES=(0 5)
NOISE_STD_NAMES=("0" "5")

for i in "${!METHODS[@]}"; do
    for j in "${!BENCHMARK_FUNCS[@]}"; do
        for k in "${!NOISE_STD_VALUES[@]}"; do
            OUTPUT_FILE="${OUTPUT_DIR}/${METHODS[$i]}_${BENCHMARK_FUNCS[$j]}_${NOISE_STD_NAMES[$k]}.tex"

            python src/generate_latex_table.py \
                --data-src-filename $INPUT_FILE \
                --table-filename $OUTPUT_FILE \
                --method "${METHODS[$i]}" \
                --benchmark-func "${BENCHMARK_FUNCS[$j]}" \
                --noise-std "${NOISE_STD_VALUES[$k]}" \
                --caption "${METHOD_NAMES[$i]} rezultatai su ${BENCHMARK_FUNC_NAMES[$j]} etalono funkcija (\$k = ${NEIGHBOUR_COUNT[$i]}$, $\mu = 0$, $\sigma = ${NOISE_STD_NAMES[$k]}$)"
        done
    done
done