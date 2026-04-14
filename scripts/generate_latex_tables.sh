#!/bin/bash

BENCHMARK_FUNCS=("sphere_func" "rastrigin_func" "rosenbrock_func")

NOISE_STD_VALUES=(0 5)
NOISE_STD_NAMES=("0" "5")

mkdir -p output/tables


python generate_latex_table.py --data-src-filename data.csv --noise-std 0 --method "KNN" --caption "Custom caption here"
