#!/bin/bash

mkdir -p output/raw

run_fnn_experiments.sh 
run_knn_experiments.sh 

# combine raw data
OUTPUT_FILE="output/raw_$(date +%Y%m%d_%H%M%S).csv"
combine_raw_results.sh $OUTPUT_FILE
echo "Output file: $OUTPUT_FILE"

rm -rf output/raw