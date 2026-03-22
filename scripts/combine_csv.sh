#!/bin/bash

OUTPUT_FILE="output/raw_data.csv"

# write header to .csv file
# echo "Method,Data size,Data function,Noise mean,Noise std. deviation,Time elapsed,Abs. error min,Abs. error max,Mean,Abs. error std. deviation" > "$OUTPUT_FILE"

# append data from existing .csv files
for FILE in output/raw/*.csv; do
    cat "$FILE" >> "$OUTPUT_FILE"
done

echo "Done: $OUTPUT_FILE"