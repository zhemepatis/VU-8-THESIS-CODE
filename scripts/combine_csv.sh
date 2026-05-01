#!/bin/bash

OUTPUT_FILE="output/raw_data.csv"

# append data from existing .csv files
for FILE in output/raw/*.csv; do
    cat "$FILE" >> "$OUTPUT_FILE"
done

echo "Done: $OUTPUT_FILE"