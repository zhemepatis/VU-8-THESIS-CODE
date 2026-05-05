#!/bin/bash

OUTPUT_FILE="${1}"

for FILE in output/raw/*.csv; do
    cat "$FILE" >> "$OUTPUT_FILE"
done