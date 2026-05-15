#!/bin/bash

INPUT_DIR="${1:-output/raw}"
OUTPUT_FILE="${2:-output/raw_$(date +%Y%m%d_%H%M%S).csv}"

# add header
echo "Method,\
Data size,\
Data function,\
Noise mean,\
Noise std. deviation,\
Absolute error min,\
Absolute error max,\
Absolute error mean,\
Absolute error std. deviation,\
Relative error min,\
Relative error max,\
Relative error mean,\
Relative error std. deviation,\
Normalized error min,\
Normalized error max,\
Normalized error mean,\
Normalized error std. deviation,\
Time elapsed" > "$OUTPUT_FILE"

# add the known rows
cat << 'EOF' >> "$OUTPUT_FILE"
fnn,10000000,sphere_func,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
fnn,10000000,sphere_func,0,5,0,0,0,0,0,0,0,0,0,0,0,0,0
fnn,10000000,rastrigin_func,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
fnn,10000000,rastrigin_func,0,5,0,0,0,0,0,0,0,0,0,0,0,0,0
fnn,10000000,rosenbrock_func,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
fnn,10000000,rosenbrock_func,0,5,0,0,0,0,0,0,0,0,0,0,0,0,0
EOF

# add actual results
for FILE in ${INPUT_DIR}/*; do
  cat "$FILE" >> "$OUTPUT_FILE"
done