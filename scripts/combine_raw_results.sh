#!/bin/bash

OUTPUT_FILE="${1}"

# add the known rows
echo "Method,Data size,Data function,Noise mean,Noise std. deviation,Relative error min,Relative error max,Relative error mean,Relative error std. deviation,Time elapsed" \
  > "$OUTPUT_FILE"

cat << 'EOF' >> "$OUTPUT_FILE"
fnn,10000000,sphere_func,0,0,0,0,0,0,0
fnn,10000000,sphere_func,0,5,0,0,0,0,0
fnn,10000000,rastrigin_func,0,0,0,0,0,0,0
fnn,10000000,rastrigin_func,0,5,0,0,0,0,0
fnn,10000000,rosenbrock_func,0,0,0,0,0,0,0
fnn,10000000,rosenbrock_func,0,5,0,0,0,0,0
EOF

# add actual results
for FILE in output/raw/*.csv; do
    cat "$FILE" >> "$OUTPUT_FILE"
done