#!/bin/bash

mkdir -p "output/graphs"
python bar_graph.py --data-src "output/raw_data.csv" --graph-name "output/graphs/1nn_0.png"
