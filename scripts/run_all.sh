#!/bin/bash

mkdir -p output/raw

for script in scripts/standalone/*.sh; 
do sbatch "$script"; 
done