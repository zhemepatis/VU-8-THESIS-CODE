#!/bin/bash
#SBATCH -p main
#SBATCH -n4

python3 -m venv venv
source venv/bin/activate

pip3 install torch 
pip3 install numpy 
pip3 install scikit-learn 

python3 knn_experiments.py