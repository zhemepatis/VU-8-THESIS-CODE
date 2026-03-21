mkdir -p "output/graphs"

for dataFile in output/raw/sphere_func/*.csv; 
do python graph_generation.py --data-src $dataFile --graph-name "output/graphs/1nn_0.png";
done

