from configs.data_set_config import DataSetConfig
from configs.data_split_config import DataSplitCofig
from configs.experiment_config import ExperimentConfig
from configs.k_nearest_neighbors_config import KNearestNeighborsConfig
from configs.noise_config import NoiseConfig
from experiment_runners.k_nearest_neighbors_runner import KNearestNeighborRunner
from utils.benchmark_funcs import BenchmarkFunctions

experiment_config :ExperimentConfig = ExperimentConfig(
    try_count = 100,
    verbose = True
)

# data set configuration
data_set_config :DataSetConfig = DataSetConfig(
    benchmark_function = BenchmarkFunctions.sphere_func,
    input_dimension = 4,
    component_domain = [-5, 5],
    data_set_size = 1_000
)

data_split_config :DataSplitCofig = DataSplitCofig(
    training_set_fraction = 0.85,
    validation_set_fraction = 0,
    test_set_fraction = 0.15
)

# k nearest neighbors configuration 
knn_config :KNearestNeighborsConfig = KNearestNeighborsConfig(
    neighbor_count = 3
)

neighbor_counts = [1, 3]
data_set_sizes = [1_000, 10_000, 100_000, 1_000_000, 10_000_000]

noise_configs = [
    None,
    NoiseConfig(
        mean = 0,
        std = 0.5
    ),
    NoiseConfig(
        mean = 0,
        std = 5
    )
]

for curr_neighbor_count in neighbor_counts:
    for curr_noise_config in noise_configs:
        for curr_data_set_size in data_set_sizes:
            
            print(f"EXPERIMENT PARAMETERS:")
            print(f"neighbor count = {curr_neighbor_count}")

            if curr_noise_config is not None:
                print(f"noise mean = {curr_noise_config.mean}")
                print(f"noise std = {curr_noise_config.std}")
            else:
                print("no noise applied")

            print(f"data set size = {curr_data_set_size:_}\n")

            # run experiment
            data_set_config :DataSetConfig = DataSetConfig(
                benchmark_function = BenchmarkFunctions.sphere_func,
                input_dimension = 4,
                component_domain = [-5, 5],
                data_set_size = curr_data_set_size
            )

            knn_config :KNearestNeighborsConfig = KNearestNeighborsConfig(
                neighbor_count = curr_neighbor_count
            ) 

            runner = KNearestNeighborRunner(
                experiment_config,
                data_set_config,
                data_split_config,
                curr_noise_config,
                knn_config
            )

            results = runner.run()

            # print results
            print(f"\nRESULTS: {results.min}\t{results.max}\t{results.mean}\t{results.std}")
            print("\n=====\n")


