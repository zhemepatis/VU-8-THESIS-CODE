from configs.data_set_config import DataSetConfig
from configs.data_split_config import DataSplitCofig
from configs.experiment_config import ExperimentConfig
from configs.k_nearest_neighbors_config import KNearestNeighborsConfig
from configs.noise_config import NoiseConfig
from experiment_runners.k_nearest_neighbors_runner import KNearestNeighborRunner
from utils.benchmark_funcs import BenchmarkFunctions

if __name__ == "__main__":
    experiment_config :ExperimentConfig = ExperimentConfig(
        process_number = 4,
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

    noise_config = None
    neighbor_count = 1
    data_set_size = 1_000_000

    # print experiment details
    print(f"EXPERIMENT PARAMETERS:")
    print(f"neighbor count = {neighbor_count}")

    if noise_config is not None:
        print(f"noise mean = {noise_config.mean}")
        print(f"noise std = {noise_config.std}")
    else:
        print("no noise applied")

    print(f"data set size = {data_set_size:_}\n")

    # run experiment
    data_set_config :DataSetConfig = DataSetConfig(
        benchmark_function = BenchmarkFunctions.sphere_func,
        input_dimension = 4,
        component_domain = [-5, 5],
        data_set_size = data_set_size
    )

    knn_config :KNearestNeighborsConfig = KNearestNeighborsConfig(
        neighbor_count = neighbor_count
    ) 

    runner = KNearestNeighborRunner(
        experiment_config,
        data_set_config,
        data_split_config,
        noise_config,
        knn_config
    )

    results = runner.run()

    # print results
    print(results)
    print(f"\nRESULTS: {results.min}\t{results.max}\t{results.mean}\t{results.std}")
    print("\n=====\n")