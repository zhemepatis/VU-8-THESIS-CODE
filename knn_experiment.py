import argparse
from configs.data_set_config import DataSetConfig
from configs.data_split_config import DataSplitCofig
from configs.experiment_config import ExperimentConfig
from configs.k_nearest_neighbors_config import KNearestNeighborsConfig
from configs.noise_config import NoiseConfig
from experiment_runners.k_nearest_neighbors_runner import KNearestNeighborRunner
from utils.benchmark_funcs import BenchmarkFunctions

if __name__ == "__main__":

    # get arguments
    parser = argparse.ArgumentParser(description = "KNN Experiment Runner")

    parser.add_argument(
        "--data-set-size",
        type = int,
        required = True,
        help = "Data set size"
    )

    parser.add_argument(
        "--neighbors",
        type = int,
        required = True,
        help = "Number of neighbors for KNN"
    )

    parser.add_argument(
        "--noise-mean",
        type = float,
        default = None,
        help = "Gaussian noise mean"
    )

    parser.add_argument(
        "--noise-std",
        type = float,
        default = None,
        help = "Gaussian noise std"
    )

    parser.add_argument(
        "--processes",
        type = int,
        default = 1,
        help = "Number of parallel processes"
    )

    args = parser.parse_args()

    # setup experiment
    experiment_config :ExperimentConfig = ExperimentConfig(
        process_number = args.processes,
        try_count = 10
    )

    data_set_config :DataSetConfig = DataSetConfig(
        benchmark_function = BenchmarkFunctions.sphere_func,
        input_dimension = 4,
        component_domain = [-5, 5],
        data_set_size = args.data_set_size
    )

    data_split_config :DataSplitCofig = DataSplitCofig(
        training_set_fraction = 0.85,
        validation_set_fraction = 0,
        test_set_fraction = 0.15
    )

    noise_config = None
    if args.noise_mean != None and args.noise_std != None:
        noise_config :NoiseConfig = NoiseConfig(
            mean = args.noise_mean,
            std = args.noise_std
        )

    knn_config :KNearestNeighborsConfig = KNearestNeighborsConfig(
        neighbor_count = args.neighbors
    )

    # print experiment details
    print(f"EXPERIMENT PARAMETERS:")
    print(f"neighbor count = {knn_config.neighbor_count}")

    if noise_config is not None:
        print(f"noise mean = {noise_config.mean}")
        print(f"noise std = {noise_config.std}")
    else:
        print("no noise applied")
    
    print(f"data set size = {data_set_config.data_set_size:_}\n")

    # run experiment
    runner = KNearestNeighborRunner(
        experiment_config,
        data_set_config,
        data_split_config,
        noise_config,
        knn_config
    )

    results = runner.run()

    # print results
    print(f"\nRESULTS: {results.min}\t{results.max}\t{results.mean}\t{results.std}")
    print("\n=====\n")