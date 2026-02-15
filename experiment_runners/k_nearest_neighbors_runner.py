import numpy as np
from configs.data_set_config import DataSetConfig
from configs.data_split_config import DataSplitCofig
from configs.experiment_config import ExperimentConfig
from configs.k_nearest_neighbors_config import KNearestNeighborsConfig
from configs.noise_config import NoiseConfig
from experiment_runners.base_runner import BaseRunner
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from models.data_set import DataSet
from models.experiment_statistics import ExperimentStatistics
from utils.data_generation_funcs import DataGenerationFunctions
from utils.noise_generation_funcs import NoiseGenerationFunctions
from utils.normalization_funcs import NormalizationFunctions

class KNearestNeighborRunner(BaseRunner):
    
    def __init__(self, 
                 experiment_config :ExperimentConfig, 
                 data_set_config :DataSetConfig, 
                 data_split_config :DataSplitCofig,
                 noise_config :NoiseConfig,
                 knn_config :KNearestNeighborsConfig) -> None:
        
        super().__init__(experiment_config, data_set_config, data_split_config, noise_config)
        self.knn_config :KNearestNeighborsConfig = knn_config
        

    def _run_experiment(self) -> ExperimentStatistics:
        data_set_raw :DataSet = DataGenerationFunctions.generate_data_set(
            self.data_set_config.input_dimension, 
            self.data_set_config.component_domain, 
            self.data_set_config.data_set_size,
            self.data_set_config.benchmark_function)
        
        # split data into training, validation, testing data sets
        splits :tuple[DataSet, DataSet, DataSet] = self._split_data_set(data_set_raw)
        
        training_set :DataSet = splits[0]
        test_set :DataSet = splits[2]

        # generate noise
        if self.noise_config != None:
            training_set = NoiseGenerationFunctions.apply_gaussian_noise(self.noise_config.mean, self.noise_config.std, training_set)

        # normalize data sets
        vector_scaler = MinMaxScaler().fit(training_set.vectors)
        scalar_scaler = MinMaxScaler().fit(training_set.scalars.reshape(-1, 1))

        training_set = NormalizationFunctions.normalize_data_set(training_set, vector_scaler, scalar_scaler)
        test_set = NormalizationFunctions.normalize_data_set(test_set, vector_scaler, scalar_scaler)
        
        # create kNN model 
        model = KNeighborsRegressor(n_neighbors = self.knn_config.neighbor_count)
        model.fit(training_set.vectors, training_set.scalars)
        
        # evaluate results
        prediction_set :DataSet = self.__test(model, test_set)

        prediction_set :DataSet = NormalizationFunctions.denormalize_data_set(prediction_set, vector_scaler, scalar_scaler)
        test_set :DataSet = NormalizationFunctions.denormalize_data_set(test_set, vector_scaler, scalar_scaler)

        abs_err_set :np.ndarray = np.abs(prediction_set.scalars - test_set.scalars)
        return self._calculate_statistics(abs_err_set)


    def __test(self, 
               model :KNeighborsRegressor, 
               test_set :DataSet) -> DataSet:

        prediction_scalars = model.predict(test_set.vectors)
        return DataSet(test_set.vectors, prediction_scalars)