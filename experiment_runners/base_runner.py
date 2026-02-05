from abc import ABC, abstractmethod

from configs.data_set_config import DataSetConfig
from configs.experiment_config import ExperimentConfig
from configs.noise_config import NoiseConfig

from models.experiment_statistics import ExperimentStatistics
from models.data_set import DataSet

from sklearn.model_selection import train_test_split
import numpy as np


class BaseRunner(ABC):
    def __init__(self, 
                 experiment_config :ExperimentConfig, 
                 data_set_config :DataSetConfig, 
                 noise_config :NoiseConfig) -> None:
        
        self.experiment_config :ExperimentConfig = experiment_config
        self.data_set_config :DataSetConfig = data_set_config
        self.noise_config :NoiseConfig = noise_config


    def run(self) -> ExperimentStatistics:
        avg_min :float = 0.0
        avg_max :float = 0.0
        avg_mean :float = 0.0
        avg_std :float = 0.0

        for iteration in range(self.experiment_config.try_count):
            if self.experiment_config.verbose and (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1} / {self.experiment_config.try_count}")

            curr_try_stats = self._run_experiment()

            # save current try statistics
            avg_min += curr_try_stats.min
            avg_max += curr_try_stats.max
            avg_mean += curr_try_stats.mean
            avg_std += curr_try_stats.std

        # calculate statistic average
        avg_min /= self.experiment_config.try_count
        avg_max /= self.experiment_config.try_count
        avg_mean /= self.experiment_config.try_count
        avg_std /= self.experiment_config.try_count

        return ExperimentStatistics(avg_min, avg_max, avg_mean, avg_std)


    @abstractmethod
    def _run_experiment(self) -> ExperimentStatistics:
        pass


    def _split_data_set(self, 
                        raw_data_set :DataSet) -> tuple[DataSet, DataSet, DataSet]:
        
        # separate training data from validation and test data
        training_vectors, temp_vectors, training_scalars, temp_scalars = train_test_split(
            raw_data_set.vectors, 
            raw_data_set.scalars, 
            test_size = (self.data_set_config.validation_set_fraction + self.data_set_config.test_set_fraction), 
            random_state = 42
        )

        # separate validation data from test data
        validation_vectors, test_vectors, validation_scalars, test_scalars = train_test_split(
            temp_vectors, 
            temp_scalars, 
            test_size = (self.data_set_config.test_set_fraction / (self.data_set_config.test_set_fraction + self.data_set_config.validation_set_fraction)), 
            random_state = 42
        )

        training_set :DataSet = DataSet(training_vectors, training_scalars)
        validation_set :DataSet = DataSet(validation_vectors, validation_scalars)
        test_set :DataSet = DataSet(test_vectors, test_scalars)

        return training_set, validation_set, test_set


    def _apply_noise(self, 
                     data_set :DataSet) -> DataSet:
        
        noise = np.random.normal(self.noise_config.mean, self.noise_config.std, len(data_set.scalars))
        data_set.scalars += noise
        return data_set


    def _calculate_statistics(self, absolute_errors) -> ExperimentStatistics:
        curr_min :float = np.min(absolute_errors)
        curr_max :float = np.max(absolute_errors)
        curr_mean :float = np.mean(absolute_errors)
        curr_std :float = np.std(absolute_errors)

        return ExperimentStatistics(curr_min, curr_max, curr_mean, curr_std)