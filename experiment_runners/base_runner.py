from abc import ABC, abstractmethod

from utils.data_generation import generate_vectors, generate_scalars

from models.experiment_statistics import ExperimentStatistics
from models.data_set import DataSet

from sklearn.model_selection import train_test_split
import numpy as np


class BaseRunner(ABC):
    def __init__(self, experiment_config, data_set_config, noise_config):
        self.experiment_config = experiment_config
        self.data_set_config = data_set_config
        self.noise_config = noise_config

        # statistics
        self.stats = ExperimentStatistics()
        self.stats.min = 0
        self.stats.max = 0
        self.stats.mean = 0
        self.stats.std = 0


    def run(self) -> ExperimentStatistics:
        for _ in range(self.experiment_config.try_count):
            curr_try_stats = self._run_experiment()

            # save current try statistics
            self.stats.min += curr_try_stats.min
            self.stats.max += curr_try_stats.max
            self.stats.mean += curr_try_stats.mean
            self.stats.std += curr_try_stats.std

        # calculate statistic average
        self.stats.min /= self.experiment_config.try_count
        self.stats.max /= self.experiment_config.try_count
        self.stats.mean /= self.experiment_config.try_count
        self.stats.std /= self.experiment_config.try_count

        return self.stats


    @abstractmethod
    def _run_experiment(self):
        pass


    def _generate_raw_data_set(self):
        vectors = generate_vectors(
            self.data_set_config.input_dimension, 
            self.data_set_config.component_domain, 
            self.data_set_config.data_set_size
        )
        
        scalars = generate_scalars(
            vectors, 
            self.data_set_config.benchmark_function
        )

        return DataSet(vectors, scalars)


    def _split_data_set(self, raw_data_set):
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

        training_set = DataSet(training_vectors, training_scalars)
        validation_set = DataSet(validation_vectors, validation_scalars)
        test_set = DataSet(test_vectors, test_scalars)

        return training_set, validation_set, test_set
    

    def _apply_noise(self, data_set):
        noise = np.random.normal(self.noise_config.mean, self.noise_config.std, len(data_set.scalars))
        data_set.scalars += noise


    def _calculate_statistics(self, absolute_errors):
        experiment_stats = ExperimentStatistics()

        experiment_stats.min = np.min(absolute_errors)
        experiment_stats.max = np.max(absolute_errors)
        experiment_stats.mean = np.mean(absolute_errors)
        experiment_stats.std = np.std(absolute_errors)

        return experiment_stats