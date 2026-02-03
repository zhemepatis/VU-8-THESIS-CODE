from abc import ABC, abstractmethod

from utils.data_generation import generate_vectors, generate_scalars

from models.experiment_statistics import ExperimentStatistics
from models.data_set import DataSet

from sklearn.model_selection import train_test_split
import numpy as np


class BaseRunner(ABC):
    def __init__(self, experiment_config, data_set_config):
        self.experiment_config = experiment_config
        self.data_set_config = data_set_config

        # statistics
        self.stats = ExperimentStatistics()
        self.stats.min = 0
        self.stats.max = 0
        self.stats.mean = 0
        self.stats.std = 0


    def run(self):
        for _ in range(self.experiment_config.try_count):
            curr_try_stats = self._run_experiment()

            # save current try statistics
            self.statistics.min += curr_try_stats.min
            self.statistics.max += curr_try_stats.max
            self.statistics.mean += curr_try_stats.mean
            self.statistics.std += curr_try_stats.std

        # calculate statistic average
        self.min /= self.experiment_config.try_count
        self.max /= self.experiment_config.try_count
        self.mean /= self.experiment_config.try_count
        self.std /= self.experiment_config.try_count


    @abstractmethod
    def _run_experiment(self):
        pass


    def _generate_raw_data_set(self):
        vectors = generate_vectors(
            self.data_set_config.input_dimention, 
            self.data_set_config.component_domain, 
            self.data_set_config.data_set_size
        )
        
        scalars = generate_scalars(
            self.vectors, 
            self.data_set_config.benchmark_function
        )
        
        return vectors, scalars
    
    def _split_data_set(self, vectors, scalars):
        # separate training data from validation and test data
        training_vectors, temp_vectors, training_scalars, temp_scalars = train_test_split(
            vectors, 
            scalars, 
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


    def _calculate_statistics(self, absolute_errors):
        experiment_stats = ExperimentStatistics()

        experiment_stats.min = np.min(absolute_errors)
        experiment_stats.max = np.max(absolute_errors)
        experiment_stats.mean = np.mean(absolute_errors)
        experiment_stats.std = np.std(absolute_errors)

        return experiment_stats