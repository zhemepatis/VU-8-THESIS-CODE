from abc import ABC, abstractmethod
from multiprocessing import Pool

from configs.data_set_config import DataSetConfig
from configs.data_split_config import DataSplitCofig
from configs.experiment_config import ExperimentConfig
from configs.noise_config import NoiseConfig

from models.error_statistics import ErrorStatistics
from models.experiment_statistics import ExperimentStatistics
from models.data_set import DataSet

from sklearn.model_selection import train_test_split
import numpy as np

class BaseRunner(ABC):
    def __init__(self, 
                 experiment_config :ExperimentConfig, 
                 data_set_config :DataSetConfig, 
                 data_split_config :DataSplitCofig,
                 noise_config :NoiseConfig) -> None:
        
        self.experiment_config :ExperimentConfig = experiment_config
        self.data_set_config :DataSetConfig = data_set_config
        self.data_split_config :DataSplitCofig = data_split_config
        self.noise_config :NoiseConfig = noise_config


    def run(self) -> ExperimentStatistics:
        
        absolute_err_stats :ErrorStatistics = ErrorStatistics(0.0, 0.0, 0.0, 0.0)
        relative_err_stats :ErrorStatistics = ErrorStatistics(0.0, 0.0, 0.0, 0.0)
        normalized_err_stats :ErrorStatistics = ErrorStatistics(0.0, 0.0, 0.0, 0.0)

        with Pool(self.experiment_config.process_number) as pool:
            results = pool.starmap(self._run_experiment, [() for _ in range(self.experiment_config.try_count)])

        # accumulate statistics
        for curr_try_stats in results:
            self.__accumulate_error_stats(absolute_err_stats, curr_try_stats.absolute_error_stats)
            self.__accumulate_error_stats(relative_err_stats, curr_try_stats.relative_error_stats)
            self.__accumulate_error_stats(normalized_err_stats, curr_try_stats.normalized_error_stats)

        # calculate statistic average
        self.__average_error_stats(absolute_err_stats, self.experiment_config.try_count)
        self.__average_error_stats(relative_err_stats, self.experiment_config.try_count)
        self.__average_error_stats(normalized_err_stats, self.experiment_config.try_count)

        return ExperimentStatistics(absolute_err_stats, relative_err_stats, normalized_err_stats)


    @abstractmethod
    def _run_experiment(self) -> ExperimentStatistics:
        pass


    def _split_data_set(self, 
                        raw_data_set :DataSet) -> tuple[DataSet, DataSet, DataSet]:
        
        if self.data_split_config.validation_set_fraction == 0:
            training_vectors, test_vectors, training_scalars, test_scalars = train_test_split(
                raw_data_set.vectors, 
                raw_data_set.scalars, 
                test_size = (self.data_split_config.validation_set_fraction + self.data_split_config.test_set_fraction), 
                random_state = 42
            )

            training_set :DataSet = DataSet(training_vectors, training_scalars)
            test_set :DataSet = DataSet(test_vectors, test_scalars)

            return training_set, None, test_set

        training_vectors, temp_vectors, training_scalars, temp_scalars = train_test_split(
            raw_data_set.vectors, 
            raw_data_set.scalars, 
            test_size = (self.data_split_config.validation_set_fraction + self.data_split_config.test_set_fraction), 
            random_state = 42
        )

        validation_vectors, test_vectors, validation_scalars, test_scalars = train_test_split(
            temp_vectors, 
            temp_scalars, 
            test_size = (self.data_split_config.test_set_fraction / (self.data_split_config.test_set_fraction + self.data_split_config.validation_set_fraction)), 
            random_state = 42
        )

        training_set :DataSet = DataSet(training_vectors, training_scalars)
        validation_set :DataSet = DataSet(validation_vectors, validation_scalars)
        test_set :DataSet = DataSet(test_vectors, test_scalars)

        return training_set, validation_set, test_set


    def _calculate_statistics(self, array) -> ExperimentStatistics:
        curr_min :float = np.min(array)
        curr_max :float = np.max(array)
        curr_mean :float = np.mean(array)
        curr_std :float = np.std(array)

        return curr_min, curr_max, curr_mean, curr_std
    

    def __accumulate_error_stats(self,
                                 target_stats: ErrorStatistics,
                                 source_stats: ErrorStatistics) -> None:
        
        target_stats.min += source_stats.min
        target_stats.max += source_stats.max
        target_stats.mean += source_stats.mean
        target_stats.std += source_stats.std


    def __average_error_stats(self,
                              stats: ErrorStatistics,
                              divisor: int) -> None:
        
        stats.min /= divisor
        stats.max /= divisor
        stats.mean /= divisor
        stats.std /= divisor