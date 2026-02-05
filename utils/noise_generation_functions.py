import numpy as np

from models.data_set import DataSet

class NoiseGenerationFunctions:

    @staticmethod
    def apply_gaussian_noise(mean :float, 
                                std :float,
                                clean_data_set :DataSet) -> DataSet:
        
        noise = np.random.normal(mean, std, len(clean_data_set.scalars))
        noisy_scalars = clean_data_set.scalars + noise

        return DataSet(clean_data_set.vectors, noisy_scalars)


    @staticmethod
    def generate_gaussian_noise(mean :float, 
                                std :float,
                                data_set_size :int) -> np.ndarray:
        
        return np.random.normal(mean, std, data_set_size)