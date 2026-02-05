import numpy as np

from models.data_set import DataSet

class DataGenerationFunctions:

    @staticmethod
    def generate_data_set(dimension :int,
                          domain_range :tuple[int, int],
                          set_size :int,
                          benchmark_function :function) -> DataSet:
    
        vectors :np.ndarray = DataGenerationFunctions.generate_vectors(dimension, domain_range, set_size)
        scalars :np.ndarray = DataGenerationFunctions.generate_scalars(vectors, benchmark_function)
        return DataSet(vectors, scalars)


    @staticmethod
    def generate_vectors(dimension :int,
                         domain_range :tuple[int, int],
                         set_size :int) -> np.ndarray:
    
        lower_end, higher_end = domain_range
        vectors = np.random.uniform(lower_end, higher_end, size = (set_size, dimension))
        return vectors


    @staticmethod
    def generate_scalars(vectors :np.ndarray,
                         benchmark_function :function) -> np.ndarray:
        
        values = np.array([benchmark_function(v) for v in vectors])
        return values
