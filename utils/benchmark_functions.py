import numpy as np

class BenchmarkFunctions:

    @staticmethod
    def sphere_func(vector):
        vector_squared = vector**2
        scalar = np.sum(vector_squared)
        return scalar