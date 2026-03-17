import numpy as np

class BenchmarkFunctions:

    @staticmethod
    def sphere_func(vector):
        vector_squared = vector**2
        scalar = np.sum(vector_squared)
        return scalar
    
    @staticmethod
    def rastrigin_func(vector, a = 5):
        vector_diff = vector**2 - a * np.cos(2 * np.pi * vector)
        scalar = a * len(vector) + np.sum(vector_diff)
        return scalar