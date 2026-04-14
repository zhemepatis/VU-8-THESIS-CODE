import numpy as np

class BenchmarkFunctions:            

    @staticmethod
    def sphere_func(vector :list[int]) -> np.ndarray:
        vector_squared = vector**2
        scalar = np.sum(vector_squared)
        return scalar
    
    @staticmethod
    def rosenbrock_func(vector :list[int]) -> np.ndarray:
    
        x = np.array(vector, dtype = float)
        scalar = np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
        return np.array(scalar)
    
    @staticmethod
    def rastrigin_func(vector :list[int]) -> np.ndarray:
    
        vector_diff = vector**2 - 10* np.cos(2 * np.pi * vector)
        scalar = 10 * len(vector) + np.sum(vector_diff)
        return scalar