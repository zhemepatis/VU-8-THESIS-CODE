import numpy as np

class BenchmarkFunctions: 

    @staticmethod
    def sphere_func(vector :list[float]) -> np.ndarray:
        
        vector_squared = vector**2
        scalar = np.sum(vector_squared)
        return scalar
    
    
    @staticmethod
    def rosenbrock_func(vector :list[float]) -> np.ndarray:
    
        x = np.array(vector, dtype = float)
        scalar = np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
        return np.array(scalar)
    
    
    @staticmethod
    def rastrigin_func(vector :list[float]) -> np.ndarray:
    
        vector_diff = vector**2 - 10* np.cos(2 * np.pi * vector)
        scalar = 10 * len(vector) + np.sum(vector_diff)
        return scalar
    
    
    @staticmethod
    def resolve_benchmark_func(func_int :int) -> tuple[callable, list[float, float]]:

        if func_int == 0:
            return BenchmarkFunctions.sphere_func, [-5.12, 5.12]
        
        if func_int == 1:
            return BenchmarkFunctions.rosenbrock_func, [-5, 10]
        
        if func_int == 2:
            return BenchmarkFunctions.rastrigin_func, [-5.12, 5.12]
        
        return None