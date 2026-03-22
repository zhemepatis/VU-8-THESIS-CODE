import numpy as np
from models.benchmark_function import BenchmarkFunction

class BenchmarkFunctions:            

    @staticmethod
    def sphere_func(vector :list[int]) -> np.ndarray:
        vector_squared = vector**2
        scalar = np.sum(vector_squared)
        return scalar
    
    @staticmethod
    def rosenbrock_func(vector :list[int], 
                       a :int = 100) -> np.ndarray:
    
        x = np.array(vector, dtype=float)
        scalar = np.sum(a * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
        return np.array(scalar)
    
    @staticmethod
    def rastrigin_func(vector :list[int], 
                       a :int = 10) -> np.ndarray:
    
        vector_diff = vector**2 - a * np.cos(2 * np.pi * vector)
        scalar = a * len(vector) + np.sum(vector_diff)
        return scalar
    
    @staticmethod   
    def resolve_benchmark_func(benchmark_function_enum :int) -> function:
        
        if benchmark_function_enum == BenchmarkFunction.SPHERE:
            return BenchmarkFunctions.sphere_func
        
        if benchmark_function_enum == BenchmarkFunction.ROSENBROCK:
            return BenchmarkFunctions.rosenbrock_func
        
        if benchmark_function_enum == BenchmarkFunction.RASTRIGIN:
            return BenchmarkFunctions.rastrigin_func