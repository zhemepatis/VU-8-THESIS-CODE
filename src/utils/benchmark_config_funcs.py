
from configs.benchmark_func_config import BenchmarkFunctionConfig
from models.benchmark_func_enum import BenchmarkFuncEnum
from utils.benchmark_funcs import BenchmarkFunctions

class BenchmarkConfigFunctions: 

    @staticmethod
    def resolve_benchmark_config(benchmark_func_str :str) -> tuple[callable, list[float, float]]:

        benchmark_func = BenchmarkConfigFunctions.map_str_to_benchmark_func(benchmark_func_str)
        match benchmark_func:
            
            case BenchmarkFuncEnum.UNDEFINED:
                raise ValueError("Inputed integer doesn't correspond to any of the benchmark functions")
            
            case BenchmarkFuncEnum.SPHERE_FUNC:
                return BenchmarkFunctionConfig(BenchmarkFunctions.sphere_func, [-5.12, 5.12], 0, 104.8576)
            
            case BenchmarkFuncEnum.ROSENBROCK_FUNC:
                return BenchmarkFunctionConfig(BenchmarkFunctions.rosenbrock_func, [-5.12, 5.12], 0, 240937.699008)
            
            case BenchmarkFuncEnum.RASTRIGIN_FUNC:
                return BenchmarkFunctionConfig(BenchmarkFunctions.rastrigin_func, [-5.12, 5.12], 0, 161.376167)


    @staticmethod
    def map_str_to_benchmark_func(benchmark_func_str :str) -> BenchmarkFuncEnum:

        benchmark_func_str = benchmark_func_str.upper()
        match benchmark_func_str:

            case BenchmarkFuncEnum.SPHERE_FUNC.name: 
                return BenchmarkFuncEnum.SPHERE_FUNC
            
            case BenchmarkFuncEnum.ROSENBROCK_FUNC.name:
                return BenchmarkFuncEnum.ROSENBROCK_FUNC
            
            case BenchmarkFuncEnum.RASTRIGIN_FUNC.name:
                return BenchmarkFuncEnum.RASTRIGIN_FUNC
            
            case _:
                return BenchmarkFuncEnum.UNDEFINED

