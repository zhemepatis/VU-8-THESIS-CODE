class BenchmarkFunctionConfig:
    def __init__(self, 
                 benchmark_func:callable,
                 component_domain :list[float],
                 min_value :float,
                 max_value :float) -> None:
        
        self.benchmark_func :callable = benchmark_func
        self.component_domain :list[float] = component_domain
        self.min :float = min_value
        self.max :float = max_value