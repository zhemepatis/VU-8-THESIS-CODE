class BenchmarkFunctionConfig:
    def __init__(self, 
                 benchmark_func:callable,
                 component_domain :list[float]) -> None:
        
        self.benchmark_func :callable = benchmark_func
        self.component_domain :list[float] = component_domain