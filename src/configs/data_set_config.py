from configs.benchmark_func_config import BenchmarkFunctionConfig

class DataSetConfig:
    def __init__(self,
                 benchmark_func_config :BenchmarkFunctionConfig,
                 input_dimension :int,  
                 data_set_size :int) -> None:
        
        self.benchmark_func_config :BenchmarkFunctionConfig = benchmark_func_config
        self.input_dimension :int = input_dimension
        self.data_set_size :int = data_set_size