class DataSetConfig:
    def __init__(self, 
                 benchmark_function :function,
                 input_dimension :int, 
                 component_domain :list[int], 
                 data_set_size :int) -> None:
        
        self.benchmark_function :function = benchmark_function
        self.input_dimension :int = input_dimension
        self.component_domain :list[int] = component_domain
        self.data_set_size :int = data_set_size