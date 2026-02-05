from dataclasses import dataclass

@dataclass(frozen = True)
class DataSetConfig:
    def __init__(self, 
                 benchmark_function :function,
                 input_dimension :int, 
                 component_domain :list[int], 
                 data_set_size :int, 
                 training_set_fraction :float, 
                 validation_set_fraction :float, 
                 test_set_fraction :float) -> None:
        
        # data generation configuration
        self.benchmark_function :function = benchmark_function
        self.input_dimension :int = input_dimension
        self.component_domain :list[int] = component_domain
        self.data_set_size :int = data_set_size

        # data split configuration
        self.training_set_fraction :float = training_set_fraction
        self.validation_set_fraction :float = validation_set_fraction
        self.test_set_fraction :float = test_set_fraction