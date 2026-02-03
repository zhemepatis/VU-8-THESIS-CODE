class DataSetConfig:
    def __init__(self, benchmark_function, input_dimention, component_domain, data_set_size):
        # data generation configuration
        self.benchmark_function = benchmark_function
        self.dimention = input_dimention
        self.component_domain = component_domain
        self.data_set_size = data_set_size

        # data split configuration
        self.training_set_fraction = 0.7
        self.validation_set_fraction = 0.15
        self.test_set_fraction = 0.15