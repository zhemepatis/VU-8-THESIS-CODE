class DataSplitCofig:

    def __init__(self,
                 training_set_fraction :float, 
                 validation_set_fraction :float, 
                 test_set_fraction :float) -> None:

        self.training_set_fraction :float = training_set_fraction
        self.validation_set_fraction :float = validation_set_fraction
        self.test_set_fraction :float = test_set_fraction
    