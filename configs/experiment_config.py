class ExperimentConfig:
    def __init__(self, 
                 process_number :int,
                 try_count :int, 
                 verbose :bool) -> None:
        
        self.process_number :int = process_number
        self.try_count :int = try_count
        self.verbose :bool = verbose