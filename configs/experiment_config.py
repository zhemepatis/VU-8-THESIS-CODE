from dataclasses import dataclass

@dataclass(frozen = True)
class ExperimentConfig:
    def __init__(self, 
                 try_count :int, 
                 verbose :bool) -> None:
        
        self.try_count :int = try_count
        self.verbose :bool = verbose