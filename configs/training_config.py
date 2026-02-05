from dataclasses import dataclass

@dataclass(frozen = True)
class TrainingConfig:
    def __init__(self, 
                 batch_size :int, 
                 delta :float, 
                 epoch_limit :int, 
                 patience_limit :int,
                 learning_rate :float,
                 verbose :bool) -> None:
        
        self.batch_size :int = batch_size
        self.delta :float = delta
        self.epoch_limit :int = epoch_limit
        self.patience_limit :int = patience_limit
        self.learning_rate :float = learning_rate

        self.verbose :bool = verbose