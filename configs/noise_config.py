from dataclasses import dataclass

@dataclass(frozen = True)
class NoiseConfig:
    def __init__(self, 
                 mean :float, 
                 std :float):

        self.mean :float = mean
        self.std :float = std