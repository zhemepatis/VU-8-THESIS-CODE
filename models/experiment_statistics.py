class ExperimentStatistics:
    def __init__(self, 
                 min :float, 
                 max :float, 
                 mean :float, 
                 std :float):
        
        self.min :float = min
        self.max :float = max
        self.mean :float = mean
        self.std :float = std