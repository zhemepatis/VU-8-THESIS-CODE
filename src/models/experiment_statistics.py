from models.error_statistics import ErrorStatistics

class ExperimentStatistics:
    def __init__(self, 
                 absolute_error_stats :ErrorStatistics, 
                 relative_error_stats :ErrorStatistics,
                 normalized_error_stats :ErrorStatistics):
        
        self.absolute_error_stats :ErrorStatistics = absolute_error_stats
        self.relative_error_stats :ErrorStatistics = relative_error_stats
        self.normalized_error_stats :ErrorStatistics = normalized_error_stats