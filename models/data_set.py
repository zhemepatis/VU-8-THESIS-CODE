import numpy as np

class DataSet:
    def __init__(self, 
                 vectors :np.ndarray, 
                 scalars :np.ndarray) -> None:
        
        self.vectors :np.ndarray = vectors
        self.scalars :np.ndarray = scalars