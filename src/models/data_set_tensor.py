from torch import FloatTensor

class DataSetTensors:
    def __init__(self, 
                 vector_tensor :FloatTensor, 
                 scalar_tensor :FloatTensor):
        
        self.vector_tensor :FloatTensor = vector_tensor
        self.scalar_tensor :FloatTensor = scalar_tensor