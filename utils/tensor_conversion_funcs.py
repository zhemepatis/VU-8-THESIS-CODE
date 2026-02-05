import torch
from models.data_set import DataSet
from models.data_set_tensor import DataSetTensors

class TensorConversionFunctions:

    @staticmethod
    def convert_to_data_set(data_set_tensors :DataSetTensors) -> DataSet:
        
        vectors = data_set_tensors.vector_tensor.numpy()
        scalars = data_set_tensors.scalar_tensor.numpy()

        return DataSet(vectors, scalars)


    @staticmethod
    def convert_to_tensors(data_set :DataSet) -> DataSetTensors:
    
        vector_tensor = torch.FloatTensor(data_set.vectors)
        scalar_tensor = torch.FloatTensor(data_set.scalars)

        return DataSetTensors(vector_tensor, scalar_tensor)
