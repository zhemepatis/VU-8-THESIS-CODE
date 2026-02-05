import numpy as np
from sklearn.preprocessing import MinMaxScaler
from models.data_set import DataSet

class NormalizationFunctions:

    @staticmethod
    def normalize_data_set(data_set :DataSet,
                           vector_scaler :MinMaxScaler,
                           scalar_scaler :MinMaxScaler) -> DataSet:
        
        vectors_normalized :np.ndarray = NormalizationFunctions.normalize_vector_set(data_set.vectors, vector_scaler)
        scalars_normalized :np.ndarray = NormalizationFunctions.normalize_scalar_set(data_set.scalars, scalar_scaler)

        return DataSet(vectors_normalized, scalars_normalized)


    @staticmethod
    def normalize_vector_set(vectors :list[float],
                             scaler :MinMaxScaler) -> np.ndarray:
        
        return scaler.transform(vectors)
    

    @staticmethod
    def normalize_scalar_set(scalars :list[float],
                             scaler :MinMaxScaler) -> np.ndarray:
        
        return scaler.transform(scalars.reshape(-1, 1))
    

    @staticmethod
    def denormalize_data_set(data_set :DataSet,
                             vector_scaler :MinMaxScaler,
                             scalar_scaler :MinMaxScaler) -> DataSet:
        
        denormalized_vectors :np.ndarray = NormalizationFunctions.denormalize_vector_set(data_set.vectors, vector_scaler)
        denormalized_scalars :np.ndarray = NormalizationFunctions.denormalize_scalar_set(data_set.vectors, scalar_scaler)
        
        return DataSet(denormalized_vectors, denormalized_scalars)
    
    
    @staticmethod
    def denormalize_vector_set(scalars :np.ndarray,
                               scaler :MinMaxScaler) -> np.ndarray:
        
        return scaler.inverse_transform(scalars)
    
    
    @staticmethod
    def denormalize_scalar_set(scalars :np.ndarray,
                               scaler :MinMaxScaler) -> np.ndarray:
        
        return scaler.inverse_transform(scalars)
