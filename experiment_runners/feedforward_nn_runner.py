import numpy as np
from configs.data_set_config import DataSetConfig
from configs.data_split_config import DataSplitCofig
from configs.experiment_config import ExperimentConfig
from configs.feedforward_nn_config import FeedforwardNNConfig
from configs.training_config import TrainingConfig
from configs.noise_config import NoiseConfig
from experiment_runners.base_runner import BaseRunner
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from models.data_set import DataSet
from models.data_set_tensor import DataSetTensors
from neural_network_models.feedforward_nn import FeedforwardNN
from torch.utils.data import DataLoader, TensorDataset
from models.experiment_statistics import ExperimentStatistics
from neural_network_models.feedforward_nn import FeedforwardNN
from utils.data_generation_funcs import DataGenerationFunctions
from utils.noise_generation_funcs import NoiseGenerationFunctions
from utils.normalization_funcs import NormalizationFunctions
from utils.tensor_conversion_funcs import TensorConversionFunctions

class FeedforwardNNRunner(BaseRunner):

    def __init__(self, 
                 experiment_config :ExperimentConfig, 
                 data_set_config :DataSetConfig, 
                 data_split_config :DataSplitCofig,
                 noise_config :NoiseConfig, 
                 training_config :TrainingConfig,
                 fnn_config :FeedforwardNNConfig) -> None:
        
        super().__init__(experiment_config, data_set_config, data_split_config, noise_config)
        self.training_config :TrainingConfig = training_config
        self.fnn_config :FeedforwardNNConfig = fnn_config


    def _run_experiment(self) -> ExperimentStatistics:
        data_set_raw :DataSet = DataGenerationFunctions.generate_data_set(
            self.data_set_config.input_dimension, 
            self.data_set_config.component_domain, 
            self.data_set_config.data_set_size,
            self.data_set_config.benchmark_function)
        
        # split data into training, validation, testing data sets
        splits :tuple[DataSet, DataSet, DataSet] = self._split_data_set(data_set_raw)
        
        training_set :DataSet = splits[0]
        validation_set :DataSet = splits[1]
        test_set :DataSet = splits[2]

        # generate noise
        if self.noise_config != None:
            training_set = NoiseGenerationFunctions.apply_gaussian_noise(self.noise_config.mean, self.noise_config.std, training_set)
            validation_set = NoiseGenerationFunctions.apply_gaussian_noise(self.noise_config.mean, self.noise_config.std, validation_set)

        # normalize data sets
        vector_scaler = MinMaxScaler().fit(training_set.vectors)
        scalar_scaler = MinMaxScaler().fit(training_set.scalars.reshape(-1, 1))

        training_set = NormalizationFunctions.normalize_data_set(training_set, vector_scaler, scalar_scaler)
        validation_set = NormalizationFunctions.normalize_data_set(validation_set, vector_scaler, scalar_scaler)
        test_set = NormalizationFunctions.normalize_data_set(test_set, vector_scaler, scalar_scaler)
        
        # convert data sets to tensors
        training_tensors = TensorConversionFunctions.convert_to_tensors(training_set)
        validation_tensors = TensorConversionFunctions.convert_to_tensors(validation_set)
        test_tensors = TensorConversionFunctions.convert_to_tensors(test_set)

        # create data loaders
        training_data_loader = self.__get_data_loader(training_tensors)
        validation_data_loader = self.__get_data_loader(validation_tensors)

        # create feedforward NN model
        model = FeedforwardNN(
            input_neuron_num = self.data_set_config.input_dimension, 
            h1_neuron_num = 70, 
            output_neuron_num = 1
        )

        loss_func = nn.MSELoss()
        loss_optimization_func = optim.Adam(model.parameters(), lr = self.training_config.learning_rate)

        model = self.__train(model, loss_func, loss_optimization_func, training_data_loader, validation_data_loader)

        # evaluate results
        prediction_tensors :DataSetTensors = self.__test(model, test_tensors)
        prediction_set :DataSet = TensorConversionFunctions.convert_to_data_set(prediction_tensors)

        prediction_set :DataSet = NormalizationFunctions.denormalize_data_set(prediction_set, vector_scaler, scalar_scaler)
        test_set :DataSet = NormalizationFunctions.denormalize_data_set(test_set, vector_scaler, scalar_scaler)

        abs_err_set :np.ndarray = np.abs(prediction_set.scalars - test_set.scalars)
        return self._calculate_statistics(abs_err_set)


    def __train(self, 
                model :FeedforwardNN, 
                loss_func, 
                loss_optimization_func, 
                training_data_loader, 
                validation_data_loader) -> None:
        
        best_validation_loss = float("inf")
        best_model_state = None
        patience_tries = 0

        for epoch in range(self.training_config.epoch_limit):
            epoch_training_loss :float = self.__epoch(model, loss_func, loss_optimization_func, training_data_loader)
            epoch_validation_loss :float = self.__validate(model, loss_func, validation_data_loader)
            
            if self.training_config.verbose and (epoch + 1) % 10 == 0:
                print(f"[{epoch + 1}] Training loss: {epoch_training_loss:.6f} | Validation loss: {epoch_validation_loss:.6f}")

            # stop condition
            if epoch_validation_loss < best_validation_loss - self.training_config.delta:
                best_validation_loss = epoch_validation_loss
                best_model_state = model.state_dict()
                patience_tries = 0
            else:
                patience_tries += 1

                if self.training_config.verbose and patience_tries >= self.training_config.patience_limit:
                    print(f"Early stopping at epoch {epoch + 1} (no improvement for {self.training_config.patience_limit} epochs).")
                    break

        # load best model
        if self.training_config.verbose:
            print(f"Loaded best model with validation loss: {best_validation_loss:.6f}")

        model.load_state_dict(best_model_state)
        return model
    

    def __epoch(self, 
                model :FeedforwardNN, 
                loss_func, 
                loss_optimization_func, 
                data_loader) -> float:
        
        model.train()
        avg_loss = 0.0

        for batch_vectors, batch_scalars in data_loader:
            # pass forward
            predictions = model(batch_vectors)
            loss = loss_func(predictions, batch_scalars)
            
            # back-propagation
            loss_optimization_func.zero_grad()
            loss.backward()
            loss_optimization_func.step()
            
            avg_loss += loss.item() * batch_vectors.size(0)

        avg_loss /= len(data_loader.dataset)
        return avg_loss


    # TODO: resolve with model mutability
    def __validate(self, 
                   model :FeedforwardNN, 
                   loss_func, 
                   data_loader) -> float:

        model.eval()
        avg_loss = 0.0

        with torch.no_grad():
            for batch_vectors, batch_scalars in data_loader:
                predictions = model(batch_vectors)

                loss = loss_func(predictions, batch_scalars)
                avg_loss += loss.item() * batch_vectors.size(0)

        avg_loss /= len(data_loader.dataset)
        return avg_loss


    def __test(self, 
               model :FeedforwardNN, 
               test_tensors :DataSetTensors) -> DataSetTensors:
        
        model.eval()

        with torch.no_grad():
            prediction_scalars_tensor = model(test_tensors.vector_tensor)

        prediction_tensors :DataSetTensors = DataSetTensors(test_tensors.vector_tensor, prediction_scalars_tensor)
        return prediction_tensors


    def __get_data_loader(self,
                          data_set_tensors :DataSetTensors) -> DataLoader:
        
        tensor_data_set :TensorDataset = TensorDataset(data_set_tensors.vector_tensor, data_set_tensors.scalar_tensor)
        data_loader :DataLoader  = DataLoader(tensor_data_set, batch_size = self.training_config.batch_size, shuffle = True)
        
        return data_loader
