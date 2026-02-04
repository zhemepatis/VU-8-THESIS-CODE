import numpy as np
from configs.data_set_config import DataSetConfig
from configs.experiment_config import ExperimentConfig
from configs.feedforward_nn_config import FeedforwardNNConfig
from configs.noise_config import NoiseConfig
from experiment_runners.base_runner import BaseRunner
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models.experiment_statistics import ExperimentStatistics
from neural_network_models.feedforward_nn import FeedforwardNN

class FeedforwardNNRunner(BaseRunner):
    def __init__(self, 
                 experiment_config :ExperimentConfig, 
                 data_set_config :DataSetConfig, 
                 noise_config :NoiseConfig, 
                 feedforward_nn_config :FeedforwardNNConfig) -> None:
        
        super().__init__(experiment_config, data_set_config, noise_config)
        self.feedforward_nn_configuration :FeedforwardNNConfig = feedforward_nn_config

        # normalization
        self.vector_scaler = None
        self.scalar_scaler = None

        # training
        self.model = None
        self.loss_func = None
        self.loss_optimization_func = None


    def _run_experiment(self) -> ExperimentStatistics:
        data_set = self._generate_raw_data_set()
        training_set, validation_set, test_set = self._split_data_set(data_set)
        
        self.__apply_noise(training_set, validation_set)
        self.__normalize(training_set, validation_set, test_set)
        
        # convert data sets to tensors
        training_vectors_tensor = torch.FloatTensor(training_set.vectors)
        training_scalars_tensor = torch.FloatTensor(training_set.scalars)

        validation_vectors_tensor = torch.FloatTensor(validation_set.vectors)
        validation_scalars_tensor = torch.FloatTensor(validation_set.scalars)

        test_vectors_tensor = torch.FloatTensor(test_set.vectors)
        test_scalars_tensor = torch.FloatTensor(test_set.scalars)

        # create data loaders
        training_data_loader = self.__get_data_loader(training_vectors_tensor, training_scalars_tensor)
        validation_data_loader = self.__get_data_loader(validation_vectors_tensor, validation_scalars_tensor)

        # create model
        self.model = FeedforwardNN(
            input_neuron_num = self.data_set_config.input_dimension, 
            h1_neuron_num = self.feedforward_nn_configuration.h1_neuron_num, 
            output_neuron_num = self.feedforward_nn_configuration.output_neuron_num
        )
        
        self.loss_func = nn.MSELoss()
        self.loss_optimization_func = optim.Adam(self.model.parameters(), lr = 0.01)

        self.__train(training_data_loader, validation_data_loader)
        predicted_scalars_tensor = self.__test(test_vectors_tensor, test_scalars_tensor)

        predicted_scalars = predicted_scalars_tensor.numpy()
        test_scalars = test_scalars_tensor.numpy()

        predicted_scalars = self.scalar_scaler.inverse_transform(predicted_scalars)
        test_scalars = self.scalar_scaler.inverse_transform(test_scalars)

        abs_err_set = np.abs(predicted_scalars - test_scalars)
        stats = self._calculate_statistics(abs_err_set)
        return stats


    def __apply_noise(self, training_set, validation_set):
        if self.noise_config != None:
            self._apply_noise(training_set)
            self._apply_noise(validation_set)


    def __init_scalers(self, data_set):
        self.vector_scaler = MinMaxScaler().fit(data_set.vectors)
        self.scalar_scaler = MinMaxScaler().fit(data_set.scalars.reshape(-1, 1))


    def __normalize(self, training_set, validation_set, test_set):
        self.__init_scalers(training_set)
        
        training_set.vectors = self.vector_scaler.transform(training_set.vectors)
        validation_set.vectors = self.vector_scaler.transform(validation_set.vectors)
        test_set.vectors = self.vector_scaler.transform(test_set.vectors)

        training_set.scalars = self.scalar_scaler.transform(training_set.scalars.reshape(-1, 1))
        validation_set.scalars = self.scalar_scaler.transform(validation_set.scalars.reshape(-1, 1))
        test_set.scalars = self.scalar_scaler.transform(test_set.scalars.reshape(-1, 1))


    def __get_data_loader(self, vectors_tensor, scalars_tensor):
        tensor_data_set = TensorDataset(vectors_tensor, scalars_tensor)
        data_loader = DataLoader(tensor_data_set, batch_size = self.feedforward_nn_configuration.batch_size, shuffle = True)
        
        return data_loader


    def __train(self, training_data_loader, validation_data_loader):
        training_losses = []
        validation_losses = []

        best_validation_loss = float("inf")
        best_model_state = None

        patience_tries = 0

        for epoch in range(self.feedforward_nn_configuration.epoch_limit):
            # training step
            self.model.train()
            epoch_training_loss = 0.0

            for batch_vectors, batch_scalars in training_data_loader:
                # pass forward
                predictions = self.model(batch_vectors)
                loss = self.loss_func(predictions, batch_scalars)
                
                # back-propagation
                self.loss_optimization_func.zero_grad()
                loss.backward()
                self.loss_optimization_func.step()
                
                epoch_training_loss += loss.item() * batch_vectors.size(0)

            # average batch loss
            epoch_training_loss /= len(training_data_loader.dataset)
            training_losses.append(epoch_training_loss)

            # validation step
            self.model.eval()
            epoch_validation_loss = 0.0
            
            with torch.no_grad():
                for batch_vectors, batch_scalars in validation_data_loader:
                    predictions = self.model(batch_vectors)
                    loss = self.loss_func(predictions, batch_scalars)
                    epoch_validation_loss += loss.item() * batch_vectors.size(0)

            epoch_validation_loss /= len(validation_data_loader.dataset)
            validation_losses.append(epoch_validation_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"[{epoch + 1}] Training loss: {epoch_training_loss:.6f} | Validation loss: {epoch_validation_loss:.6f}")

            # stop condition
            if epoch_validation_loss < best_validation_loss - self.feedforward_nn_configuration.delta:
                best_validation_loss = epoch_validation_loss
                best_model_state = self.model.state_dict()
                patience_tries = 0
            else:
                patience_tries += 1

                if patience_tries >= self.feedforward_nn_configuration.patience_limit:
                    print(f"Early stopping at epoch {epoch + 1} (no improvement for {self.feedforward_nn_configuration.patience_limit} epochs).")
                    break

        # load best model
        self.model.load_state_dict(best_model_state)
        print(f"Loaded best model with validation loss: {best_validation_loss:.6f}")


    def __test(self, test_vectors_tensor, test_scalars_tensor):
        self.model.eval()

        # get predictions
        with torch.no_grad():
            test_set_predictions = self.model(test_vectors_tensor)
            self.loss_func(test_set_predictions, test_scalars_tensor)

        return test_set_predictions

