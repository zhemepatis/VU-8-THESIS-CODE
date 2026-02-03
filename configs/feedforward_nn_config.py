class FeedforwardNNConfig:
    def __init__(self, h1_neuron_number, output_neuron_number, learning_rate, batch_size, delta, epoch_limit, patience_limit):
        # architecture configuration
        self.h1_neuron_number = h1_neuron_number
        self.output_neuron_number = output_neuron_number
        self.learning_rate = learning_rate
        
        # training configuration
        self.batch_size = batch_size
        self.delta = delta
        self.epoch_limit = epoch_limit
        self.patience_limit = patience_limit