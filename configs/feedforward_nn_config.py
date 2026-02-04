class FeedforwardNNConfig:
    def __init__(self, 
                 h1_neuron_num :int, 
                 output_neuron_num :int, 
                 learning_rate :float, 
                 batch_size :int, 
                 delta :float, 
                 epoch_limit :int, 
                 patience_limit :int) -> None:
        
        # architecture configuration
        self.h1_neuron_num :int = h1_neuron_num
        self.output_neuron_num :int = output_neuron_num
        self.learning_rate :float = learning_rate
        
        # training configuration
        self.batch_size :int = batch_size
        self.delta :float = delta
        self.epoch_limit :int = epoch_limit
        self.patience_limit :int = patience_limit