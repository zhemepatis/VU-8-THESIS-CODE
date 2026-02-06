class FeedforwardNNConfig:
    def __init__(self,
                 input_neuron_num :int,
                 h1_neuron_num :int,
                 output_neuron_num :int) -> None:
        
        self.input_neuron_num :int = input_neuron_num, 
        self.h1_neuron_num :int = h1_neuron_num, 
        self.output_neuron_num :int = output_neuron_num