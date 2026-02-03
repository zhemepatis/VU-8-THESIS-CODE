from experiment_runners.base_runner import BaseRunner

class FeedforwardNNRunner(BaseRunner):
    def __init__(self, experiment_config, data_set_config, feedforward_nn_config, model):
        super().__init__(experiment_config, data_set_config)
        self.feedforward_nn_configuration = feedforward_nn_config
        
        # TODO: perhaps move
        self.model = model

    #  TODO:
    def _run_experiment(self):
        pass