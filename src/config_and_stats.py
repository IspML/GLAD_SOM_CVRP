import numpy as np


class config_and_stats:
    def __init__(self, path_to_config):
        self.nodes_per_orders = 10
        self.number_of_iterations = 100
        self.gain_parameter = 0.5
        self.number_of_neurons = 100

    def set_number_of_neurons(self, number_of_neurons):
        self.number_of_neurons = number_of_neurons

    def current_iteration(self):
        return 1

    def learning_rate(self):
        return 0.1

    def val_lambda(self):
        return 0.1

    def F(self,  discrete_distance_to_winning):
        H = self.number_of_neurons * 0.2
        G = self.gain_parameter
        m1 = -discrete_distance_to_winning * discrete_distance_to_winning / (G ** 2)
        discrete_distance_to_winning2 = np.where((discrete_distance_to_winning < H), 0, discrete_distance_to_winning)
        result = np.exp(m1, where=(discrete_distance_to_winning2 < H))
        return result

    def log_solution_quality(self):
        pass

    def log_current_iteration(self, iteration):
        pass

    def log_capacity_bias(self, capacity_bias):
        pass

    def log_change_vector(self, change_vector):
        pass
