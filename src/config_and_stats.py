import numpy as np


class config_and_stats:
    def __init__(self, number_of_neuorons=100, alfa=0.03, mi=0.6, lambda_val=0.03):
        self.current_iteration_val = 1
        self.nodes_per_orders = 10
        self.G = 0.2 * number_of_neuorons
        self.number_of_neurons = number_of_neuorons
        self.alfa = alfa
        self.mi = mi
        self.learning_rate = 0.5
        self.lambda_val = lambda_val

    def __str__(self):
        return f"conf:\nalfa:{self.alfa} mi:{self.mi} learning_rate:{self.learning_rate} lambda:{self.lambda_val} G:{self.G}"

    def set_number_of_neurons(self, number_of_neurons):
        self.number_of_neurons = number_of_neurons

    def current_iteration(self):
        return self.current_iteration_val

    def get_learning_rate(self):
        return 1

    def get_lambda(self):
        return self.lambda_val

    def F(self, discrete_distance_to_winning):
        H = self.number_of_neurons * 0.2
        G = self.G
        m1 = -(discrete_distance_to_winning * discrete_distance_to_winning) / (G ** 2)
        result = np.exp(m1)
        result = np.where((discrete_distance_to_winning > H), 0, result)
        result = result * self.mi
        return result

    def log_solution_quality(self):
        pass

    def next_iteration(self):
        self.current_iteration_val += 1
        self.G *= (1 - self.alfa)
        self.mi *= (1 - self.alfa)
        self.learning_rate *= (1 - self.alfa) ** 2
        self.lambda_val *= (1 - self.alfa) **3

    def log_capacity_bias(self, capacity_bias):
        pass

    def log_change_vector(self, change_vector):
        pass
