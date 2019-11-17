import numpy as np
import src.distance_calculators


class config_and_stats:
    # alfa - decay rate
    # mi - learning rate towards chosen node
    # lambda - learning rate towards neigbours
    # v - capacity penalty weight
    def __init__(self, orders, number_of_neuorons=100, decay_rate=1, mi=0.6, mi_decay=0.05, lambda_val=0.1,
                 lambda_val_decay=0.15, v=0.1, v_decay=0.05, expected_penalty_ratio=1.0, expected_penalty_ratio_decay=0,
                 blocking_period=3, blocking_frequency=7, plotting_frequency=10, selfcalculate_v=True,
                 G_neurons_percentage=0.2, G_decay=0.05, F_neurons_percentage=0.3, learninig_rate=1,
                 learning_rate_decay=-0.01):
        self.orders = orders
        self.current_iteration_val = 1
        self.G = G_neurons_percentage * number_of_neuorons
        self.G_decay = G_decay
        self.F_neurons_percentage = F_neurons_percentage
        self.number_of_neurons = number_of_neuorons

        self.decay_rate = decay_rate
        self.learning_rate = learninig_rate
        self.learning_rate_decay = learning_rate_decay
        self.mi = mi
        self.mi_decay = mi_decay
        self.v = v
        self.v_decay = v_decay
        self.lambda_val = lambda_val
        self.lambda_val_decay = lambda_val_decay
        self.blocking_period = blocking_period
        self.blocking_frequency = blocking_frequency
        self.expected_penalty_ratio = expected_penalty_ratio
        self.expected_penalty_ratio_decay = expected_penalty_ratio_decay
        self.plotting_frequency = plotting_frequency

        self.closest_node_distance = np.ndarray((orders.shape[0]))
        self.sum_of_distances = 0.0
        self.sum_of_penalties = 0.0
        self.selfcalculate_penalty_weight = selfcalculate_v


    def __str__(self):
        return f"conf:\nalfa:{self.alfa} mi:{self.mi} learning_rate:{self.learning_rate} lambda:{self.lambda_val} G:{self.G}"

    def calculate_distances_to_routes(self, routes):
        all_trues = np.ones((routes.shape[0], routes.shape[1]), dtype=bool)
        all_distances = src.distance_calculators.distances_all_route_all_orders(routes=routes, use_node_mask=all_trues,
                                                                                orders=self.orders)
        self.closest_node_distance = all_distances.min(axis=1).min(axis=1)

    def MSE(self):
        mse = np.mean(self.closest_node_distance * self.closest_node_distance)
        return mse

    def current_iteration(self):
        return self.current_iteration_val

    def get_learning_rate(self):
        return self.learning_rate

    def get_lambda(self):
        return self.lambda_val

    def F(self, discrete_distance_to_winning):
        H = self.number_of_neurons * self.F_neurons_percentage
        G = self.G
        m1 = -(discrete_distance_to_winning * discrete_distance_to_winning) / (G ** 2)
        result = np.exp(m1)
        result = np.where((discrete_distance_to_winning > H), 0, result)
        result = result * self.mi
        return result

    def log_solution_quality(self):
        pass

    def next_iteration(self):
        expected_ratio = self.expected_penalty_ratio
        self.current_iteration_val += 1
        self.G *= (1 - self.G_decay)
        self.mi *= (1 - self.mi_decay)
        self.lambda_val *= (1 - self.lambda_val_decay)

        self.learning_rate *= (1 - self.learning_rate_decay)
        if self.selfcalculate_penalty_weight:
            if self.sum_of_penalties > 0:
                distance_to_penalty_ratio = self.sum_of_distances / self.sum_of_penalties
                self.v *= expected_ratio * distance_to_penalty_ratio
        else:
            self.v *= (1 - self.v_decay) ** 0.7
            # self.v *= (1 - self.alfa) ** 0.7

        self.sum_of_distances = 0.0
        self.sum_of_penalties = 0.0

    def log_capacity_bias(self, capacity_bias):
        self.sum_of_penalties += float(capacity_bias)

    def log_distance_to_chosen(self, distance: float):
        self.sum_of_distances += float(distance)

    def log_change_vector(self, change_vector):
        pass
