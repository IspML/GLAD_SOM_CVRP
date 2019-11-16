import numpy as np
import src.distance_calculators


class config_and_stats:
    # alfa - decay rate
    # mi - learning rate towards chosen node
    # lambda - learning rate towards neigbours
    # v - capacity penalty weight
    def __init__(self, orders, number_of_neuorons=100, alfa=0.03, mi=0.6, lambda_val=0.1, v=0.1):
        self.orders = orders
        self.current_iteration_val = 1
        self.G = 0.2 * number_of_neuorons
        self.number_of_neurons = number_of_neuorons
        self.alfa = alfa
        self.mi = mi
        self.v = v
        # self.learning_rate = 0.5
        self.lambda_val = lambda_val
        self.closest_node_distance = np.ndarray((orders.shape[0]))
        self.sum_of_distances = 0.0
        self.sum_of_penalties = 0.0
        self.selfcalculate_penalty_weight = True

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
        return 1.0

    def get_lambda(self):
        return self.lambda_val

    def F(self, discrete_distance_to_winning):
        H = self.number_of_neurons * 0.3
        G = self.G
        m1 = -(discrete_distance_to_winning * discrete_distance_to_winning) / (G ** 2)
        result = np.exp(m1)
        result = np.where((discrete_distance_to_winning > H), 0, result)
        result = result * self.mi
        return result

    def log_solution_quality(self):
        pass

    def next_iteration(self):
        expected_ratio = 1.0
        self.current_iteration_val += 1
        self.G *= (1 - self.alfa)
        self.mi *= (1 - self.alfa)
        self.lambda_val *= (1 - self.alfa) ** 3


        if self.selfcalculate_penalty_weight:
            if self.sum_of_penalties >0:
                distance_to_penalty_ratio = self.sum_of_distances / self.sum_of_penalties
                self.v *= expected_ratio * distance_to_penalty_ratio
        else:
            self.v *= (1 - self.alfa) ** 0.7

        self.sum_of_distances = 0.0
        self.sum_of_penalties = 0.0


    def log_capacity_bias(self, capacity_bias):
        self.sum_of_penalties += float(capacity_bias)

    def log_distance_to_chosen(self, distance:float):
        self.sum_of_distances += float(distance)

    def log_change_vector(self, change_vector):
        pass
