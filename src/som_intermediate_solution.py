import numpy as np
import math
import src.config_and_stats

# Returns [node_id] = dist to order
from src import capacity_penalty
from src.distance_calculators import distances_all_routes_single_order


# Returns [order,route_id,node_id] = dist to order


# SOM składa się z epok. Każda epoka składa się z zaprezentowani każdego wierzchołka sieci,
# zaprezentowania każdego zamówienie siecie, aktualiacji wag. Na razie olewamy prezentowanie miast sieci.
class som_intermediate_solution:
    def __init__(self):
        self.blocking_period = 0
        self.number_of_petals = 0
        self.number_of_nodes_per_petal = 0
        self.number_of_dimensions = 2
        self.roads = np.ndarray(
            shape=(self.number_of_petals, self.number_of_nodes_per_petal, self.number_of_dimensions))
        self.last_iteration_blocked = np.ndarray((self.number_of_petals, self.number_of_nodes_per_petal))
        self.route_remaining_capacity = []
        self.route_max_capacity = []
        self.order_to_route_assignments = []
        self.demands = []

        self.last_iteration_chosen = np.ndarray((self.number_of_petals, self.number_of_nodes_per_petal))

    def set_routes_capacity(self, routes_max_capacity):
        self.route_max_capacity = routes_max_capacity
        self.route_remaining_capacity = routes_max_capacity

    def set_orders(self, orders, demands):
        self.orders = orders
        self.order_to_route_assignments = np.zeros((orders.shape[0]), dtype=int) - 1
        self.demands = demands

    def init_with_petaloid(self, number_of_petals: int, number_of_nodes_per_petal: int, scale: float,
                           number_of_dimensions: int):
        self.number_of_petals = number_of_petals
        self.number_of_dimensions = number_of_dimensions
        self.number_of_nodes_per_petal = number_of_nodes_per_petal
        self.roads = np.ndarray(
            shape=(self.number_of_petals, self.number_of_nodes_per_petal, self.number_of_dimensions),
            dtype=float)
        self.roads = np.ndarray(
            shape=(self.number_of_petals, self.number_of_nodes_per_petal, self.number_of_dimensions),
            dtype=float)

        for petal in range(self.number_of_petals):
            alfa_0 = math.pi * (float(petal) / self.number_of_petals - 1.0 / (2 * self.number_of_petals))
            for i in range(self.number_of_nodes_per_petal):
                alfa_d = i * math.pi / (self.number_of_petals * self.number_of_nodes_per_petal)
                alfa = alfa_0 + alfa_d
                x = math.cos(self.number_of_petals * alfa) * math.cos(alfa) * scale
                y = math.cos(self.number_of_petals * alfa) * math.sin(alfa) * scale

                self.roads[petal, i, 0] = x
                self.roads[petal, i, 1] = y

        # Last iteration node was blocked
        self.last_iteration_blocked = np.ndarray((self.number_of_petals, self.number_of_nodes_per_petal))
        self.route_remaining_capacity = np.ndarray(self.number_of_petals)
        self.last_iteration_chosen = np.ndarray((self.number_of_petals, self.number_of_nodes_per_petal))

    def present_order_to_solution(self, order_id: int,
                                  config_and_logger: src.config_and_stats):
        order = self.orders[order_id]
        use_node_mask = (self.last_iteration_blocked) < (config_and_logger.current_iteration())
        # TODO: overcapacitated routes must be prohibited
        distances = distances_all_routes_single_order(self.roads, use_node_mask, order)

        capacity_penalties = capacity_penalty.calculate_penalty_one_order_all_nodes(self.demands[order_id],
                                                                                    self.number_of_nodes_per_petal,
                                                                                    self.route_max_capacity,
                                                                                    self.route_remaining_capacity)
        goodness_of_node = distances + capacity_penalties * config_and_logger.v

        chosen_node_ind = np.unravel_index(np.argmin(goodness_of_node), goodness_of_node.shape)
        self.last_iteration_blocked[chosen_node_ind] = config_and_logger.current_iteration()
        self.move_node_to_order(chosen_node_ind[0], chosen_node_ind[1], order_id, config_and_logger)

    # Muszę najpierw policyczyć odległość dyskretną od wygranego wierzchołka
    def move_node_to_order(self, which_route: int, which_node: int, order_id: int,
                           config_and_logger: src.config_and_stats):
        order = self.orders[order_id]
        indexes = np.arange(self.number_of_nodes_per_petal)
        d_1 = np.abs(indexes - which_node)
        d_2 = self.number_of_nodes_per_petal - d_1
        d_matrix = np.min(np.vstack((d_1, d_2)), 0)
        F = config_and_logger.F(d_matrix)

        change_vector_neighbour = (np.roll(self.roads[which_route], 1) + np.roll(self.roads[which_route], -1) -
                                   2 * self.roads[which_route])
        change_vector_neighbour = change_vector_neighbour * config_and_logger.get_lambda()

        change_vector_distance = (order - self.roads[which_route])
        change_vector_distance = change_vector_distance * np.transpose(np.vstack((F, F)))
        change_vector_distance = change_vector_distance * config_and_logger.get_learning_rate()

        change_vector = change_vector_distance + change_vector_neighbour * config_and_logger.get_learning_rate()

        self.roads[which_route] = self.roads[which_route] + change_vector
        if self.order_to_route_assignments[order_id] != -1:
            self.route_remaining_capacity[self.order_to_route_assignments[order_id]] += self.demands[order_id]
        self.route_remaining_capacity[which_route] -= self.demands[order_id]
        self.order_to_route_assignments[order_id] = which_route

    def block_overcapacitated_routes(self, epochs_to_block, config_and_logger: src.config_and_stats):
        routes_to_block = np.argwhere(self.route_remaining_capacity < 0)
        self.last_iteration_chosen[routes_to_block, :] =  config_and_logger.current_iteration() + epochs_to_block
        pass

    def straighten_routes(self):
        weight = [0.5, 0.15, 0.15, 0.1, 0.1]
        self.roads = weight[0] * self.roads + \
                     np.roll(self.roads, axis=1, shift=1) * weight[1] + \
                     np.roll(self.roads, axis=1, shift=-1) * weight[2] + \
                     np.roll(self.roads, axis=1, shift=2) * weight[3] + \
                     np.roll(self.roads, axis=1, shift=-2) * weight[4]

    def present_depote_to_solution(self, depote: np.ndarray, config_and_logger: src.config_and_stats):
        pass

    def to_vrp_solution(self):
        pass
