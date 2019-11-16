import sys

import numpy as np
import math
import src.config_and_stats

# Returns [node_id] = dist to order
from src import capacity_penalty
from src.distance_calculators import distances_all_routes_single_order
from src.distance_calculators import distances_all_route_all_orders


# Returns [order,route_id,node_id] = dist to order


# SOM składa się z epok. Każda epoka składa się z zaprezentowani każdego wierzchołka sieci,
# zaprezentowania każdego zamówienie siecie, aktualiacji wag. Na razie olewamy prezentowanie miast sieci.


# Każda droga (route) składa się z wierzchołków (node) i staramy się dopasować drogę (przesuwając wierzchołki) do zamówień (order)
class som_intermediate_solution:
    def __init__(self):
        self.blocking_period = 0
        self.number_of_petals = 0
        self.number_of_nodes_per_petal = 0
        self.number_of_dimensions = 2

        self.routes = np.ndarray(
            shape=(self.number_of_petals, self.number_of_nodes_per_petal, self.number_of_dimensions))
        self.blocked_until = np.zeros((self.number_of_petals, self.number_of_nodes_per_petal))
        self.route_remaining_capacity = []
        self.route_max_capacity = []
        self.order_to_route_assignments = []
        self.demands = []

        self.last_iteration_chosen = np.zeros((self.number_of_petals, self.number_of_nodes_per_petal))

    def set_routes_capacity(self, routes_max_capacity: np.ndarray):
        self.route_max_capacity = routes_max_capacity.copy()
        self.route_remaining_capacity = routes_max_capacity.copy()

    def set_orders(self, orders, demands):
        self.orders = orders
        self.order_to_route_assignments = np.zeros((orders.shape[0]), dtype=int) - 1
        self.demands = demands

    def init_with_petaloid(self, number_of_petals: int, number_of_nodes_per_petal: int, scale: float,
                           number_of_dimensions: int):
        self.number_of_petals = number_of_petals
        self.number_of_dimensions = number_of_dimensions
        self.number_of_nodes_per_petal = number_of_nodes_per_petal
        self.routes = np.ndarray(
            shape=(self.number_of_petals, self.number_of_nodes_per_petal, self.number_of_dimensions),
            dtype=float)
        self.routes = np.ndarray(
            shape=(self.number_of_petals, self.number_of_nodes_per_petal, self.number_of_dimensions),
            dtype=float)

        for petal in range(self.number_of_petals):
            alfa_0 = math.pi * (float(petal) / self.number_of_petals - 1.0 / (2 * self.number_of_petals))
            for i in range(self.number_of_nodes_per_petal):
                alfa_d = i * math.pi / (self.number_of_petals * self.number_of_nodes_per_petal)
                alfa = alfa_0 + alfa_d
                x = math.cos(self.number_of_petals * alfa) * math.cos(alfa) * scale
                y = math.cos(self.number_of_petals * alfa) * math.sin(alfa) * scale

                self.routes[petal, i, 0] = x
                self.routes[petal, i, 1] = y

        # Last iteration node was blocked
        self.blocked_until = np.zeros((self.number_of_petals, self.number_of_nodes_per_petal))
        self.last_iteration_chosen = np.zeros((self.number_of_petals, self.number_of_nodes_per_petal))
        self.route_remaining_capacity = np.ndarray((self.number_of_petals))

    def present_order_to_solution(self, order_id: int,
                                  config_and_logger: src.config_and_stats):
        order = self.orders[order_id]

        use_node_mask = (self.blocked_until < (config_and_logger.current_iteration())) & (
                self.last_iteration_chosen < (config_and_logger.current_iteration()))
        # Calculate distances and capacity penalties
        distances = distances_all_routes_single_order(self.routes, use_node_mask, order)

        if np.min(distances) == sys.float_info.max:
            return

        capacity_penalties = capacity_penalty.calculate_penalty_one_order_all_nodes(self.demands[order_id],
                                                                                    self.number_of_nodes_per_petal,
                                                                                    self.route_max_capacity,
                                                                                    self.route_remaining_capacity) \
                             * config_and_logger.v

        goodness_of_node = distances + capacity_penalties

        # Chose node
        chosen_node_ind = np.unravel_index(np.argmin(goodness_of_node), goodness_of_node.shape)
        # Log statistics
        config_and_logger.log_capacity_bias(capacity_penalties[chosen_node_ind])
        config_and_logger.log_distance_to_chosen(distances[chosen_node_ind])
        # Block node for the rest of the epoch and move node
        self.last_iteration_chosen[chosen_node_ind] = config_and_logger.current_iteration()
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

        change_vector_neighbour = (np.roll(self.routes[which_route], 1) + np.roll(self.routes[which_route], -1) -
                                   2 * self.routes[which_route])
        change_vector_neighbour = change_vector_neighbour * config_and_logger.get_lambda()

        change_vector_distance = (order - self.routes[which_route])
        change_vector_distance = change_vector_distance * np.transpose(np.vstack((F, F)))
        change_vector_distance = change_vector_distance

        change_vector = change_vector_distance + change_vector_neighbour

        change_vector = change_vector * config_and_logger.get_learning_rate()

        self.routes[which_route] = self.routes[which_route] + change_vector
        # if self.order_to_route_assignments[order_id] != -1:
        #     self.route_remaining_capacity[self.order_to_route_assignments[order_id]] += self.demands[order_id]
        # self.route_remaining_capacity[which_route] -= self.demands[order_id]
        self.order_to_route_assignments[order_id] = which_route

    def block_overcapacitated_routes(self, epochs_to_block, config_and_logger: src.config_and_stats):
        routes_to_block = np.argwhere(self.route_remaining_capacity < 0)
        self.blocked_until[routes_to_block, :] = config_and_logger.current_iteration() + epochs_to_block

    def block_overcapacitated_routes_force_solution(self, epochs_to_block, config_and_logger: src.config_and_stats):
        forced_solution = self.to_vrp_solution()
        for route_id in range(self.number_of_petals):
            load = 0
            for node_id in range(self.number_of_nodes_per_petal):
                load += len(forced_solution[route_id][node_id])
            if load > self.route_max_capacity[route_id]:
                self.blocked_until[route_id, :] = config_and_logger.current_iteration() + epochs_to_block
            self.route_remaining_capacity[route_id] = self.route_max_capacity[route_id] - load

    def straighten_routes(self, config: src.config_and_stats):
        straighten_node = np.argwhere(self.last_iteration_chosen < config.current_iteration() +1)
        straighten_node_next = (straighten_node+1)
        straighten_node_next[0,:]-=1
        straighten_node_prev = straighten_node-1
        straighten_node_prev[0,:]+=1


        self.routes[straighten_node] = 0.5 * (self.routes[straighten_node+1]+self.routes[straighten_node-1])
        # self.routes = np.where(np.vstack([straighten_node, straighten_node]),
        #                        0.5 * (np.roll(self.routes, axis=1, shift=1) + np.roll(self.routes, axis=1, shift=-1)),
        #                        self.routes)

        # weight = [0.7, 0.1, 0.1, 0.05, 0.05]
        # self.routes = weight[0] * self.routes + \
        #               np.roll(self.routes, axis=1, shift=1) * weight[1] + \
        #               np.roll(self.routes, axis=1, shift=-1) * weight[2] + \
        #               np.roll(self.routes, axis=1, shift=2) * weight[3] + \
        #               np.roll(self.routes, axis=1, shift=-2) * weight[4]

    def present_depote_to_solution(self, depote: np.ndarray, config_and_logger: src.config_and_stats):
        pass

    def to_vrp_solution(self):
        all_trues = np.ones((self.routes.shape[0], self.routes.shape[1]), dtype=bool)

        all_distances = distances_all_route_all_orders(self.routes, all_trues, self.orders)
        all_distances_flatten = all_distances.reshape(
            (all_distances.shape[0], all_distances.shape[1] * all_distances.shape[2]))

        closest_node_flatten_id = np.argmin(all_distances_flatten, axis=1)
        closest_node_id = closest_node_flatten_id % self.number_of_nodes_per_petal
        closest_route_id = closest_node_flatten_id / self.number_of_nodes_per_petal

        solution = [[[] for node_id in range(self.number_of_nodes_per_petal)] for route_id in
                    range(self.number_of_petals)]

        mapped_orders = {}
        for order_id in range(self.orders.shape[0]):
            route_id = int(closest_route_id[order_id])
            node_id = int(closest_node_id[order_id])
            solution[route_id][node_id].append(order_id)
        return solution
