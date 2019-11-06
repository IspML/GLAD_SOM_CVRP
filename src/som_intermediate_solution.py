import sys

import numpy as np
import math


# Returns [node_id] = dist to order
def distances_single_route_single_order(routes, which_route, use_node_mask, order):
    number_of_dimensions = routes.shape[2]
    number_of_nodes = routes.shape[1]
    calculated_distances = np.ndarray(shape=(number_of_nodes))
    for node_id in range(number_of_nodes):
        dist = 0
        if use_node_mask[which_route, node_id] == 1:
            for dim_id in range(number_of_dimensions):
                dist = dist + (order[dim_id] - routes[which_route, node_id, dim_id]) ** 2
            dist = math.sqrt(dist)
        else:
            dist = sys.float_info.max
        calculated_distances[node_id] = dist
    return calculated_distances


def distances_all_routes_single_order(routes, use_node_mask, order):
    number_of_dimensions = routes.shape[2]
    number_of_nodes = routes.shape[1]
    number_of_routes = routes.shape[0]
    calculated_distances = [distances_single_route_single_order(routes, route_id, use_node_mask, order) for route_id in
                            range(number_of_routes)]
    calculated_distances = np.stack(calculated_distances)
    return calculated_distances


# Returns [order,route_id,node_id] = dist to order
def distances_all_route_all_orders(routes, use_node_mask, orders):
    number_of_dimensions = routes.shape[2]
    number_of_nodes = routes.shape[1]
    number_of_routes = routes.shape[0]
    number_of_orders = orders.shape[0]
    distances_from_orders = [distances_all_routes_single_order(routes, use_node_mask, orders[order_id]) for order_id in
                             range(number_of_orders)]
    distances_from_orders = np.stack(distances_from_orders)
    return distances_from_orders


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
        self.route_remaining_capacity = np.ndarray(self.number_of_petals)
        self.last_iteration_chosen = np.ndarray((self.number_of_petals, self.number_of_nodes_per_petal))

    def init_with_petaloid(self, number_of_petals, number_of_nodes_per_petal, scale, number_of_dimensions):
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

    def present_order_to_solution(self, order, config_and_logger):
        use_node_mask = (config_and_logger.current_iteration() - self.last_iteration_blocked) > self.blocking_period
        # TODO: overcapacitated routes
        distances = distances_all_routes_single_order(self.roads,
                                                      use_node_mask,
                                                      order)
        print(f"Distances from order {order} to all routes {distances}")
        chosen_node_ind = np.unravel_index(np.argmin(distances), distances.shape)
        self.last_iteration_blocked[chosen_node_ind] = config_and_logger.current_iteration()
        self.move_node_to_order(chosen_node_ind[0], chosen_node_ind[1], config_and_logger)

    # Muszę najpierw policyczyć odległość dyskretną od wygranego wierzchołka
    def move_node_to_order(self, which_route, which_node, config_and_logger):
        indexes = np.arange(self.number_of_nodes_per_petal)
        d_1 = np.abs(indexes - which_node)
        d_2 = self.number_of_nodes_per_petal - d_1
        d_matrix = np.max(np.vstack((d_1, d_2)), 0)
        F = config_and_logger.learning_rate() * config_and_logger.F(d_matrix)
        change_vector_neighbour_part = (np.roll(self.roads[which_route], 1) + np.roll(self.roads[which_route], -1) -
                                        self.roads[which_route])
        change_vector_neighbour_part = change_vector_neighbour_part * config_and_logger.val_lambda()
        change_vector = (self.roads[which_route] - self.roads[which_route, which_node,:])
        change_vector = change_vector*np.transpose(np.vstack((F,F)))
        change_vector = change_vector + change_vector_neighbour_part
        self.roads[which_route] = self.roads[which_route] + change_vector

    # Returns solution as list of lists of orders
    def present_depote_to_solution(self, depote, config_and_logger):
        pass

    def to_vrp_solution(self):
        pass
