import math
import sys

import numpy as np


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


def distances_all_route_all_orders(routes, use_node_mask, orders):
    number_of_dimensions = routes.shape[2]
    number_of_nodes = routes.shape[1]
    number_of_routes = routes.shape[0]
    number_of_orders = orders.shape[0]
    distances_from_orders = [distances_all_routes_single_order(routes, use_node_mask, orders[order_id]) for order_id in
                             range(number_of_orders)]
    distances_from_orders = np.stack(distances_from_orders)
    return distances_from_orders