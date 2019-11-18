import random

import numpy as np

from src.config_and_stats import config_and_stats
from src.som_intermediate_solution import som_intermediate_solution
from src.visualisation import plot_network


def one_some_epoch(som_solution: som_intermediate_solution,
                   orders: np.ndarray,
                   depote: np.ndarray,
                   config_and_logger: config_and_stats):
    som_solution.prepare_new_epoch()
    number_of_orders = orders.shape[0]
    # Randomize order in which orders are presented to solution
    orders_id = []
    for i in range(number_of_orders):
        orders_id.append(i)
    random.shuffle(orders_id)

    # Present each order to solution in a randomized way
    for order_id in orders_id:
        som_solution.present_order_to_solution(order_id, config_and_logger)

    som_solution.present_depote_to_solution(depote, config_and_logger)

    som_solution.straighten_routes(config_and_logger)

    config_and_logger.next_iteration()


def solve_using_som(orders: np.ndarray, demands: np.ndarray, depote: np.ndarray,
                    number_of_routes: int, capacity: int,
                    config: config_and_stats, number_of_epochs):
    routes_capacity = np.ones((number_of_routes), dtype=int) * capacity
    number_of_neurons_per_ring = config.number_of_neurons

    som_solution = som_intermediate_solution()
    som_solution.init_with_petaloid(number_of_routes, number_of_neurons_per_ring, 0.3, orders.shape[1])
    som_solution.set_orders(orders, demands)
    som_solution.set_routes_capacity(routes_capacity)

    for i in range(number_of_epochs):
        if i % config.plotting_frequency == 0:
            plot_network(orders, som_solution.routes)
        print(f" iteration {i}")
        one_some_epoch(som_solution, orders, depote, config)

    final_solution, route_loads, length = som_solution.to_vrp_solution(depote, demands)
    return final_solution,route_loads, length