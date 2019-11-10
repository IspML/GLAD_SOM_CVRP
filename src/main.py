import numpy as np
import random
# import pandas as pd
from src.config_and_stats import config_and_stats
from src.io_helper import read_tsp
from src.som_intermediate_solution import som_intermediate_solution
from src.visualisation import plot_network


def one_some_epoch(som_solution: som_intermediate_solution, orders: np.ndarray,
                   config_and_logger: config_and_stats):
    number_of_orders = orders.shape[0]
    np.random.shuffle(orders)
    for order_id in range(number_of_orders):
        som_solution.present_order_to_solution(orders[order_id], config_and_logger)
        # plot_network(orders, solution.roads)

    config_and_logger.next_iteration()


if __name__ == '__main__':
    number_of_neurons_per_ring = 50
    config_and_logger = config_and_stats(number_of_neuorons=number_of_neurons_per_ring, alfa=0.01, mi=0.9,
                                         lambda_val=0.03)
    solution = som_intermediate_solution()
    solution.init_with_petaloid(number_of_petals=3, number_of_nodes_per_petal=number_of_neurons_per_ring, scale=3,
                                number_of_dimensions=2)
    (tsp_map, number_of_dimensions, number_of_cities) = read_tsp("maps/my_test.tsp")

    # plot_network(tsp_map, solution.roads)
    for i in range(1, 300):
        one_some_epoch(solution, tsp_map, config_and_logger)
        print(config_and_logger.__str__())
        if i % 10 == 1:
            plot_network(tsp_map, solution.roads)
    # solution.present_order_to_solution(tsp_map[3], config_and_logger)
    plot_network(tsp_map, solution.roads)
