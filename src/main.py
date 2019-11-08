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
    for order_id in range(number_of_orders):
        som_solution.present_order_to_solution(orders[order_id], config_and_logger)
        if order_id%10==1:
            plot_network(orders, solution.roads)


if __name__ == '__main__':
    # for i in range(100):
    #     print(f"{i+1} {(random.random()-0.5)*10} {(random.random()-0.5)*10}")
    config_and_logger = config_and_stats("todo")
    solution = som_intermediate_solution()
    solution.init_with_petaloid(number_of_petals=3, number_of_nodes_per_petal=20, scale=3, number_of_dimensions=2)
    (tsp_map, number_of_dimensions, number_of_cities) = read_tsp("maps/my_test.tsp")

    # plot_network(tsp_map, solution.roads)
    one_some_epoch(solution, tsp_map, config_and_logger)
    # solution.present_order_to_solution(tsp_map[3], config_and_logger)
    plot_network(tsp_map, solution.roads)
