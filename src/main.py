import numpy as np
import random
# import pandas as pd
from src.config_and_stats import config_and_stats
from src.io_helper import read_tsp
from src.som_intermediate_solution import som_intermediate_solution
from src.visualisation import plot_network, generate_gif


def one_some_epoch(som_solution: som_intermediate_solution, orders: np.ndarray,
                   config_and_logger: config_and_stats):
    number_of_orders = orders.shape[0]
    # np.random.shuffle(orders)
    orders_id = []
    for i in range(number_of_orders):
        orders_id.append(i)
    random.shuffle(orders_id)
    for order_id in orders_id:
        som_solution.present_order_to_solution(order_id, config_and_logger)

    config_and_logger.next_iteration()


if __name__ == '__main__':
    images_for_gif = []
    number_of_neurons_per_ring = 100
    number_of_rings = 3

    (tsp_map, demands, number_of_dimensions, number_of_cities) = read_tsp("maps/my_test.tsp")

    routes_capacity = np.ones((number_of_rings)) * (tsp_map.shape[0] / number_of_rings + 2)

    config_and_logger = config_and_stats(number_of_neuorons=number_of_neurons_per_ring,
                                         orders=tsp_map,
                                         alfa=0.01,
                                         mi=0.6,
                                         lambda_val=0.03)

    solution = som_intermediate_solution()

    solution.init_with_petaloid(number_of_petals=number_of_rings,
                                number_of_nodes_per_petal=number_of_neurons_per_ring,
                                scale=4,
                                number_of_dimensions=2)

    solution.set_routes_capacity(routes_capacity)

    solution.set_orders(tsp_map, demands)

    for i in range(400):
        one_some_epoch(solution, tsp_map, config_and_logger)
        if i % 10 == 1:
            plot_network(tsp_map, solution.roads, images_for_gif=images_for_gif)
            solution.block_overcapacitated_routes(10, config_and_logger)
        # if i%10==1 and i<200:
        #     solution.straighten_routes()
        config_and_logger.calculate_distances_to_routes(solution.roads)
        print(
            f"itaration {i} mse {config_and_logger.MSE()} routes routes remaingin capacity {solution.route_remaining_capacity} \n   last  chosen {solution.last_iteration_chosen[:,0]}")
