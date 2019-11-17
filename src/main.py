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
    # Randomize order in which orders are presented to solution
    orders_id = []
    for i in range(number_of_orders):
        orders_id.append(i)
    random.shuffle(orders_id)
    # Present each order to solution in a randomized way
    for order_id in orders_id:
        som_solution.present_order_to_solution(order_id, config_and_logger)

    # Block overcapacitated routes
    if config_and_logger.current_iteration() % config_and_logger.blocking_frequency == 0 and \
            config_and_logger.current_iteration() > 30:
        solution.block_overcapacitated_routes_force_solution(config_and_logger.blocking_period, config_and_logger)

    # solution.straighten_routes(config_and_logger)

    config_and_logger.next_iteration()


if __name__ == '__main__':
    np.seterr(all='warn')
    images_for_gif = []
    number_of_neurons_per_ring = 90
    number_of_rings = 3

    (orders, demands, number_of_dimensions, number_of_cities) = read_tsp("maps/my_test.tsp")
    # Regularizing
    orders /= 5

    #
    routes_capacity = np.ones((number_of_rings)) * int((orders.shape[0] / number_of_rings + 2))

    config_and_logger = config_and_stats(number_of_neuorons=number_of_neurons_per_ring,
                                         orders=orders,
                                         decay_rate=0.03,
                                         mi=0.7,
                                         lambda_val=0.03,
                                         v=0.05,
                                         learning_rate_decay=0.01,
                                         blocking_frequency=1,
                                         blocking_period=2)

    solution = som_intermediate_solution()

    solution.init_with_petaloid(number_of_petals=number_of_rings,
                                number_of_nodes_per_petal=number_of_neurons_per_ring,
                                scale=0.5,
                                number_of_dimensions=2)

    solution.set_routes_capacity(routes_capacity)

    solution.set_orders(orders, demands)

    for i in range(250):
        one_some_epoch(solution, orders, config_and_logger)
        if i % 10 == 0:
            plot_network(orders, solution.routes, images_for_gif=images_for_gif)

        print(
            f"iteration {i} mse {config_and_logger.MSE()} routes routes "+
            f"remaining capacity {solution.route_remaining_capacity}")
    final_solution,length = solution.to_vrp_solution()
    print(f"final length {length}")