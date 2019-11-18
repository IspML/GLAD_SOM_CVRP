import numpy as np
import random
import os
from datetime import datetime

from src.SOM_algorithm import solve_using_som
from src.config_and_stats import config_and_stats
from src.io_helper import read_tsp

if __name__ == '__main__':
    np.seterr(all='warn')
    (orders, demands, depote, capacity, number_of_rings, number_of_dimensions, number_of_cities) = read_tsp(
        "maps/Christophides/M-n151-k12")

    output_path = f"results/{str(datetime.now()).replace(' ', '')}"
    os.mkdir(output_path)

    nr_of_tests = 3
    for i in range(nr_of_tests):
        with open(f"{output_path}/{i}th_test", "w+") as output_file:
            number_of_neurons_per_ring = int(orders.shape[0] * 4 / number_of_rings)

            config_and_logger = config_and_stats(number_of_neuorons=number_of_neurons_per_ring,
                                                 orders=orders,
                                                 decay_rate=0.5,
                                                 mi=0.7 * random.uniform(0.5, 2),
                                                 lambda_val=0.03 * random.uniform(0.5, 2),
                                                 v=0.05,
                                                 learninig_rate=random.uniform(0.3, 1),
                                                 learning_rate_decay=0,
                                                 blocking_frequency=10,
                                                 blocking_period=5,
                                                 begin_blocking=100,
                                                 plotting_frequency=10,
                                                 expected_penalty_ratio=random.uniform(0.5, 20))
            output_file.write(
                "mi : " + str(config_and_logger.mi) + "\n" +
                "lambda_val : " + str(config_and_logger.lambda_val) + "\n" +
                "learning_rate : " + str(config_and_logger.learning_rate) + "\n" +
                "blocking_period : " + str(config_and_logger.blocking_period) + "\n" +
                "blocking_frequency : " + str(config_and_logger.blocking_frequency) + "\n" +
                "expected_penalty_ratio : " + str(config_and_logger.expected_penalty_ratio) + "\n\n"
            )

            final_solution, route_loads, length = solve_using_som(orders, demands, depote, number_of_rings,
                                                                  capacity, config_and_logger, 130)

            overload = max(route_loads) - capacity
            print(f" \n\nfinal length {length}\n\nfinal loads {route_loads}\n\noverload {overload}\n\n")
            output_file.write("final length : " + str(length) + "\n" +
                              "overload : " + str(overload) + "\n" +
                              "solution : \n" + str(final_solution))
