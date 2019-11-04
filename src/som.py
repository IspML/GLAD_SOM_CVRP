import numpy as np
import random
# import pandas as pd
from src.som_intermediate_solution import som_intermediate_solution


def one_som_epoch(problem, som_solution, config_and_logger):
    orders = problem.orders
    depotes = problem.depotes

    random.shuffle(orders)
    for order in orders:
        som_solution.present_order_to_solution(order, config_and_logger)
    for depote in depotes:
        som_solution.present_depote_to_solution(depote, config_and_logger)
        config_and_logger.log_solution(som_solution)


def solve_som(problem, config_and_logger):
    calculated_vrp_solutions = []
    capacity_penaly_const = config_and_logger.capacity_penalty

    solution = som_intermediate_solution()
    solution.init_with_petaloid(problem.number_of_routes, problem, config_and_logger)
    orders = problem.orders
    depotes = problem.depotes
    for iteration in range(1, config_and_logger.number_of_iterations):
        one_som_epoch(problem, solution, config_and_logger)

    return solution
