import numpy as np
import random
# import pandas as pd
from src.som_intermediate_solution import som_intermediate_solution
from src.stats_logger import stats_logger

def one_some_epoch(problem, som_solution, config, logger):
    orders = problem.orders
    depotes = problem.depotes

    random.shuffle(orders)
    for order in orders:
        som_solution.present_order_to_solution(order, logger)
    for depote in depotes:
        som_solution.present_depote_to_solution(depote, logger)
    logger.log_solution(som_solution)

def solve_som(problem, config):
    logger = stats_logger()
    calculated_vrp_solutions = []
    capacity_penaly_const = config.capacity_penalty

    solution = som_intermediate_solution()
    solution.init_with_petaloid(problem.number_of_routes, problem, config)
    orders = problem.orders
    depotes = problem.depotes
    for iteration in range(1, config.number_of_iterations):
        random.shuffle(orders)
        for order in orders:
            solution.present_order_to_solution(order, logger)
        for depote in depotes:
            solution.present_depote_to_solution(depote, logger)
        calculated_vrp_solutions.append(solution.to_vrp_solution())
