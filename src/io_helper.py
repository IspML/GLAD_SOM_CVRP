# import pandas as pd
import numpy as np


def read_tsp(filename):
    with open(filename) as f:
        node_coord_start = None
        number_of_cities = None
        number_of_dimensions = None
        lines = f.readlines()

        # Obtain the information about the .tsp
        i = 0
        while not number_of_cities or not node_coord_start or not number_of_dimensions:
            line = lines[i]
            if line.startswith('NODES :'):
                number_of_cities = int(line.split()[-1])
            if line.startswith('NODE_COORD_SECTION'):
                node_coord_start = i
            if line.startswith('DIMENSIONS :'):
                number_of_dimensions = int(line.split()[-1])
            i = i + 1

        orders = np.ndarray(shape=(number_of_cities, number_of_dimensions))
        print('Problem with {} cities read.'.format(number_of_cities))

        order_id = 0
        while order_id < number_of_cities:
            line = lines[i + order_id]
            for id in range(number_of_dimensions):
                orders[order_id, id] = float(line.split(' ')[id + 1])
            order_id = order_id + 1

        demands = np.ones(orders.shape[0], dtype=int)
        return (orders, demands, number_of_dimensions, number_of_cities)
