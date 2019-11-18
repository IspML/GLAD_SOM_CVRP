# import pandas as pd
import numpy as np


def read_tsp(filename):
    with open(filename) as f:
        node_coord_start = None
        number_of_cities = None
        number_of_dimensions = None
        lines = f.readlines()
        capacity = None
        number_of_vehicles = None

        # Obtain the information about the .tsp
        i = 0
        while not number_of_vehicles or not capacity \
                or not number_of_cities or not node_coord_start \
                or not number_of_dimensions:
            line = lines[i]
            if line.startswith('CAPACITY :'):
                capacity = int(line.split()[-1])
            if line.startswith('VEHICLES :'):
                number_of_vehicles = int(line.split()[-1])
            if line.startswith('NODES :'):
                number_of_cities = int(line.split()[-1])
            if line.startswith('NODE_COORD_SECTION'):
                node_coord_start = i
            if line.startswith('DIMENSIONS :'):
                number_of_dimensions = int(line.split()[-1])
            i = i + 1

        orders = np.ndarray(shape=(number_of_cities - 1, number_of_dimensions))
        print('Problem with {} cities read.'.format(number_of_cities))

        order_id = 0
        while order_id < number_of_cities:
            line = lines[i + order_id]
            order = np.ndarray((number_of_dimensions), float)
            for dim_id in range(number_of_dimensions):
                order[dim_id] = float(line.split(' ')[dim_id + 1])
            if order_id == 0:
                depote = order
            else:
                orders[order_id - 1] = order
            order_id = order_id + 1

        demands = np.ndarray((orders.shape[0]), dtype=int)
        for j in range(number_of_cities - 1):
            line = lines[i + order_id + j + 1]
            dim_id = int(line.split(' ')[0])
            demand = int(line.split(' ')[1])
            demands[dim_id - 1] = demand

        depote -= np.min(orders)
        orders -= np.min(orders)
        scale = np.max(orders)
        depote /= scale
        orders /= np.max(scale)

        return (orders, demands, depote, capacity,number_of_vehicles, number_of_dimensions, number_of_cities)
