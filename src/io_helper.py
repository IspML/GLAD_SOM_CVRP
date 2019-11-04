import pandas as pd
import numpy as np


def read_tsp(filename):
    """
    Read a file in .tsp format into a pandas DataFrame
    The .tsp files can be found in the TSPLIB project. Currently, the library
    only considers the possibility of a 2D map.
    """
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

        cities = np.ndarray(shape=(number_of_cities, number_of_dimensions))
        print('Problem with {} cities read.'.format(number_of_cities))

        order_id = 0
        while order_id < number_of_cities:
            line = lines[i + order_id]
            for id in range(number_of_dimensions):
                cities[order_id, id] = float(line.split(' ')[id + 1])
            order_id = order_id + 1

        # for cord_id in range(number_of_dimensions):
        #     cord_sum = cities[:, cord_id].sum()
        #     cord_sum = cord_sum / number_of_cities
        #     cities[:, cord_id] -= cord_sum

        return (cities, number_of_dimensions, number_of_cities)


def normalize(points):
    """
    Return the normalized version of a given vector of points.
    For a given array of n-dimensions, normalize each dimension by removing the
    initial offset and normalizing the points in a proportional interval: [0,1]
    on y, maintining the original ratio on x.
    """
    ratio = (points.x.max() - points.x.min()) / (points.y.max() - points.y.min()), 1
    ratio = np.array(ratio) / max(ratio)
    norm = points.apply(lambda c: (c - c.min()) / (c.max() - c.min()))
    return norm.apply(lambda p: ratio * p, axis=1)
