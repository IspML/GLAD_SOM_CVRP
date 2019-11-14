import math
import sys

import numpy as np


# Potrzebuję dla każdego wierzchołka znać na której trasie jest.
def calculate_penalty_one_order_all_nodes(current_demand: int,
                                          nodes_per_ring:int,
                                          routes_max_capacity: np.ndarray,
                                          routes_current_capacity: np.ndarray):

    # nodes_penalty powinno byc (ilosć_tras,ilość wierzchołków na trasie)
    nodes_penalty = (1 + (routes_max_capacity-routes_current_capacity + current_demand) / routes_max_capacity) ** 2
    return np.transpose(np.vstack([nodes_penalty for i in range(nodes_per_ring)]))
