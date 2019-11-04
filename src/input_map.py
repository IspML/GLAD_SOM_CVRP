import numpy as np
import gem

class input_map:
    def __init__(self, path_to_file):
        self.number_of_dimensions = 2
        self.number_of_vehicles = 0
        self.number_of_orders = 0
        self.max_capacity = np.ndarray((self.number_of_vehicles,))
        self.orders = np.ndarray((self.number_of_orders, self.number_of_dimensions))
        self.demands = np.ndarray((self.number_of_orders,))
        self.depote = np.ndarray((self.number_of_dimensions,))

