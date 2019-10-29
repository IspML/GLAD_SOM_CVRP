import numpy as np


class som_intermediate_solution:
    def __init__(self):
        self.number_of_petals = 0
        self.number_of_nodes_per_petal = 0
        self.number_of_dimensions = 2
        self.roads = np.ndarray((self.number_of_petals, self.number_of_nodes_per_petal, self.number_of_dimensions))

    def init_with_petaloid(self, number_of_petals, number_of_nodes_per_petal, number_of_dimensions):
        self.number_of_petals = number_of_petals
        self.number_of_dimensions = number_of_dimensions
        self.number_of_nodes_per_petal = number_of_nodes_per_petal
        self.roads = np.ndarray((self.number_of_petals, self.number_of_nodes_per_petal, self.number_of_dimensions))
        # TODO: Init nodes on petals

        # Last iteration node was blocked
        self.last_iteration_blocked = np.ndarray((self.number_of_petals, self.number_of_nodes_per_petal))
        self.route_remaining_capacity = np.ndarray(self.number_of_petals)
        self.last_iteration_chosen = np.ndarray((self.number_of_petals, self.number_of_nodes_per_petal))

    def present_order_to_solution(self,order,logger):
        pass
    # Returns solution as list of lists of orders
    def present_depote_to_solution(self, depote,logger):
        pass

    def to_vrp_solution(self):
        pass