class config_and_stats:
    def __init__(self, path_to_config):
        self.nodes_per_orders = 10
        self.number_of_iterations = 100
        self.learning_rate = 0.5

    def current_iteration(self):
        return 1
    def log_solution_quality(self):
        pass

    def log_current_iteration(self, iteration):
        pass

    def log_capacity_bias(self, capacity_bias):
        pass

    def log_change_vector(self, change_vector):
        pass
