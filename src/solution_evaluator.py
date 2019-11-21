# Should evaluate solution's cost
def evaluate_solution(solution, path_to_distance_matrix, depote_id):
    distance_matrix = []
    with open(path_to_distance_matrix) as file:
        lines = file.readlines()
        for line in lines:
            distance_matrix.append(map(float, line.split(" ")))

    def distance_between_two_orders(a, b):
        if a == -1:
            a = depote_id
        if b == -1:
            b = depote_id
        return distance_matrix[a - 1][b - 1]

    sum_of_distances = 0
    for route in solution:
        for order_id in range(len(route) - 1):
            sum_of_distances += distance_between_two_orders(route[order_id], route[order_id + 1])

    return sum_of_distances
