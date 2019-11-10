import matplotlib.pyplot as plt
import matplotlib as mpl


def plot_network(cities, roads, name='diagram.png'):
    """Plot a graphical representation of the problem"""
    x_1 = [cities[i, 0] for i in range(cities.shape[0])]
    x_2 = [cities[i, 1] for i in range(cities.shape[0])]
    plt.plot(x_1, x_2, 'ro')
    plt.axis([-5, 5, -5, 5])
    for road_id in range(roads.shape[0]):
        # road_id = 0
        x_1 = [roads[road_id, i, 0] for i in range(roads.shape[1])]
        x_2 = [roads[road_id, i, 1] for i in range(roads.shape[1])]
        plt.plot(x_1, x_2,linestyle="--",marker="o")
    #     road_id = 2
    #     x_1 = [roads[road_id, i, 0] for i in range(roads.shape[1])]
    #     x_2 = [roads[road_id, i, 1] for i in range(roads.shape[1])]
    #     plt.plot(x_1, x_2)
    #     break
    plt.show()
#