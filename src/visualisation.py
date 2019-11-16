import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation

"""Plot a graphical representation of the problem"""


def plot_network(cities, roads, name='diagram.png',images_for_gif=[]):
    fig = plt.figure()
    ims = []
    x_1 = [cities[i, 0] for i in range(cities.shape[0])]
    x_2 = [cities[i, 1] for i in range(cities.shape[0])]
    plt.plot(x_1, x_2, 'ro', marker = 'd')
    plt.axis([-1, 1, -1, 1])
    for road_id in range(roads.shape[0]):
        x_1 = [roads[road_id, i, 0] for i in range(roads.shape[1])]
        x_2 = [roads[road_id, i, 1] for i in range(roads.shape[1])]
        plt.plot(x_1, x_2, linestyle="--",label=f"route {road_id}")
        plt.legend()
        # plt.plot(x_1, x_2, linestyle="--",marker='o')
        # plt.imshow(x_1, x_2,  animated=True)
        # ims.append( plt.imshow(x_1, x_2, linestyle="--", animated=True))
    # images_for_gif.append(ims)
    plt.show()
#

def generate_gif(images_for_gif):
    # Nawet nie zaczęte
    fig = plt.figure()


    ani = animation.ArtistAnimation(fig, images_for_gif, interval=30, blit=True,
                                    repeat_delay=1000)

    # ani.save('dynamic_images.mp4')

    plt.show()