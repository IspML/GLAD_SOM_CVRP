import networkx as nx
from node2vec import Node2Vec


def read_graph_from_adjacency_matrix(path):
    def create_graph(nodes, edges):
        g = nx.Graph()
        g.add_nodes_from(nodes)
        g.add_weighted_edges_from(edges)
        return g

    pass
    edges = []
    row = 0
    with open(path) as file:
        lines = file.readlines()
        for line in lines:
            distances = map(float, lines[row].split(" "))
            for destination, distance in enumerate(distances):
                if destination != row:
                    edges.append((row, destination, distance))
            row += 1

        nodes = range(row)
    return create_graph(nodes, edges)


if __name__ == '__main__':
    dimensions = 3
    nr_of_vehicles = 7
    path = "/home/miron/PycharmProjects/GLAD_SOM_CVRP/src/maps/quantum1-100-1/quantum1-100-1"
    graph = read_graph_from_adjacency_matrix(path)
    # FILES
    EMBEDDING_FILENAME = path + "node2vec"
    EMBEDDING_MODEL_FILENAME = './embeddings.model'

    # Create a graph

    # Precompute probabilities and generate walks
    node2vec = Node2Vec(graph, dimensions=dimensions, walk_length=1, num_walks=300, workers=4)

    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    # Any keywords acceptable by gensim.Word2Vec can be passed, `diemnsions` and `workers` are automatically passed (from the Node2Vec constructor)

    # Look for most similar nodes
    model.wv.most_similar('2')  # Output node names are always strings

    # # Save embeddings for later use
    model.wv.save_word2vec_format(EMBEDDING_FILENAME + ".tmp")

    with open(EMBEDDING_FILENAME + ".tmp") as graph_file:
        with open(EMBEDDING_FILENAME, "w+") as final_graph:
            final_graph.write("NAME : node2vec\n")
            final_graph.write("TYPE : CVRP\n")
            final_graph.write(f"NODES : {len(graph.nodes)}\n")
            final_graph.write(f"DIMENSIONS : {dimensions}\n")
            final_graph.write(f"VEHICLES : {nr_of_vehicles}\n")
            final_graph.write(f"CAPACITY : {int(len(graph.nodes) / nr_of_vehicles) + 10}\n")
            final_graph.write("NODE_COORD_SECTION\n")
            lines = graph_file.readlines()
            for line in lines:
                if len(line.split(" "))>2:
                    splitted_line = line.split(" ")
                    splitted_line[0] = int(splitted_line[0])+1
                    splitted_line = map(str,splitted_line)
                    final_graph.write(" ".join(splitted_line))
            final_graph.write("DEMAND_SECTION\n")
            for i in range(len(graph.nodes)):
                if i == 0:
                    final_graph.write("1 0\n")
                else:
                    final_graph.write(f"{i + 1} 1\n")
            final_graph.write("DEPOT_SECTION\n1\n-1\n")


# # Save model for later use
# model.save(EMBEDDING_MODEL_FILENAME)
