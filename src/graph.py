import numpy as np
import json
import networkx
import pandas as pd
from src.distance_calculators import euclidean_distance

def create_graph(nodes, edges):
    g = networkx.Graph()
    g.add_nodes_from(nodes)
    g.add_weighted_edges_from(edges)
    return g

def graph_distance_matrix(graph, nodes):
    dist_matrix = np.zeros((len(nodes), len(nodes)))
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            dist_matrix[i, j] = networkx.shortest_path_length(graph, source=nodes[i], target=nodes[j],
                                                              weight=None)
    return dist_matrix


# def eucl_distance_matrix(coords):
#     dist_matrix = np.zeros((len(coords), len(coords)))
#     for i in range(len(coords)):
#         for j in range(len(coords)):
#             dist_matrix[i, j] = euclidean_distance(coords[i][0], coords[j][0], coords[i][1], coords[j][1])
#     return dist_matrix


class JSONGraphConstructor:
    def __init__(self, list_of_nodes, nodes_file, edge_file):
        """
        initialize graph
        """
        assert len(list_of_nodes) > 0
        assert nodes_file is not None and nodes_file != ''
        assert edge_file is not None and edge_file != ''

        self.list_nodes = list_of_nodes
        self.nodes, self.edges = self.get_graph_data(nodes_file, edge_file)
        self.graph = create_graph(self.nodes, self.edges)
        self.adj_matrix = networkx.adjacency_matrix(self.graph)
        self.dist_matrix = graph_distance_matrix(self.graph, self.nodes)

    def get_graph_data(self, nodes_file, edge_file):
        """
        Wczytuje 2 pliki json
        wybiera wierzchołki z listy oraz krawędzie które łączą 2 wierzchołki z listy
        waga krawędzi - odległość Euklidesowa
        :param nodes_file:
        :param edge_file:
        :return:
        """
        with open(nodes_file) as file:
            all_nodes = json.load(file)

        selected_nodes = {k: all_nodes[k] for k in self.list_nodes if k in all_nodes}
        distance_matrix_tmp = np.zeros((len(selected_nodes), len(selected_nodes)))  # temporary distance matrix
        i = 0
        for d1, v1 in selected_nodes:
            j = 0
            for d2, v2 in selected_nodes:
                distance_matrix_tmp[i, j] = euclidean_distance(float(v1['lat']), float(v2['lat']), float(v1['lon']),
                                                               float(v2['lon']))
                j += 1
            i += 1

        list_selected_nodes = [k for k in selected_nodes]
        nodes, edges = [], []
        for k, v in selected_nodes.items():
            nodes.append(k)
            edges.extend(v['ways'])

        with open(edge_file) as file:
            all_edges = json.load(file)
            selected_edges = {k: all_edges[k] for k in edges if k in all_edges}

        for k, v in selected_edges.items():  # usuwamy krawędzie które łączą się z tylko 1 wierzchołkiem
            for el in selected_edges[k]:
                if el not in nodes:
                    print('Error')
                    del selected_edges[k]
                    break
        edges_list = [tuple(selected_edges[k]) for k in selected_edges]
        distances = []
        for el in edges_list:
            i1 = list_selected_nodes.index(el[0])
            i2 = list_selected_nodes.index(el[1])
            distances.append(distances[i1, i2])
        edges_list = [k + (d,) for k, d in zip(edges_list, distances)]

        return list_selected_nodes, edges_list

    def get_graph(self):
        return self.graph

    def get_adj_dist_matrix(self):
        return self.adj_matrix, self.dist_matrix


class ChristophidesGraphConstructor:
    def __init__(self, list_of_nodes, nodes_file):
        """
        initialize graph
        """
        assert len(list_of_nodes) > 0
        assert nodes_file is not None and nodes_file != ''

        self.list_nodes = list_of_nodes
        self.nodes, self.edges,  self.dist_matrix = self.get_graph_data(nodes_file)
        self.graph = create_graph(self.nodes, self.edges)
        self.adj_matrix = networkx.adjacency_matrix(self.graph)

    def get_graph_data(self, nodes_file):
        data = pd.read_csv(nodes_file, header=None)
        nodes = data[0]
        nodes = nodes.to_list()
        dist_matrix = np.zeros((len(data), len(data)))
        edges = []
        for i in range(len(data)):
            for j in range(len(data)):
                dist_matrix[i, j] = np.linalg.norm((data.values[i, 1:] - data.values[j, 1:]), ord=2)
                edges.append((i, j, dist_matrix[i, j]))
        return nodes, edges, dist_matrix

    def get_graph(self):
        return self.graph

    def get_adj_dist_matrix(self):
        return self.adj_matrix, self.dist_matrix