from queue import PriorityQueue

import networkx as nx
import matplotlib.pyplot as plt
import random as rd
import time
import pandas as pd
import numpy as np


def get_random_graph():
    num_vertices = rd.randint(5, 8)
    adj_mat = [[0 for x in range(num_vertices)] for y in range(num_vertices)]
    for i in range(num_vertices):
        for j in range(i):
            adj_mat[i][j] = rd.randint(1, 25)
            adj_mat[j][i] = adj_mat[i][j]
    return adj_mat


def print_adj_mat(adj_mat):
    print("Adjacency matrix")
    np_matrice = np.array(adj_mat)
    print(np_matrice)


def draw_graph_from_adj_mat(adj):
    G = nx.Graph()
    weight_tuples = []
    labels = {}
    for i in range(len(adj)):
        labels[i] = i
    for i in range(len(adj)):
        for j in range(i):
            if adj[i][j] != 0:
                weight_tuples.append((i, j, adj[i][j]))
    for i in range(len(adj)):
        for j in range(len(adj[i])):
            if adj[i][j] != 0:
                break
        else:
            labels.pop(i)

    G.add_weighted_edges_from(weight_tuples)
    pos = nx.spring_layout(G)
    plt.plot("-b", label="Discovered Neighbour")
    plt.plot("-r", label="Discovered Path")
    plt.legend(loc='upper right', bbox_to_anchor=(1, 0))

    nx.draw(G, pos, node_color='black', node_size=1000, with_labels=True, labels=labels,
            font_color='w',
            font_weight=800, font_size=18, width=3)
    labels_weights = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels_weights)
    fig_size = plt.gcf().get_size_inches()
    size_factor = 1.6
    plt.gcf().set_size_inches(size_factor * fig_size)
    return G, pos


def draw_graph_from_new_path(path):
    new_graph = nx.Graph()
    labels = {}
    for i in range(len(pi)):
        labels[i] = i

    for i in range(len(Adj)):
        for j in range(len(Adj[i])):
            if Adj[i][j] != 0:
                break
        else:
            labels.pop(i)
    new_graph = nx.Graph()
    root = None
    for i in range(0, len(pi)):
        if pi[i] != -1:
            new_graph.add_edge(i, pi[i], weight=dist[i])
        else:
            root = i
    position = nx.spring_layout(new_graph)
    nx.draw(new_graph, position, node_color='black', node_size=1000, with_labels=True, labels=labels,
            font_color='w',
            font_weight=800, font_size=18, edge_color='r', width=6)
    nx.draw(new_graph, position, nodelist=[root], node_color='r', node_size=1000, width=6,edge_color='r')
    labels_weights = nx.get_edge_attributes(new_graph, 'weight')
    nx.draw_networkx_edge_labels(new_graph, position, edge_labels=labels_weights)
    return new_graph


def dijkstra_on_graph(root, adj_mat):
    global pi, dist
    dist = []
    S = []
    path = []
    inf = float('inf')
    for i in range(len(adj_mat)):
        pi.append(-1)
        dist.append(inf)
        S.append(0)
    dist[root] = 0
    q = PriorityQueue()
    G, position = draw_graph_from_adj_mat(adj_mat)
    nx.draw(G, position, nodelist=[root], node_color='r', node_size=1000, width=3)
    plt.ion()
    for i in range(len(adj_mat)):
        q.put((dist[i], i))
    while not q.empty():
        u = q.get()[1]
        if S[u] == 1:
            continue
        else:
            S[u] = 1
        elist = []
        for v in range(len(adj_mat[u])):
            if adj_mat[u][v] != 0:
                elist.append((u, v))
                nx.draw_networkx_edges(G, position, edgelist=[(u, v)], edge_color='deepskyblue', width=3)
                time.sleep(0.5)
                plt.pause(0.5)
                if dist[v] > dist[u] + adj_mat[u][v]:
                    dist[v] = dist[u] + adj_mat[u][v]
                    pi[v] = u
                    q.put((dist[v], v))
        nx.draw_networkx_edges(G, position, edgelist=elist, edge_color='black', width=3)
        if pi[u] != -1:
            path.append((u, pi[u]))
        nx.draw_networkx_edges(G, position, edgelist=path, edge_color='r', width=6)
        plt.plot()
        plt.pause(0.1)
    time.sleep(3)
    plt.close()


def print_table_pi_costs():
    dict = {'Vertex': [i for i in range(len(Adj))], 'Cost': [dist[i] for i in range(len(dist))],
            'PI': [pi[i] for i in range(len(pi))]}
    vertices_caracts = pd.DataFrame(dict)
    print(vertices_caracts)
    return vertices_caracts


def run():
    global pi, dist, Adj
    root = -1
    print("!!!DIJKSTRA VISUALIZATION MADE BY JASON ABI SAAD!!!")
    graph_input = input("Would you like to input a graph or generate a random one?(Y|N): ")
    if graph_input == 'Y' or graph_input == 'y':
        num_vertices = int(input("Input number of vertices: "))
        Adj = [[0 for i in range(num_vertices)] for i in range(num_vertices)]
        input_vertices = 'True'
        while input_vertices:
            u, v, w = -1, -1, -1
            while u < 0 or u > len(Adj) - 1:
                u = int(input('Input u: '))
                if 0 <= u < len(Adj):
                    break
                print("u must be between 0 and %d" % (len(Adj) - 1))

            while v < 0 or v > len(Adj) - 1:
                v = int(input('Input v: '))
                if 0 <= v < len(Adj):
                    break
                print("v must be between 0 and %d" % (len(Adj) - 1))

            while w < 0:
                w = int(input('Input w: '))
                if w >= 0:
                    break
                print("Dijkstra accepts only positive weights")

            Adj[u][v] = w
            Adj[v][u] = w
            c = input('Would you like to enter more: (Y|N)')
            if c == 'N' or c == 'n':
                input_vertices = False
    else:
        time.sleep(1)
        Adj = get_random_graph()
    print_adj_mat(Adj)
    while root < 0 or root > len(Adj) - 1:
        root = int(input("Input root: "))
    dijkstra_on_graph(root, Adj)
    draw_graph_from_new_path(pi)
    print_table_pi_costs()
    plt.ioff()
    plt.show()


pi = []
dist = []
Adj = []
run()