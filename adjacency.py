import networkx as nx
import numpy as numpy


def edgetoadj():

    # Getting info out of txt file using edgelist function
    g = nx.read_edgelist("facebook_combined.txt", create_using=nx.DiGraph(), nodetype=int)
    nx.info(g)
    # Number of nodes
    g.number_of_nodes()
    # Turn directed graph to undirected
    h = g.to_directed()
    # Making adjacency matrix
    a = nx.to_numpy_array(h, nonedge=float('inf'))
    # Making self node values = 0
    a[numpy.diag_indices_from(a)] = 0