import networkx as nx
import numpy as numpy
from mpi4py import MPI

# @param adj_local:       The sub-matrix (numpy ndarray) that each processor has
# @param n:               The original matrix size, or total number of nodes
# @param rows_per_proc:   A list contains the numbers of rows each processor has
# @param comm:            The mpi4py communication protocol
# @param p:               The number of processors being used
# @param a:               The adjacency matrix made out of the data

def adjmatrixtoprocessor(n, p):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    # p=8
    # n= 4038
    # If rank 0 read data and distribute submatrix to other processors
    if rank == 0:
        mod = n % p
        # Rows per processor
        rows_per_proc = [int(n / p) for i in range(p)]
        for rem in range(mod):
            rows_per_proc[rem] += 1
        # Getting info out of txt file using edgelist function
        g = nx.read_edgelist("facebook_combined.txt", create_using=nx.DiGraph(), nodetype=int)
        # Turn directed graph to undirected
        h = g.to_directed()
        # Making adjacency matrix
        a = nx.to_numpy_array(h, nonedge=float('inf'))
        # Making self node values = 0
        a[numpy.diag_indices_from(a)] = 0
        for i in range(p):
            splitmatrix = numpy.array_split(a, size, axis=0)

    else:
        splitmatrix = None

    # Scatter data to processors
    adj_local = comm.scatter(splitmatrix, root=0)
    # Print what each processors data is
    print('Processor {} has data:'.format(rank), adj_local)


adjmatrixtoprocessor(4038, 8)
