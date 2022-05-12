import networkx as nx
import numpy as numpy
from mpi4py import MPI

# @param adj_local:       The sub-matrix (numpy ndarray) that each processor has
# @param n:               The original matrix size, or total number of nodes
# @param rows_per_proc:   A list contains the numbers of rows each processor has
# @param comm:            The mpi4py communication protocol
# @param p:               The number of processors being used


def adjmatrixtoprocessor(n, p):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()
    # p=8
    # n= 4038
    # If rank 0 read data and distribute submatrix to other processors, else read submatrix
    if rank == 0:
        mod = n % p
        # Rows per processor
        rows_per_proc = [int(n / p) for i in range(p)]
        for rem in range(mod):
            rows_per_proc[rem] += 1
        print(rows_per_proc)
        # Getting info out of txt file using edgelist function
        g = nx.read_edgelist("facebook_combined.txt", create_using=nx.DiGraph(), nodetype=int)
        # Turn directed graph to undirected
        h = g.to_directed()
        # Making adjacency matrix
        a = nx.to_numpy_array(h, nonedge=float('inf'))
        # Making self node values = 0
        a[numpy.diag_indices_from(a)] = 0
        data = a
        adj_local = numpy.array_split(data, p)
        for i in range(p):
            adj_local = numpy.array_split(data, p)[i-1]
        print(adj_local)

        req = comm.Isend(adj_local, dest=1, tag=11)
        req.wait()
    elif rank == 1:
        req = comm.Irecv(source=0, tag=11)
        adj_local = req.wait()
    data = comm.scatter(adj_local, root=0)

    print('Processor {} has data:'.format(rank), data)
    #
    # start = sum(rows_per_proc[:(rank - 1)]) if rank != 0 else 0
    # # local submatrix each processor will use
    # adj_local = a[start: start + rows_per_proc[rank], :]
    # print(adj_local)


adjmatrixtoprocessor(4038, 8)
