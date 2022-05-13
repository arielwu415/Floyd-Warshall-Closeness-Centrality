# -*- coding: utf-8 -*-
"""
Created on Fri May 13 10:37:26 2022

@author: Ariel
"""
from mpi4py import MPI
import networkx as nx
import numpy as np
import os

# @dict:        Dictionary containing centrality scores for network nodes
def produce_report(dict):
    report = ''
    for key, value in dict.items():
        report += f"{key}:\t{value}\n"

    with open('output.txt', 'w', encoding="utf-8") as file:
        file.write(report)

    top_five_list = list()
    try:
        top_five_list = sorted(dict, key=dict.get, reverse=True)[:5]
    except:
        print('oops, something went wrong!')

    print("Top five nodes: ", end='')
    total = 0
    for i, item in enumerate(top_five_list):
        if i == len(top_five_list) - 1:
            total += dict[item]
            print(f'{item}', end='')
        else:
            total += dict[item]
            print(f'{item}, ', end='')

    print(f'\nAverage of top five is: {total / len(top_five_list)}')


# Communication Creation
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# p = 8
n = 4038
# If rank 0 read data and distribute submatrix to other processors
if rank == 0:

    # Rows per processor
    rows_per_proc = [int(n / size)+1 if i < n %
                     size else int(n / size) for i in range(size)]

    # Getting info out of txt file using edgelist function
    G = nx.read_edgelist("facebook_combined.txt",
                         create_using=nx.DiGraph(), nodetype=int)
    # Turn directed graph to undirected
    h = G.to_directed()

    # Making adjacency matrix
    A = nx.to_numpy_array(h, nonedge=float('inf'))
    # Making self node values = 0
    A[np.diag_indices_from(A)] = 0

    for i in range(size):
        splitmatrix = np.array_split(A, size, axis=0)

else:
    splitmatrix = None
    rows_per_proc = None


# Scatter and broadcast data to processors
adj_local = comm.scatter(splitmatrix, root=0)
rows_per_proc = comm.bcast(rows_per_proc, root=0)


# Print what each processors data is
print('Processor {} has data:'.format(rank), adj_local)


# Floyd-Warshall Algorithm
for t in range(0, size):
    print("Task:{}; Processor {}\n".format(t, rank))
    adj_rproc = adj_local
    rproc = (rank - t + size) % size

    if t != 0:
        sproc = (rank + t) % size

        adj_rproc = np.empty(shape=(rows_per_proc[rproc], n), dtype='i')

        sreq = comm.Isend([adj_local, MPI.INT], dest=sproc, tag=sproc)
        rreq = comm.Irecv([adj_rproc, MPI.INT], source=rproc, tag=rank)
        sreq.wait()
        rreq.wait()

    local_num_rows = rows_per_proc[rank]
    rproc_num_rows = rows_per_proc[rproc]

    row_offset = sum(rows_per_proc[:rproc])

    for k in range(0, rproc_num_rows):
        for i in range(0, local_num_rows):
            for j in range(0, n):
                adj_local[i, j] = min(
                    adj_local[i, j], adj_local[i, k + row_offset] + adj_rproc[k, j])


# Closeness Centrality Calculation
Cr = np.zeros(rows_per_proc[rank])

# iterate through all rows in one processor
for i in range(rows_per_proc[rank]):

    # sum up each row except the self node to get the total distance
    total_dist = sum(adj_local[i, :]) - adj_local[i, i]
    dist = total_dist / (n - 1)
    centrality = 1 / dist

    Cr[i] = centrality

# Gather Centrality from All Processors
C = comm.gather(Cr, root=0)

# Result Output
if rank == 0:
    sorted_C = C.sort
    print("----------------------------------------------------")
    print(sorted_C[:30])
