# -*- coding: utf-8 -*-
"""
Created on Fri May 13 10:37:26 2022

@author: Ariel
"""
import enum
from mpi4py import MPI
import networkx as nx
import numpy as np
import time
import sys


def produce_report(_list, time):
    report = ''
    
    dic = dict()
    num_cols = len(_list[0])
    for i, row in enumerate(_list):
        for j, val in enumerate(row):
            dic[j + (num_cols * i)] = _list[i][j]
    print(dic)
    for node, value in dic.items():
        report += f"node {node}: {value}\n"
        
    with open('output.txt', 'w', encoding="utf-8") as file:
        file.write(report)
            
    top_keys = sorted(dic, key=dic.get, reverse=True)[:5]

    print("\nTop five nodes: ", end='')

    total = 0
    for i, key in enumerate(top_keys):
        if i == len(top_keys) - 1:
            total += dic[key]
            print(f'{key}:{dic[key]}', end='')
        else:
            total += dic[key]
            print(f'{key}:{dic[key]}, ', end='')

    print(f'\nAverage of top five is: {total / len(top_keys)}')
    print(f'\nExecution time: {time}')


args = sys.argv
edge_list_file = args[1] if len(args) > 1 else "facebook_combined.txt"

# Communication Creation
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# p = 8
# If rank 0 read data and distribute submatrix to other processors
if rank == 0:
    # @_list:        list containing centrality scores for network nodes
    start_time = time.time()

    # Getting info out of txt file using edgelist function
    G = nx.read_edgelist(edge_list_file,
                         create_using=nx.DiGraph(), nodetype=int)

    # Making adjacency matrix
    A = nx.to_numpy_array(G, nonedge=8000, dtype='i')
    # Making self node values = 0
    A[np.diag_indices_from(A)] = 0
    n = A.shape[0]

    # Rows per processor
    rows_per_proc = [int(n / size)+1 if i < n %
                     size else int(n / size) for i in range(size)]

    splitmatrix = np.array_split(A, size, axis=0)

else:
    splitmatrix = None
    rows_per_proc = None
    n = 0


# Scatter and broadcast data to processors
adj_local = comm.scatter(splitmatrix, root=0)
rows_per_proc = comm.bcast(rows_per_proc, root=0)
n = comm.bcast(n, root=0)


# Print what each processors data is
# print('Processor {} has data:'.format(rank), adj_local, "\nn=", n)


# Floyd-Warshall Algorithm
for t in range(0, size+size):
    # print("Task:{}; Processor {}\n".format(t, rank))
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

# print(rank, adj_local)

# Closeness Centrality Calculation
Cr = np.zeros(rows_per_proc[rank])

# iterate through all rows in one processor
for i in range(rows_per_proc[rank]):

    # sum up each row except the self node to get the total distance
    total_dist = sum(adj_local[i, :])
    dist = total_dist / (n - 1)
    centrality = 1 / dist

    Cr[i] = centrality

# Gather Centrality from All Processors
C = comm.gather(Cr, root=0)

# Result Output
if rank == 0:
    print(C)
    execution_time = time.time() - start_time
    produce_report(C, execution_time)